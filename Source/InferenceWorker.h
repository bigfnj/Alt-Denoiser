#pragma once

#include <juce_core/juce_core.h>

#include <array>
#include <atomic>
#include <cstring>
#include <vector>

#include "DeepFilterNetProcessor.h"

/**
    M1: moves DeepFilterNet inference off the audio callback.

    Before this, processBlock ran the model inline: a 2048-sample block at 48 kHz
    performed four to five back-to-back inference calls inside one callback, with
    no worker and no deadline fallback. When it missed, the output FIFO underran
    and digital silence was spliced into the signal.

    The transport carries whole model hops only, so both queues are fixed-size
    and never allocate. Sub-hop accumulation stays on the audio thread in
    buffers it owns privately.

    This also completes H3: the DFState is now owned exclusively by the worker,
    which is the only thing that ever calls into libDF. Nothing else can observe
    a half-published or freed model pointer.

    No extra latency is introduced. The output FIFO is already primed with one
    hop (H5/H6), which is 10 ms at 48 kHz of slack against a measured inference
    cost of 0.32 ms per hop, so the existing cushion serves as the runway.
*/

//==============================================================================
/** Single-producer, single-consumer queue of fixed-size audio frames.

    Monotonic counters rather than wrapped indices, so full and empty are
    unambiguous without a spare slot. Frame storage is allocated once by
    prepare(); push and pop only memcpy.
*/
template <int Capacity>
class FrameQueue
{
public:
    static_assert (Capacity > 1, "need room for at least one in-flight frame");

    /** Allocates frame storage. Not real-time safe; call before starting. */
    void prepare (int frameSize)
    {
        for (auto& f : frames)
            f.assign ((size_t) frameSize, 0.0f);
        size = frameSize;
        reset();
    }

    void reset()
    {
        readCount.store (0, std::memory_order_relaxed);
        writeCount.store (0, std::memory_order_relaxed);
    }

    /** Producer side. Returns false when full, dropping nothing.

        Each frame carries the sequence number of the input hop it derives from,
        so a consumer can identify and discard work that predates a reset
        without stopping the worker or waiting on it.
    */
    bool push (unsigned sequence, const float* source)
    {
        const auto w = writeCount.load (std::memory_order_relaxed);
        const auto r = readCount.load (std::memory_order_acquire);
        if (w - r >= (unsigned) Capacity)
            return false;

        const auto slot = w % Capacity;
        sequences[slot] = sequence;
        std::memcpy (frames[slot].data(), source, sizeof (float) * (size_t) size);
        writeCount.store (w + 1, std::memory_order_release);
        return true;
    }

    /** Consumer side. Returns false when empty. */
    bool pop (unsigned& sequence, float* destination)
    {
        const auto r = readCount.load (std::memory_order_relaxed);
        const auto w = writeCount.load (std::memory_order_acquire);
        if (w == r)
            return false;

        const auto slot = r % Capacity;
        sequence = sequences[slot];
        std::memcpy (destination, frames[slot].data(), sizeof (float) * (size_t) size);
        readCount.store (r + 1, std::memory_order_release);
        return true;
    }

    int getNumReady() const
    {
        return (int) (writeCount.load (std::memory_order_acquire)
                      - readCount.load (std::memory_order_acquire));
    }

private:
    std::array<std::vector<float>, Capacity> frames;
    std::array<unsigned, Capacity> sequences {};
    int size = 0;
    std::atomic<unsigned> readCount { 0 };
    std::atomic<unsigned> writeCount { 0 };
};

//==============================================================================
class InferenceWorker : private juce::Thread
{
public:
    InferenceWorker() : juce::Thread ("AltDenoiser inference") {}
    ~InferenceWorker() override { stop(); }

    /** Takes ownership of the model for the worker's lifetime. Not RT safe. */
    void start (DeepFilterNetProcessor* modelToOwn, int frameSize)
    {
        stop();

        model = modelToOwn;
        size = frameSize;
        scratchIn.assign ((size_t) frameSize, 0.0f);
        scratchOut.assign ((size_t) frameSize, 0.0f);
        inbound.prepare (frameSize);
        outbound.prepare (frameSize);
        droppedFrames.store (0, std::memory_order_relaxed);
        framesAccepted.store (0, std::memory_order_relaxed);
        framesCollected.store (0, std::memory_order_relaxed);
        framesDroppedAfterAccept.store (0, std::memory_order_relaxed);

        if (model != nullptr && size > 0)
        {
            // Above normal, below the host's audio thread: this must keep up
            // with real time but must never compete with the callback itself.
            startThread (juce::Thread::Priority::high);
        }
    }

    void stop()
    {
        if (isThreadRunning())
        {
            signalThreadShouldExit();
            wakeUp.signal();
            stopThread (2000);
        }
        model = nullptr;
    }

    bool isActive() const { return isThreadRunning(); }

    /** Audio thread. Hands one hop to the worker and wakes it. */
    bool submit (unsigned sequence, const float* frame)
    {
        const bool accepted = inbound.push (sequence, frame);
        if (accepted)
            framesAccepted.fetch_add (1, std::memory_order_release);
        else
            droppedFrames.fetch_add (1, std::memory_order_relaxed);

        // Signalling an OS event rather than letting the worker poll matters on
        // Windows, where a timed wait rounds up to the system timer resolution.
        // Measured on this machine at default resolution, a 100 us sleep
        // actually takes 15.5 ms, which would exceed the entire cushion.
        wakeUp.signal();
        return accepted;
    }

    /** Audio thread. Retrieves one finished hop and the input sequence it came
        from, if the worker has produced one.
    */
    bool collect (unsigned& sequence, float* frame)
    {
        if (! outbound.pop (sequence, frame))
            return false;
        framesCollected.fetch_add (1, std::memory_order_relaxed);
        return true;
    }

    /** Frames handed over but not yet collected, INCLUDING one being inferred
        right now.

        Queue occupancy alone is not enough: between the worker popping an input
        and pushing its output, both queues read empty, so a caller waiting on
        "is anything outstanding" would exit early and fall back unnecessarily.
        Counting accepted minus collected closes that window.
    */
    int getInFlight() const
    {
        const auto accepted  = framesAccepted.load (std::memory_order_acquire);
        const auto collected = framesCollected.load (std::memory_order_acquire);
        const auto dropped   = framesDroppedAfterAccept.load (std::memory_order_acquire);
        return (int) (accepted - collected - dropped);
    }

    /** Blocks until the worker delivers a hop, or the timeout expires.

        Only for non-realtime rendering. An offline bounce calls processBlock as
        fast as the CPU allows, so the cushion the realtime path relies on never
        gets any wall-clock time to fill: measured before this existed, an
        offline render fell back to dry for 20000 samples and dropped 25 frames.
        Blocking is correct there and forbidden in the realtime path.

        Returns false on timeout, which the caller must treat as a failure and
        degrade rather than hang.
    */
    bool waitForOutput (int timeoutMs)
    {
        if (! isThreadRunning())
            return false;

        // Loops against a deadline rather than trusting a single wait. The event
        // is auto-reset and the worker signals once per frame, so a signal left
        // over from frames already collected makes wait() return immediately
        // with nothing ready. Treating that as failure caused a spurious dry
        // fallback and made the measured latency jump between 483 and 675
        // samples from run to run.
        const auto deadline = juce::Time::getMillisecondCounter() + (juce::uint32) timeoutMs;
        for (;;)
        {
            if (outbound.getNumReady() > 0)
                return true;

            const auto now = juce::Time::getMillisecondCounter();
            if (now >= deadline)
                return false;

            outputReady.wait ((int) (deadline - now));
        }
    }

    int getPendingInput()  const { return inbound.getNumReady(); }
    int getReadyOutput()   const { return outbound.getNumReady(); }
    int getDroppedFrames() const { return (int) droppedFrames.load (std::memory_order_relaxed); }

private:
    void run() override
    {
        while (! threadShouldExit())
        {
            bool didWork = false;

            unsigned sequence = 0;
            while (inbound.pop (sequence, scratchIn.data()))
            {
                if (threadShouldExit())
                    return;

                model->processFrame (scratchIn.data(), scratchOut.data());

                // A full outbound queue means the audio thread has stopped
                // collecting, which happens while the transport is stopped.
                // Dropping is correct there; the queue is reset on the next
                // prepare or reset.
                if (! outbound.push (sequence, scratchOut.data()))
                {
                    droppedFrames.fetch_add (1, std::memory_order_relaxed);
                    // Accepted but never collectable: account for it so
                    // getInFlight() cannot stay permanently positive.
                    framesDroppedAfterAccept.fetch_add (1, std::memory_order_release);
                }

                outputReady.signal();

                didWork = true;
            }

            if (! didWork)
                wakeUp.wait (20);
        }
    }

    DeepFilterNetProcessor* model = nullptr;
    int size = 0;

    // Capacity is generous relative to the one hop of cushion the pipeline
    // actually relies on, so a scheduling hiccup queues work rather than losing
    // it. 16 hops is 160 ms at 48 kHz and costs about 30 KB per queue.
    static constexpr int kCapacity = 16;
    FrameQueue<kCapacity> inbound;
    FrameQueue<kCapacity> outbound;

    std::vector<float> scratchIn, scratchOut;
    juce::WaitableEvent wakeUp { false };        // auto-reset
    juce::WaitableEvent outputReady { false };   // auto-reset, for offline waits
    std::atomic<unsigned> droppedFrames { 0 };
    std::atomic<unsigned> framesAccepted { 0 };
    std::atomic<unsigned> framesCollected { 0 };
    std::atomic<unsigned> framesDroppedAfterAccept { 0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (InferenceWorker)
};
