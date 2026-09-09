#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "DeepFilterNetProcessor.h"
#include "InferenceWorker.h"
#include "Resampler.hpp"
#include <algorithm>
#include <atomic>
#include <vector>
#include <memory>

class SimpleFifo {
public:
    void setSize(int size) { 
        buffer.resize(size, 0.0f); 
        writePos = 0; readPos = 0; samplesInFifo = 0; 
    }
    
    void push(const float* data, int numSamples) {
        // A default-constructed FIFO has an empty buffer, so `% buffer.size()`
        // below is a division by zero and `buffer[writePos]` writes out of
        // bounds. reset() could reach exactly that state after a refused
        // prepareToPlay; a review probe crashed with exit 139.
        if (buffer.empty() || numSamples <= 0) { jassertfalse; return; }
        for (int i = 0; i < numSamples; ++i) {
            buffer[writePos] = data[i];
            writePos = (writePos + 1) % buffer.size();
        }
        samplesInFifo += numSamples;
        if (samplesInFifo > buffer.size()) samplesInFifo = buffer.size();
    }
    
    void peek(float* dest, int numSamples) {
        int tempRead = readPos;
        for (int i = 0; i < numSamples; ++i) {
            dest[i] = buffer[tempRead];
            tempRead = (tempRead + 1) % buffer.size();
        }
    }
    
    void discard(int numSamples) {
        readPos = (readPos + numSamples) % buffer.size();
        samplesInFifo -= numSamples;
    }
    
    int getAvailable() const { return samplesInFifo; }

    void clear() {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        writePos = 0; readPos = 0; samplesInFifo = 0;
    }

private:
    std::vector<float> buffer;
    int writePos = 0;
    int readPos = 0;
    int samplesInFifo = 0;
};

class AltDenoiserProcessor : public juce::AudioProcessor {
public:
    AltDenoiserProcessor();
    ~AltDenoiserProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    // M12: without this the base class accepts anything, and pluginval confirmed
    // the plugin was advertising Mono through 7.1 Surround on both buses. On a
    // 5.1 instantiation channels 2-5 passed through un-denoised and undelayed
    // while the host shifted the track by the reported latency.
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    // M2: hosts call reset() on a transport locate without re-preparing. Without
    // it, up to one hop of audio from the previous playhead position stayed in
    // the FIFOs and was replayed at the new position.
    void reset() override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Alt Denoiser"; }
    
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    // M5: was 0.0, which tells the host it may stop calling processBlock the
    // moment input ends, truncating the tail of an offline bounce. The pipeline
    // holds the reported latency (1920 samples at 48 kHz, and it scales with the
    // rate, so 40 ms at any rate) before anything reaches the output.
    double getTailLengthSeconds() const override { return 1920.0 / 48000.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Default"; }
    void changeProgramName(int, const juce::String&) override {}

    /** L7: bumped whenever the saved shape changes incompatibly. A state
        written by an OLDER build carries no attribute at all, reads as 0, and
        still loads, so backwards compatibility is preserved. The version exists
        for the forward direction: a state from a NEWER build is refused whole
        rather than half-applied.
    */
    static constexpr int kStateSchemaVersion = 1;

    void getStateInformation(juce::MemoryBlock& destData) override
    {
        auto state = apvts.copyState();
        std::unique_ptr<juce::XmlElement> xml(state.createXml());

        // No null check on xml: ValueTree::createXml() returns null only for an
        // invalid tree, and apvts.copyState() is always valid, so the branch
        // would be unreachable and could never be mutation-tested. Writing a
        // zero-length block would also be worse than crashing, since a host
        // cannot distinguish it from "this plugin has no state".
        xml->setAttribute("schemaVersion", kStateSchemaVersion);
        copyXmlToBinary(*xml, destData);
    }

    void setStateInformation(const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

        if (xmlState.get() != nullptr)
        {
            if (xmlState->hasTagName(apvts.state.getType()))
            {
                // Absent attribute reads 0, which is <= current, so states from
                // builds predating this check still load.
                if (xmlState->getIntAttribute("schemaVersion", 0) > kStateSchemaVersion)
                {
                    jassertfalse;   // written by a newer build; keep defaults
                    return;
                }

                apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
            }
        }
    }

    /** Running peak that the editor consumes and resets.

        M10/L4: the previous scheme applied a one-pole EMA to channel 0's RMS
        once per block, so the time constant swung 32x with buffer size (1.9 ms
        at 64 samples, 61.6 ms at 2048) and at small buffers roughly 92% of
        blocks were never observed by the 60 Hz UI. A running max drops nothing
        at any buffer size, because the UI's window is exactly one frame.

        Peak rather than RMS is also what makes the meter scale honest: it is
        drawn with a red band above 0 dBFS that an RMS value can never reach,
        since a full-scale sine is -3.01 dB RMS.
    */
    class PeakProbe
    {
    public:
        /** Audio thread. */
        void push(float blockPeak) noexcept
        {
            float prev = value.load(std::memory_order_relaxed);
            // CAS rather than load/store: the UI consumer can exchange between
            // a plain read and write, which would silently drop a peak.
            while (blockPeak > prev
                   && ! value.compare_exchange_weak(prev, blockPeak,
                                                    std::memory_order_relaxed))
            {
            }
        }

        /** UI thread. Returns the peak since the last call and clears it. */
        float takeAndReset() noexcept { return value.exchange(0.0f, std::memory_order_acquire); }

    private:
        std::atomic<float> value { 0.0f };
    };

    PeakProbe inputLevel;
    PeakProbe outputLevel;
    juce::AudioProcessorValueTreeState apvts;

private:
    std::unique_ptr<DeepFilterNetProcessor> dfProcessor;

    // M1: owns the model and runs inference off the audio callback.
    InferenceWorker worker;

    SimpleFifo inputFifo;
    SimpleFifo outputFifo;
    std::vector<float> tempInputFrame;
    std::vector<float> tempOutputFrame;
    // H3: written from the prepareToPlay thread and read from the audio thread.
    // A plain bool here is a data race; the underlying DFState pointer has the
    // same problem and is only fully resolved by moving inference to a worker
    // that owns it exclusively (M1).
    std::atomic<bool> modelLoaded { false };
    float lastAttenLim = -1.0f;   // sentinel: forces the first block to apply the real value
    int preparedBlockSize = 0;    // C5: what the resample buffers were sized for

    std::unique_ptr<Resampler<1, 1>> resamplerHandler;
    std::vector<float> resampleInBuffer;
    std::vector<float> resampleOutBuffer;
    std::vector<float> monoBuffer;   // H4: host-rate mono sum fed to the model
    int modelFrameLength = 480;      // M3: hop size reported by the loaded model
    std::atomic<float>* attenParam = nullptr;   // L5: cached, not looked up per block

    // M1: latency-aligned dry signal, used when the worker has not produced a
    // hop in time. Emitting the delayed dry input is a dropout the listener may
    // not notice; emitting digital silence is a click they always will.
    SimpleFifo dryDelay;
    std::vector<float> dryScratch;

    // M1: monotonic hop counter, and the barrier below which collected output is
    // discarded. Both are touched only by the audio thread and reset().
    unsigned nextInputSequence = 0;
    unsigned acceptFromSequence = 0;

public:
    /** Diagnostics for the M1 worker path. Read from any thread.

        fallbackSamples counts output samples the worker failed to deliver in
        time, which were filled from the latency-aligned dry signal. In steady
        state this should stop increasing entirely: one hop of cushion is 10 ms
        at 48 kHz against a measured 0.32 ms of inference per hop.
    */
    std::atomic<unsigned> fallbackSamples { 0 };
    int getWorkerDroppedFrames() const { return worker.getDroppedFrames(); }

    /** L6: the attenuation limit actually pushed into the model so far,
        which trails the parameter while a change is slewing.
    */
    float getAppliedAttenLim() const { return worker.getAppliedAttenLim(); }

private:

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AltDenoiserProcessor)
};