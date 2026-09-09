#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "DeepFilterNetProcessor.h"
#include "InferenceWorker.h"
#include "Resampler.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>
#include <memory>

/** Single-threaded ring buffer for sub-hop accumulation.

    M8: every operation is now bounds-checked and reports refusal, and the class
    caches its own capacity rather than consulting buffer.size().

    The original had three ways to corrupt itself. `samplesInFifo > buffer.size()`
    compared a signed int against a size_t, so the usual arithmetic conversions
    turned any NEGATIVE count into a huge unsigned value, which compared greater
    and clamped the FIFO to "completely full". `discard` had no floor, so it was
    the thing that could produce that negative count. And every wrap used
    `% buffer.size()`, which is a division by zero before setSize has run: a
    review probe reached exactly that state through reset() after a refused
    prepareToPlay and crashed with exit 139.

    None of those were reachable through the call sites, which all guard
    correctly, but the guards lived in the callers rather than the class. They
    live here now, and the counters record whether they ever fired.
*/
class SimpleFifo {
public:
    void setSize(int size) {
        capacity = juce::jmax(0, size);
        buffer.assign((size_t) capacity, 0.0f);
        writePos = 0; readPos = 0; samplesInFifo = 0;
        overflows = 0; underflows = 0;
    }

    /** Returns false and stores nothing if the request does not fit. */
    bool push(const float* data, int numSamples) {
        if (capacity <= 0 || numSamples <= 0 || data == nullptr) { jassertfalse; return false; }
        if (numSamples > getFreeSpace()) { jassertfalse; ++overflows; return false; }

        for (int i = 0; i < numSamples; ++i) {
            buffer[(size_t) writePos] = data[i];
            if (++writePos == capacity) writePos = 0;
        }
        samplesInFifo += numSamples;
        return true;
    }

    /** Pushes numSamples of silence. Used by the three priming sites. */
    bool pushSilence(int numSamples) {
        if (capacity <= 0 || numSamples <= 0) { jassertfalse; return false; }
        if (numSamples > getFreeSpace()) { jassertfalse; ++overflows; return false; }

        for (int i = 0; i < numSamples; ++i) {
            buffer[(size_t) writePos] = 0.0f;
            if (++writePos == capacity) writePos = 0;
        }
        samplesInFifo += numSamples;
        return true;
    }

    /** Returns false and writes nothing if fewer than numSamples are available. */
    bool peek(float* dest, int numSamples) const {
        if (capacity <= 0 || numSamples <= 0 || dest == nullptr) { jassertfalse; return false; }
        if (numSamples > samplesInFifo) { jassertfalse; return false; }

        int tempRead = readPos;
        for (int i = 0; i < numSamples; ++i) {
            dest[i] = buffer[(size_t) tempRead];
            if (++tempRead == capacity) tempRead = 0;
        }
        return true;
    }

    /** Returns false and advances nothing if fewer than numSamples are available,
        so the count can never go negative.
    */
    bool discard(int numSamples) {
        if (capacity <= 0 || numSamples <= 0) { jassertfalse; return false; }
        if (numSamples > samplesInFifo) { jassertfalse; ++underflows; return false; }

        readPos = (readPos + numSamples) % capacity;
        samplesInFifo -= numSamples;
        return true;
    }

    int getAvailable() const noexcept { return samplesInFifo; }
    int getCapacity()  const noexcept { return capacity; }
    int getFreeSpace() const noexcept { return capacity - samplesInFifo; }

    // Diagnostics: non-zero means a caller's own guard failed.
    int getOverflows()  const noexcept { return overflows; }
    int getUnderflows() const noexcept { return underflows; }

    void clear() {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        writePos = 0; readPos = 0; samplesInFifo = 0;
    }

private:
    std::vector<float> buffer;
    int capacity = 0;
    int writePos = 0;
    int readPos = 0;
    int samplesInFifo = 0;
    int overflows = 0;
    int underflows = 0;
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

    /** M7: a real bypass parameter, so no wrapper ever synthesises one.

        JUCE's default processBlockBypassed is a bare passthrough carrying
        `jassert (getLatencySamples() == 0)`. This plugin reports 1920, so the
        default would land a bypassed track 40 ms EARLY against the host's
        compensation and break on the assertion in a Debug build.

        Declaring the parameter also means APVTS saves it. A wrapper-synthesised
        bypass is stored in VST3's private state and not stored at all by AU or
        LV2, so it would not survive a session reload on two of three formats.
    */
    juce::AudioParameterBool* getBypassParameter() const override { return bypassParam; }
    void processBlockBypassed(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Alt Denoiser"; }
    
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    // M5: was 0.0, which tells the host it may stop calling processBlock the
    // moment input ends, truncating the tail of an offline bounce. The pipeline
    // holds the whole reported latency before anything reaches the output.
    //
    // 7a: set by prepareToPlay from the DERIVED latency and the actual rate, so
    // it cannot disagree with getLatencySamples(). It used to be a second
    // hardcoded 1920/48000, which meant the failed-load path reported zero
    // latency and a 40 ms tail simultaneously. Zero before the first prepare is
    // correct: nothing is loaded, so nothing is held.
    double getTailLengthSeconds() const override
    {
        return tailLengthSeconds.load (std::memory_order_relaxed);
    }

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
    /** Maps the attenuation-limit parameter, in dB, to the dry proportion of the
        output mix.

        This is libDF's own formula (tract.rs): a limit of 100 dB or more means
        "no limit", i.e. fully enhanced; below 0.01 dB means no reduction at all,
        i.e. fully dry; in between the dry share is 10^(-db/20). Reproducing it
        here rather than inside the model is what makes the limit smoothable per
        sample, because the plugin already holds the aligned dry signal.
    */
    static float attenLimitToDryMix(float db) noexcept
    {
        if (! std::isfinite(db)) return 0.0f;    // treat nonsense as "no limit"
        const float lim = std::abs(db);
        if (lim >= 100.0f) return 0.0f;          // fully wet
        if (lim < 0.01f)   return 1.0f;          // fully dry
        return std::pow(10.0f, -lim / 20.0f);
    }

    /** Dry proportion, smoothed across the block. Ramp length is in SECONDS, so
        it is identical at every sample rate and block size, unlike the
        hop-quantised worker slew this replaced.
    */
    juce::SmoothedValue<float> attenMix;
    static constexpr double kAttenRampSeconds = 0.05;
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
    int primedDryDelay = 0;   // what prepareToPlay primed, so reset() matches it

    // 7a: reported latency in the 48 kHz domain, derived from the loaded model's
    // geometry. The single source for setLatencySamples, the dry delay priming
    // and the tail length.
    int derivedLatency48k = 0;
    std::atomic<double> tailLengthSeconds { 0.0 };

    // M7: the bypass path needs the ORIGINAL input, per channel, at HOST rate,
    // delayed by exactly the reported latency. dryDelay cannot serve: it is mono
    // and lives in the 48 kHz domain, so bypassing through it would collapse the
    // channels and pass the signal through the resampler twice.
    juce::AudioParameterBool* bypassParam = nullptr;

    // 7b: read in prepareToPlay only. Never touched from processBlock.
    juce::AudioParameterChoice* modelParam = nullptr;
    std::vector<SimpleFifo> bypassDelay;
    std::vector<float> bypassScratch;
    juce::SmoothedValue<float> bypassMix;          // 0 = processed, 1 = bypassed
    static constexpr double kBypassRampSeconds = 0.01;
    bool forcedBypass = false;                     // set by processBlockBypassed

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

    /** L6: the dry proportion currently being mixed, 0 = fully enhanced and
        1 = fully dry. Trails the parameter only for the ramp length.
    */
    float getAttenDryMix() const { return attenMix.getCurrentValue(); }

private:

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AltDenoiserProcessor)
};