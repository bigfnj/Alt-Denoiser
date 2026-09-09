/*
    Offline render harness for AltDenoiserProcessor.

    pluginval passes this plugin at strictness level 5 with every defect in
    BACKLOG.md present, because it checks that the plugin is a well-formed VST3
    rather than that it processes audio correctly. These tests check the audio.

    Each test names the backlog item it pins down. Tests start life as
    CHARACTERISATION tests, declared Expect::FailUntilFixed so the harness stays
    green while a known defect is still known, and are flipped to Expect::Pass
    once the item is fixed, at which point they become regression tests. All of
    them currently pass; the mechanism is kept for the items still open.

    Run:  AltDenoiserTests[.exe] [blockSize] [sampleRate]
    Exit: 0 if every test matched its expectation, 1 otherwise. A test that
          starts passing without its Expect being updated is also reported, so
          a silently-fixed defect cannot go unnoticed.
*/

#include <juce_audio_processors/juce_audio_processors.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "DeepFilterNetProcessor.h"
#include "BinaryData.h"

namespace
{

double kSampleRate = 48000.0;
int    kBlockSize  = 512;   // both overridable from argv

//==============================================================================
// Minimal result plumbing. Each test declares whether it currently passes, so
// the harness exits 0 while known-broken behaviour is still known-broken.

enum class Expect { Pass, FailUntilFixed };

struct Outcome
{
    std::string name;
    std::string backlogId;
    bool        passed = false;
    Expect      expectation = Expect::Pass;
    std::string detail;
};

std::vector<Outcome> results;

void record (std::string name, std::string backlogId, bool passed,
             Expect expectation, std::string detail)
{
    results.push_back ({ std::move (name), std::move (backlogId), passed,
                         expectation, std::move (detail) });
}

//==============================================================================
// Harness helpers

/** Drives the processor for a number of blocks, calling a generator to fill
    each input block and a consumer to observe each output block.
*/
template <typename FillFn, typename ObserveFn>
void render (AltDenoiserProcessor& proc, int numBlocks, FillFn fill, ObserveFn observe)
{
    juce::AudioBuffer<float> buffer (2, kBlockSize);
    juce::MidiBuffer midi;

    for (int block = 0; block < numBlocks; ++block)
    {
        buffer.clear();
        fill (buffer, block);
        midi.clear();
        proc.processBlock (buffer, midi);
        observe (buffer, block);
    }
}

void prepare (AltDenoiserProcessor& proc)
{
    // This harness renders far faster than realtime, which is exactly what a
    // host does during an offline bounce. Declaring it makes the plugin take
    // its non-realtime path and wait for the inference worker instead of
    // falling back to dry, which is the correct behaviour for a render and is
    // also the only way these correctness assertions mean anything.
    proc.setNonRealtime (true);
    proc.setPlayConfigDetails (2, 2, kSampleRate, kBlockSize);
    proc.prepareToPlay (kSampleRate, kBlockSize);
}

/** Sets the attenuation-limit parameter through the host-facing API, the same
    path a DAW or the editor would use.
*/
void setAttenuation (AltDenoiserProcessor& proc, float valueDb)
{
    auto* param = proc.apvts.getParameter ("atten_lim");
    jassert (param != nullptr);
    param->setValueNotifyingHost (param->convertTo0to1 (valueDb));
}

void fillSine (juce::AudioBuffer<float>& buffer, int block, float freqL, float freqR)
{
    const auto startSample = (juce::int64) block * kBlockSize;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        const auto t = (double) (startSample + i) / kSampleRate;
        buffer.setSample (0, i, 0.25f * (float) std::sin (juce::MathConstants<double>::twoPi * freqL * t));
        if (buffer.getNumChannels() > 1)
            buffer.setSample (1, i, 0.25f * (float) std::sin (juce::MathConstants<double>::twoPi * freqR * t));
    }
}

/** Longest run of exactly-zero samples in a channel. Digital silence spliced
    into a continuous tone is the signature of an output-FIFO underrun.
*/
int longestZeroRun (const std::vector<float>& samples, int fromIndex = 0)
{
    int longest = 0, current = 0;
    for (size_t i = (size_t) fromIndex; i < samples.size(); ++i)
    {
        if (samples[i] == 0.0f) { ++current; longest = std::max (longest, current); }
        else current = 0;
    }
    return longest;
}

int countZeros (const std::vector<float>& samples, int fromIndex, int toIndex)
{
    int n = 0;
    for (int i = fromIndex; i < toIndex && i < (int) samples.size(); ++i)
        if (samples[i] == 0.0f) ++n;
    return n;
}

//==============================================================================
// T1 / H5 - output FIFO underrun splices digital silence into the signal.

void testStartupZeroSplice()
{
    AltDenoiserProcessor proc;
    prepare (proc);
    setAttenuation (proc, 0.0f);          // spectral passthrough

    std::vector<float> out;
    out.reserve (60 * kBlockSize);
    render (proc, 60,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
            [&out] (const juce::AudioBuffer<float>& b, int)
            {
                const auto* p = b.getReadPointer (0);
                out.insert (out.end(), p, p + b.getNumSamples());
            });

    // The reported latency is legitimately silent; only look past it.
    const int latency = proc.getLatencySamples();
    const int zerosAfterLatency = countZeros (out, latency, (int) out.size());
    const int worstRun = longestZeroRun (out, latency);

    // Before the fix this was geometry-dependent: clean at 480 and 960, and
    // 448 spliced samples at 512. Priming the output FIFO with one hop removes
    // the dependence, so every geometry is now expected to be clean.
    // Asserts the longest RUN, not the count. The signal past the latency is now
    // the dry sine (attenuation 0 is fully dry since L6), which crosses zero
    // legitimately, so isolated zeros are expected and meaningless. The defect
    // this guards against produced runs of 32.
    const bool passed = (worstRun <= 4);
    record ("startup zero-splice", "H5", passed, Expect::Pass,
            "reported latency " + std::to_string (latency)
              + ", zero samples after it " + std::to_string (zerosAfterLatency)
              + ", longest zero run " + std::to_string (worstRun));
}

//==============================================================================
// T2 / H4 - the right input channel was discarded outright.
//
// The model is mono, so the output is legitimately the same on both channels.
// The defect was that only channel 0 was ever READ, so anything present only in
// the right channel vanished. This feeds silence left and a tone right: if the
// right input reaches the model at all, the output is non-silent.

void testRightChannelReachesTheModel()
{
    AltDenoiserProcessor proc;
    prepare (proc);
    setAttenuation (proc, 0.0f);          // spectral passthrough

    double inputRightPeak = 0.0;
    double outputPeak = 0.0;
    render (proc, 40,
            [&inputRightPeak] (juce::AudioBuffer<float>& b, int blk)
            {
                fillSine (b, blk, 440.0f, 440.0f);
                b.clear (0, 0, b.getNumSamples());          // silence on the left
                for (int i = 0; i < b.getNumSamples(); ++i)
                    inputRightPeak = std::max (inputRightPeak, (double) std::abs (b.getSample (1, i)));
            },
            [&outputPeak] (const juce::AudioBuffer<float>& b, int blk)
            {
                if (blk < 15) return;     // let the pipeline fill
                for (int i = 0; i < b.getNumSamples(); ++i)
                    outputPeak = std::max (outputPeak, (double) std::abs (b.getSample (0, i)));
            });

    // Silent output would ALSO result from the harness never putting signal on
    // the right channel, so require a real input peak before believing it.
    const bool inputWasRightOnly = (inputRightPeak > 0.1);
    const bool passed = inputWasRightOnly && (outputPeak > 0.01);
    record ("right channel reaches the model", "H4", passed, Expect::Pass,
            "input right peak " + std::to_string (inputRightPeak)
              + (inputWasRightOnly ? "" : "  <-- HARNESS BUG: no signal on the right")
              + ", output peak " + std::to_string (outputPeak)
              + " (0 means the right input was discarded)");
}

//==============================================================================
// T3 / H2 - lastAttenLim is never reset, so the knob stops reaching the model
// after any re-prepare.

void testAttenuationSurvivesReprepare()
{
    AltDenoiserProcessor proc;

    auto renderRms = [&proc]()
    {
        double sumSquares = 0.0;
        int counted = 0;
        render (proc, 40,
                [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
                [&] (const juce::AudioBuffer<float>& b, int blk)
                {
                    if (blk < 15) return;
                    for (int i = 0; i < b.getNumSamples(); ++i)
                    {
                        const auto s = (double) b.getSample (0, i);
                        sumSquares += s * s;
                        ++counted;
                    }
                });
        return counted > 0 ? std::sqrt (sumSquares / counted) : 0.0;
    };

    // Attenuation 0 is a passthrough, so the tone survives at full level.
    prepare (proc);
    setAttenuation (proc, 0.0f);
    const double rmsFirst = renderRms();

    // A buffer-size or sample-rate change re-enters prepareToPlay. The knob has
    // not moved, so the same passthrough level must come back.
    prepare (proc);
    const double rmsAfterReprepare = renderRms();

    const double ratio = rmsFirst > 0.0 ? rmsAfterReprepare / rmsFirst : 0.0;
    const bool passed = (ratio > 0.9 && ratio < 1.1);
    record ("attenuation survives re-prepare", "H2", passed, Expect::Pass,
            "RMS before " + std::to_string (rmsFirst)
              + ", after " + std::to_string (rmsAfterReprepare)
              + ", ratio " + std::to_string (ratio)
              + " (a drop means the model reverted to 100 dB while the knob still reads 0)");
}

//==============================================================================
// T4 / H6 - reported latency versus the delay actually introduced.

void testReportedLatencyMatchesMeasured()
{
    AltDenoiserProcessor proc;
    prepare (proc);
    setAttenuation (proc, 0.0f);

    const int reported = proc.getLatencySamples();
    const int burstBlock = 20;
    const int burstStart = burstBlock * kBlockSize;

    std::vector<float> out;
    out.reserve (60 * kBlockSize);
    render (proc, 60,
            [burstBlock] (juce::AudioBuffer<float>& b, int blk)
            {
                if (blk < burstBlock) return;                   // silence first
                fillSine (b, blk, 1000.0f, 1000.0f);
            },
            [&out] (const juce::AudioBuffer<float>& b, int)
            {
                const auto* p = b.getReadPointer (0);
                out.insert (out.end(), p, p + b.getNumSamples());
            });

    int firstAudible = -1;
    for (int i = burstStart; i < (int) out.size(); ++i)
        if (std::abs (out[(size_t) i]) > 0.01f) { firstAudible = i; break; }

    const int measured = firstAudible >= 0 ? firstAudible - burstStart : -1;

    // At attenuation 0 the plugin is fully dry, so the output is the INPUT
    // delayed by the whole reported latency. That is exactly what a host
    // compensates for, which makes this an end-to-end check of the claim rather
    // than of one component of it.
    //
    // This assertion USED to subtract the model's 1440 samples, because
    // attenuation 0 made the model return its input immediately and the output
    // arrived at 483. That was a real defect the test was encoding: the plugin
    // reported 1920 and delivered at 483, so a host compensating by 1920 played
    // the track 1437 samples EARLY. Making the limit a proper crossfade (L6)
    // fixed it, and the honest expectation is now the reported figure itself.
    const int expected = reported;
    const bool passed = (measured >= 0 && std::abs (measured - expected) <= 32);
    record ("reported latency matches measured", "H6", passed, Expect::Pass,
            "[worker fallback " + std::to_string (proc.fallbackSamples.load())
              + " samples, dropped " + std::to_string (proc.getWorkerDroppedFrames())
              + " frames] reported " + std::to_string (reported)
              + ", measured end-to-end " + std::to_string (measured)
              + ", error " + (measured >= 0 ? std::to_string (measured - expected) : std::string ("n/a")));
}

//==============================================================================
// T5 / C5 - a host delivering a larger block than it declared must not make the
// resampler write past the end of buffers sized in prepareToPlay.

void testOversizedBlockIsRefused()
{
    AltDenoiserProcessor proc;

    // Declare a small block, then hand over a much larger one.
    const int declared = 128;
    const int oversized = declared * 4;
    proc.setPlayConfigDetails (2, 2, kSampleRate, declared);
    proc.prepareToPlay (kSampleRate, declared);
    setAttenuation (proc, 0.0f);

    juce::AudioBuffer<float> buffer (2, oversized);
    juce::MidiBuffer midi;
    std::vector<float> before ((size_t) oversized);

    for (int i = 0; i < oversized; ++i)
    {
        const auto v = 0.25f * (float) std::sin (juce::MathConstants<double>::twoPi * 440.0 * i / kSampleRate);
        buffer.setSample (0, i, v);
        buffer.setSample (1, i, v);
        before[(size_t) i] = v;
    }

    proc.processBlock (buffer, midi);

    // The guard passes audio through untouched rather than corrupting the heap.
    // Without it this call writes past resampleInBuffer; the assertion is that
    // we survive AND that the signal was not mangled.
    double maxDelta = 0.0;
    for (int i = 0; i < oversized; ++i)
        maxDelta = std::max (maxDelta, (double) std::abs (buffer.getSample (0, i) - before[(size_t) i]));

    const bool passed = (maxDelta < 1.0e-6);
    record ("oversized block refused safely", "C5", passed, Expect::Pass,
            "declared " + std::to_string (declared)
              + ", delivered " + std::to_string (oversized)
              + ", max |out-in| = " + std::to_string (maxDelta)
              + " (0 means passed through untouched)");
}

//==============================================================================
// T6 / H7 - an unusable plugin must pass audio through, never mute the track.
//
// The old code called buffer.clear() whenever the model was unavailable, so a
// failure to stage the model silently killed the track with no UI indication.
// This drives processBlock with no prepareToPlay first, which is reachable
// during host scanning and leaves the plugin in exactly that unusable state.

void testUnpreparedPassesAudioThrough()
{
    AltDenoiserProcessor proc;
    proc.setPlayConfigDetails (2, 2, kSampleRate, kBlockSize);
    // deliberately NO prepareToPlay

    juce::AudioBuffer<float> buffer (2, kBlockSize);
    juce::MidiBuffer midi;
    std::vector<float> before ((size_t) kBlockSize);

    for (int i = 0; i < kBlockSize; ++i)
    {
        const auto v = 0.25f * (float) std::sin (juce::MathConstants<double>::twoPi * 440.0 * i / kSampleRate);
        buffer.setSample (0, i, v);
        buffer.setSample (1, i, v);
        before[(size_t) i] = v;
    }

    proc.processBlock (buffer, midi);

    double maxDelta = 0.0, outPeak = 0.0;
    for (int i = 0; i < kBlockSize; ++i)
    {
        const auto out = (double) buffer.getSample (0, i);
        maxDelta = std::max (maxDelta, std::abs (out - before[(size_t) i]));
        outPeak  = std::max (outPeak, std::abs (out));
    }

    const bool passed = (outPeak > 0.01) && (maxDelta < 1.0e-6);
    record ("unprepared passes audio through", "H7", passed, Expect::Pass,
            "output peak " + std::to_string (outPeak)
              + ", max |out-in| = " + std::to_string (maxDelta)
              + " (peak 0 means the track was muted)");
}

//==============================================================================
// T7 / M2 - reset() must not leave stale audio to be replayed.
//
// Hosts call reset() on a transport locate without re-preparing. Anything left
// in the FIFOs would then be emitted at the new playhead position.

void testResetDropsStaleAudio()
{
    AltDenoiserProcessor proc;
    prepare (proc);
    setAttenuation (proc, 0.0f);          // spectral passthrough

    // Fill the pipeline with a loud tone.
    double loudPeak = 0.0;
    render (proc, 30,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
            [&loudPeak] (const juce::AudioBuffer<float>& b, int blk)
            {
                if (blk < 15) return;
                for (int i = 0; i < b.getNumSamples(); ++i)
                    loudPeak = std::max (loudPeak, (double) std::abs (b.getSample (0, i)));
            });

    proc.reset();

    // Now feed silence. Anything audible is a replay of the pre-reset content.
    double leakedPeak = 0.0;
    int leakedSamples = 0;      // how long the leak lasts, not just how loud
    int lastLeakIndex = -1;
    int sampleIndex = 0;
    render (proc, 10,
            [] (juce::AudioBuffer<float>& b, int) { b.clear(); },
            [&] (const juce::AudioBuffer<float>& b, int)
            {
                for (int i = 0; i < b.getNumSamples(); ++i, ++sampleIndex)
                {
                    const auto v = (double) std::abs (b.getSample (0, i));
                    leakedPeak = std::max (leakedPeak, v);
                    if (v > 1.0e-3) { ++leakedSamples; lastLeakIndex = sampleIndex; }
                }
            });

    // Silent output would ALSO result from the pipeline never having carried
    // signal, so require a real pre-reset peak before believing it.
    const bool wasLoud = (loudPeak > 0.1);
    // A handful of samples is the resampler's 4-sample interpolation window,
    // which is unavoidable without resetting it and is inaudible. A leak lasting
    // hundreds of samples is the model's overlap-add and lookahead state being
    // replayed at the new playhead position, which is the actual M2 defect.
    const bool passed = wasLoud && (leakedSamples <= 8);
    record ("reset drops stale audio", "M2", passed, Expect::Pass,
            "pre-reset peak " + std::to_string (loudPeak)
              + (wasLoud ? "" : "  <-- HARNESS BUG: pipeline was never loud")
              + ", leaked " + std::to_string (leakedSamples) + " samples"
              + ", peak " + std::to_string (leakedPeak)
              + ", last at index " + std::to_string (lastLeakIndex));
}

//==============================================================================
// T8 / M1 - the realtime fallback emits dry audio, never digital silence.
//
// This harness renders far faster than realtime, so the realtime path cannot
// keep its cushion filled and WILL fall back. That is the design, not a bug:
// what matters is what it falls back TO. Before M1 the deficit was zero-filled,
// which is an audible click. It now splices the latency-aligned dry signal.

void testRealtimeFallbackIsDryNotSilence()
{
    AltDenoiserProcessor proc;
    proc.setNonRealtime (false);                 // force the realtime path
    proc.setPlayConfigDetails (2, 2, kSampleRate, kBlockSize);
    proc.prepareToPlay (kSampleRate, kBlockSize);
    setAttenuation (proc, 0.0f);

    const int latency = proc.getLatencySamples();
    std::vector<float> out;
    out.reserve (80 * (size_t) kBlockSize);
    render (proc, 80,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
            [&out] (const juce::AudioBuffer<float>& b, int)
            {
                const auto* p = b.getReadPointer (0);
                out.insert (out.end(), p, p + b.getNumSamples());
            });

    // Past the reported latency every sample should carry signal, whether it
    // came from the worker or from the dry fallback.
    const int zeros = countZeros (out, latency, (int) out.size());
    const int worstRun = longestZeroRun (out, latency);
    const auto fellBack = proc.fallbackSamples.load();

    int firstZero = -1, lastZero = -1;
    for (int i = latency; i < (int) out.size(); ++i)
        if (out[(size_t) i] == 0.0f) { if (firstZero < 0) firstZero = i; lastZero = i; }

    // The resampler's 4-tap window can produce a short exact-zero region where a
    // wet run meets a dry splice. A handful of samples is 0.08 ms and inaudible;
    // a run of hundreds would mean the deficit is still being zero-filled, which
    // is the defect. Bound it tightly enough that a regression to zero-filling
    // (which produced 448 samples in runs of 32 before H5) cannot pass.
    const bool passed = (worstRun <= 8 && zeros <= 32);
    record ("realtime fallback is dry not silence", "M1", passed, Expect::Pass,
            "fell back for " + std::to_string (fellBack) + " samples"
              + ", zeros past latency " + std::to_string (zeros)
              + ", longest run " + std::to_string (worstRun)
              + ", first at " + std::to_string (firstZero)
              + ", last at " + std::to_string (lastZero));
}

//==============================================================================
// T9 / N2 - re-preparing while the worker is busy must not free the model from
// under it.
//
// initialize() calls df_free on the live DFState (H1). If the worker from the
// previous configuration is still running, it may be inside processFrame on that
// exact pointer. Realtime mode is used deliberately: the offline path waits for
// the worker and drains the queue, which hides the window entirely.

void testReprepareWhileWorkerBusy()
{
    AltDenoiserProcessor proc;
    double lastPeak = 0.0;

    for (int round = 0; round < 12; ++round)
    {
        proc.setNonRealtime (false);            // do not drain; let frames pile up
        proc.setPlayConfigDetails (2, 2, kSampleRate, kBlockSize);
        proc.prepareToPlay (kSampleRate, kBlockSize);
        setAttenuation (proc, 0.0f);

        // Enough blocks to fill the worker's inbound queue, few enough that it
        // cannot have drained it.
        render (proc, 6,
                [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
                [&lastPeak] (const juce::AudioBuffer<float>& b, int)
                {
                    for (int i = 0; i < b.getNumSamples(); ++i)
                        lastPeak = std::max (lastPeak, (double) std::abs (b.getSample (0, i)));
                });
        // Next loop iteration re-prepares immediately, with work still in flight.
    }

    // Surviving is most of the assertion; under ASAN the use-after-free aborts
    // here rather than passing. Also confirm the plugin still works afterwards.
    proc.setNonRealtime (true);
    prepare (proc);
    setAttenuation (proc, 0.0f);
    double afterPeak = 0.0;
    render (proc, 30,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
            [&afterPeak] (const juce::AudioBuffer<float>& b, int blk)
            {
                if (blk < 15) return;
                for (int i = 0; i < b.getNumSamples(); ++i)
                    afterPeak = std::max (afterPeak, (double) std::abs (b.getSample (0, i)));
            });

    const bool passed = (afterPeak > 0.1);
    record ("re-prepare while worker busy", "N2", passed, Expect::Pass,
            "survived 12 re-prepares with work in flight, peak during " + std::to_string (lastPeak)
              + ", peak after " + std::to_string (afterPeak)
              + " (build with -fsanitize=address to turn the race into an abort)");
}

//==============================================================================
// T10-T13 / M9, M10, M11, L1, L3, L4 - meter behaviour.
//
// DbMeter is a plain juce::Component and the harness already installs a
// ScopedJuceInitialiser_GUI, so it can be driven directly with no editor.

void testMeterDecayIsFrameRateIndependent()
{
    // The same wall-clock second at two frame rates must reach the same level.
    // The old code multiplied by a fixed factor per tick, so 60 ticks and 30
    // ticks of the same duration disagreed by tens of dB (M9).
    DbMeter fast { true }, slow { true };
    fast.setSize (70, 240);
    slow.setSize (70, 240);

    fast.update (1.0f, 0.0);
    slow.update (1.0f, 0.0);

    for (int i = 0; i < 60; ++i) fast.update (0.0f, 1.0 / 60.0);
    for (int i = 0; i < 30; ++i) slow.update (0.0f, 1.0 / 30.0);

    const double diff = std::abs (fast.getLevelDb() - slow.getLevelDb());
    const bool passed = (diff < 0.5);
    record ("meter decay is frame-rate independent", "M9", passed, Expect::Pass,
            "after 1 s: 60 Hz -> " + std::to_string (fast.getLevelDb())
              + " dB, 30 Hz -> " + std::to_string (slow.getLevelDb())
              + " dB, difference " + std::to_string (diff) + " dB");
}

void testMeterDecayRateIsStandard()
{
    // A digital peak meter falls about 20 dB/s. The old code fell 116 dB/s.
    DbMeter m { true };
    m.setSize (70, 240);
    m.update (1.0f, 0.0);
    const float start = m.getLevelDb();
    for (int i = 0; i < 60; ++i) m.update (0.0f, 1.0 / 60.0);
    const float fell = start - m.getLevelDb();

    const bool passed = (fell > 19.0f && fell < 21.0f);
    record ("meter decay rate is standard", "M9", passed, Expect::Pass,
            "fell " + std::to_string (fell) + " dB in 1 s (target 20, old code 116)");
}

void testMeterHoldNeverDropsBelowBar()
{
    // M11: bar and readout used to decay from separate state at different rates,
    // so they disagreed after every transient. L3: the readout also had no floor
    // and decremented without bound, reaching about -108,000 after an hour.
    DbMeter m { true };
    m.setSize (70, 240);

    bool invariantHeld = true;
    m.update (1.0f, 0.0);
    for (int i = 0; i < 600; ++i)          // 10 s of silence
    {
        m.update (0.0f, 1.0 / 60.0);
        if (m.getHoldDb() < m.getLevelDb() - 1.0e-3f) invariantHeld = false;
    }

    const bool floored = (m.getLevelDb() >= DbMeter::kFloorDb - 1.0e-3f)
                      && (m.getHoldDb()  >= DbMeter::kFloorDb - 1.0e-3f);
    const bool passed = invariantHeld && floored;
    record ("meter hold tracks bar and both floor", "M11/L3", passed, Expect::Pass,
            "after 10 s silence: level " + std::to_string (m.getLevelDb())
              + " dB, hold " + std::to_string (m.getHoldDb())
              + " dB, floor " + std::to_string (DbMeter::kFloorDb)
              + (invariantHeld ? "" : ", INVARIANT VIOLATED"));
}

void testMeterRepaintsOnlyOnChange()
{
    // L1: repaint() was unconditional, so two meters fully repainted 60 times a
    // second forever, including in total silence with an identical result.
    DbMeter m { true };
    m.setSize (70, 240);

    for (int i = 0; i < 200; ++i) m.update (0.0f, 1.0 / 60.0);   // settle to floor
    const int before = m.getRepaintRequests();
    for (int i = 0; i < 120; ++i) m.update (0.0f, 1.0 / 60.0);   // 2 s more silence
    const int during = m.getRepaintRequests() - before;

    // Counts repaint REQUESTS. An earlier version of this test counted paint()
    // calls and was vacuous: headless, paint() never runs, so the counter stayed
    // at zero whether the gate worked or not. It passed against a mutation that
    // made repaint() unconditional, which is exactly what it exists to catch.
    const bool passed = (during == 0);
    record ("meter repaints only on change", "L1", passed, Expect::Pass,
            "repaints during 120 silent updates: " + std::to_string (during));
}

void testMetersSeeAllChannels()
{
    // L4: only channel 0 was metered, so a right-channel-only source read as
    // silence on the input meter.
    AltDenoiserProcessor proc;
    prepare (proc);
    setAttenuation (proc, 0.0f);

    render (proc, 20,
            [] (juce::AudioBuffer<float>& b, int blk)
            {
                fillSine (b, blk, 440.0f, 440.0f);
                b.clear (0, 0, b.getNumSamples());     // silence on the LEFT
            },
            [] (const juce::AudioBuffer<float>&, int) {});

    const float seen = proc.inputLevel.takeAndReset();
    const bool passed = (seen > 0.1f);
    record ("meters see all channels", "L4", passed, Expect::Pass,
            "right-channel-only input, meter saw peak " + std::to_string (seen));
}

//==============================================================================
// T15 / L8 - impossible geometry must be refused, not acted on.
//
// prepareToPlay(0, 0) previously produced an infinite resample ratio, and the
// resampler's inner while loop never terminates when the target sample time is
// infinite. The failure mode is a HUNG host, not wrong audio, which is why the
// mutation test for this one cannot simply be "remove the guard and run".

void testInvalidGeometryIsRefused()
{
    AltDenoiserProcessor proc;

    // Prepare VALIDLY first. Without this the test is vacuous: a virgin
    // processor already reports latency 0 with preparedBlockSize 0, so deleting
    // the guard's clean-up leaves every assertion green. A review mutation
    // proved exactly that.
    prepare (proc);
    const int latencyWhenValid = proc.getLatencySamples();

    // Each of these must refuse and clear the state the valid prepare set.
    proc.prepareToPlay (0.0, 0);
    const int latencyAfterZero = proc.getLatencySamples();

    // reset() after a refused prepare must not crash. The refusal returns before
    // the FIFOs are sized, and reset() re-primes them; a review probe crashed
    // here with exit 139 before SimpleFifo::push was hardened. Hosts reach this
    // through the VST3 wrapper's setProcessing(false).
    proc.reset();

    // The geometry that actually detonates: a zero rate with a VALID block size
    // is the only combination that reaches the resampler with an infinite
    // source sample time. The original test never tried it.
    proc.prepareToPlay (0.0, kBlockSize);
    proc.reset();

    proc.prepareToPlay (48000.0, -1);
    proc.prepareToPlay (1.0, 512);          // below the supported floor

    // With preparedBlockSize cleared, processBlock refuses every block and the
    // H7 passthrough carries the audio.
    juce::AudioBuffer<float> buffer (2, 512);
    juce::MidiBuffer midi;
    std::vector<float> before ((size_t) 512);
    for (int i = 0; i < 512; ++i)
    {
        const auto v = 0.25f * (float) std::sin (juce::MathConstants<double>::twoPi * 440.0 * i / 48000.0);
        buffer.setSample (0, i, v);
        buffer.setSample (1, i, v);
        before[(size_t) i] = v;
    }
    proc.processBlock (buffer, midi);

    double maxDelta = 0.0;
    for (int i = 0; i < 512; ++i)
        maxDelta = std::max (maxDelta, (double) std::abs (buffer.getSample (0, i) - before[(size_t) i]));

    // A valid prepare afterwards must still work: the refusal cannot leave the
    // plugin permanently broken.
    prepare (proc);
    setAttenuation (proc, 0.0f);
    double recovered = 0.0;
    render (proc, 30,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
            [&recovered] (const juce::AudioBuffer<float>& b, int blk)
            {
                if (blk < 15) return;
                for (int i = 0; i < b.getNumSamples(); ++i)
                    recovered = std::max (recovered, (double) std::abs (b.getSample (0, i)));
            });

    // latencyWhenValid > 0 is what makes the latencyAfterZero == 0 clause mean
    // something: it proves the guard CLEARED a non-zero value rather than
    // observing one that was already zero.
    const bool passed = (latencyWhenValid > 0) && (latencyAfterZero == 0)
                     && (maxDelta < 1.0e-6) && (recovered > 0.1);
    record ("invalid geometry is refused", "L8", passed, Expect::Pass,
            "latency when valid " + std::to_string (latencyWhenValid)
              + " -> after refusal " + std::to_string (latencyAfterZero)
              + ", survived reset() twice, passthrough delta " + std::to_string (maxDelta)
              + ", recovered peak " + std::to_string (recovered));
}

//==============================================================================
// T16 / L6 - the attenuation limit is a wet/dry crossfade applied per sample.
//
// libDF implements atten_lim as (1-lim)*enh + lim*noisy before a linear
// synthesis, so at 0 dB ("no reduction") the output must be the INPUT delayed by
// exactly the reported latency. That is a far stronger assertion than comparing
// RMS levels, and it is only true if the crossfade and the dry alignment are
// both right.

void testAttenuationIsACrossfade()
{
    AltDenoiserProcessor proc;
    prepare (proc);
    setAttenuation (proc, 0.0f);          // fully dry

    const int latency = proc.getLatencySamples();
    std::vector<float> in, out;
    in.reserve (60 * (size_t) kBlockSize);
    out.reserve (60 * (size_t) kBlockSize);
    render (proc, 60,
            [&in] (juce::AudioBuffer<float>& b, int blk)
            {
                fillSine (b, blk, 440.0f, 440.0f);
                const auto* p = b.getReadPointer (0);
                in.insert (in.end(), p, p + b.getNumSamples());
            },
            [&out] (const juce::AudioBuffer<float>& b, int)
            {
                const auto* p = b.getReadPointer (0);
                out.insert (out.end(), p, p + b.getNumSamples());
            });

    // Compare out[latency + n] against in[n], well past the ramp and the
    // startup region.
    // Search a small window for the offset that best fits, rather than assuming
    // the reported latency is exact to the sample. The resampler contributes a
    // few samples of group delay even at a 1:1 ratio, and a fixed-offset compare
    // turns that phase shift into a large amplitude error on a sine: 4 samples
    // at 440 Hz is already 0.04 of full scale.
    int bestOffset = -1;
    double bestWorst = 1.0e9;
    for (int off = latency - 16; off <= latency + 16; ++off)
    {
        if (off <= 0) continue;
        double worst = 0.0;
        const int from = off + 2000;
        const int to = (int) std::min (out.size(), in.size() + (size_t) off) - 1;
        if (to <= from) continue;
        for (int i = from; i < to; ++i)
            worst = std::max (worst, (double) std::abs (out[(size_t) i] - in[(size_t) (i - off)]));
        if (worst < bestWorst) { bestWorst = worst; bestOffset = off; }
    }

    // ALIGNMENT is what L6 is about, and it is asserted tightly at every rate.
    // The RESIDUAL is a resampler-fidelity question, not an alignment one, so its
    // tolerance depends on the rate: at 48 kHz the converter runs 1:1 and the dry
    // path is near-exact, while at any other rate the signal has been through
    // 96k->48k and 48k->96k Catmull-Rom conversions before the comparison.
    //
    // Measured residual at 96 kHz is about 0.076 against a 0.25 peak, i.e. 30%.
    // That is the interpolator's own error on a 440 Hz tone nowhere near Nyquist,
    // and it is worth knowing: see the resampler quality note in BACKLOG.md.
    const bool nativeRate = (std::abs (kSampleRate - 48000.0) < 1.0);
    const double residualTolerance = nativeRate ? 0.01 : 0.12;

    const bool primed = (std::abs (proc.getAttenDryMix() - 1.0f) < 1.0e-3f);
    const bool aligned = (bestOffset > 0) && (std::abs (bestOffset - latency) <= 8);
    const bool passed = primed && aligned && (bestWorst < residualTolerance);
    record ("attenuation is a latency-aligned crossfade", "L6", passed, Expect::Pass,
            "dry mix " + std::to_string (proc.getAttenDryMix())
              + " (want 1.0 at 0 dB), best-fit offset " + std::to_string (bestOffset)
              + " vs reported latency " + std::to_string (latency)
              + ", residual " + std::to_string (bestWorst)
              + " (tolerance " + std::to_string (residualTolerance) + ")");
}

void testAttenuationMixIsSmoothedAndBounded()
{
    AltDenoiserProcessor proc;
    prepare (proc);

    // 100 dB is "no limit": fully enhanced, dry share 0.
    setAttenuation (proc, 100.0f);
    render (proc, 20,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
            [] (const juce::AudioBuffer<float>&, int) {});
    const float atNoLimit = proc.getAttenDryMix();

    // A change must RAMP, not jump: one block is far shorter than the 50 ms ramp
    // at every geometry the gate runs.
    setAttenuation (proc, 0.0f);
    render (proc, 1,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
            [] (const juce::AudioBuffer<float>&, int) {});
    const float afterOneBlock = proc.getAttenDryMix();

    // And it must arrive. 50 ms is at most 5 blocks at the largest geometry.
    render (proc, 40,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
            [] (const juce::AudioBuffer<float>&, int) {});
    const float settled = proc.getAttenDryMix();

    const bool noLimitOk = (atNoLimit < 1.0e-3f);
    const bool rampedOk  = (afterOneBlock > 0.0f && afterOneBlock < 0.999f);
    const bool settledOk = (std::abs (settled - 1.0f) < 1.0e-3f);

    const bool passed = noLimitOk && rampedOk && settledOk;
    record ("attenuation mix ramps and settles", "L6", passed, Expect::Pass,
            "at 100 dB " + std::to_string (atNoLimit) + " (want 0), after one block of a "
              "100->0 change " + std::to_string (afterOneBlock)
              + " (want a partial ramp), settled " + std::to_string (settled) + " (want 1)");
}

//==============================================================================
// T17 / L7 - state round-trips, and a state from a newer build is refused whole.

void testStateSchema()
{
    juce::MemoryBlock saved;
    {
        AltDenoiserProcessor proc;
        setAttenuation (proc, 42.0f);
        proc.getStateInformation (saved);
    }

    const bool wroteSomething = (saved.getSize() > 0);

    float restored = -1.0f;
    {
        AltDenoiserProcessor proc;
        proc.setStateInformation (saved.getData(), (int) saved.getSize());
        restored = proc.apvts.getParameter ("atten_lim")->convertFrom0to1 (
                       proc.apvts.getParameter ("atten_lim")->getValue());
    }

    // A state claiming a future schema must leave defaults untouched rather than
    // loading half of it.
    float afterFuture = -1.0f;
    bool parsedFutureXml = false;
    {
        AltDenoiserProcessor proc;
        auto xml = juce::parseXML (R"(<Parameters schemaVersion="99"><PARAM id="atten_lim" value="0.0"/></Parameters>)");
        // Assert rather than branch: wrapping this in `if (xml != nullptr)` made
        // a typo in the literal silently skip the only assertion that covers the
        // version check, reporting PASS having exercised nothing.
        jassert (xml != nullptr);
        parsedFutureXml = (xml != nullptr);
        if (xml != nullptr)
        {
            juce::MemoryBlock future;
            juce::AudioProcessor::copyXmlToBinary (*xml, future);
            proc.setStateInformation (future.getData(), (int) future.getSize());
        }
        afterFuture = proc.apvts.getParameter ("atten_lim")->convertFrom0to1 (
                          proc.apvts.getParameter ("atten_lim")->getValue());
    }

    const bool roundTripped = std::abs (restored - 42.0f) < 0.5f;
    const bool futureRefused = std::abs (afterFuture - 100.0f) < 0.5f;   // default survived
    const bool passed = wroteSomething && roundTripped && futureRefused && parsedFutureXml;
    record ("state round-trips and rejects the future", "L7", passed, Expect::Pass,
            "saved " + std::to_string ((int) saved.getSize()) + " bytes, restored "
              + std::to_string (restored) + " (want 42), after a schemaVersion=99 state "
              + std::to_string (afterFuture) + " (want the 100 default)");
}

//==============================================================================
// T19 / M8 - SimpleFifo, tested directly.
//
// Every one of these defects was unreachable through the plugin's call sites,
// because the guards lived in the callers. That is exactly why the class needs
// its own test: the next caller does not inherit those guards.

void testSimpleFifo()
{
    std::vector<std::string> failures;
    auto check = [&failures] (bool ok, const char* what)
    {
        if (! ok) failures.push_back (what);
    };

    const float ramp[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float out[8] = {};

    // An unsized FIFO must refuse everything rather than divide by zero. This is
    // the crash a review probe hit through reset() after a refused prepare.
    {
        SimpleFifo f;
        check (! f.push (ramp, 4),    "unsized push should be refused");
        check (! f.pushSilence (4),   "unsized pushSilence should be refused");
        check (! f.discard (1),       "unsized discard should be refused");
        check (f.getAvailable() == 0, "unsized FIFO should report nothing available");
    }

    // Wrap-around must preserve order.
    {
        SimpleFifo f;
        f.setSize (5);
        check (f.push (ramp, 4), "push of 4 into 5 should succeed");
        check (f.discard (3),    "discard of 3 should succeed");
        check (f.push (ramp + 4, 3), "push of 3 after discard should succeed");   // wraps
        check (f.getAvailable() == 4, "should hold 4 after wrap");
        check (f.peek (out, 4), "peek of 4 should succeed");
        check (out[0] == 4 && out[1] == 5 && out[2] == 6 && out[3] == 7,
               "wrapped read should preserve order");
    }

    // Overflow must be refused, not silently clamped. The original advanced
    // writePos past readPos while clamping the count, which left the FIFO
    // reporting full and returning newest-then-oldest.
    {
        SimpleFifo f;
        f.setSize (4);
        check (f.push (ramp, 4), "fill to capacity should succeed");
        check (! f.push (ramp, 1), "push beyond capacity should be refused");
        check (f.getAvailable() == 4, "refused push must not change the count");
        check (f.getOverflows() == 1, "refused push should be counted");
        check (f.peek (out, 4) && out[0] == 1 && out[3] == 4,
               "contents must survive a refused push intact");
    }

    // Under-run must be refused so the count can never go negative. A negative
    // count was the input to the signed/unsigned comparison that clamped the
    // FIFO to "full", so this is the assertion that pins the original bug.
    {
        SimpleFifo f;
        f.setSize (4);
        check (f.push (ramp, 2), "push of 2 should succeed");
        check (! f.discard (3), "discard beyond available should be refused");
        check (f.getAvailable() == 2, "refused discard must not change the count");
        check (f.getUnderflows() == 1, "refused discard should be counted");

        // The original would now report 4 (capacity) instead of 2. Pushing one
        // sample and reading it back is what catches that.
        check (f.push (ramp + 7, 1), "push after a refused discard should succeed");
        check (f.getAvailable() == 3, "count should be 3, not the capacity");
        check (f.peek (out, 3) && out[2] == 8, "the newly pushed sample should read back");
    }

    // Negative and null arguments must be refused rather than looped on.
    {
        SimpleFifo f;
        f.setSize (4);
        check (! f.push (ramp, -5),   "negative push should be refused");
        check (! f.push (nullptr, 2), "null push should be refused");
        check (! f.discard (-5),      "negative discard should be refused");
        check (! f.peek (out, -1),    "negative peek should be refused");
        check (f.getAvailable() == 0, "refusals must leave the FIFO empty");
    }

    // setSize(0) must leave it inert rather than armed with a zero modulus.
    {
        SimpleFifo f;
        f.setSize (4);
        f.setSize (0);
        check (f.getCapacity() == 0, "setSize(0) should report zero capacity");
        check (! f.push (ramp, 1),   "push into a zero-capacity FIFO should be refused");
    }

    std::string detail = failures.empty()
        ? std::string ("all bounds, wrap-around, overflow, underflow and null cases behave")
        : (std::to_string (failures.size()) + " failed: " + failures.front());
    record ("SimpleFifo is bounds-safe", "M8", failures.empty(), Expect::Pass, detail);
}

//==============================================================================
// T24-T26 / C1, C2, M3 - the alt_df shim.
//
// These tests could not be written against libDF's C API. Every failure there
// was an .expect() inside an extern "C" function, so the failure mode was an
// abort of the whole test process: not a red test, no output, no exit code to
// inspect. "Reaching the next line" is the assertion.

void testCorruptModelDegradesToBypass()
{
    std::vector<std::string> failures;
    auto check = [&failures] (bool ok, const char* what)
    {
        if (! ok) failures.push_back (what);
    };

    const auto* good = reinterpret_cast<const uint8_t*> (
                           AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gz);
    const int goodSize = AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gzSize;

    check (good != nullptr && goodSize > 4096, "the embedded model should be present");

    // 1. Nothing at all.
    {
        DeepFilterNetProcessor p;
        check (! p.initializeFromMemory (nullptr, 1024), "a null buffer should be refused");
        check (! p.isReady(), "a refused load must leave the processor unusable");
        check (p.getLastStatus() == ALT_DF_NULL_ARG, "a null buffer should report NULL_ARG");
    }

    // 2. Zero length.
    {
        DeepFilterNetProcessor p;
        check (! p.initializeFromMemory (good, 0), "a zero-length buffer should be refused");
        check (p.getLastStatus() == ALT_DF_NULL_ARG, "zero length should report NULL_ARG");
    }

    // 3. Random bytes. Not gzip, so this fails in the decompressor.
    {
        std::vector<uint8_t> garbage (8192);
        juce::Random rng (12345);
        for (auto& b : garbage) b = (uint8_t) rng.nextInt (256);

        DeepFilterNetProcessor p;
        check (! p.initializeFromMemory (garbage.data(), (int) garbage.size()),
               "random bytes should be refused");
        check (p.getLastStatus() == ALT_DF_BAD_MODEL, "random bytes should report BAD_MODEL");
    }

    // 4. A TRUNCATED copy of the real archive. This is the one that matters:
    //    the gzip header is valid, so the failure happens deep inside the tar
    //    reader rather than at the first byte. It is exactly what a short write
    //    to the old temp file produced, and it used to abort the host.
    {
        DeepFilterNetProcessor p;
        check (! p.initializeFromMemory (good, goodSize / 2),
               "a truncated archive should be refused");
        const auto st = p.getLastStatus();
        check (st == ALT_DF_BAD_MODEL || st == ALT_DF_INIT,
               "a truncated archive should report BAD_MODEL or INIT");
    }

    // 5. A valid gzip stream that is not a DeepFilterNet archive: the first
    //    32 KB of the real file's *decompressed* content is not reachable
    //    without inflating, so use a gzip of something else entirely. A gzip
    //    member of a single stored empty tar is the cheapest such thing, and
    //    the point is that the archive parses but yields no enc.onnx.
    {
        juce::MemoryBlock gz;
        {
            juce::MemoryOutputStream raw (gz, false);
            juce::GZIPCompressorOutputStream zip (raw, 6);
            const std::vector<char> emptyTar (1024, 0);   // two zero blocks = end of archive
            zip.write (emptyTar.data(), emptyTar.size());
            zip.flush();
        }

        DeepFilterNetProcessor p;
        check (! p.initializeFromMemory (gz.getData(), (int) gz.getSize()),
               "a gzip containing no model should be refused");
        check (p.getLastStatus() == ALT_DF_BAD_MODEL || p.getLastStatus() == ALT_DF_INIT,
               "an empty archive should report BAD_MODEL or INIT");
    }

    // 6. After all of that, a good load must still work. A failed attempt
    //    cannot poison the object: release() runs first on every path.
    {
        DeepFilterNetProcessor p;
        check (! p.initializeFromMemory (good, 64), "a 64-byte archive should be refused");
        check (p.initialize(), "the real model should load after a refused one");
        check (p.isReady(), "the real model should report ready");
        check (p.getFrameLength() == 480, "the loaded model should report a 480-sample hop");
    }

    const std::string detail = failures.empty()
        ? std::string ("six malformed archives refused, process still alive, good load after")
        : (std::to_string (failures.size()) + " failed: " + failures.front());
    record ("corrupt model degrades to bypass", "C1/C2", failures.empty(), Expect::Pass, detail);
}

void testModelMetadataIsReported()
{
    // libDF's C API exposed only the hop size, which is why the plugin's
    // reported latency had to be the hardcoded 1920. alt_df_info reports the
    // whole geometry, so it can be derived. This test pins the derivation
    // against the model that is actually embedded.

    DeepFilterNetProcessor p;
    const bool loaded = p.initialize();
    const auto& info = p.getInfo();

    const int modelDelay = p.getModelDelaySamples();
    const int derived    = modelDelay + (int) info.hop_size;   // + the plugin's own cushion

    const bool passed = loaded
                     && info.sr == 48000u
                     && info.hop_size == 480u
                     && info.fft_size == 960u
                     && info.lookahead == 2u
                     && info.ch == 1u
                     && modelDelay == 1440
                     && derived == 1920;

    record ("model metadata is reported", "C2", passed, Expect::Pass,
            "sr " + std::to_string (info.sr)
              + ", hop " + std::to_string (info.hop_size)
              + ", fft " + std::to_string (info.fft_size)
              + ", lookahead " + std::to_string (info.lookahead)
              + ", ch " + std::to_string (info.ch)
              + " -> model delay " + std::to_string (modelDelay)
              + ", derived latency " + std::to_string (derived) + " (want 1920)");
}

void testWrongFrameLengthIsRefused()
{
    // M3. libDF's df_process_frame built its ndarray views with from_shape_ptr,
    // which performs no bounds check, and libDF's only guard was a
    // debug_assert compiled out in release. Handing it a buffer shorter than
    // the model's hop was therefore a silent heap read past the end, not an
    // error. alt_df range-checks the lengths before dereferencing anything.
    //
    // Called through the C API directly: the C++ wrapper always passes the
    // model's own hop, so the guard is unreachable from there by construction.

    std::vector<std::string> failures;
    auto check = [&failures] (bool ok, const char* what)
    {
        if (! ok) failures.push_back (what);
    };

    AltDf* st = nullptr;
    const auto created = alt_df_create (
        reinterpret_cast<const uint8_t*> (AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gz),
        (size_t) AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gzSize,
        100.0f, &st);

    check (created == ALT_DF_OK && st != nullptr, "the model should load");

    if (st != nullptr)
    {
        AltDfInfo info {};
        check (alt_df_info (st, &info) == ALT_DF_OK, "info should succeed");

        std::vector<float> in ((size_t) info.hop_size, 0.1f);
        std::vector<float> out ((size_t) info.hop_size, 0.0f);
        const size_t hop = (size_t) info.hop_size;

        check (alt_df_process_frame (st, in.data(), hop - 1, out.data(), hop, nullptr)
                   == ALT_DF_BAD_LENGTH, "a short input length should be refused");
        check (alt_df_process_frame (st, in.data(), hop, out.data(), hop - 1, nullptr)
                   == ALT_DF_BAD_LENGTH, "a short output length should be refused");
        check (alt_df_process_frame (st, in.data(), hop + 1, out.data(), hop, nullptr)
                   == ALT_DF_BAD_LENGTH, "an over-long input length should be refused");
        check (alt_df_process_frame (st, nullptr, hop, out.data(), hop, nullptr)
                   == ALT_DF_NULL_ARG, "a null input should be refused");
        check (alt_df_process_frame (nullptr, in.data(), hop, out.data(), hop, nullptr)
                   == ALT_DF_NULL_ARG, "a null state should be refused");

        // The correct call must still work after all those refusals, and must
        // actually write something.
        float lsnr = -999.0f;
        check (alt_df_process_frame (st, in.data(), hop, out.data(), hop, &lsnr) == ALT_DF_OK,
               "a correctly sized frame should succeed");
        check (lsnr > -900.0f, "the local SNR should have been written");

        alt_df_free (st);
    }

    // Freeing null must be a no-op, not a crash.
    alt_df_free (nullptr);

    // Every status must have a description; a missing arm would be a null deref
    // in the DBG path exactly when something has already gone wrong.
    for (int s = ALT_DF_OK; s <= ALT_DF_PANIC; ++s)
        check (alt_df_status_str ((AltDfStatus) s) != nullptr, "every status needs a description");

    const std::string detail = failures.empty()
        ? std::string ("short, long, and null arguments all refused; correct call still works")
        : (std::to_string (failures.size()) + " failed: " + failures.front());
    record ("wrong frame length is refused", "M3", failures.empty(), Expect::Pass, detail);
}

//==============================================================================
// T20 / M7 - bypass must be latency-aligned, exposed, and click-free.

void setBypass (AltDenoiserProcessor& proc, bool on)
{
    auto* p = proc.apvts.getParameter ("bypass");
    jassert (p != nullptr);
    p->setValueNotifyingHost (on ? 1.0f : 0.0f);
}

void testBypassIsLatencyAligned()
{
    AltDenoiserProcessor proc;
    prepare (proc);
    setBypass (proc, true);

    const int latency = proc.getLatencySamples();
    std::vector<float> in, out;
    render (proc, 60,
            [&in] (juce::AudioBuffer<float>& b, int blk)
            {
                fillSine (b, blk, 440.0f, 880.0f);
                const auto* p = b.getReadPointer (0);
                in.insert (in.end(), p, p + b.getNumSamples());
            },
            [&out] (const juce::AudioBuffer<float>& b, int)
            {
                const auto* p = b.getReadPointer (0);
                out.insert (out.end(), p, p + b.getNumSamples());
            });

    // Bypassed, the output must be the input delayed by EXACTLY the reported
    // latency, bit-for-bit: the bypass path is a delay line, not a resampler, so
    // unlike the L6 crossfade there is no interpolation error to allow for.
    double worst = 0.0;
    const int from = latency + 2000;
    const int to = (int) std::min (out.size(), in.size() + (size_t) latency) - 1;
    for (int i = from; i < to; ++i)
        worst = std::max (worst, (double) std::abs (out[(size_t) i] - in[(size_t) (i - latency)]));

    const bool passed = (to > from) && (worst < 1.0e-6);
    record ("bypass is latency-aligned", "M7", passed, Expect::Pass,
            "reported latency " + std::to_string (latency)
              + ", worst |out[n+latency] - in[n]| = " + std::to_string (worst)
              + " over " + std::to_string (to - from) + " samples (want bit-exact)");
}

void testBypassPreservesStereo()
{
    // The processed path is mono by design (H4). Bypass must NOT be: it is a
    // per-channel delay line, so a bypassed stereo track has to stay stereo.
    AltDenoiserProcessor proc;
    prepare (proc);
    setBypass (proc, true);

    double maxDiff = 0.0;
    render (proc, 40,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 880.0f); },
            [&maxDiff] (const juce::AudioBuffer<float>& b, int blk)
            {
                if (blk < 20) return;
                for (int i = 0; i < b.getNumSamples(); ++i)
                    maxDiff = std::max (maxDiff, (double) std::abs (b.getSample (0, i) - b.getSample (1, i)));
            });

    const bool passed = (maxDiff > 0.1);
    record ("bypass preserves stereo", "M7", passed, Expect::Pass,
            "max |L-R| bypassed = " + std::to_string (maxDiff)
              + " (0 would mean bypass collapsed the channels like the wet path does)");
}

void testBypassToggleDoesNotClick()
{
    AltDenoiserProcessor proc;
    prepare (proc);
    setBypass (proc, false);

    // ONE render, toggling inside it. Two separate render calls would restart
    // fillSine's block index, putting a phase jump in the INPUT and making the
    // test fail on a discontinuity it created itself. That is what the first
    // version of this test did, reporting steps of 0.25 against a 0.25 signal.
    std::vector<float> out;
    render (proc, 150,
            [] (juce::AudioBuffer<float>& b, int blk) { fillSine (b, blk, 440.0f, 440.0f); },
            [&out, &proc] (const juce::AudioBuffer<float>& b, int blk)
            {
                const auto* p = b.getReadPointer (0);
                out.insert (out.end(), p, p + b.getNumSamples());
                if (blk == 90) setBypass (proc, true);
            });

    // Two things a bad bypass produces: a silent gap where stale audio was
    // flushed, and a step discontinuity at the switch. Neither is allowed.
    const int latency = proc.getLatencySamples();
    const int worstRun = longestZeroRun (out, latency);

    double worstStep = 0.0;
    for (size_t i = (size_t) latency + 1; i < out.size(); ++i)
        worstStep = std::max (worstStep, (double) std::abs (out[i] - out[i - 1]));

    // A 440 Hz sine at 0.25 peak steps at most 0.25*2*pi*440/rate per sample,
    // about 0.009 at 48 kHz. Allow generous headroom for the crossfade itself
    // but nothing like a full-scale jump.
    const double stepLimit = 0.05;
    const bool passed = (worstRun <= 4) && (worstStep < stepLimit);
    record ("bypass toggle does not click", "M7", passed, Expect::Pass,
            "longest zero run " + std::to_string (worstRun)
              + ", worst sample-to-sample step " + std::to_string (worstStep)
              + " (limit " + std::to_string (stepLimit) + ")");
}

void testBypassParameterIsExposedAndSaved()
{
    AltDenoiserProcessor proc;

    const bool exposed = (proc.getBypassParameter() != nullptr);

    // It must be a real parameter, not a wrapper-synthesised one, so it survives
    // a state round-trip. A synthesised bypass is stored in VST3's private state
    // and not stored at all by AU or LV2.
    setBypass (proc, true);
    juce::MemoryBlock saved;
    proc.getStateInformation (saved);

    AltDenoiserProcessor fresh;
    fresh.setStateInformation (saved.getData(), (int) saved.getSize());
    const auto* p = fresh.getBypassParameter();
    const bool restored = (p != nullptr) && p->get();

    const bool passed = exposed && restored;
    record ("bypass parameter is exposed and saved", "M7", passed, Expect::Pass,
            std::string ("getBypassParameter() ") + (exposed ? "non-null" : "NULL")
              + ", survived a state round-trip: " + (restored ? "yes" : "no"));
}

} // namespace

//==============================================================================
int main (int argc, char** argv)
{
    if (argc > 1) kBlockSize  = std::atoi (argv[1]);
    if (argc > 2) kSampleRate = std::atof (argv[2]);

    juce::ScopedJuceInitialiser_GUI juceInit;

    std::printf ("AltDenoiser offline harness  (%.0f Hz, %d-sample blocks)\n\n",
                 kSampleRate, kBlockSize);

    testStartupZeroSplice();
    testRightChannelReachesTheModel();
    testAttenuationSurvivesReprepare();
    testReportedLatencyMatchesMeasured();
    testOversizedBlockIsRefused();
    testUnpreparedPassesAudioThrough();
    testResetDropsStaleAudio();
    testRealtimeFallbackIsDryNotSilence();
    testReprepareWhileWorkerBusy();
    testMeterDecayIsFrameRateIndependent();
    testMeterDecayRateIsStandard();
    testMeterHoldNeverDropsBelowBar();
    testMeterRepaintsOnlyOnChange();
    testMetersSeeAllChannels();
    testInvalidGeometryIsRefused();
    testAttenuationIsACrossfade();
    testAttenuationMixIsSmoothedAndBounded();
    testStateSchema();
    testSimpleFifo();
    testBypassIsLatencyAligned();
    testBypassPreservesStereo();
    testBypassToggleDoesNotClick();
    testBypassParameterIsExposedAndSaved();
    testCorruptModelDegradesToBypass();
    testModelMetadataIsReported();
    testWrongFrameLengthIsRefused();

    int unexpected = 0;
    for (const auto& r : results)
    {
        const bool asExpected = (r.expectation == Expect::Pass) ? r.passed : true;
        const char* tag = r.passed ? "PASS"
                        : (r.expectation == Expect::FailUntilFixed ? "FAIL (expected)" : "FAIL");

        if (! r.passed && r.expectation == Expect::Pass) ++unexpected;
        if (r.passed && r.expectation == Expect::FailUntilFixed)
            std::printf ("  NOTE: %s now passes; flip its Expect to Pass.\n", r.name.c_str());

        std::printf ("[%-15s] %-34s %-4s  %s\n",
                     tag, r.name.c_str(), r.backlogId.c_str(), r.detail.c_str());
        juce::ignoreUnused (asExpected);
    }

    std::printf ("\n%d test(s), %d unexpected failure(s)\n", (int) results.size(), unexpected);
    return unexpected == 0 ? 0 : 1;
}
