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
    const bool passed = (zerosAfterLatency == 0);
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

    // Attenuation 0 makes the model return its input immediately, so its 1440
    // samples of algorithmic delay ((fft 960 - hop 480) + lookahead 2 * 480) do
    // not appear here. The pipeline is accountable for the remainder, which
    // after priming is one 480-sample hop of cushion plus the resampler's own
    // group delay. Scale to the host rate the same way the plugin does.
    const int modelDelay = (int) std::lround (1440.0 * kSampleRate / 48000.0);
    const int expected = reported - modelDelay;
    const bool passed = (measured >= 0 && std::abs (measured - expected) <= 32);
    record ("reported latency matches measured", "H6", passed, Expect::Pass,
            "reported " + std::to_string (reported)
              + " - model " + std::to_string (modelDelay)
              + " = expected pipeline " + std::to_string (expected)
              + ", measured " + std::to_string (measured)
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
