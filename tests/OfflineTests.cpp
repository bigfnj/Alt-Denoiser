/*
    Offline render harness for AltDenoiserProcessor.

    pluginval passes this plugin at strictness level 5 with every defect in
    BACKLOG.md present, because it checks that the plugin is a well-formed VST3
    rather than that it processes audio correctly. These tests check the audio.

    They are CHARACTERISATION tests: several are expected to fail against the
    current code, and each names the backlog item it pins down. A failure here
    is the bug being detected, not the harness being wrong. As each item is
    fixed, the corresponding EXPECT flips to pass and becomes a regression test.

    Run:  AltDenoiserTests[.exe]
    Exit: 0 if every test matched its expectation, 1 otherwise.
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

    // Measured: a 48 kHz host whose block size is an exact multiple of the
    // 480-sample model hop drains the output FIFO evenly and never underruns.
    // Every other geometry splices silence in during startup. So the expected
    // result is configuration-dependent, and saying so keeps a clean run at
    // 480 or 960 from looking like the bug was fixed.
    const bool cleanGeometry = (kBlockSize % 480) == 0;
    const bool passed = (zerosAfterLatency == 0);
    record ("startup zero-splice", "H5", passed,
            cleanGeometry ? Expect::Pass : Expect::FailUntilFixed,
            "reported latency " + std::to_string (latency)
              + ", zero samples after it " + std::to_string (zerosAfterLatency)
              + ", longest zero run " + std::to_string (worstRun));
}

//==============================================================================
// T2 / H4 - the right channel is overwritten with a copy of the left.

void testStereoIsPreserved()
{
    AltDenoiserProcessor proc;
    prepare (proc);
    setAttenuation (proc, 0.0f);

    double maxAbsDiff = 0.0;
    double maxInputDiff = 0.0;   // non-vacuity guard, see below
    render (proc, 40,
            [&maxInputDiff] (juce::AudioBuffer<float>& b, int blk)
            {
                fillSine (b, blk, 440.0f, 880.0f);
                for (int i = 0; i < b.getNumSamples(); ++i)
                    maxInputDiff = std::max (maxInputDiff,
                                             (double) std::abs (b.getSample (0, i) - b.getSample (1, i)));
            },
            [&maxAbsDiff] (const juce::AudioBuffer<float>& b, int blk)
            {
                if (blk < 10) return;     // let the pipeline fill
                for (int i = 0; i < b.getNumSamples(); ++i)
                    maxAbsDiff = std::max (maxAbsDiff,
                                           (double) std::abs (b.getSample (0, i) - b.getSample (1, i)));
            });

    // An output diff of zero would ALSO be the result if the harness never fed
    // distinct channels, so require a large input diff before believing it.
    // Without this the test could pass vacuously forever.
    const bool inputWasStereo = (maxInputDiff > 0.1);
    const bool passed = inputWasStereo && (maxAbsDiff > 1.0e-6);
    record ("stereo channels stay distinct", "H4", passed, Expect::FailUntilFixed,
            "input max |L-R| = " + std::to_string (maxInputDiff)
              + (inputWasStereo ? "" : "  <-- HARNESS BUG: input was not stereo")
              + ", output max |L-R| = " + std::to_string (maxAbsDiff)
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
    record ("attenuation survives re-prepare", "H2", passed, Expect::FailUntilFixed,
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
    const bool passed = (measured >= 0 && std::abs (measured - reported) <= 32);
    record ("reported latency matches measured", "H6", passed, Expect::FailUntilFixed,
            "reported " + std::to_string (reported)
              + ", measured " + std::to_string (measured)
              + ", error " + (measured >= 0 ? std::to_string (measured - reported) : std::string ("n/a"))
              + " samples (at attenuation 0, so this excludes model-dependent delay)");
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
    testStereoIsPreserved();
    testAttenuationSurvivesReprepare();
    testReportedLatencyMatchesMeasured();

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
