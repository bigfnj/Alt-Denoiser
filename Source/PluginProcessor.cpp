#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <juce_core/juce_core.h>

AltDenoiserProcessor::AltDenoiserProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    // The old constructor took a sample rate that nothing ever read: the model
    // is fixed at its own rate and the resampler owns the conversion. Dropped
    // along with the rest of the class's rewrite.
    dfProcessor = std::make_unique<DeepFilterNetProcessor>();

    // L5: cached once. This was a string-keyed hash and lookup on the audio
    // thread every single block.
    attenParam = apvts.getRawParameterValue("atten_lim");
    jassert(attenParam != nullptr);

    bypassParam = dynamic_cast<juce::AudioParameterBool*>(apvts.getParameter("bypass"));
    jassert(bypassParam != nullptr);
}

AltDenoiserProcessor::~AltDenoiserProcessor() {
}

juce::AudioProcessorValueTreeState::ParameterLayout AltDenoiserProcessor::createParameterLayout() 
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    auto range = juce::NormalisableRange<float>(0.0f, 100.0f, 0.1f);
    range.setSkewForCentre(20.0f); 
    // M13: without attributes the host's generic editor and automation lane
    // show a bare "20.0" while the plugin UI shows "20.0 dB", and 100 is a
    // sentinel meaning "no limit" that an automating host had no way to know.
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "atten_lim", 1 },
        "Reduction Limit",
        range,
        100.0f,
        juce::AudioParameterFloatAttributes()
            .withLabel("dB")
            .withStringFromValueFunction([](float value, int) {
                return value >= 99.95f ? juce::String("No limit")
                                       : juce::String(value, 1) + " dB";
            })
    ));

    // M7: added last. VST3 hashes parameter IDs from the string plus version
    // hint rather than from index, so appending one does not renumber atten_lim
    // and existing automation survives.
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { "bypass", 1 }, "Bypass", false));

    return layout;
}

void AltDenoiserProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    // MUST come first. initialize() below calls df_free on the live DFState (H1),
    // and on a re-prepare the worker from the previous configuration is still
    // running and may be inside processFrame on that exact pointer: a
    // use-after-free.
    //
    // H1 and M1 were each safe in isolation. H1 introduced the df_free when no
    // worker existed; M1 introduced the worker when nothing freed the state.
    // Together they opened this window. releaseResources() already had the
    // right order, which is why the leak-and-teardown path never showed it.
    worker.stop();

    // L8: refuse geometry that cannot work, before anything is constructed from
    // it. A plugin scanner calling prepareToPlay(0, 0) previously produced an
    // infinite resample ratio, and the resampler's inner loop
    // (while acc >= target_sample_time) never terminates when the target sample
    // time is infinite, so the host hangs rather than misbehaving.
    //
    // preparedBlockSize is cleared explicitly rather than left at whatever a
    // previous prepare set, so the C5 guard in processBlock refuses every block
    // and the plugin degrades to the H7 passthrough.
    if (! (sampleRate >= 8000.0 && sampleRate <= 384000.0) || samplesPerBlock <= 0) {
        jassertfalse;
        preparedBlockSize = 0;
        derivedLatency48k = 0;
        setLatencySamples(0);
        tailLengthSeconds.store(0.0, std::memory_order_relaxed);
        modelLoaded.store(false, std::memory_order_release);
        return;
    }

    // Build the model: everything below is sized from what it reports.
    const bool loaded = dfProcessor->initialize();

    // M3: take the hop size from the model rather than assuming 480. Every
    // archive libDF can load uses 480 today, but df_process_frame builds its
    // ndarray views from the model's own hop_size with no bounds check, and the
    // only guard inside libDF is a debug_assert compiled out in release. A
    // mismatch would therefore be a silent heap overrun, not a clean failure.
    const size_t reportedHop = dfProcessor->getFrameLength();
    const bool hopIsSane = (reportedHop > 0 && reportedHop <= 4096);
    if (loaded && ! hopIsSane)
        DBG("Model reports an unusable hop size; bypassing");

    const bool usable = loaded && hopIsSane;
    modelFrameLength = usable ? (int) reportedHop : 480;

    tempInputFrame.assign((size_t) modelFrameLength, 0.0f);
    tempOutputFrame.assign((size_t) modelFrameLength, 0.0f);

    inputFifo.setSize(48000);
    outputFifo.setSize(48000);

    // H5/H6: prime the output FIFO with one model hop of silence.
    //
    // The drain loop reads whatever the FIFO holds and zero-fills the deficit.
    // With no priming the cushion accumulated by accident, so startup underran
    // and spliced digital silence into the signal (measured: 448 samples at
    // 48 kHz / 512, in runs of 32), and the resulting delay then depended on
    // block geometry (4 samples at a 480-multiple, 451 at 512) while the
    // reported latency stayed a constant.
    //
    // Input and output rates are equal and the model is 1:1, so the only
    // mismatch is quantisation to hop-sized frames. One hop of cushion bounds
    // the worst-case deficit, and the FIFO can no longer underrun.
    //
    // This is also what the existing 1920 figure already assumed: 1440 samples
    // of model algorithmic delay ((960-480) + 2*480) plus 480 of cushion. The
    // number was right; the priming that would have made it true was missing.
    outputFifo.pushSilence(modelFrameLength);

    // 7a: DERIVED from the loaded model, not hardcoded.
    //
    //   model delay = (fft_size - hop_size) + lookahead * hop_size
    //   reported    = model delay + one hop of quantisation cushion
    //
    // For the standard archive that is (960-480) + 2*480 + 480 = 1920, which is
    // exactly the literal this replaces. The literal was never wrong; it was
    // unmaintainable. The low-latency archive has lookahead 0 and therefore a
    // true latency of 960, so shipping it against a hardcoded 1920 would
    // over-report by 20 ms and a compensating host would play the track early:
    // the same defect class as H6, which is what motivated deriving it.
    //
    // libDF's C API could not supply fft_size or lookahead. alt_df_info can,
    // which is why this had to wait for the shim.
    derivedLatency48k = usable
        ? dfProcessor->getModelDelaySamples() + modelFrameLength
        : 0;

    // Report zero latency when the model could not be loaded. The bypass path
    // (H7) passes audio through untouched, so asking the host to delay every
    // other track by 40 ms for a plugin that is not processing is simply wrong.
    const int latencyInHost = juce::roundToInt(derivedLatency48k * (sampleRate / 48000.0));
    setLatencySamples(latencyInHost);

    // M5: the tail must agree with the latency. It used to be a second hardcoded
    // 1920/48000, so on the failed-load path the plugin reported zero latency
    // and a 40 ms tail at the same time.
    tailLengthSeconds.store(latencyInHost / sampleRate, std::memory_order_relaxed);

    // L6/H2: adopt the parameter's CURRENT value with no ramp. This is the
    // priming case, and here it is one call rather than the flag-and-race the
    // first L6 attempt needed: setCurrentAndTargetValue cannot be defeated by
    // thread ordering because no other thread is involved.
    //
    // H2 was that a fresh model reverted to its hardcoded 100 dB while the UI
    // still showed the user's setting. The model is now always at 100 and the
    // limit lives entirely on this side, so the failure has no mechanism left.
    // M7: one delay line per channel, primed with the reported latency so a
    // bypassed track stays aligned with the host's compensation.
    {
        const int channels = juce::jmax(1, getTotalNumInputChannels());
        bypassDelay.clear();
        bypassDelay.resize((size_t) channels);
        for (auto& d : bypassDelay) {
            d.setSize(latencyInHost + samplesPerBlock + 1);
            d.pushSilence(latencyInHost);
        }
        bypassScratch.assign((size_t) samplesPerBlock, 0.0f);
    }
    bypassMix.reset(sampleRate, kBypassRampSeconds);
    bypassMix.setCurrentAndTargetValue(
        (bypassParam != nullptr && bypassParam->get()) ? 1.0f : 0.0f);

    attenMix.reset(sampleRate, kAttenRampSeconds);
    attenMix.setCurrentAndTargetValue(attenLimitToDryMix(attenParam->load(std::memory_order_relaxed)));

    // C5: remember what we sized the resample buffers for, so processBlock can
    // refuse a block larger than we allocated instead of writing past the end.
    preparedBlockSize = samplesPerBlock;

    // init resampler
    resamplerHandler = std::make_unique<Resampler<1, 1>>(sampleRate, 48000.0);
    double maxRatio = 48000.0 / sampleRate;
    int maxResampledSize = (int)(samplesPerBlock * maxRatio) + 128; // +128 for safety margin
    resampleInBuffer.resize(maxResampledSize);
    resampleOutBuffer.resize(maxResampledSize);
    monoBuffer.assign((size_t) samplesPerBlock, 0.0f);

    // M1: the latency-aligned dry path. Priming with the full reported latency in
    // the 48 kHz domain means a read of N samples returns the input from N
    // samples ago at exactly the delay the host is compensating for, so a
    // fallback splice stays phase-aligned with the wet signal.
    // 48 kHz domain, so this is the derived figure directly rather than the
    // host-rate one.
    const int dryDelaySamples = derivedLatency48k;
    dryDelay.setSize(48000);
    dryDelay.pushSilence(dryDelaySamples);
    primedDryDelay = dryDelaySamples;   // so reset() re-primes the same amount
    dryScratch.assign((size_t) (maxResampledSize + modelFrameLength), 0.0f);

    // M1: hand the model to the worker. It becomes the sole owner, so nothing
    // else can observe a half-published or freed DFState (completing H3).
    if (usable)
        worker.start(dfProcessor.get(), modelFrameLength);
    else
        worker.stop();

    // H3: published LAST, with release ordering, after every buffer and FIFO it
    // guards has been sized. The audio thread acquires it in processBlock, so it
    // can never observe modelLoaded == true against half-built geometry.
    modelLoaded.store(usable, std::memory_order_release);
}

bool AltDenoiserProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& in  = layouts.getMainInputChannelSet();
    const auto& out = layouts.getMainOutputChannelSet();

    if (in.isDisabled() || out.isDisabled())
        return false;

    // The processing chain is mono in the middle and fans back out, so anything
    // beyond mono or stereo would leave the extra channels unprocessed.
    if (in != out)
        return false;

    return in == juce::AudioChannelSet::mono()
        || in == juce::AudioChannelSet::stereo();
}

void AltDenoiserProcessor::releaseResources() {
    // M12/M1: the worker holds the model, so it must be stopped before the
    // plugin is torn down or reconfigured.
    worker.stop();
    modelLoaded.store(false, std::memory_order_release);
}

void AltDenoiserProcessor::processBlockBypassed(juce::AudioBuffer<float>& buffer,
                                                juce::MidiBuffer& midi) {
    // M7: the base implementation is a bare passthrough that asserts the plugin
    // reports zero latency. This one reports 1920, so the default would land a
    // bypassed track 40 ms early against the host's compensation and break on
    // the assertion in a Debug build.
    //
    // Routing through processBlock instead means the bypassed signal takes the
    // same latency-aligned path as the parameter-driven bypass, so a host that
    // engages bypass through either route gets identical, aligned audio.
    forcedBypass = true;
    processBlock(buffer, midi);
    forcedBypass = false;
}

void AltDenoiserProcessor::reset() {
    // M2: a transport locate calls reset() without re-preparing, so anything
    // still buffered here would be replayed at the new playhead position. Clear
    // both FIFOs and restore the H5/H6 priming cushion so the first block after
    // the locate does not underrun and splice in silence.
    inputFifo.clear();
    outputFifo.clear();

    // M1: refuse every hop the worker is still carrying from before the locate.
    // Draining the queues instead would mean waiting on the worker from a path a
    // host may call on the audio thread. Measured before this barrier existed:
    // 961 leaked samples, about two hops, ending at index 1442.
    acceptFromSequence = nextInputSequence;

    // The dry delay holds a full reported-latency window of pre-locate audio.
    // Leaving it would replay that content through the fallback path: measured
    // 4631 leaked samples before this line existed.
    // M8: pushSilence instead of allocating a vector. reset() may be called on
    // the audio thread by some hosts, and this path allocated twice per call.
    // The amount comes from what prepareToPlay actually primed rather than a
    // second copy of the 1920 literal, which could drift out of step with it.
    dryDelay.clear();
    dryDelay.pushSilence(primedDryDelay);

    // M7: the bypass delays hold a full latency window of pre-locate audio too.
    for (auto& d : bypassDelay) {
        const int primed = juce::jmax(0, getLatencySamples());
        d.clear();
        d.pushSilence(primed);
    }

    if (modelFrameLength <= 0)
        return;

    // Restore the H5/H6 priming cushion so the first block after the locate does
    // not underrun and splice in silence.
    outputFifo.pushSilence(modelFrameLength);

    // The model's own state is deliberately NOT reset. libDF exposes no reset
    // entry point, and df_free plus df_create would re-read and re-parse the
    // archive, far too slow for a locate and a crash surface besides (C1).
    //
    // An earlier version of this function flushed silent frames through the
    // model on the theory that its overlap-add tail and lookahead would
    // otherwise be replayed. Measurement refuted that: with the flush disabled
    // the residue after a locate is identical, 3 samples ending at index 482,
    // which is the resampler's 4-sample Catmull-Rom window arriving just after
    // the priming cushion. The flush was removed rather than left as unjustified
    // work on a path a host may call from the audio thread.
    //
    // Measured at attenuation 0, where the model returns its input immediately.
    // The fully-processing case is not covered by that measurement.
}


void AltDenoiserProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // H7: when the model is unavailable, pass audio through unchanged rather
    // than calling buffer.clear(). Muting the track reads as a broken DAW and
    // gives the user nothing to diagnose; a denoiser that cannot load should
    // simply not denoise. The meters are still updated below so the UI shows
    // signal arriving and leaving, instead of freezing at their last values and
    // making a silent plugin look healthy.
    const bool modelAvailable = modelLoaded.load(std::memory_order_acquire)
                                && dfProcessor != nullptr && dfProcessor->isReady();

    // M7: capture the untouched input FIRST. Everything below works in place,
    // so this is the only point at which the dry signal still exists.
    {
        const int n = buffer.getNumSamples();
        for (int ch = 0; ch < totalNumInputChannels && ch < (int) bypassDelay.size(); ++ch)
            bypassDelay[(size_t) ch].push(buffer.getReadPointer(ch), n);
    }

    // M10/L4: sample peak across ALL input channels, held until the editor
    // consumes it. AudioBuffer::getMagnitude(start, num) scans every channel, so
    // a right-channel-only source no longer reads as silence on the input meter.
    // Pushed before the guard below, so a refused block still shows signal.
    if (totalNumInputChannels > 0)
        inputLevel.push(buffer.getMagnitude(0, buffer.getNumSamples()));

    // C5: the resample buffers were sized from the samplesPerBlock the host
    // declared in prepareToPlay. A host that then delivers a larger block would
    // make the resampler write past the end of those vectors: preparing for 128
    // at 44.1 kHz allocates 267 floats, and a 512-sample block writes 292 floats
    // (1168 bytes) beyond it. VST3 treats maxSamplesPerBlock as binding, but
    // nothing here should depend on the host honouring it. Pass the audio
    // through untouched rather than corrupting the heap.
    if (buffer.getNumSamples() > preparedBlockSize) {
        jassertfalse;
        // Passing through untouched, so what leaves equals what arrived.
        if (totalNumOutputChannels > 0)
            outputLevel.push(buffer.getMagnitude(0, buffer.getNumSamples()));
        return;
    }

    // clear and parameter update
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

  if (modelAvailable) {
    // No null fallback: attenParam is set in the constructor from a parameter
    // createParameterLayout guarantees exists, so the branch is unreachable.
    attenMix.setTargetValue(attenLimitToDryMix(attenParam->load(std::memory_order_relaxed)));

    int hostNumSamples = buffer.getNumSamples();

    // Latched once per block: the resampler callback may run several times and
    // this must not change underneath it.
    const bool offlineRender = isNonRealtime();

    // H4: the model has one channel. The old code read only channel 0 and later
    // overwrote channel 1 with a copy, so anything present only in the right
    // input was discarded before inference: hard-panned content vanished and an
    // M/S or dual-mic recording lost half its information. Sum to mono first so
    // every input channel reaches the model. The wet result is still mono and is
    // fanned back out below, which is inherent to a one-channel model.
    {
        auto* mono = monoBuffer.data();
        if (totalNumInputChannels >= 2) {
            const auto* left  = buffer.getReadPointer(0);
            const auto* right = buffer.getReadPointer(1);
            for (int i = 0; i < hostNumSamples; ++i)
                mono[i] = 0.5f * (left[i] + right[i]);
        } else {
            juce::FloatVectorOperations::copy(mono, buffer.getReadPointer(0), hostNumSamples);
        }
    }

    // resample
    float* sourceInputPtrs[] = { monoBuffer.data() }; float* sourceOutputPtrs[] = { monoBuffer.data() };
    float* targetInputPtrs[] = { resampleInBuffer.data() };   float* targetOutputPtrs[] = { resampleOutBuffer.data() };
    resamplerHandler->process(
        sourceInputPtrs,
        sourceOutputPtrs,
        targetInputPtrs,
        targetOutputPtrs,
        hostNumSamples,
        // lambda callback
        [&](float* const* input_buffers, float* const* output_buffers, int sample_count_48k) {
            
            auto* readPtr = input_buffers[0];   // 48kHz input
            auto* writePtr = output_buffers[0]; // 48kHz output

            // Keep the latency-aligned dry copy advancing in lockstep, whether
            // or not it ends up being needed this block.
            dryDelay.push(readPtr, sample_count_48k);
            dryDelay.peek(dryScratch.data(), sample_count_48k);
            dryDelay.discard(sample_count_48k);

            // M1: slice to whole hops and hand them to the worker. The audio
            // thread no longer calls the model; it only accumulates and memcpys.
            inputFifo.push(readPtr, sample_count_48k);
            while (inputFifo.getAvailable() >= modelFrameLength) {
                inputFifo.peek(tempInputFrame.data(), modelFrameLength);
                inputFifo.discard(modelFrameLength);
                worker.submit(nextInputSequence++, tempInputFrame.data());
            }

            // Collect whatever the worker has finished, discarding anything that
            // predates the last reset. Tagging frames rather than draining the
            // queues means a locate costs no wait on this thread.
            unsigned producedSequence = 0;
            while (worker.collect(producedSequence, tempOutputFrame.data()))
                if (producedSequence >= acceptFromSequence)
                    outputFifo.push(tempOutputFrame.data(), modelFrameLength);

            // M1: offline rendering must WAIT rather than fall back.
            //
            // A bounce calls processBlock as fast as the CPU allows, so the one
            // hop of cushion the realtime path relies on never receives any
            // wall-clock time to refill. Measured before this branch existed, an
            // offline render fell back to dry for 20000 samples and dropped 25
            // frames, i.e. it produced almost no processed audio at all.
            //
            // Blocking here is safe precisely because this is not the realtime
            // path. The wait is bounded so a stalled worker degrades to the dry
            // fallback below instead of hanging the render.
            if (offlineRender) {
                const int deadlineMs = 2000;
                while (outputFifo.getAvailable() < sample_count_48k
                       && worker.getInFlight() > 0) {
                    if (! worker.waitForOutput(deadlineMs))
                        break;
                    while (worker.collect(producedSequence, tempOutputFrame.data()))
                        if (producedSequence >= acceptFromSequence)
                            outputFifo.push(tempOutputFrame.data(), modelFrameLength);
                }
            }

            int samplesAvailable = outputFifo.getAvailable();
            int samplesToRead = juce::jmin(samplesAvailable, sample_count_48k);
            if (samplesToRead > 0) {
                outputFifo.peek(writePtr, samplesToRead);
                outputFifo.discard(samplesToRead);
            }
            if (samplesToRead < sample_count_48k) {
                // The worker was late. Splice in the latency-aligned dry signal
                // rather than digital silence: an unprocessed moment is far less
                // audible than a click, and it stays phase-aligned because the
                // dry delay matches the latency the host compensates for.
                const int deficit = sample_count_48k - samplesToRead;
                juce::FloatVectorOperations::copy(writePtr + samplesToRead,
                                                  dryScratch.data() + samplesToRead,
                                                  deficit);
                fallbackSamples.fetch_add((unsigned) deficit, std::memory_order_relaxed);
            }

            // L6: the attenuation limit, applied HERE rather than inside the model.
            //
            // libDF implements atten_lim as spec_enh = (1-lim)*enh + lim*noisy
            // with lim = 10^(-db/20), before a linear WOLA synthesis, so it is
            // exactly a wet/dry crossfade in the time domain. dryScratch already
            // holds that same latency-aligned dry signal (built for M1's
            // fallback), so doing the mix here is equivalent to letting the model
            // do it, and it can be smoothed PER SAMPLE.
            //
            // The first attempt at L6 slewed a cross-thread parameter on the
            // worker instead. It froze at hop-aligned block sizes, varied its
            // rate 3.5x-100x with host buffer size, and reintroduced H2. None of
            // those failures is expressible in this shape: there is no
            // cross-thread value, no hop quantisation and no priming race.
            {
                auto* dry = dryScratch.data();
                for (int i = 0; i < sample_count_48k; ++i) {
                    const float mix = attenMix.getNextValue();   // 0 = wet, 1 = dry
                    writePtr[i] += mix * (dry[i] - writePtr[i]);
                }
            }
        }
    );

    // Fan the mono wet result back out to every output channel.
    for (int ch = 0; ch < totalNumOutputChannels; ++ch)
        juce::FloatVectorOperations::copy(buffer.getWritePointer(ch), monoBuffer.data(), hostNumSamples);
  }   // if (modelAvailable); otherwise the buffer passes through untouched

    // M7: crossfade to the latency-aligned dry signal.
    //
    // The model keeps running while bypassed. That is deliberate: it is the only
    // way un-bypassing cannot flush stale audio or re-trigger the H5 startup
    // splice, because nothing is ever flushed and the FIFOs never leave their
    // target fill. The cost is that bypass saves no CPU.
    {
        const int n = buffer.getNumSamples();
        const bool wantBypass = forcedBypass
                             || (bypassParam != nullptr && bypassParam->get());
        bypassMix.setTargetValue(wantBypass ? 1.0f : 0.0f);

        for (int ch = 0; ch < totalNumOutputChannels && ch < (int) bypassDelay.size(); ++ch) {
            auto& delay = bypassDelay[(size_t) ch];
            if (! delay.peek(bypassScratch.data(), n))
                continue;

            // Each channel walks its own copy of the smoother so they ramp
            // identically; the master advances once, below.
            auto chMix = bypassMix;
            auto* w = buffer.getWritePointer(ch);
            for (int i = 0; i < n; ++i) {
                const float m = chMix.getNextValue();
                w[i] += m * (bypassScratch[(size_t) i] - w[i]);
            }
        }
        bypassMix.skip(n);
        for (auto& d : bypassDelay)
            d.discard(n);
    }

    // Measured after processing (or after the H7 bypass), so the meter reflects
    // what actually leaves the plugin.
    if (totalNumOutputChannels > 0)
        outputLevel.push(buffer.getMagnitude(0, buffer.getNumSamples()));
}

juce::AudioProcessorEditor* AltDenoiserProcessor::createEditor() {return new AltDenoiserEditor(*this, apvts);}
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter(){return new AltDenoiserProcessor();}