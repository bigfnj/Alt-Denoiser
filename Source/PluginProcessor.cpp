#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <juce_core/juce_core.h>

AltDenoiserProcessor::AltDenoiserProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    dfProcessor = std::make_unique<DeepFilterNetProcessor>(48000);
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
    {
        const std::vector<float> primingSilence((size_t) modelFrameLength, 0.0f);
        outputFifo.push(primingSilence.data(), modelFrameLength);
    }

    // Report zero latency when the model could not be loaded. The bypass path
    // (H7) passes audio through untouched, so asking the host to delay every
    // other track by 40 ms for a plugin that is not processing is simply wrong.
    const int latencyInHost = usable
        ? juce::roundToInt(1920.0 * (sampleRate / 48000.0))
        : 0;
    setLatencySamples(latencyInHost);

    // H2: initialize() builds a fresh DFState at a hardcoded 100 dB, but
    // lastAttenLim kept its old value, so the change-detector in processBlock
    // saw no delta and never re-applied the user's setting. Measured effect: a
    // knob left at 0 dB rendered a tone 45 dB quieter after a re-prepare, so
    // the plugin silently applied maximum reduction while the UI read zero.
    // Resetting the sentinel forces the next block to push the real value.
    lastAttenLim = -1.0f;

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
    const int dryDelaySamples = (int) std::lround(1920.0);
    dryDelay.setSize(48000);
    {
        const std::vector<float> primingSilence((size_t) dryDelaySamples, 0.0f);
        dryDelay.push(primingSilence.data(), dryDelaySamples);
    }
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
    dryDelay.clear();
    if (dryDelay.getAvailable() == 0) {
        const std::vector<float> primingSilence((size_t) 1920, 0.0f);
        dryDelay.push(primingSilence.data(), 1920);
    }

    if (modelFrameLength <= 0)
        return;

    // Restore the H5/H6 priming cushion so the first block after the locate does
    // not underrun and splice in silence.
    const std::vector<float> primingSilence((size_t) modelFrameLength, 0.0f);
    outputFifo.push(primingSilence.data(), modelFrameLength);

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
    float newAttenLim = *apvts.getRawParameterValue("atten_lim");
    if (std::abs(newAttenLim - lastAttenLim) > 0.01f) {
        // H3 residue: published to the worker rather than applied here, because
        // the worker is concurrently inside processFrame on the same DFState.
        worker.setAttenuationLimit(newAttenLim);
        lastAttenLim = newAttenLim;
    }

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
        }
    );

    // Fan the mono wet result back out to every output channel.
    for (int ch = 0; ch < totalNumOutputChannels; ++ch)
        juce::FloatVectorOperations::copy(buffer.getWritePointer(ch), monoBuffer.data(), hostNumSamples);
  }   // if (modelAvailable); otherwise the buffer passes through untouched

    // Measured after processing (or after the H7 bypass), so the meter reflects
    // what actually leaves the plugin.
    if (totalNumOutputChannels > 0)
        outputLevel.push(buffer.getMagnitude(0, buffer.getNumSamples()));
}

juce::AudioProcessorEditor* AltDenoiserProcessor::createEditor() {return new AltDenoiserEditor(*this, apvts);}
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter(){return new AltDenoiserProcessor();}