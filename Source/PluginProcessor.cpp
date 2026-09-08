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
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        "atten_lim", 
        "Attenuation Limit", 
        range, 
        100.0f 
    ));

    return layout;
}

void AltDenoiserProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    inputFifo.setSize(48000);
    outputFifo.setSize(48000);
    tempInputFrame.resize(480, 0.0f);
    tempOutputFrame.resize(480, 0.0f);

    int latencyInHost = juce::roundToInt(1920.0 * (sampleRate / 48000.0));
    setLatencySamples(latencyInHost);

    if (dfProcessor->initialize()) {
        modelLoaded = true;
    } else {
        modelLoaded = false;
    }

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
    const bool modelAvailable = modelLoaded && dfProcessor != nullptr && dfProcessor->isReady();

    // C5: the resample buffers were sized from the samplesPerBlock the host
    // declared in prepareToPlay. A host that then delivers a larger block would
    // make the resampler write past the end of those vectors: preparing for 128
    // at 44.1 kHz allocates 267 floats, and a 512-sample block writes 292 floats
    // (1168 bytes) beyond it. VST3 treats maxSamplesPerBlock as binding, but
    // nothing here should depend on the host honouring it. Pass the audio
    // through untouched rather than corrupting the heap.
    if (buffer.getNumSamples() > preparedBlockSize) {
        jassertfalse;
        return;
    }
 
    // input rms
    const float smoothAlpha = 0.5f; // smoothing factor for RMS
    float currentInRMS = 0.0f;
    if (totalNumInputChannels > 0)
        currentInRMS = buffer.getRMSLevel(0, 0, buffer.getNumSamples());
    float oldIn = inputRmsLevel.load();
    inputRmsLevel.store(oldIn * (1.0f - smoothAlpha) + currentInRMS * smoothAlpha);

    // clear and parameter update
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

  if (modelAvailable) {
    float newAttenLim = *apvts.getRawParameterValue("atten_lim");
    if (std::abs(newAttenLim - lastAttenLim) > 0.01f) {
        dfProcessor->setAttenLim(newAttenLim);
        lastAttenLim = newAttenLim;
    }

    int hostNumSamples = buffer.getNumSamples();

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
            inputFifo.push(readPtr, sample_count_48k);

            // predict
            while (inputFifo.getAvailable() >= 480) {
                inputFifo.peek(tempInputFrame.data(), 480);
                inputFifo.discard(480);
                dfProcessor->processFrame(tempInputFrame.data(), tempOutputFrame.data());
                outputFifo.push(tempOutputFrame.data(), 480);
            }
            // outputfifo ---> writePtr
            int samplesAvailable = outputFifo.getAvailable();
            int samplesToRead = juce::jmin(samplesAvailable, sample_count_48k);
            if (samplesToRead > 0) {
                outputFifo.peek(writePtr, samplesToRead);
                outputFifo.discard(samplesToRead);
            }
            if (samplesToRead < sample_count_48k) {
                juce::FloatVectorOperations::clear(writePtr + samplesToRead, sample_count_48k - samplesToRead);
            }
        }
    );

    // Fan the mono wet result back out to every output channel.
    for (int ch = 0; ch < totalNumOutputChannels; ++ch)
        juce::FloatVectorOperations::copy(buffer.getWritePointer(ch), monoBuffer.data(), hostNumSamples);
  }   // if (modelAvailable); otherwise the buffer passes through untouched

    // output RMS
    float currentOutRMS = 0.0f;
    if (totalNumOutputChannels > 0)
        currentOutRMS = buffer.getRMSLevel(0, 0, buffer.getNumSamples());        
    float oldOut = outputRmsLevel.load();
    outputRmsLevel.store(oldOut * (1.0f - smoothAlpha) + currentOutRMS * smoothAlpha);
}

juce::AudioProcessorEditor* AltDenoiserProcessor::createEditor() {return new AltDenoiserEditor(*this, apvts);}
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter(){return new AltDenoiserProcessor();}