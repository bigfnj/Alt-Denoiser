#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "DeepFilterNetProcessor.h"
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

    void getStateInformation(juce::MemoryBlock& destData) override
    {
        auto state = apvts.copyState();
        std::unique_ptr<juce::XmlElement> xml(state.createXml());
        copyXmlToBinary(*xml, destData);
    }

    void setStateInformation(const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

        if (xmlState.get() != nullptr)
        {
            if (xmlState->hasTagName(apvts.state.getType()))
            {
                apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
            }
        }
    }

    std::atomic<float> inputRmsLevel { 0.0f };
    std::atomic<float> outputRmsLevel { 0.0f };
    juce::AudioProcessorValueTreeState apvts;

private:
    std::unique_ptr<DeepFilterNetProcessor> dfProcessor;

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

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AltDenoiserProcessor)
};