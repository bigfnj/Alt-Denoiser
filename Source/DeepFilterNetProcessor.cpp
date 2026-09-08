#include "DeepFilterNetProcessor.h"
#include "BinaryData.h"
#include <fstream>
#include <iostream>
#include <string>
#include <juce_core/juce_core.h>

DeepFilterNetProcessor::DeepFilterNetProcessor(uint32_t sampleRate)
    : sampleRate(sampleRate) {
}

DeepFilterNetProcessor::~DeepFilterNetProcessor() {
    if (state != nullptr) {
        df_free(state);
    }
}

bool DeepFilterNetProcessor::initialize()
{
    const char* modelData = AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gz;
    const int modelSize   = AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gzSize;

    if (modelSize <= 0 || modelData == nullptr)
    {
        DBG("Embedded model data invalid or missing");
        return false;
    }

    // H1: release any previous state before replacing it. prepareToPlay calls
    // initialize() on every sample-rate and buffer-size change, and the only
    // other df_free is in the destructor, so each change leaked a whole model.
    if (state != nullptr)
    {
        df_free(state);
        state = nullptr;
    }

    // C3/C4/M6: one unique file per call, deleted as soon as it has been read.
    // The old code wrote to a fixed name in the shared temp directory, so two
    // plugin instances raced on the same path; JUCE's FileOutputStream seeks to
    // end-of-file, so repeated runs appended instead of replacing, meaning one
    // short write poisoned the file permanently; and nothing ever deleted it.
    auto tempModel = juce::File::createTempFile(".tar.gz");

    bool staged = false;
    {
        juce::FileOutputStream stream(tempModel);
        if (stream.openedOk())
        {
            // write()'s bool used to be discarded, which turned a disk-full or
            // antivirus-blocked write into a panic inside Rust rather than a
            // clean failure here. flush() returns void, so the write status is
            // read back from getStatus() instead.
            staged = stream.write(modelData, (size_t) modelSize);
            stream.flush();
            staged = staged && stream.getStatus().wasOk();
        }
        else
        {
            DBG("Failed to open temp file for model");
        }
    }

    // C1 mitigation: df_create CANNOT report failure. DFState::new returns Self,
    // every error inside is an .expect(), and a panic crossing extern "C" aborts
    // the host process. So validate the archive here, where failure is still
    // recoverable, rather than letting Rust discover it.
    if (! staged || tempModel.getSize() != (juce::int64) modelSize)
    {
        DBG("Failed to stage the embedded model");
        tempModel.deleteFile();
        return false;
    }

    state = df_create(tempModel.getFullPathName().toRawUTF8(), 100.0f, nullptr);

    // DfParams::new reads the whole archive during df_create, so nothing needs
    // the file afterwards.
    tempModel.deleteFile();

    return state != nullptr;
}

void DeepFilterNetProcessor::setAttenLim(float limitDB) {
    if (state != nullptr) {
        // 调用 df.h 中定义的 C 接口
        df_set_atten_lim(state, limitDB);
    }
}

void DeepFilterNetProcessor::processFrame(const float* input, float* output) {
    if (state) {
        df_process_frame(state, (float*)input, output);
    }
}