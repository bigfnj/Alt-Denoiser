#include "DeepFilterNetProcessor.h"
#include "BinaryData.h"
#include <cstring>
#include <juce_core/juce_core.h>

DeepFilterNetProcessor::~DeepFilterNetProcessor() {
    release();
}

void DeepFilterNetProcessor::release()
{
    if (state != nullptr)
    {
        alt_df_free (state);
        state = nullptr;
    }
    info = {};
    processFailures = 0;
}

bool DeepFilterNetProcessor::initialize()
{
    return initializeFromMemory (AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gz,
                                 AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gzSize);
}

bool DeepFilterNetProcessor::initializeFromMemory (const void* modelData, int modelSize)
{
    // H1: release any previous state before replacing it. prepareToPlay calls
    // this on every sample-rate and buffer-size change, and the only other free
    // is in the destructor, so each change used to leak a whole model.
    release();

    if (modelData == nullptr || modelSize <= 0)
    {
        lastStatus = ALT_DF_NULL_ARG;
        DBG ("Model data invalid or missing");
        return false;
    }

    // Straight from memory. There is no temp file any more, which is what
    // closed C1/C3/C4/M6 outright rather than mitigating them: the old path
    // wrote the embedded archive to a file, and two plugin instances raced on
    // the same name, JUCE's FileOutputStream appended rather than replaced so
    // one short write poisoned it permanently, and nothing deleted it.
    //
    // alt_df_create reads the bytes during the call and does not retain them,
    // so the BinaryData pointer needs no particular lifetime beyond this line.
    lastStatus = alt_df_create (static_cast<const uint8_t*> (modelData),
                                (size_t) modelSize,
                                100.0f,
                                &state);

    if (lastStatus != ALT_DF_OK || state == nullptr)
    {
        DBG ("alt_df_create failed: " << alt_df_status_str (lastStatus));
        state = nullptr;
        return false;
    }

    // Cache the geometry once. Everything downstream sizes itself from this,
    // and it cannot change while a model is loaded.
    const auto infoStatus = alt_df_info (state, &info);
    if (infoStatus != ALT_DF_OK)
    {
        lastStatus = infoStatus;
        DBG ("alt_df_info failed: " << alt_df_status_str (infoStatus));
        release();
        return false;
    }

    return true;
}

bool DeepFilterNetProcessor::processFrame (const float* input, float* output) {
    if (state == nullptr || input == nullptr || output == nullptr)
        return false;

    const size_t hop = (size_t) info.hop_size;

    // lsnr is the frame's local SNR in dB. Discarded for now; BACKLOG.md notes
    // it as the natural feed for a meaningful meter.
    float lsnr = 0.0f;
    const auto status = alt_df_process_frame (state, input, hop, output, hop, &lsnr);

    if (status != ALT_DF_OK)
    {
        // Copy the input through rather than leaving the caller's buffer at
        // whatever the previous frame wrote, which would repeat that frame
        // forever. A dry frame is the correct degradation: it is what the
        // plugin already does when no model is loaded at all.
        std::memcpy (output, input, hop * sizeof (float));
        ++processFailures;
        lastStatus = status;
        return false;
    }

    return true;
}
