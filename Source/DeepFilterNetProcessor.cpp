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
    processFailures.store (0, std::memory_order_relaxed);
}

// Which archives this build embedded. Set from CMake's ALTDENOISER_MODELS.
// Defaulted here so the file still compiles if it is ever built outside that
// CMakeLists, rather than silently taking a branch nobody intended.
#ifndef ALTDENOISER_HAS_STANDARD_MODEL
 #define ALTDENOISER_HAS_STANDARD_MODEL 1
#endif
#ifndef ALTDENOISER_HAS_LL_MODEL
 #define ALTDENOISER_HAS_LL_MODEL 0
#endif

#if ! (ALTDENOISER_HAS_STANDARD_MODEL || ALTDENOISER_HAS_LL_MODEL)
 #error "No model archive was embedded. Check ALTDENOISER_MODELS."
#endif

const char* DeepFilterNetProcessor::getModelDisplayName (DfnModel which) noexcept
{
    switch (which)
    {
        case DfnModel::Standard:   return "Standard";
        case DfnModel::LowLatency: return "Low latency";
    }
    return "Standard";
}

bool DeepFilterNetProcessor::isModelAvailable (DfnModel which) noexcept
{
    switch (which)
    {
        case DfnModel::Standard:   return ALTDENOISER_HAS_STANDARD_MODEL != 0;
        case DfnModel::LowLatency: return ALTDENOISER_HAS_LL_MODEL != 0;
    }
    return false;
}

const char* DeepFilterNetProcessor::getEmbeddedModel (DfnModel which, int& sizeOut) noexcept
{
    sizeOut = 0;

    switch (which)
    {
        case DfnModel::Standard:
           #if ALTDENOISER_HAS_STANDARD_MODEL
            sizeOut = AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gzSize;
            return AltDenoiserBinaryData::DeepFilterNet3_onnx_tar_gz;
           #else
            return nullptr;
           #endif

        case DfnModel::LowLatency:
           #if ALTDENOISER_HAS_LL_MODEL
            sizeOut = AltDenoiserBinaryData::DeepFilterNet3_ll_onnx_tar_gzSize;
            return AltDenoiserBinaryData::DeepFilterNet3_ll_onnx_tar_gz;
           #else
            return nullptr;
           #endif
    }

    return nullptr;
}

DfnModel DeepFilterNetProcessor::getFirstAvailableModel() noexcept
{
    for (int i = 0; i < kNumDfnModels; ++i)
        if (isModelAvailable (static_cast<DfnModel> (i)))
            return static_cast<DfnModel> (i);

    jassertfalse;   // unreachable: CMake refuses to configure with no models
    return DfnModel::Standard;
}

bool DeepFilterNetProcessor::initialize (DfnModel which)
{
    // A build that did not embed the requested archive falls back to one it did,
    // rather than refusing to load anything. Losing 20 ms of latency
    // improvement is a far better outcome than a silent bypass, and the
    // alternative would make a session saved by a full build unusable in a slim
    // one.
    if (! isModelAvailable (which))
    {
        DBG ("Model '" << getModelDisplayName (which) << "' is not embedded in this build; falling back");
        which = getFirstAvailableModel();
    }

    int size = 0;
    const char* data = getEmbeddedModel (which, size);

    const bool ok = initializeFromMemory (data, size);
    if (ok)
        loadedModel = which;
    return ok;
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

        // release() rather than a bare null assignment. alt_df_create does
        // guarantee *out_state is NULL on every non-OK return, but relying on
        // that makes this line a leak the moment the Rust side ever writes the
        // handle before a fallible step. Freeing a null is a no-op.
        release();
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
        processFailures.fetch_add (1, std::memory_order_relaxed);
        lastStatus = status;
        return false;
    }

    return true;
}
