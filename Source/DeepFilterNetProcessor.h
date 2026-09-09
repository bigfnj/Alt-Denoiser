#pragma once
#include <cstddef>
#include <cstdint>
#include "alt_df.h"

/** Owns one DeepFilterNet runtime.

    Since the alt_df shim (BACKLOG.md C1/C2) every call below reports failure
    rather than panicking out of Rust, so a corrupt model archive or an
    inference error degrades to bypass instead of aborting the host. The old
    libDF C API could not do that: df_create returned a Box and never NULL, and
    both it and df_process_frame ended in .expect().
*/
/** Which embedded archive to load.

    Both share sr 48000, hop 480 and fft 960, so a swap needs no resampler
    change, no FIFO resize and no reallocation. Only `lookahead` differs, 2
    against 0, which moves the derived latency from 1920 to 960.

    Do not renumber: the values are the AudioParameterChoice indices and are
    persisted in saved sessions.
*/
enum class DfnModel
{
    Standard   = 0,
    LowLatency = 1
};

inline constexpr int kNumDfnModels = 2;

class DeepFilterNetProcessor {
public:
    /** Human-readable name, as shown in the parameter and the editor. Valid for
        every enumerator whether or not that archive was embedded.
    */
    static const char* getModelDisplayName (DfnModel which) noexcept;

    /** Whether this build actually embedded that archive.

        The parameter always offers both choices regardless, so a session saved
        by a full build still restores its selection into a slim one instead of
        being clamped into a different model. A slim build hides the row in the
        editor and falls back to whatever it does have.
    */
    static bool isModelAvailable (DfnModel which) noexcept;

    /** The embedded archive's bytes, or nullptr/0 if this build did not embed
        it. The only place BinaryData is named, so nothing else has to be
        recompiled against which archives a given build happens to carry.
    */
    static const char* getEmbeddedModel (DfnModel which, int& sizeOut) noexcept;

    /** The first archive this build actually embedded. Always valid: CMake
        refuses to configure with none, and a #error backs that up.
    */
    static DfnModel getFirstAvailableModel() noexcept;

    DeepFilterNetProcessor() = default;
    ~DeepFilterNetProcessor();

    DeepFilterNetProcessor (const DeepFilterNetProcessor&) = delete;
    DeepFilterNetProcessor& operator= (const DeepFilterNetProcessor&) = delete;

    /** Loads the embedded model. Returns false and leaves the object unusable
        if the archive cannot be read; the caller is expected to bypass.
    */
    bool initialize (DfnModel which = DfnModel::Standard);

    /** Which model actually got loaded. Not necessarily what was asked for: a
        build that did not embed the requested archive falls back.
    */
    DfnModel getLoadedModel() const noexcept { return loadedModel; }

    /** Loads a model from an arbitrary buffer.

        initialize() is this with the embedded archive. It is separate so the
        offline harness can hand over a deliberately corrupt buffer, which is
        the only way to exercise the failure path that the whole shim exists
        for: with the old API that test could not be written, because the
        failure was an abort.
    */
    bool initializeFromMemory (const void* modelData, int modelSize);

    // There is no setAttenLim. It went dead when L6 moved the attenuation
    // crossfade onto the audio thread, and alt_df has no setter to call.

    /** Denoises one frame of exactly getFrameLength() samples.

        Returns false if inference failed, having copied input to output so the
        caller emits the dry signal rather than repeating the previous frame.
    */
    bool processFrame (const float* input, float* output);

    bool isReady() const noexcept { return state != nullptr; }

    /** Hop size the loaded model expects, in samples at 48 kHz. 0 when nothing
        is loaded.

        M3: every archive libDF can load uses 480 today, but the caller must
        still size its frames from this rather than assuming. alt_df now range
        checks the buffer length itself, so a mismatch is a refused call rather
        than the silent heap overrun it was through libDF's from_shape_ptr.
    */
    size_t getFrameLength() const noexcept { return state != nullptr ? (size_t) info.hop_size : 0; }

    /** Geometry of the loaded model. Only meaningful while isReady(). */
    const AltDfInfo& getInfo() const noexcept { return info; }

    /** Algorithmic delay of the loaded model in samples at its own rate:
        (fft_size - hop_size) + lookahead * hop_size.

        This is the model's contribution only. The plugin adds one hop of
        quantisation cushion on top; see AltDenoiserProcessor::prepareToPlay.
        1440 for the standard archive, 480 for the low-latency one.
    */
    int getModelDelaySamples() const noexcept
    {
        return state != nullptr
                 ? (int) (info.fft_size - info.hop_size) + (int) (info.lookahead * info.hop_size)
                 : 0;
    }

    /** Status of the last call that returned one. For diagnostics and tests. */
    AltDfStatus getLastStatus() const noexcept { return lastStatus; }

    /** How many frames have failed inference since load. Non-zero means the
        model is returning errors and the wet path is passing dry audio.
    */
    int getProcessFailures() const noexcept { return processFailures; }

private:
    void release();

    AltDf*      state  = nullptr;
    AltDfInfo   info   {};
    AltDfStatus lastStatus = ALT_DF_OK;
    DfnModel    loadedModel = DfnModel::Standard;
    int         processFailures = 0;
};
