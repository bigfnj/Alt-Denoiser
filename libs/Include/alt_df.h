/* alt_df.h - C ABI for the alt_df shim crate (libs/alt_df).
 *
 * This replaces libDF's own df.h. That API could not report failure: df_create
 * returned a Box and never NULL, and both it and df_process_frame ended in
 * .expect(), so a corrupt model archive or a tract error panicked across
 * extern "C" and aborted the host. See libs/alt_df/src/lib.rs for the detail.
 *
 * Everything here returns a status, validates its arguments before touching
 * memory, and cannot panic out of Rust.
 *
 * Keep this in sync with libs/alt_df/src/lib.rs by hand. It is small and
 * changes rarely, so it is not generated; the offline harness has a test that
 * fails if the two disagree about hop size or about the status of a rejected
 * archive.
 */
#ifndef ALT_DF_H
#define ALT_DF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a loaded model. */
typedef struct AltDf AltDf;

/* Result of every entry point. 0 is success. */
typedef enum AltDfStatus {
    ALT_DF_OK          = 0,
    ALT_DF_NULL_ARG    = 1,  /* a required pointer argument was NULL          */
    ALT_DF_BAD_MODEL   = 2,  /* the archive is not a readable DFN tar.gz      */
    ALT_DF_INIT        = 3,  /* the archive parsed but tract could not build  */
    ALT_DF_BAD_LENGTH  = 4,  /* a buffer length != the model's hop size       */
    ALT_DF_PROCESS     = 5,  /* inference failed on this frame                */
    ALT_DF_PANIC       = 6   /* a panic was caught; the host is still alive   */
} AltDfStatus;

/* Geometry of the loaded model.
 *
 * lookahead is the field that differs between the standard archive (2) and the
 * low-latency one (0), and is why the plugin's reported latency is derived
 * rather than hardcoded:
 *
 *     latency = (fft_size - hop_size) + lookahead * hop_size + hop_size
 *
 * which is 1920 for the standard model and 960 for the low-latency one. The
 * trailing hop is the plugin's own quantisation cushion, not the model's.
 */
typedef struct AltDfInfo {
    uint32_t sr;
    uint32_t hop_size;
    uint32_t fft_size;
    uint32_t lookahead;
    uint32_t ch;
} AltDfInfo;

/* Builds a runtime from a .tar.gz DeepFilterNet ONNX archive held in memory.
 *
 * The bytes are read during the call and not retained, so they may be freed
 * immediately afterwards. On failure *out_state is set to NULL and nothing is
 * allocated.
 *
 * atten_lim is in dB; 100 or above disables the limit.
 */
AltDfStatus alt_df_create(const uint8_t* model_data,
                          size_t         model_len,
                          float          atten_lim,
                          AltDf**        out_state);

/* Reports the loaded model's geometry. */
AltDfStatus alt_df_info(const AltDf* st, AltDfInfo* out_info);

/* Denoises one hop.
 *
 * input_len and output_len must both equal info.hop_size; they are checked
 * before any dereference. This is the bounds check libDF did not have: its
 * df_process_frame built ndarray views with from_shape_ptr and its only guard
 * was a debug_assert compiled out in release, so a short buffer was a silent
 * heap overrun rather than an error.
 *
 * out_lsnr receives the frame's local SNR in dB and may be NULL.
 * Real-time safe: no allocation, no locking, no I/O on the success path.
 */
AltDfStatus alt_df_process_frame(AltDf*       st,
                                 const float* input,
                                 size_t       input_len,
                                 float*       output,
                                 size_t       output_len,
                                 float*       out_lsnr);

/* There is deliberately no alt_df_set_atten_lim. libDF implements atten_lim as
 * a plain wet/dry crossfade in the linear domain just before WOLA synthesis
 * (BACKLOG.md L6), so the plugin does that crossfade itself on the audio thread
 * where it is sample-accurate. The model is created at 100 dB, libDF's "no
 * limit" value, and never told otherwise.
 */

/* Frees a handle. NULL is accepted and ignored. */
void alt_df_free(AltDf* st);

/* A short static description of a status, for logging. Never NULL. */
const char* alt_df_status_str(AltDfStatus status);

#ifdef __cplusplus
}
#endif

#endif /* ALT_DF_H */
