//! A non-panicking C ABI over DeepFilterNet's tract runtime.
//!
//! # Why this crate exists
//!
//! libDF ships its own C API behind the `capi` feature, and that API cannot
//! report failure. In the pinned `libDF/src/capi.rs`:
//!
//! * `DFState::new` returns `Self`, not `Result`. Every failure inside it is an
//!   `.expect()`, and it ends in `Box::new(self)`, so `df_create` can never
//!   return NULL. A corrupt or truncated model archive panics.
//! * `df_process_frame` ends in `.expect("Failed to process DF frame")`, so a
//!   tract error mid-playback panics.
//! * `df_process_frame` builds its ndarray views with `from_shape_ptr`, which
//!   performs no bounds check, and libDF's only length guard is a
//!   `debug_assert` that is compiled out in release.
//!
//! A panic crossing an `extern "C"` boundary aborts the process. In a plugin
//! that means the DAW dies, taking the user's unsaved session with it, because
//! a file on disk was a few bytes short.
//!
//! Every entry point here returns an [`AltDfStatus`], validates its arguments
//! before touching memory, and wraps its body in [`catch_unwind`] so a panic
//! anywhere below becomes [`AltDfStatus::Panic`] rather than an abort.
//!
//! # What else it buys
//!
//! * **Loading from memory.** `alt_df_create` takes a byte range, so the host
//!   side no longer writes the embedded archive to a temp file and hands over a
//!   path. That deletes the whole staging path along with the races and
//!   leftovers that came with it (BACKLOG.md C1/C3/C4/M6).
//! * **Model metadata.** `alt_df_info` reports `sr`, `hop_size`, `fft_size`,
//!   `lookahead` and `ch`. libDF's C API exposes only the hop, which is why the
//!   plugin's reported latency had to be a hardcoded constant.
//! * **No second model in the binary.** `Cargo.toml` disables libDF's default
//!   features, so `default-model` never embeds an archive into this static
//!   library.

// The whole non-abort guarantee rests on unwinding being available. Under
// `panic = "abort"` catch_unwind compiles to a plain call and every panic below
// aborts the host, silently: the code still builds, the tests still pass, and
// the protection is simply gone. Fail the build instead.
//
// This fires for `--profile release-lto`, which the DeepFilterNet workspace
// defines with `panic = "abort"`. We build `--release`.
#[cfg(panic = "abort")]
compile_error!(
    "alt_df must be built with panic = \"unwind\". Under panic = \"abort\" the \
     catch_unwind guards at every FFI entry point are inert and a panic inside \
     tract will abort the host process."
);

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

// `df`, not `deep_filter`: libDF's Cargo.toml sets `[lib] name = "df"`, and the
// extern crate name comes from the lib name rather than the package name.
use df::tract::{DfParams, DfTract, ReduceMask, RuntimeParams};
use ndarray::{ArrayView2, ArrayViewMut2};

/// Result of every entry point. 0 is success; everything else is a failure the
/// caller is expected to handle by bypassing.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AltDfStatus {
    Ok = 0,
    /// A required pointer argument was NULL.
    NullArg = 1,
    /// The model archive could not be parsed: truncated, not gzip, or missing
    /// one of enc.onnx / erb_dec.onnx / df_dec.onnx / config.ini.
    BadModel = 2,
    /// The archive parsed but the tract runtime could not be built from it.
    Init = 3,
    /// An input or output buffer length did not equal the model's hop size.
    BadLength = 4,
    /// Inference failed on this frame.
    Process = 5,
    /// A panic was caught at the FFI boundary. The host is still alive; the
    /// state should be considered unusable and freed.
    Panic = 6,
}

/// Geometry of the loaded model. Everything the host needs to size its buffers
/// and derive its reported latency.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AltDfInfo {
    /// Model sample rate, 48000 for every DeepFilterNet3 archive.
    pub sr: u32,
    /// Samples consumed and produced per `alt_df_process_frame` call.
    pub hop_size: u32,
    /// STFT window length.
    pub fft_size: u32,
    /// Frames of lookahead. This is the term that differs between the standard
    /// and low-latency archives, and the reason the plugin's latency cannot be
    /// a constant.
    pub lookahead: u32,
    /// Channels the runtime was built for.
    pub ch: u32,
}

/// Opaque handle. Owns the tract runtime.
pub struct AltDf {
    model: DfTract,
}

/// Runs `f`, converting a panic into [`AltDfStatus::Panic`].
///
/// The default panic hook still runs first, so the message and location reach
/// stderr and end up in the host's log. That is deliberate: the status code
/// tells the plugin what to do, and stderr tells us what happened.
fn guard<F: FnOnce() -> AltDfStatus>(f: F) -> AltDfStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(_) => {
            eprintln!("alt_df: caught a panic at the FFI boundary; returning Panic");
            AltDfStatus::Panic
        }
    }
}

/// Builds a DeepFilterNet runtime from a model archive held in memory.
///
/// `model_data` / `model_len` describe a `.tar.gz` DeepFilterNet ONNX archive.
/// The bytes are read during this call and not retained, so the caller may free
/// them immediately afterwards.
///
/// `atten_lim` is the attenuation limit in dB, matching libDF's semantics: 100
/// or above disables the limit entirely.
///
/// On success writes a handle to `*out_state` and returns [`AltDfStatus::Ok`].
/// On failure `*out_state` is set to NULL and nothing is allocated.
///
/// # Safety
/// `model_data` must point to at least `model_len` readable bytes, and
/// `out_state` must point to a writable `*mut AltDf`.
#[no_mangle]
pub unsafe extern "C" fn alt_df_create(
    model_data: *const u8,
    model_len: usize,
    atten_lim: f32,
    out_state: *mut *mut AltDf,
) -> AltDfStatus {
    if out_state.is_null() {
        return AltDfStatus::NullArg;
    }
    *out_state = std::ptr::null_mut();

    if model_data.is_null() || model_len == 0 {
        return AltDfStatus::NullArg;
    }

    guard(|| {
        let bytes = slice::from_raw_parts(model_data, model_len);

        let params = match DfParams::from_bytes(bytes) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("alt_df: could not parse the model archive: {e:#}");
                return AltDfStatus::BadModel;
            }
        };

        // Identical to what libDF's own capi.rs configured, so the audio this
        // produces is unchanged from the version that went through df_create.
        let runtime = RuntimeParams::default_with_ch(1)
            .with_atten_lim(atten_lim)
            .with_thresholds(-15.0, 35.0, 35.0)
            .with_post_filter(0.0)
            .with_mask_reduce(ReduceMask::MAX);

        let model = match DfTract::new(params, &runtime) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("alt_df: could not initialise the DeepFilter runtime: {e:#}");
                return AltDfStatus::Init;
            }
        };

        *out_state = Box::into_raw(Box::new(AltDf { model }));
        AltDfStatus::Ok
    })
}

/// Reports the loaded model's geometry.
///
/// # Safety
/// `st` must be a handle from [`alt_df_create`] that has not been freed, and
/// `out_info` must point to a writable [`AltDfInfo`].
#[no_mangle]
pub unsafe extern "C" fn alt_df_info(st: *const AltDf, out_info: *mut AltDfInfo) -> AltDfStatus {
    if st.is_null() || out_info.is_null() {
        return AltDfStatus::NullArg;
    }

    guard(|| {
        let m = &(*st).model;
        *out_info = AltDfInfo {
            sr: m.sr as u32,
            hop_size: m.hop_size as u32,
            fft_size: m.fft_size as u32,
            lookahead: m.lookahead as u32,
            ch: m.ch as u32,
        };
        AltDfStatus::Ok
    })
}

/// Denoises one hop of audio.
///
/// `input_len` and `output_len` must both equal the model's `hop_size`. They
/// are checked here, before any pointer is dereferenced: this is the bounds
/// check libDF does not have, and passing a short buffer to libDF's
/// `df_process_frame` is a heap overrun rather than an error.
///
/// `out_lsnr` receives the frame's local SNR in dB and may be NULL.
///
/// Real-time safe: no allocation, no locking, no I/O on the success path.
///
/// # Safety
/// `st` must be a live handle. `input` must point to `input_len` readable
/// floats and `output` to `output_len` writable floats. The two must not
/// overlap.
#[no_mangle]
pub unsafe extern "C" fn alt_df_process_frame(
    st: *mut AltDf,
    input: *const f32,
    input_len: usize,
    output: *mut f32,
    output_len: usize,
    out_lsnr: *mut f32,
) -> AltDfStatus {
    if st.is_null() || input.is_null() || output.is_null() {
        return AltDfStatus::NullArg;
    }

    let hop = (*st).model.hop_size;
    if input_len != hop || output_len != hop {
        return AltDfStatus::BadLength;
    }

    guard(|| {
        let state = &mut *st;

        let noisy = slice::from_raw_parts(input, input_len);
        let enh = slice::from_raw_parts_mut(output, output_len);

        // from_shape, not from_shape_ptr: this one returns Err on a shape that
        // does not match the slice instead of building a view over memory that
        // is not there.
        let noisy = match ArrayView2::from_shape((1, hop), noisy) {
            Ok(v) => v,
            Err(_) => return AltDfStatus::BadLength,
        };
        let enh = match ArrayViewMut2::from_shape((1, hop), enh) {
            Ok(v) => v,
            Err(_) => return AltDfStatus::BadLength,
        };

        match state.model.process(noisy, enh) {
            Ok(lsnr) => {
                if !out_lsnr.is_null() {
                    *out_lsnr = lsnr;
                }
                AltDfStatus::Ok
            }
            Err(e) => {
                eprintln!("alt_df: inference failed on this frame: {e:#}");
                AltDfStatus::Process
            }
        }
    })
}

// There is deliberately no alt_df_set_atten_lim.
//
// libDF exposes one, and the plugin used to call it every block. It turned out
// (BACKLOG.md L6) that libDF implements atten_lim as
//     spec_enh = (1 - lim) * enh + lim * noisy,   lim = 10^(-db/20)
// immediately before a linear WOLA synthesis, which is a plain wet/dry
// crossfade wearing a model parameter's clothes. The plugin now performs that
// crossfade itself on the audio thread, where it is sample-accurate and cannot
// be reordered against the worker. The model is created at 100 dB, which is
// libDF's "no limit" value, and never told otherwise.
//
// A setter would therefore be an FFI entry point with no reachable caller and a
// second, redundant way to attenuate.

/// Frees a handle. NULL is accepted and ignored, and freeing twice is a
/// use-after-free exactly as with `free`.
///
/// # Safety
/// `st` must be a handle from [`alt_df_create`] that has not already been
/// freed, and must not be in use on another thread.
#[no_mangle]
pub unsafe extern "C" fn alt_df_free(st: *mut AltDf) {
    if st.is_null() {
        return;
    }

    // Dropping a DfTract drops tract models and their tensors. That is
    // ordinary Rust, but it is still user code running inside a Drop impl, so
    // it gets the same guard as everything else.
    let _ = guard(|| {
        drop(Box::from_raw(st));
        AltDfStatus::Ok
    });
}

/// A short static description of a status, for logging. Never NULL.
#[no_mangle]
pub extern "C" fn alt_df_status_str(status: AltDfStatus) -> *const std::os::raw::c_char {
    let s: &'static [u8] = match status {
        AltDfStatus::Ok => b"ok\0",
        AltDfStatus::NullArg => b"a required argument was null\0",
        AltDfStatus::BadModel => b"the model archive could not be parsed\0",
        AltDfStatus::Init => b"the DeepFilter runtime could not be initialised\0",
        AltDfStatus::BadLength => b"a buffer length did not match the model hop size\0",
        AltDfStatus::Process => b"inference failed on this frame\0",
        AltDfStatus::Panic => b"a panic was caught at the FFI boundary\0",
    };
    s.as_ptr() as *const std::os::raw::c_char
}
