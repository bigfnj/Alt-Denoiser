# Backlog

Findings from a code audit of `5b6eaf4` (upstream `main`, 2026-03-01), performed 2026-09-08.

Items have stable IDs so they can be referenced from commits and PRs. Severity reflects
consequence, not effort. "Upstream" marks items that are plain bug fixes worth sending to
`Altinus/Alt-Denoiser` rather than keeping on this fork.

## Method and coverage

Four domains were audited: the real-time audio path, model lifecycle and the Rust FFI
binding, the GUI, and conformance to the JUCE plugin/host contract. Claims were verified
against the pinned submodule sources (DeepFilterNet `d375b2d`, JUCE `501c076`) rather than
against the tagged releases, which differ.

Two domains were **not** audited. See [Coverage gap](#coverage-gap).

Numbers for the FIFO and latency behaviour come from simulating the real
`Resampler-Cpp.inl` phase accumulator over 3000 blocks across multiple sample rates and
block sizes, plus an independent analytic trace. Model geometry was measured by loading the
archives through `DfParams::new` + `DfTract::new` and reading live metadata.

---

## Critical

These terminate the host process or corrupt memory.

### C1. A bad model file aborts the DAW instead of failing gracefully

`Source/DeepFilterNetProcessor.cpp:43-44`

`df_create` can never return `NULL`. In the pinned `libDF/src/capi.rs`, `DFState::new`
returns `Self` rather than `Result`, every internal failure is an `.expect()`
(`DfParams::new(...).expect("Could not load model from path")`,
`DfTract::new(...).expect("Could not initialize DeepFilter runtime.")`), and it ends in
`Box::new(self)`. So the `state != nullptr` check is dead code, and a missing, truncated or
wrong-format model panics across an `extern "C"` boundary and calls `abort()`. The host dies
instantly with unsaved session data lost. The only diagnostic is a `DBG()` compiled out in
release builds.

Fix: catch the failure before it reaches Rust (validate the archive first), or add a
non-panicking entry point to `capi.rs` upstream. Until then, C3 and C4 are the reachable
triggers and should be fixed first.

### C2. An inference error aborts the DAW mid-playback

`Source/PluginProcessor.cpp:108`, mechanism in pinned `capi.rs`

`df_process_frame` wraps `state.m.process(...).expect("Failed to process DF frame")`. Any
error returned by the tract runtime during live playback terminates the host from inside the
audio callback. Same abort-on-unwind path as C1.

### C3. Fixed shared temp-file path makes C1 reachable in normal use  — Upstream

`Source/DeepFilterNetProcessor.cpp:29-30`

The model is written to `tempDir/alt_denoiser_model.tar.gz`, a constant name shared by every
plugin instance and every process on the machine. Two instances initialising concurrently
means one can be writing while the other reads, and the reader then panics per C1 and takes
the whole DAW with it. Two DAWs open at once hit the same window. Multiple instances is the
normal use case for a denoiser.

Fix: `juce::TemporaryFile`, or skip the filesystem entirely per M4.

### C4. The temp file appends instead of truncating — Upstream

`Source/DeepFilterNetProcessor.cpp:32-41`

JUCE's `FileOutputStream` sets the write position to end-of-file and the code never calls
`setPosition(0)` or `truncate()`. One short write from a full disk or an antivirus hook
leaves a truncated first gzip member. Because `GzDecoder` reads only the first member and
later runs only ever append, the broken member is never replaced: every subsequent load
panics and aborts the DAW until the user manually deletes the file. The file also grows by
7,983,136 bytes per `initialize()` call and is never deleted.

### C5. Heap overrun when a host exceeds its declared block size

`Source/PluginProcessor.cpp:51-53`, written at `:88-89` and `:119`

`resampleInBuffer` and `resampleOutBuffer` are sized from `prepareToPlay`'s
`samplesPerBlock`, and `hostNumSamples`-derived counts are written into them with no clamp
or assertion. `prepareToPlay(44100, 128)` allocates 267 floats; a later 512-sample block
resamples to 559 and writes 292 floats (1168 bytes) past the end.

A conforming VST3 host will not do this, since `maxSamplesPerBlock` is a binding ceiling.
VST2's `effSetBlockSize` is advisory, and this project builds VST2 unconditionally
(`CMakeLists.txt:91`), so it is a live risk there. Clamp regardless.

---

## High

### H1. Every `prepareToPlay` leaks an entire model — Upstream

`Source/DeepFilterNetProcessor.cpp:43`

`state = df_create(...)` with no prior `df_free` and no guard. `releaseResources()` is empty,
so the only free is in the destructor, and only for the last instance. Every sample-rate or
buffer-size change leaks a full ONNX session plus weights.

### H2. The attenuation knob stops working after any re-prepare — Upstream

`Source/PluginProcessor.h:102`, `Source/PluginProcessor.cpp:82`,
`Source/DeepFilterNetProcessor.cpp:43`

`lastAttenLim` is initialised once as a member and never reset in `prepareToPlay`, while each
new `DFState` is built with a hardcoded `100.0f`. Set the knob to 20 dB, change the buffer
size, and the change-detector computes `abs(20 - 20) < 0.01` and skips the re-apply. The
engine runs at 100 while the UI and the saved state both still read 20.

Fix: reset `lastAttenLim` in `prepareToPlay`, or pass the current parameter value into
`df_create` instead of the hardcoded constant.

### H3. `state` and `modelLoaded` are shared across threads with no synchronisation

`Source/DeepFilterNetProcessor.h:18`, `Source/PluginProcessor.h:101`

A plain pointer and a plain `bool`, written from the `prepareToPlay` thread and read from the
audio thread, with no atomic, mutex or lock-free handoff anywhere. On a sample-rate change
while audio is still draining, the audio thread can read a stale or half-published pointer.

### H4. Stereo input is destroyed

`Source/PluginProcessor.cpp:88`, `:124-125`

Only channel 0 is ever read, and channel 1 is overwritten with a copy of the processed
result. The right channel is discarded rather than summed. Anything panned hard right
vanishes; an M/S or dual-mic recording loses half its information. This happens on the
plugin's own declared stereo layout, and neither the UI nor the README mentions it.

Fix options, in preference order: downmix `(L+R)/2` for inference and keep per-channel dry
signal, or run two inference streams, or redeclare the bus as mono-in/stereo-out so the
behaviour is at least honest.

### H5. A zero-splice at the top of every playback

`Source/PluginProcessor.cpp:112-120`

The drain loop emits whatever the output FIFO holds and pads the remainder with silence
rather than waiting for a cushion to establish. Measured:

| Configuration | Zero samples | Where |
| :--- | ---: | :--- |
| 48000 / 512 | 448 | 14 events of 32, blocks 0-13 |
| 44100 / 512 | 478 | mostly blocks 0-5, small residue after |
| 48000 / 480, 960, 1440 | 0 | escapes entirely |

Audible as a buzz or zipper on every play and after every re-prepare. It is a startup
artifact, not a recurring one: in steady state the output-FIFO cushion walks 448 down to 32
in steps of 32, so `min(out_before_read)` is exactly 32 + 480 = 512, which precisely meets
the request, and fills stop after block 14.

A 48 kHz host whose block size is an exact multiple of 480 escapes completely. That is the
only escape and it does not survive a sample-rate change. This is the most likely explanation
for the contradictory reports on upstream issue #7, where one user sees a broken plugin and
another sees a working one.

Fix: hold the FIFO at a fixed target fill before producing output, and declare that fill as
part of the latency (see H6).

### H6. Reported latency is wrong, and no single constant can be correct

`Source/PluginProcessor.cpp:39-40`

`setLatencySamples(round(1920 * sr/48000))` is hardcoded. The embedded model is DeepFilterNet3
**standard**, measured as `sr=48000, hop=480, fft=960, lookahead=2`, giving an algorithmic
delay of `(960-480) + 2*480 = 1440` samples. So 1920 is 480 samples over the model's own delay
while omitting both resampler stages.

Worse, the steady-state output-FIFO cushion is real undeclared delay that sawtooths across
the 14-block cycle rather than sitting at a constant:

| Configuration | Cushion range |
| :--- | ---: |
| 48000 / 480, 960, 1440 | 0 |
| 48000 / 512 | 0 to 448 |
| 48000 / 1024 | 0 to 448 |
| 44100 / 480 | 0 to 476 |
| 44100 / 512 | 0 to 478 |

A time-varying delay cannot be delay-compensated by any fixed number, so parallel or blend
chains get moving comb filtering, and the error changes when the user changes buffer size.
Fixing H5 by holding a fixed target fill also makes this number well-defined.

### H7. Load failure produces silence instead of bypass, and the meters conceal it

`Source/PluginProcessor.cpp:65-68`, with `Source/PluginEditor.h:57`

`buffer.clear(); return;` mutes the track. Given C1, this path is reachable only through a
temp-file open failure (read-only temp directory, disk full, permissions), not through model
failure, so it is narrower than it appears but still real.

Two things compound it. The early return precedes both RMS stores, so the meter atomics keep
their last values and the `DbMeter` attack/release cycle parks them at a plausible-looking
level with a permanent 1.94 dB flicker. The one on-screen indicator that would reveal the
failure actively hides it. Separately, `setLatencySamples` is called at `:40` before
`initialize()` at `:42`, so the host delay-compensates every other track by 40 ms for a
plugin emitting silence.

A denoiser that cannot load its model should pass audio through unchanged and report zero
latency.

---

## Medium

| ID | Location | Issue |
| :--- | :--- | :--- |
| M1 | `PluginProcessor.cpp:105-110` | Inference runs on the audio thread with no bound on iterations: a 2048-sample block does 4-5 back-to-back model evaluations in one callback, with no worker thread and no deadline fallback. This is the architectural root cause behind H5 and the dropout reports. |
| M2 | `PluginProcessor.cpp:56` | No `reset()` override and an empty `releaseResources()`, so roughly 450 samples of audio from the previous playhead plus stale model recurrent state survive every transport locate. |
| M3 | `libs/Include/df.h:25` | `df_get_frame_length` is declared but never called; 480 is hardcoded in five places. `df_process_frame` builds its views with `from_shape_ptr`, which does no bounds check, and the guarding `debug_assert` is compiled out in release. Any model swap becomes silent heap corruption. |
| M4 | `CMakeLists.txt:86` | The model is embedded twice at source level, because `--features capi` also enables `default-model`, so libDF carries its own `include_bytes!` copy alongside the `juce_add_binary_data` one. **Measured: this does not reach the shipped binary.** The 7,983,136-byte archive appears exactly once in the released VST3 and once in the Standalone, so the linker drops the unreferenced copy. The cost is build time and intermediate size, not distribution size. It becomes real bloat the moment anything references `DfParams::default()`. Separately, `DfParams::from_bytes` would load the embedded copy with no filesystem involvement, deleting C3, C4 and M6 outright, but the pinned C API exposes only the path-based `df_create`, so that needs one new entry point upstream. |
| M5 | `PluginProcessor.h:62` | `getTailLengthSeconds()` returns 0.0 despite roughly 40-50 ms of held state, so offline bounces can truncate the tail. |
| M6 | `DeepFilterNetProcessor.cpp:29-41` | The temp file is never deleted and grows by ~8 MB per `initialize()`, persisting across DAW restarts. |
| M7 | `PluginProcessor.h` | No bypass parameter and no `processBlockBypassed`. Un-bypassing flushes stale pre-bypass audio and re-triggers the H5 splice. |
| M8 | `PluginProcessor.h:16-35` | `SimpleFifo` has no capacity check on `push`, no floor on `discard`, and a signed/unsigned comparison at `:22` that converts a negative count into a full FIFO. Not reachable through current call sites because the guards live in the callers, but the class is unsafe as written. |
| M9 | `PluginEditor.h:54-70` | Meter ballistics decay at -116 dB/s against -20 to -26 dB/s for a standard digital peak meter, with no `dt` term, on a timer JUCE documents as imprecise at exactly this timescale. |
| M10 | `PluginProcessor.cpp:71-76` | The audio-side RMS EMA is applied once per block, so its time constant swings 32x with buffer size (1.9 ms at 64 samples, 61.6 ms at 2048). At small buffers roughly 92% of blocks are never sampled by the 60 Hz UI. A running max reset by the UI after reading would drop nothing. |
| M11 | `PluginEditor.h:100`, `:152` | The meter bar reads from `smoothedLevel` and the number printed above it from `displayedDb`, on different decay rates, so they disagree after every transient. |
| M12 | `PluginProcessor.h:47-110` | No `isBusesLayoutSupported`. Mono is genuinely safe (every index traced and guarded), but any layout with 3 or more outputs leaves channels 2 and up unprocessed and undelayed while the host shifts the whole track by the reported latency. |
| M13 | `PluginProcessor.cpp:21-28` | The parameter has no unit label and no string-from-value function, so hosts show a bare "20.0"; the automation lane reads "Attenuation Limit" while the visible knob reads "Reduction"; and 100 is an undocumented sentinel meaning "no limit". |

---

## Low

| ID | Location | Issue |
| :--- | :--- | :--- |
| L1 | `PluginEditor.h:70` | `repaint()` is unconditional, so two meters fully repaint 60 times a second forever, including in total silence with an identical rendered result. |
| L2 | `PluginEditor.h:117-175` | `paint()` heap-allocates roughly 1000 times a second: a `Path`, a `ColourGradient`, and six `String`s per meter per frame, plus glyph layout. The clip path and the five tick labels are static for a given size and belong in `resized()`. |
| L3 | `PluginEditor.h:64-68` | `displayedDb` has no lower clamp and decrements without bound, reaching about -108,000 after an hour. Masked only because the readout prints "-inf" below -90. |
| L4 | `PluginProcessor.cpp:74`, `:131` | RMS values are drawn on a scale styled for peaks, with a red band above 0 dB that cannot light: a full-scale sine is -3.01 dB RMS. Only channel 0 is metered, so a right-channel-only source reads as silence on the input meter. |
| L5 | `PluginProcessor.cpp:81` | String-keyed parameter lookup on the audio thread every block, rather than caching the `std::atomic<float>*` once in the constructor. |
| L6 | `PluginProcessor.cpp:82-85` | Attenuation changes are applied per block with no smoothing, so a fast automation ramp steps audibly. |
| L7 | `PluginProcessor.h:70-88` | `setStateInformation` checks only the tag name, with no schema version, and `*xml` is dereferenced without checking `createXml()` for null. |
| L8 | `PluginProcessor.h:19`, `:29`, `:34`, `PluginProcessor.cpp:50` | No guard against `sampleRate == 0` or `processBlock` preceding `prepareToPlay`; a scanner calling `prepareToPlay(0, 0)` yields an infinite ratio and an unbounded resampler write loop. |

---

## Dead code

- `PluginEditor.h:54` — `const float attack = 1.0f;` is never used. Attack is implemented as an instantaneous jump at `:57`.
- `DeepFilterNetProcessor.h:19` — the `sampleRate` member is stored by the constructor and never read. `df_create` has no sample-rate parameter.
- `libs/Include/df.h:25` — `df_get_frame_length` is never called (see M3).
- `PluginProcessor.cpp:79-80` — the channel-clear loop never iterates in the declared stereo-in/stereo-out layout.
- `PluginProcessor.cpp:56-57` — `releaseResources()` has an empty body (see M2).
- `PluginEditor.h:189`, `:198` — the `apvts` constructor parameter and reference member are redundant; the same object is reachable as `audioProcessor.apvts`.
- `DeepFilterNetProcessor.cpp:56` — the return value of `df_process_frame` is discarded. It is the per-frame local SNR and is the natural feed for a gate or a meaningful meter.
- `df.h` omits four exported symbols: `df_set_post_filter_beta`, `df_next_log_msg`, `df_free_log_msg`, `df_process_frame_raw`.

---

## Size and latency

Measured against the shipped `v1.0.1` Windows release and the model archives, 2026-09-08.

### PKG1. The release zip ships 44 MB of build artifacts

`.github/workflows/build.yml:60-64`

The workflow uploads the whole `build/AltDenoiserPlugin_artefacts/Release/` directory. The
published `Alt-Denoiser-windows.zip` is 52,657,139 bytes and contains:

| File | Bytes | Should ship |
| :--- | ---: | :--- |
| `Alt Denoiser_SharedCode.lib` | 44,041,594 | no, JUCE intermediate static library |
| `Standalone/Alt Denoiser.exe` | 28,941,312 | yes |
| `VST/Alt Denoiser.dll` | 27,619,328 | only if VST2 is kept |
| `VST3/.../Alt Denoiser.vst3` | 27,713,536 | yes |
| `VST/.exp`, `VST/.lib`, `VST3/.exp`, `VST3/.lib` | 5,835 | no, MSVC import/export stubs |

The 44 MB `_SharedCode.lib` is a link-time intermediate that no user can use, and the `.exp`
and `.lib` files are MSVC import/export stubs. Restricting the upload path to the plugin
bundles costs one line of YAML.

Measured candidate payloads, all recompressed with the same tool so the comparison is fair
(the published zip was produced by a different implementation, hence 51.0 rather than 52.7):

| Payload | zip | solid 7z |
| :--- | ---: | ---: |
| A: everything, as published | 51.0 MB | |
| B: drop `_SharedCode.lib` + stubs | 42.5 MB | 24.0 MB |
| C: B, and drop VST2 | 28.5 MB | 23.3 MB |
| E: VST3 only | 14.0 MB | |

Two results worth keeping, because both contradict the obvious guess:

**Removing the 44 MB `.lib` saves only 8.6 MB of download.** A static library is highly
redundant and compresses about 5:1. Remove it because it is meaningless to users, not because
it is the big lever.

**The real cost is shipping three near-identical binaries.** VST3, VST2 and Standalone each
carry the same model, the same ~16 MB tract runtime and the same JUCE, and zip's 32 KB DEFLATE
window cannot deduplicate across files. A solid 7z holding all three is 24.0 MB, smaller than
a zip holding only two. In a solid archive, dropping VST2 saves just 0.8 MB, so the size
argument for removing VST2 disappears entirely once compression is solid.

Therefore: drop VST2 for the licensing and build-correctness reasons in the coverage gap, not
for size. For download size, prefer per-format downloads at ~14 MB each, which is standard
practice for plugin vendors and avoids making users fetch 24 MB to get one format. If a single
combined archive is wanted instead, use solid 7z, noting that Windows 11 23H2+ extracts it
natively but macOS and Linux users need a tool.

### SIZE1. Where the 26 MB actually goes

The model archive is 7,983,136 bytes, verified present exactly once in the VST3 at offset
16,954,672. Isolating the rest by building `deep_filter` + tract into a bare `main()` with no
model embedded (16,975,872 bytes) gives the split:

| Component | Size | Share of the 27,713,536-byte VST3 |
| :--- | ---: | ---: |
| tract ONNX inference runtime | ~16.2 MB | 58% |
| DeepFilterNet3 model archive | 7.98 MB | 29% |
| JUCE, wrappers, GUI and plugin code | ~3.5 MB | 13% |

**JUCE is not the size problem, and replacing it would make things worse.** Rebuilding the
comparable Rust plugin (nice-plug + egui) with the identical standard model produced a
31,546,880-byte VST3 against JUCE's 27,713,536, so that stack costs ~7.4 MB where JUCE costs
~3.5 MB. 87% of the binary is inference runtime plus model weights that any implementation
has to carry.

The consequence for planning: PKG1 is the only meaningful distribution-size lever, and the
reason to consider leaving JUCE is LIC1 (AGPLv3), not bytes.

### LAT1. The reported latency over-states the real delay by roughly 10 ms

See H6 for the mechanism. Reported is 1920 samples at 48 kHz; the standard model's
algorithmic delay is 1440, and at a block size that is a multiple of 480 the FIFO cushion is
exactly 0, so the true delay is about 1443 samples. The plugin does not hold audio back to
match its claim, so the host over-compensates and the denoised track lands roughly 477
samples (9.9 ms) late relative to everything else. Correcting the number is both an alignment
fix and a session-latency reduction.

### LAT2. The low-latency model saves 20 ms, and costs 29 MB

Measured metadata for the two archives libDF can load:

| Model | lookahead | Algorithmic delay | Cost per hop | ONNX size |
| :--- | ---: | ---: | ---: | ---: |
| DeepFilterNet3 standard (current) | 2 | 1440 (30.0 ms) | 0.320 ms | 8.2 MB |
| DeepFilterNet3 LL | 0 | 480 (10.0 ms) | 0.459 ms | 37.2 MB |

Switching to the LL model is by far the largest latency win available, cutting the model's
own contribution from 30 ms to 10 ms. Note the naming is counterintuitive: LL is the *bigger
and slower* network, so this trades roughly 29 MB of binary and 43% more CPU per hop for 20 ms
of delay. Shipping both and letting the user choose is possible, but changing model at runtime
changes reported latency, which hosts handle inconsistently.

### LAT3. Choosing a block size that is a multiple of 480 costs nothing

At 48 kHz with a block size of 480, 960 or 1440 the output-FIFO cushion is exactly 0 rather
than sawtoothing to 448, and the H5 startup splice disappears entirely. This is a zero-code
recommendation for users today and worth putting in the README. It does not survive a
sample-rate change, since 44.1 kHz never yields a multiple of 480.

### LAT4. Threading the inference does not reduce latency

Worth recording because it is easy to assume otherwise, including from the analysis in the
`ChungkeLee` fork. Moving inference off the audio thread (M1) is a stability and CPU-headroom
fix. It generally *increases* reported latency, because a worker pipeline needs a runway to
absorb scheduling jitter. For reference, the comparable Rust plugin reserves two host quanta,
20 ms at 48 kHz, for exactly this. Do M1 for dropout resistance, not for latency.

### Achievable total

At 48 kHz with a 480-multiple block size, fixing LAT1 alone takes the honest figure from 1920
to about 1443 samples (30 ms). Adding LAT2 takes it to about 483 samples (10 ms), a 4x
improvement on the current claim, at the cost of a 29 MB larger binary.

---

## Licensing

### LIC1. No licence file, and the stated licence is wrong for the pinned JUCE

The repository has no `LICENSE` or `COPYING` file; the GitHub API reports `license: null`. The
only claim is a README badge reading GPLv3.

The pinned JUCE submodule `501c076` (2026-01-16) is **JUCE 8**, whose `LICENSE.md` states the
modules are dual-licensed **AGPLv3** or commercial. A JUCE 8 derivative cannot be licensed as
plain GPLv3 with the AGPLv3 §13 obligation dropped.

Fix: add the AGPLv3 text, correct the README, and add the AGPLv3 §5(a) modification notice.
Worth sending upstream as a standalone PR, since their labelling is wrong independently of
anything on this fork.

---

## Verified correct

Recorded so these are not "fixed" into breakage later.

- The in-place aliasing at `PluginProcessor.cpp:88`, where input and output pointers are both
  `buffer.getWritePointer(0)`, is safe: `Resampler-Cpp.inl` fully consumes its input before
  writing its output.
- `SimpleFifo::peek` + `discard` is coherent, and both call sites pass matched counts.
- `setLookAndFeel` is paired correctly and `modernLook` outlives every component using it.
  `stopTimer()` is called. The `SliderAttachment` is constructed after and destroyed before
  its slider. No dangling LookAndFeel and no lifetime bug in the editor.
- A mono layout reads and writes nothing out of bounds; every index is guarded.
- Non-ASCII temp paths are safe, since `juce::String` is UTF-8 internally on Windows.
- Every signature in the hand-vendored `df.h` matches the pinned Rust exactly, including the
  three-argument `df_create`. It would have been an ABI mismatch against the v0.5.6 tag, so
  whoever vendored it tracked the submodule pin correctly.
- The resampler's cross-block phase accumulation is correct.

---

## Coverage gap

Two domains were not audited. Open questions:

**Build system** (`CMakeLists.txt`)
- Whether JUCE 8 can build VST2 at all, given `FORMATS VST` at `:91` and
  `juce_set_vst2_sdk_path` at `:90`. This decides whether the CI matrix is currently green by
  luck. Note `modules/plugin_sdk` is a third-party mirror of an SDK Steinberg stopped
  licensing in 2018, and is a distribution risk independent of whether it builds.
- Whether the link step can race ahead of the Rust staticlib under `--parallel`, given
  `add_custom_target(build_libdf ALL)` plus `add_dependencies` on an IMPORTED target.
- Whether CMake rebuilds when Rust sources change (`BYPRODUCTS` correctness).
- The MSVC Debug/Release CRT mismatch, since cargo is always invoked `--release`.
- Whether `RUST_SYSTEM_LIBS` is complete per platform.

**CI** (`.github/workflows/build.yml`)
- No `pluginval` and no `auval` on any platform, so nothing in this backlog is currently
  caught before release.
- Whether the artifact path `build/AltDenoiserPlugin_artefacts/Release/` is correct on all
  three runners, given the workflow passes both `-DCMAKE_BUILD_TYPE=Release` (single-config)
  and `--config Release` (multi-config), and given `project()` is `AltDenoiser` while the
  plugin target is `AltDenoiserPlugin`. The path demonstrably resolves on Windows, since
  `v1.0.1` published a populated zip (see PKG1), but no macOS or Linux release has ever been
  published, so those two remain unverified.
- No Rust build caching, so every run recompiles libDF from scratch.

**Resampler and latency derivation** (`modules/Resampler/`)
- The Catmull-Rom group delay per conversion stage, needed to close out H6.
- Whether the downsampling direction can overrun the buffers from the other side: `maxRatio`
  is `48000/sampleRate`, so at 96 kHz the buffers are sized at half `samplesPerBlock` while
  the 48k-to-96k output direction expands. Needs tracing at 96000 and 192000.
- Which rates the library actually supports, versus the README's "any host sample rate".
- The library's licence and its compatibility with LIC1.

---

## Suggested first milestone

Three one-line changes with outsized effect, all upstreamable:

1. H1 — free `state` before reassigning it.
2. C3 — give the temp file a unique name per instance via `juce::TemporaryFile`.
3. H2 — reset `lastAttenLim` in `prepareToPlay`.

That removes the leak, the crash-on-second-instance, and the dead knob.

Then the item that changes the product's failure mode: H7, bypass instead of `buffer.clear()`,
and guard the `df_create` call so a panic cannot cross the FFI boundary. A denoiser that mutes
the track is worse than one that passes it through, and one that kills the DAW is worse than
both.
