# Backlog

Findings from a code audit of `5b6eaf4` (upstream `main`, 2026-03-01), performed 2026-09-08.

Items have stable IDs so they can be referenced from commits and PRs. Severity reflects
consequence, not effort. "Upstream" marks items that are plain bug fixes worth sending to
`Altinus/Alt-Denoiser` rather than keeping on this fork. The owner's decision is to fork and
deviate, so nothing here is being sent upstream.

## Status, 2026-09-09

All 33 original findings are closed: C1-C5, H1-H7, M1-M13, L1-L8. Earlier revisions of this file said 30, which never matched the list. Verified by `scripts/gate.sh`: a build, the offline
harness across five geometries (480/48000, 512/48000, 1024/48000, 512/44100, 1024/96000),
`pluginval` at strictness 5, and a packaging shape check, run identically locally and in CI on
Windows, macOS and Linux.

35 harness tests on the default build, 31 on each slim one. Every guard added this session was
mutation tested: the guard was broken, the resulting failure was checked to name the right
item, and the guard was restored. Those results are recorded in the commit messages rather
than in a green run, because a green run proves nothing about whether the check can fail.

What that discipline caught, in this repo, this session:
- Three vacuous tests of my own. One counted `paint()` calls in a headless harness, where
  `paint()` never runs. One ran a refusal against a virgin processor whose latency was already
  zero. One wrapped its assertion in `if (xml != nullptr)`, so a typo would have reported PASS.
- Three real defects of my own: a use-after-free between `worker.stop()` and `df_free`, an
  unconditional `setLatencySamples` that reported 40 ms after a failed load, and attenuation
  left on the audio thread after I claimed it had moved.
- A coverage gap that no amount of reading would have found: H6 measures the plugin at
  attenuation 0, which since L6 is fully dry, and the dry path is delayed by the derived
  latency. H6 was comparing the derivation against itself and would have passed with a wrong
  number. See 7a.
- Two build configurations that do not compile or do not pass, found only by building all
  three. See 7b.

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

**FIXED** (`48f4ed4`). `libs/alt_df` replaces libDF's C API entirely. `alt_df_create` returns a
status, validates its arguments before dereferencing anything, and wraps its body in
`catch_unwind`, so a corrupt archive is a refused load and a bypassed plugin rather than a dead
host. It also loads from memory, which deleted the temp-file staging that made C3, C4 and M6
possible in the first place.

Mutation tested in two parts, because the guard's value is the difference between them: a
deliberate `panic!()` inside `alt_df_create` with `catch_unwind` intact runs all 26 tests and
exits 1, while the same panic without it produces "thread caused non-unwinding panic. aborting.",
exit 127, and zero test output. A `compile_error!` on `cfg(panic = "abort")` fails the build if
the profile is ever changed, since `catch_unwind` is inert there and would fail silently.

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

**FIXED** (`48f4ed4`). `alt_df_process_frame` returns a status, and a failed frame copies the
input through so the caller emits the dry signal rather than repeating the previous frame
forever. Same `catch_unwind` guard as C1.

`Source/PluginProcessor.cpp:108`, mechanism in pinned `capi.rs`

`df_process_frame` wraps `state.m.process(...).expect("Failed to process DF frame")`. Any
error returned by the tract runtime during live playback terminates the host from inside the
audio callback. Same abort-on-unwind path as C1.

### C3. Fixed shared temp-file path makes C1 reachable in normal use (upstreamable)

**FIXED** (`388abbe`). Staged via `juce::File::createTempFile`, so every instance and process gets
a unique path.

`Source/DeepFilterNetProcessor.cpp:29-30`

The model is written to `tempDir/alt_denoiser_model.tar.gz`, a constant name shared by every
plugin instance and every process on the machine. Two instances initialising concurrently
means one can be writing while the other reads, and the reader then panics per C1 and takes
the whole DAW with it. Two DAWs open at once hit the same window. Multiple instances is the
normal use case for a denoiser.

Fix: `juce::TemporaryFile`, or skip the filesystem entirely per M4.

### C4. The temp file appends instead of truncating (upstreamable)

**FIXED** (`388abbe`). Written fresh and deleted as soon as `df_create` has read it, which also
closes M6. Confirmed in the wild: `%TEMP%lt_denoiser_model.tar.gz` had reached 582,768,928
bytes, exactly 73 x 7,983,136, purely from this session's test runs. Deleted.

`Source/DeepFilterNetProcessor.cpp:32-41`

JUCE's `FileOutputStream` sets the write position to end-of-file and the code never calls
`setPosition(0)` or `truncate()`. One short write from a full disk or an antivirus hook
leaves a truncated first gzip member. Because `GzDecoder` reads only the first member and
later runs only ever append, the broken member is never replaced: every subsequent load
panics and aborts the DAW until the user manually deletes the file. The file also grows by
7,983,136 bytes per `initialize()` call and is never deleted.

### C5. Heap overrun when a host exceeds its declared block size

**FIXED** (`388abbe`). processBlock refuses a block larger than `preparedBlockSize` and passes
audio through untouched. Mutation-tested: with the guard disabled the harness terminates with
`0xC0000374`, `STATUS_HEAP_CORRUPTION`, confirming the overflow was real and reachable.

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

### H1. Every `prepareToPlay` leaks an entire model (upstreamable)

**FIXED** (`388abbe`). `initialize()` frees the previous `DFState` before replacing it.

`Source/DeepFilterNetProcessor.cpp:43`

`state = df_create(...)` with no prior `df_free` and no guard. `releaseResources()` is empty,
so the only free is in the destructor, and only for the last instance. Every sample-rate or
buffer-size change leaks a full ONNX session plus weights.

### H2. The attenuation knob stops working after any re-prepare (upstreamable)

**FIXED** (`388abbe`). `prepareToPlay` resets `lastAttenLim` to its sentinel. Harness now
measures RMS 0.176713 before and after a re-prepare, ratio exactly 1.000000.

`Source/PluginProcessor.h:102`, `Source/PluginProcessor.cpp:82`,
`Source/DeepFilterNetProcessor.cpp:43`

`lastAttenLim` is initialised once as a member and never reset in `prepareToPlay`, while each
new `DFState` is built with a hardcoded `100.0f`. Set the knob to 20 dB, change the buffer
size, and the change-detector computes `abs(20 - 20) < 0.01` and skips the re-apply. The
engine runs at 100 while the UI and the saved state both still read 20.

**Measured:** with the knob at 0 dB (passthrough), a 440 Hz tone renders at 0.1767 RMS. After
a re-prepare with the knob untouched it renders at 0.000909, a ratio of 0.005 and a drop of
about 45 dB. The consequence is worse than a dead control: the plugin silently switches to
maximum noise reduction and starts destroying content while the UI still reads 0.

Fix: reset `lastAttenLim` in `prepareToPlay`, or pass the current parameter value into
`df_create` instead of the hardcoded constant.

### H3. `state` and `modelLoaded` are shared across threads with no synchronisation

**FIXED** (`93bf963` + `0d389e6`). `modelLoaded` is `std::atomic<bool>`, published with release
ordering at the very end of `prepareToPlay` after every buffer it guards. The `DFState` pointer
is now owned exclusively by the inference worker, so no other thread calls into libDF at all.

`Source/DeepFilterNetProcessor.h:18`, `Source/PluginProcessor.h:101`

A plain pointer and a plain `bool`, written from the `prepareToPlay` thread and read from the
audio thread, with no atomic, mutex or lock-free handoff anywhere. On a sample-rate change
while audio is still draining, the audio thread can read a stale or half-published pointer.

### H4. Stereo input is destroyed

**FIXED** (`9fd5b6f`), partially. Inputs are summed to `(L+R)/2` so nothing is discarded before
inference. Output remains mono, which is inherent to a one-channel model. True stereo via two
model instances is still open; see the note at the end of this item.

`Source/PluginProcessor.cpp:88`, `:124-125`

Only channel 0 is ever read, and channel 1 is overwritten with a copy of the processed
result. The right channel is discarded rather than summed. **Measured:** feeding 440 Hz left
and 880 Hz right, the input channels differ by 0.440 peak and the output channels are
bit-identical (max |L-R| = 0.000000). The harness asserts the input diff first, so this cannot
pass vacuously. Anything panned hard right
vanishes; an M/S or dual-mic recording loses half its information. This happens on the
plugin's own declared stereo layout, and neither the UI nor the README mentions it.

Fix options, in preference order: downmix `(L+R)/2` for inference and keep per-channel dry
signal, or run two inference streams, or redeclare the bus as mono-in/stereo-out so the
behaviour is at least honest.

### H5. A zero-splice at the top of every playback

**FIXED** (`960cd5f`). The output FIFO is primed with one model hop of silence, which bounds the
worst-case deficit. Zero spliced samples measured across 48000/480, 48000/512, 48000/1024,
44100/512, 44100/480 and 96000/1024.

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
only escape and it does not survive a sample-rate change. **Confirmed by the offline harness:**
at 480 and 960 the output contains zero silent samples past the reported latency; at 512 it
contains 352, in runs of exactly 32, matching the predicted per-event size. This is the most likely explanation
for the contradictory reports on upstream issue #7, where one user sees a broken plugin and
another sees a working one.

Fix: hold the FIFO at a fixed target fill before producing output, and declare that fill as
part of the latency (see H6).

### H6. Reported latency is wrong, and no single constant can be correct

**FIXED** (`960cd5f`). The same priming makes the delay deterministic. The reported 1920 needed no
change: it was always 1440 of model delay plus 480 of cushion, and only the priming was missing.
Measured error after the fix is 1-4 samples across six rate/block combinations.

`Source/PluginProcessor.cpp:39-40`

`setLatencySamples(round(1920 * sr/48000))` is hardcoded. The embedded model is DeepFilterNet3
**standard**, measured as `sr=48000, hop=480, fft=960, lookahead=2`, giving an algorithmic
delay of `(960-480) + 2*480 = 1440` samples.

**Measured by the offline harness at attenuation 0** (a spectral passthrough, so this isolates
pipeline delay from model delay), driving a silence-then-burst signal and finding the first
audible output sample:

| Block size | Pipeline delay | Implied total (pipeline + 1440) | Reported | Error |
| ---: | ---: | ---: | ---: | ---: |
| 480 | 4 | 1444 | 1920 | +476 (9.9 ms too much) |
| 512 | 451 | 1891 | 1920 | +29 |
| 960 | 4 | 1444 | 1920 | +476 |

This corrects an earlier claim in this document. 1920 is not uniformly wrong: it is nearly
right at 512-sample blocks and is 476 samples out at 480 and 960. The constant appears to have
been tuned against one buffer size. The real defect is that the true delay is a function of
block geometry (4 samples versus 451 of pipeline delay, purely from the FIFO cushion) while
the report is a constant, so the error swings by 447 samples with a setting the user changes
freely.

The 4-sample floor at clean geometries is the Catmull-Rom resampler's own group delay for both
conversion stages combined, which also settles one of the open resampler questions below.

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

**FIXED** (`1b74600`). Audio passes through untouched when the model is unavailable, and the meter
updates moved outside that gate so the UI shows real signal instead of freezing. Mutation-tested:
restoring `buffer.clear()` makes the harness fail with output peak 0.000000 and exit 1.

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
| M1 | `PluginProcessor.cpp:105-110` | **FIXED** (`0d389e6`); inference moved to a worker thread in `Source/InferenceWorker.h`, with a bounded offline wait, sequence-tagged frames so a locate discards in-flight work, and a latency-aligned dry fallback instead of zero-fill. No latency added. Was: inference runs on the audio thread with no bound on iterations: a 2048-sample block does 4-5 back-to-back model evaluations in one callback, with no worker thread and no deadline fallback. This is the architectural root cause behind H5 and the dropout reports. |
| M2 | `PluginProcessor.cpp:56` | **FIXED** (`f6455ab` + `0d389e6`). Was: no `reset()` override and an empty `releaseResources()`, so roughly 450 samples of audio from the previous playhead plus stale model recurrent state survive every transport locate. |
| M3 | `libs/Include/df.h:25` | **FIXED** (`f6455ab`); the hop is taken from `df_get_frame_length` and an implausible value falls back to bypass. Was: `df_get_frame_length` is declared but never called; 480 is hardcoded in five places. `df_process_frame` builds its views with `from_shape_ptr`, which does no bounds check, and the guarding `debug_assert` is compiled out in release. Any model swap becomes silent heap corruption. |
| M4 | `CMakeLists.txt:86` | **FIXED** (`48f4ed4`); `alt_df` sets `default-features = false`, so `default-model` never embeds anything. Was: the model is embedded twice at source level, because `--features capi` also enables `default-model`, so libDF carries its own `include_bytes!` copy alongside the `juce_add_binary_data` one. **Measured: this does not reach the shipped binary.** The 7,983,136-byte archive appears exactly once in the released VST3 and once in the Standalone, so the linker drops the unreferenced copy. The cost is build time and intermediate size, not distribution size. It becomes real bloat the moment anything references `DfParams::default()`. Separately, `DfParams::from_bytes` would load the embedded copy with no filesystem involvement, deleting C3, C4 and M6 outright, but the pinned C API exposes only the path-based `df_create`, so that needs one new entry point upstream. |
| M5 | `PluginProcessor.h:62` | **FIXED** (`f6455ab`); returns 0.04 s. Was: `getTailLengthSeconds()` returns 0.0 despite roughly 40-50 ms of held state, so offline bounces can truncate the tail. |
| M6 | `DeepFilterNetProcessor.cpp:29-41` | **FIXED** (`388abbe`) with C4. Was: the temp file is never deleted and grows by ~8 MB per `initialize()`, persisting across DAW restarts. |
| M7 | `PluginProcessor.h` | **FIXED** (`e06ea90`); a real `AudioParameterBool` with `getBypassParameter()` returning it, a per-channel host-rate delay primed to the reported latency, and a 10 ms crossfade. Bit-exact alignment, worst error 0.000000 over 26799 samples. `processBlockBypassed` routes through `processBlock` because the base implementation asserts zero latency. Note the trade-off: the model keeps running while bypassed, so BYPASS SAVES NO CPU. That is the only way un-bypassing cannot flush stale audio or re-trigger the H5 splice, because nothing is ever flushed. Was: no bypass parameter and no `processBlockBypassed`. |
| M8 | `PluginProcessor.h:16-35` | **FIXED** (`ae692f0`); every operation bounds-checked and returning a refusal, with overflow and underflow counters, and a direct unit test. Was: no capacity check on `push`, no floor on `discard`, and a signed/unsigned comparison at `:22` that converted a negative count into a full FIFO. |
| M9 | `PluginEditor.h:54-70` | **FIXED** (`8135ca5`); dt-based decay at 20 dB/s. Was: Meter ballistics decay at -116 dB/s against -20 to -26 dB/s for a standard digital peak meter, with no `dt` term, on a timer JUCE documents as imprecise at exactly this timescale. |
| M10 | `PluginProcessor.cpp:71-76` | **FIXED** (`8135ca5`); running peak consumed by the UI. Was: The audio-side RMS EMA is applied once per block, so its time constant swings 32x with buffer size (1.9 ms at 64 samples, 61.6 ms at 2048). At small buffers roughly 92% of blocks are never sampled by the 60 Hz UI. A running max reset by the UI after reading would drop nothing. |
| M11 | `PluginEditor.h:100`, `:152` | **FIXED** (`8135ca5`); one value drives both. Was: The meter bar reads from `smoothedLevel` and the number printed above it from `displayedDb`, on different decay rates, so they disagree after every transient. |
| M12 | `PluginProcessor.h:47-110` | **FIXED** (`9fd5b6f`); pluginval now reports only `Mono, Stereo`. Was: no `isBusesLayoutSupported`. Mono is genuinely safe (every index traced and guarded), but any layout with 3 or more outputs leaves channels 2 and up unprocessed and undelayed while the host shifts the whole track by the reported latency. |
| M13 | `PluginProcessor.cpp:21-28` | **FIXED** (`f6455ab`); label, string-from-value, and the display name reconciled with the knob. Was: the parameter has no unit label and no string-from-value function, so hosts show a bare "20.0"; the automation lane reads "Attenuation Limit" while the visible knob reads "Reduction"; and 100 is an undocumented sentinel meaning "no limit". |

---

## Low

| ID | Location | Issue |
| :--- | :--- | :--- |
| L1 | `PluginEditor.h:70` | **FIXED** (`8135ca5`); repaint gated on a changed render, mutation-tested. Was: `repaint()` is unconditional, so two meters fully repaint 60 times a second forever, including in total silence with an identical rendered result. |
| L2 | `PluginEditor.h:117-175` | **FIXED** (`8135ca5`); statics hoisted into resized(). Was: `paint()` heap-allocates roughly 1000 times a second: a `Path`, a `ColourGradient`, and six `String`s per meter per frame, plus glyph layout. The clip path and the five tick labels are static for a given size and belong in `resized()`. |
| L3 | `PluginEditor.h:64-68` | **FIXED** (`8135ca5`); clamped at the -100 dB floor. Was: `displayedDb` has no lower clamp and decrements without bound, reaching about -108,000 after an hour. Masked only because the readout prints "-inf" below -90. |
| L4 | `PluginProcessor.cpp:74`, `:131` | **FIXED** (`8135ca5`); sample peak across all channels. Was: RMS values are drawn on a scale styled for peaks, with a red band above 0 dB that cannot light: a full-scale sine is -3.01 dB RMS. Only channel 0 is metered, so a right-channel-only source reads as silence on the input meter. |
| L5 | `PluginProcessor.cpp:81` | **FIXED** (`58e120d`); pointer cached in the constructor. Was: String-keyed parameter lookup on the audio thread every block, rather than caching the `std::atomic<float>*` once in the constructor. |
| L6 | `PluginProcessor.cpp:541-547` | **FIXED** (`1f495d3`) as an audio-thread crossfade, pinned by two harness tests. The first attempt was **ATTEMPTED AND REVERTED** (`58e120d`). A hop-rate slew in the worker failed the gate at hop-aligned block sizes, varied 3.5x-100x with buffer size, and reintroduced H2. See the commit. The correct shape is an audio-thread crossfade: `atten_lim` is implemented in tract.rs as `(1-lim)*wet + lim*dry`, and the plugin already holds the aligned dry signal. Was: Attenuation changes are applied per block with no smoothing, so a fast automation ramp steps audibly. |
| L7 | `PluginProcessor.h:70-88` | **FIXED** (`58e120d`); schemaVersion added. The unreachable createXml() null check was deliberately NOT added. Was: `setStateInformation` checks only the tag name, with no schema version, and `*xml` is dereferenced without checking `createXml()` for null. |
| L8 | `PluginProcessor.h:19`, `:29`, `:34`, `PluginProcessor.cpp:50` | **FIXED** (`58e120d`); geometry refused before anything is built from it, plus SimpleFifo hardened against an unsized buffer. Was: No guard against `sampleRate == 0` or `processBlock` preceding `prepareToPlay`; a scanner calling `prepareToPlay(0, 0)` yields an infinite ratio and an unbounded resampler write loop. |

---

## Dead code

Re-verified 2026-09-09 against the current tree, because acting on the original list
blindly would have removed things that had since become live and left things the original
sweep never saw. Line numbers in the original entries were stale.

### Closed

| Original claim | Outcome |
| :--- | :--- |
| `PluginEditor.h:54` unused `attack` constant | Gone with the `DbMeter` rewrite (M9/M11/L3). The identifier no longer exists. |
| `DeepFilterNetProcessor.h:19` unread `sampleRate` member | Removed with the shim rewrite. Nothing read it; the model is fixed at its own rate and the resampler owns conversion. |
| `df_get_frame_length` never called | Became live under M3, then superseded: the shim reads `hop_size` from `alt_df_info`. `libs/Include/df.h` is deleted. |
| channel-clear loop never iterates | Removed. `isBusesLayoutSupported` refuses any layout where the input and output sets differ, and one bus of each is declared, so the counts are always equal. |
| `releaseResources()` empty | Stale. It calls `worker.stop()` and clears `modelLoaded`. |
| `df.h` omits four exported symbols | Moot. `df.h` is deleted and `alt_df.h` declares exactly what `alt_df` exports. |

### Found by the re-verification, now closed

| Item | Outcome |
| :--- | :--- |
| `DeepFilterNetProcessor::setAttenLim` had zero callers | Removed, along with the `alt_df` setter it would have called. It went dead when L6 moved the attenuation crossfade onto the audio thread, leaving an FFI entry point reachable from nothing. |
| `InferenceWorker::isActive`, `getPendingInput`, `getReadyOutput` | Removed. No callers anywhere, tests included. |
| `DbMeter::getPaintCount` and its counter | Removed. It was the vacuous counter L1's test was originally written against: headless, `paint()` never runs, so it could not move whether the gate worked or not. `getRepaintRequests` replaced it. |
| `PluginProcessor.cpp` `if (modelFrameLength <= 0) return;` in `reset()` | Removed. `modelFrameLength` is assigned only from a hop already range-checked `> 0 && <= 4096`, and its initialiser is 480, so no value can reach the branch. |
| `libs/libDF/*` in `.gitignore` | Removed. That path has not existed for some time. |
| The `.gitignore` line `*.dll *.lib *.a *.exp *.ilk *.pdb` | Split one per line. Gitignore does not split on whitespace, so it was a single pattern matching a file literally named that, and none of those extensions were ignored at all. Nothing leaked only because every build output already sat under an ignored directory. |
| The About dialog's hardcoded "v1.0" | Now `JucePlugin_VersionString`. It had already drifted past the v1.0.1 release. |

### Kept deliberately

- `PluginProcessor.h` `if (! std::isfinite(db)) return 0.0f;` in `attenLimitToDryMix`. Currently
  unreachable: `db` comes from a `NormalisableRange<float>(0, 100, 0.1)`, which cannot produce a
  non-finite value. Kept because the cost is one instruction and the failure it prevents is a NaN
  entering a `SmoothedValue`, which poisons the output permanently rather than glitching once.
- The `apvts` reference in the editor. Read once, to build the `SliderAttachment`. Redundant with
  `audioProcessor.apvts` but not dead.
- `dryScratch` is sized `maxResampledSize + modelFrameLength` and indexed only to
  `maxResampledSize`. The extra hop is unused headroom. Harmless, and shrinking it removes margin
  from the one buffer that a resampler phase error would overrun first.

### Still open

- `alt_df_process_frame` writes the frame's local SNR to `out_lsnr` and
  `DeepFilterNetProcessor::processFrame` discards it. It is the natural feed for a meaningful
  meter or a gate, which is a feature rather than a fix.

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

### LAT2. The low-latency model saves 20 ms, and costs far more than first measured

**SHIPPED** (`62d6374`, `4c9c590`). Both archives are embedded and selectable, the reported
latency derives from the loaded model's geometry, and the editor states the cost before the
user picks.

Re-measured 2026-09-09 against the shim, timing 200 warmed-up frames per model rather than
extrapolating:

| Model | lookahead | Reported latency | Cost per 480-sample hop | % of one core | Embedded archive | Uncompressed ONNX |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepFilterNet3 standard | 2 | 1920 (40 ms) | 0.907 ms | 9.1% | 7.98 MB | 8.59 MB |
| DeepFilterNet3 LL | 0 | 960 (20 ms) | 2.670 ms | 26.7% | 36.36 MB | 39.0 MB |

The size that matters is the compressed archive, since that is what JUCE embeds. The two are
listed separately because the earlier estimate mixed them.

The first estimate of "43% more CPU per hop" was wrong by a wide margin. It is about **2.9x**,
and the archive is **4.6x** larger, not 29 MB more. The naming stays counterintuitive: LL names
the delay, and it is the bigger network (`emb_hidden_dim` 512 against 256, `df_num_layers` 3
against 2). Embedding both takes the Windows VST3 from 27.9 MB to 64.3 MB, which is why
`ALTDENOISER_MODELS=standard` exists.

Runtime switching is still not done, and still for the reason given: it would change the
reported latency mid-session. The choice applies on the next `prepareToPlay`, and the parameter
is marked non-automatable to say so.

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

**FIXED** (`b169d88`). Canonical AGPLv3 text added as `LICENSE` (661 lines, SHA-256
`0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`, includes section 13 Remote
Network Interaction). README badge and licence section corrected, component licences tabulated,
and an AGPLv3 section 5(a) modification notice added. The release workflow now copies `LICENSE`
and `README.md` into the package and hard-fails if `LICENSE` is absent, mutation-tested.

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

**Build system** (`CMakeLists.txt`): RESOLVED 2026-09-08 by building locally on Windows
(MSVC 19.51, Visual Studio generator). Both Release and Debug configurations built clean,
exit 0, zero errors.

- VST2 under JUCE 8: moot, the format was removed.
- `build_libdf` racing the link under `--parallel`: **not an issue.** Cargo completed well
  before the link step, and `add_dependencies(AltDenoiserPlugin build_libdf)` enforces
  ordering in any generator, not just the IMPORTED `df` target.
- MSVC Debug/Release CRT mismatch: **no warnings observed.** No `LNK4098`, no `defaultlib`
  conflict, in either configuration. It is also safe by design rather than by luck: ownership
  never crosses the CRT boundary, because the C++ side only ever returns pointers to
  `df_free` and never frees Rust-allocated memory itself.
- `RUST_SYSTEM_LIBS` completeness on Windows: confirmed, both configurations linked.
- `-DCMAKE_BUILD_TYPE` plus `--config`: **correct, not redundant.** CMake warned that
  `CMAKE_BUILD_TYPE` was unused because the Windows generator is multi-config, which means
  single-config generators on Linux need it and multi-config ones need `--config`. The
  workflow passing both is what makes the matrix work.
- ~~`BYPRODUCTS` correctness (whether CMake rebuilds when Rust sources change).~~
  **RESOLVED 2026-09-09.** It does rebuild and relink correctly, but `BYPRODUCTS` is not what
  makes that true, and the mechanism is worth knowing. The `add_custom_target(... ALL)` has a
  stamp file that never exists plus `VerifyInputsAndOutputsExist=false` and no
  `AdditionalInputs`, so it is unconditionally out of date and cargo does the source
  fingerprinting. The relink happens because `IMPORTED_LOCATION` lands on the link line as an
  absolute path that MSBuild's file tracker records as a read input (confirmed in
  `link.read.1.tlog`); Ninja and Makefiles reach the same result through an explicit
  dependency on the imported path. `BYPRODUCTS` itself only marks the file `GENERATED`, adds it
  to the clean list, and on Ninja declares it an output of the edge. On the Visual Studio
  generator the byproduct path appears nowhere in the generated project at all.
- ~~A stale library could be linked silently.~~ **FIXED 2026-09-09.** `TARGET_RELEASE_DIR` is a
  hardcoded path, so a `CARGO_TARGET_DIR` in the environment or a `[build] target-dir` in any
  `.cargo/config.toml` above the crate sent cargo's output elsewhere while `IMPORTED_LOCATION`
  kept pointing at whatever happened to be here. `--target-dir` is now passed explicitly.
- Still open: `cargo build --release` is unconditional, so a `--config Debug` C++ build links
  the Release Rust library and no configuration switch ever rebuilds it. Harmless today, since
  ownership never crosses the CRT boundary, but it means a Debug build is not debuggable into
  Rust.
- Still open: everything above re-verified on macOS and Linux runners.

**Validation**: `pluginval` 1.0.3 at strictness level 5, run 2026-09-08 against the Release
VST3: **PASS, exit 0, zero warnings.**

That result is important mainly for what it does *not* mean. It passed with every finding in
this backlog present. It did not detect H1 (the leak), H2 (the dead knob), H4 (the stereo
collapse), H5 (the startup splice), or C1 (a corrupt model aborting the host). `pluginval`
verifies that the plugin is a well-formed VST3, not that it processes audio correctly, so it
is a necessary gate and not a sufficient one.

It did confirm two findings empirically:
- **M12 upgrades to CONFIRMED-reachable.** `pluginval` enumerated the advertised layouts as
  Mono, Stereo, LCR, Quadraphonic, 5.0, 5.1, 7.0 and 7.1 on both input and output, and
  successfully enabled all of them. The missing `isBusesLayoutSupported` is therefore not
  hypothetical: a 5.1 instantiation is accepted, and channels 2-5 pass through un-denoised
  and undelayed while the host shifts the track by the reported latency.
- **M5 confirmed:** `Reported taillength: 0`.

**Consequence for the plan.** There is currently no way to detect a regression in the audio
path, and every Phase 2 fix touches it. An offline render harness is needed before changing
DSP code: a console target linking the processor directly, driving `prepareToPlay` and
`processBlock` with known input, and asserting impulse position against reported latency,
absence of zero-runs in steady state, right-channel survival, and that the parameter reaches
the model. Roughly the coverage the comparable Rust project gets from its 28 tests.

**CI** (`.github/workflows/build.yml`)
- ~~No `pluginval` and no `auval` on any platform.~~ **FIXED.** `scripts/gate.sh` runs the
  build, the harness across five geometries, `pluginval` at strictness 5 and a packaging
  shape check, identically locally and in CI, and reports `GATE PASSED (DEGRADED)` rather
  than silently skipping when a stage cannot run. `auval` is still absent on macOS.
- The artefacts path is no longer hardcoded: the staging step added in `a01cd73` locates the
  bundle and hard-fails on zero or multiple matches, so a wrong path on macOS or Linux now
  breaks the build loudly instead of publishing an empty archive. Confirmed working against a
  real Windows build tree.
- ~~No Rust build caching.~~ **FIXED**, the workflow caches the cargo registry and target
  directory.

**Still open, and worth surfacing**
- There is no ASAN or TSan job. The N2 use-after-free was found by reasoning and confirmed by
  a 1-in-6 `STATUS_HEAP_CORRUPTION`, which is not a reliable detector. A sanitiser build of
  the harness on Linux would turn that class of bug into a deterministic failure.
- `setStateInformation` refusing a future schema keeps the CURRENT state rather than
  restoring defaults, so switching to a preset saved by a newer build silently does nothing
  instead of visibly resetting.
- `schemaVersion` is written as an attribute on the APVTS root, which round-trips back into
  the live parameter tree on load.

**Resampler and latency derivation** (`modules/Resampler/`)
- ~~The Catmull-Rom group delay per conversion stage, needed to close out H6.~~ **Measured at
  4 samples total for both stages combined**, via the offline harness at a block size that is a
  multiple of 480 (where the FIFO cushion is 0, so the residual delay is the resampler alone).
- ~~Whether the downsampling direction can overrun the buffers from the other side.~~
  **RESOLVED 2026-09-09: it cannot, at any rate the guard permits.** The premise was wrong.
  The expanding direction never writes into a ratio-scaled buffer: stage 3 of
  `resampler::process` READS `resampleOutBuffer` and WRITES `monoBuffer`, which is sized to
  the full host block. Stage 1 is the only writer into the ratio-scaled buffers, and with
  `phi` the entering phase bounded in `[0, ts)` its output count is
  `M <= floor(N * 48000 / sr) + 1 <= maxResampledSize - 127`, so both keep 127 spare floats
  in the worst case at every rate from the 8 kHz floor to the 384 kHz ceiling.
- ~~Which rates the library actually supports, versus the README's "any host sample rate".~~
  **RESOLVED: 8 kHz to 384 kHz inclusive**, enforced in `prepareToPlay`. Outside that the
  plugin refuses every block and passes audio through undenoised. The README claim is
  corrected.
- ~~The library's licence and its compatibility with LIC1.~~ **CC0 1.0 Universal** (public
  domain dedication), so compatible with anything, including AGPLv3.

---

## Where to pick up

Read this section and the ranked table below it; everything above is history.

Session closed 2026-09-09 with a clean tree, all work on `main`, CI green on three platforms.
Nothing is half-finished and no branch is outstanding.

To rebuild and verify, from the repo root:

    CMAKE="/c/Program Files/CMake/bin/cmake.exe" PLUGINVAL=<path to pluginval.exe> bash scripts/gate.sh build

The `CMAKE` override is not optional under Git Bash, which resolves a bundled cmake 3.31 that
shadows the real install and cannot load the Visual Studio generator. Without `PLUGINVAL` the
gate reports `GATE PASSED (DEGRADED)`, which is not a pass.

`libs/alt_df/Cargo.lock` is committed and load-bearing: it pins tract to 0.21.4, and a fresh
resolution picks 0.21.18, against which the vendored libDF does not compile. Seed it from
`modules/DeepFilterNet/Cargo.lock` rather than regenerating it.

Highest-value next items, in order: A17 (nothing measures denoising), A10 (JUCE compiles
twice), A11 (cargo cache key omits the toolchain), A22 (the model-embedding default, which is
an owner decision rather than a fix).

---

## Post-merge audit, 2026-09-09

A four-domain audit of `094fe39` after everything above was on `main`: memory and threading,
test quality, dead code and regressions, optimisation. Fixed in `6f353d1` unless marked OPEN.

### Defects found and fixed

| ID | Finding |
| :--- | :--- |
| A1 | The attenuation ramp was not sample-rate independent, and its comment claimed it was. `attenMix` was primed with the host rate while `getNextValue()` is called once per 48 kHz sample, so the ramp ran 100 ms at 96 kHz against an intended 50 ms. |
| A2 | The bypass delay desynchronised permanently after one oversized block. `processBlock` pushed into `bypassDelay` before the C5 guard and discarded after it, so a refused block was pushed and never drained. Silent: the push SUCCEEDED, so not even the overflow counter moved. |
| A3 | `reset()` after a refused prepare called `pushSilence(0)`, which `SimpleFifo` refuses with `jassertfalse`, breaking Debug builds on a path the harness drives. |
| A4 | `getInFlight()` could read negative, because `framesAccepted` was incremented after `inbound.push`. That exited the offline drain loop early and spliced an unnecessary dry frame. |
| A5 | `worker.submit()` took a mutex on the audio thread once per hop. `juce::WaitableEvent` is not an OS event in JUCE 8 on any platform; `signal()` locks a `std::mutex` and calls `notify_all()` under it. No priority inheritance bounds a stall if the worker is preempted mid-hold. It now signals only when the worker publishes that it is waiting. |
| A6 | Inference failures were invisible. `processFrame` returns `bool`, the worker discarded it, and the counter behind it had no reader, so a model failing every frame produced dry audio with nothing to show for it. `fallbackSamples` stays 0 in that case, because a frame WAS delivered. |
| A7 | Teardown depended on member declaration order. Reordering `worker` and `dfProcessor` would give a use-after-free at host shutdown. |
| A8 | `alt_df_status_str` matched a `repr(C)` enum received from C exhaustively, which is only sound for a discriminant the enum declares. It takes an `int` with a default arm now. |
| A9 | The CI cache still pointed at `modules/DeepFilterNet/target`, which nothing has built into since the shim. Every run since Phase 6 recompiled tract from scratch, taking run times from about 11 to about 17 minutes. |

### The test audit's central finding

At attenuation 0 the L6 crossfade computes `writePtr[i] += 1.0 * (dry[i] - writePtr[i])`,
which erases the wet path arithmetically. Eleven tests ran there, so they asserted things
about a plain delay line and nothing else.

Measured: the M1 fallback test passed with the dry splice replaced by `zeromem`, and the H5
test passed with the output FIFO priming deleted. Each was the only regression cover for its
item. Both now run at attenuation 100, and the M1 test asserts `fallbackSamples > 0` rather
than printing it. The mutation that was previously invisible now produces 39040 zeros
against 4.

This is the same trap 7a hit from the other direction: a test whose expected value comes from
the code path under test. Worth stating as a rule for this repo. **Any assertion made at
attenuation 0 is an assertion about `dryDelay` and nothing else.**

### Still open, ranked

| ID | Item | Measurement | Recommendation |
| :--- | :--- | :--- | :--- |
| A10 | JUCE is compiled TWICE per build. `AltDenoiserPlugin.dir` and `AltDenoiserTests.dir` each hold their own `juce_gui_basics.obj` (23.78 MB), `juce_graphics.obj` (14.44 MB) and 18 more. `juce_add_console_app` makes an independent JUCE target rather than reusing the plugin's shared code. | About half the C++ compile time, on all three CI platforms, on every clean build | DO IT. The largest single build-time win. |
| A11 | The cargo cache key omits the toolchain. `dtolnay/rust-toolchain@stable` moves, every fingerprint then misses, and nothing prunes the 1.5 GB target directory. It also holds `df.lib` at 209 MB and `df.dll`, both built and discarded, because libDF declares `crate-type = ["cdylib", "rlib", "staticlib"]` and the shim links only the rlib. | 209 MB of dead staticlib per cache | DO IT. `Swatinem/rust-cache@v2` keys on the toolchain and prunes. |
| A12 | The resampler has no ratio-1 short circuit. At 48 kHz it runs 960 spline evaluations and 960 double divisions per hop to produce a near-identity filter that is not bit-exact and adds about one sample of unreported delay. | ~12.8 us of ~17 us non-model audio-thread cost, plus a bit-exact 48 kHz path | WORTH TRYING. The CPU win is negligible against a 907 us model call; the bit-exactness is the real prize. |
| A13 | `BinaryData` is compiled from 159.8 MB of generated C++ (`BinaryData2.cpp` alone is 131 MB), producing a 44.35 MB `.lib` via `/bigobj`. A linker resource would take the bytes verbatim. | 159.8 MB of parse work per clean build | WORTH TRYING. Three platforms, three mechanisms, so not a one-liner. |
| A14 | `lto = "fat"` plus `codegen-units = 1` on the shim, and `opt-level = "z"` on `tract-onnx`, `tract-hir`, `tract-onnx-opl` and `tract-pulse`, which total about 1.46 MB and run only inside `alt_df_create`. | 0.8 to 1.6 MB combined, unmeasured | WORTH TRYING, measure before keeping. |
| A15 | `stopThread(2000)` calls `TerminateThread` on timeout and its return value is discarded, so `prepareToPlay` proceeds to free a model the killed thread may be inside. Needs a worker stalled for over 2 s. | Low probability, host hang if the killed thread held the CRT heap lock | WORTH FIXING. Check the return and refuse to re-initialise. |
| A16 | No ASAN or TSan job. N2 was found by reasoning and confirmed by a 1-in-6 `STATUS_HEAP_CORRUPTION`, which is not a detector. | | WORTH TRYING. A sanitiser build of the harness on Linux. |
| A17 | Nothing measures denoising. No test asserts noise reduction, SNR improvement or a noise-floor drop. A model returning plausible-but-wrong audio at the right latency passes all 35 tests. The per-frame local SNR is computed by the shim and discarded. | | The most valuable gap in the suite. Feed speech plus known noise, assert the floor drops. |
| A18 | `attenLimitToDryMix`'s interpolating arm is never exercised. Every test uses 0 or 100, and the parameter's useful range is everything between. | | Cheap test. |
| A19 | Mono operation is never exercised. `render()` always builds two channels and `prepare()` always declares 2-in/2-out, though `isBusesLayoutSupported` advertises mono as supported. | | Cheap test. |
| A20 | `processBlockBypassed` is never called by any test; all four bypass tests go through the parameter. `releaseResources()` is never called either. | | Cheap test. |
| A21 | The editor hardcodes "40 ms"/"9%" and "20 ms"/"27%" as strings. 7a exists so the latency is derived from the model; the UI re-hardcodes it. An archive with different geometry would make the UI lie while `getLatencySamples()` stayed correct. | | Derive the latency half. The CPU half has to stay a measured constant. |
| A22 | `ALTDENOISER_MODELS=standard` takes the VST3 from 64,290,304 to 27,930,624 bytes. Measured from three builds of the same tree. | 36,359,680 bytes, 56.6% | Owner's call. This is 75x the best compressor result, and larger than every other size item on this list combined. |

### Measured, and deliberately NOT done

- Byte-plane shuffling the model weights before compression saves about 3.32 MB (5.2%),
  measured on the real tar streams. It forks the archive format away from upstream and adds a
  failure mode to the load path that C1 and C2 were about.
- A stronger compressor (brotli -q11) saves 486,387 bytes, 0.76%, and a third of that goes
  straight back into the decoder it needs.
- `panic = "abort"` would remove 2,241,372 bytes of Rust unwind tables, 3.5% of the file. It
  also removes the C1/C2 guarantee, which is why `lib.rs` carries a `compile_error!` on it.
- Aggressive `/OPT:ICF` saves 68,865 bytes and breaks function-pointer identity.
- Dropping unused JUCE modules for SIZE saves exactly zero. `/OPT:REF` already deleted all of
  it: no FLAC, Ogg, vorbis, WAVE, AIFF, ASIO or WASAPI literal survives anywhere in the DLL.
  It is still worth doing for build time.
- Shrinking the 480-sample cushion. The quantisation arithmetic does allow 0 at a
  480-multiple block size at 48 kHz, which is what LAT1 observed. It stopped being available
  at M1: the audio thread calls `collect()` microseconds after `submit()` in the same
  callback, so the worker is exactly one hop behind in steady state and the cushion is its
  runway rather than a safety margin. A single short block would also move the residue off
  zero permanently.

### Where the binary actually goes

PE section sizes sum to the byte. Non-model total 19,947,508, of which Rust is 16,424,126
(82.3%) and C++ 3,465,631 (17.4%). Within the Rust half, attributed by the crate each
function's panic location names: ndarray 32.0%, tract-core 28.7%, tract-data 7.9%, rustfft
6.5%, tract-onnx 5.0%, tract-hir 4.0%, smallvec 3.7%, std 3.7%, tract-linalg 1.7%, libDF
0.9%, the shim under 0.05%. This agrees with SIZE1's earlier estimate, which was reached by
building a bare `main()`, so treat it as confirmed rather than new.

---

## Historical: the suggested first milestone

Kept for the record, because it is a fair snapshot of what the audit thought mattered on
2026-09-08 and all of it shipped within a day.

It proposed three one-line changes: H1, free `state` before reassigning it; C3, give the temp
file a unique name per instance; H2, reset the attenuation sentinel in `prepareToPlay`. Then
the item it called the one that changes the product's failure mode: H7, bypass instead of
`buffer.clear()`, and guard `df_create` so a panic cannot cross the FFI boundary.

All four are closed. Two of them turned out differently than proposed and the difference is
worth keeping. C3 was not fixed by a unique temp name in the end; the temp file was deleted
outright when the shim gained the ability to load from memory. And the `df_create` guard could
not be written on the C++ side at all, because the panic originated inside Rust: it needed the
shim (C1/C2).
