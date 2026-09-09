# Alt Denoiser

![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-brightgreen)
![Format](https://img.shields.io/badge/Format-VST3%20%7C%20AU%20%7C%20LV2-blue)
![License](https://img.shields.io/badge/License-AGPLv3-red)

**Alt Denoiser** is a real-time AI audio noise suppression plugin based on **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)**. It wraps the open-source DeepFilterNet3 models into a ready-to-use audio plugin, supporting Windows, macOS, and Linux platforms in VST3, AU (macOS), and LV2 (Linux) formats.

**Alt Denoiser** 是一个基于 **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** 的实时 AI 语音降噪插件。它将开源的实时语音增强模型打包成一个开箱即用的音频插件，支持 Windows、macOS、Linux 平台，以及 VST3、AU (macOS)、LV2 (Linux) 多种格式。

---

## Features / 特性

* Two DeepFilterNet3 models, chosen from a **Model** selector in the UI: Standard at 40 ms latency, Low latency at 20 ms. Both are compiled into the plugin, so nothing is downloaded at runtime.
    * 内置两个 DeepFilterNet3 模型，可在界面的 **Model** 下拉框中选择：Standard 延迟 40 ms，Low latency 延迟 20 ms。两者都已嵌入插件，运行时无需下载。
* One **Reduction** knob, 0 to 100 dB, where 100 reads "No limit". The limit is a per-sample dry/wet crossfade on the audio thread with a 50 ms ramp, so moving it does not click.
    * 单个 **Reduction** 旋钮，0 到 100 dB，100 显示为 "No limit"。该限制在音频线程上以逐采样的干湿交叉淡化实现，斜坡长度 50 ms，因此调节时不会产生爆音。
* Input and output sample **peak** meters, with a 1.5 s peak hold and a 20 dB/s fall. The scale runs from -60 dB to +6 dB and the readout follows the hold value.
    * 输入/输出**采样峰值**电平表：峰值保持 1.5 秒，回落 20 dB/秒。刻度范围 -60 dB 到 +6 dB，数字读数跟随保持值。
* A real **Bypass** parameter that the host can see, automate and save, delay-matched to the reported latency so toggling it does not shift the track.
    * 真实的 **Bypass** 参数，宿主可读取、自动化并保存，并按插件上报的延迟对齐，切换时音轨不会前后错位。

## Latency and CPU / 延迟与 CPU

| Model | Reported latency | CPU per 480-sample hop | Share of one core | Embedded archive |
| :--- | ---: | ---: | ---: | ---: |
| Standard | 1920 samples (40 ms) | 0.907 ms | 9.1% | 7.98 MB |
| Low latency | 960 samples (20 ms) | 2.670 ms | 26.7% | 36.36 MB |

The archive column is the compressed `.tar.gz` that gets compiled in, which is what costs
binary size. Uncompressed the two ONNX sets are 8.59 MB and 39.0 MB.

The realtime budget is one hop of wall time per hop of audio, which is 10 ms at 48 kHz. Both models fit inside it, and `scripts/gate.sh` fails the build if either exceeds half of it.

Read the table carefully, because the naming is backwards. "Low latency" names the **delay**, not the cost. It is the bigger network (`emb_hidden_dim` 512 against 256, `df_num_layers` 3 against 2), so it buys 20 ms less delay for roughly 3x the CPU and roughly 4.6x the download.

请注意这里的命名容易误解："Low latency" 指的是**延迟**更低，而不是开销更低。它其实是更大的网络，用大约 3 倍的 CPU 和大约 4.6 倍的体积换来 20 ms 的延迟改善。

The latency figure is derived from the loaded model rather than hardcoded: `(fft_size - hop_size) + lookahead * hop_size`, plus one hop of the plugin's own quantisation cushion. Both archives share a 480-sample hop and a 960-sample FFT at 48 kHz and differ only in `lookahead`, 2 against 0. The plugin converts that figure to the host rate before reporting it, so 40 ms and 20 ms hold at every supported sample rate. The resampler adds about 4 samples on top, measured, which the reported figure does not include.

The per-hop timings come from the "model keeps up with realtime" case in `tests/OfflineTests.cpp`, which runs on every gate run rather than from a one-off benchmark. They were measured on one developer machine; yours will differ.

Switching model is **not automatable** and takes effect when the plugin is **reloaded**, not when you change the box. The two archives report different latencies, and a plugin that changes its reported latency mid-session is among the least reliably handled things in VST3, so the choice is only read when the host next prepares the plugin.

切换模型**不可自动化**，并且需要**重新加载插件**后才生效。两个模型上报的延迟不同，而插件在会话中途改变上报延迟是宿主处理得最不可靠的行为之一。

## Sample rate / 采样率

The model runs at its native 48 kHz, and the plugin converts to and from the host rate around it. `prepareToPlay` accepts rates from **8000 Hz to 384000 Hz**. Outside that range the plugin loads no model, reports zero latency, and passes audio through unchanged: the track is not denoised, and nothing announces it.

Conversion is a 4-point Catmull-Rom interpolator (`modules/Resampler`) with **no anti-alias filter before downsampling and no anti-image filter after upsampling**. Running at 96 kHz therefore folds content above 24 kHz back into the audible band instead of removing it. Run the plugin at 48 kHz where you can, since there is then no ratio to convert.

插件在 48 kHz 下运行模型，并在宿主采样率之间来回转换。`prepareToPlay` 只接受 **8000 Hz 到 384000 Hz**；超出该范围时插件不加载模型、上报零延迟并原样放行音频，不会有任何提示。重采样使用 4 点 Catmull-Rom 插值，**没有抗混叠和抗镜像滤波器**，因此在 96 kHz 下 24 kHz 以上的内容会折返到可听频段。建议尽量在 48 kHz 下使用。

## Channels / 声道

Alt Denoiser is mono in the middle. The model has one channel, so a stereo input is summed to `(L + R) / 2` before inference and the single wet result is copied to every output channel. **Stereo material comes back mono.** Only mono and stereo layouts are accepted, and the input layout must match the output layout.

处理链中间是单声道：立体声输入先求和为 `(L + R) / 2` 再送入模型，单声道结果再复制到每个输出声道，**立体声素材会变成单声道**。插件只接受单声道和立体声，且输入与输出布局必须一致。

## Interface / 界面

<img width="685" height="472" alt="Alt Denoiser UI" src="https://github.com/user-attachments/assets/e71a463d-8c70-4efd-b598-79a90e1edc9f" />

This screenshot is from upstream and predates the Model row, which adds 80 px to the bottom of
the window.

该截图来自上游，早于 Model 选择行，实际窗口比图中高 80 px。

## Download / 下载

Visit the **[Releases](https://github.com/bigfnj/Alt-Denoiser/releases)** page to download the latest version.

前往 **[Releases](https://github.com/bigfnj/Alt-Denoiser/releases)** 页面下载最新版本。

CI builds and uploads on Windows, macOS and Linux. A tagged release carries one zip per format per operating system: VST3 on all three, AU on macOS, LV2 on Linux. Each zip holds the plugin bundle plus `LICENSE` and `README.md`, and nothing else. There is no Standalone build, no VST2, and no link intermediates such as `_SharedCode.lib` or MSVC `.exp` / `.lib` stubs.

CI 会在 Windows、macOS 和 Linux 上构建并上传产物。每次打标签的发布中，每个平台的每种格式各有一个 zip，内含插件本体以及 `LICENSE` 和 `README.md`。不提供 Standalone 和 VST2。

---

## Build Instructions / 构建指南

If you want to build from source, please follow these steps.
如果您想自己编译代码，请参考以下步骤。

### 0. Prerequisites / 环境准备

Ensure you have the following installed:
你需要确保安装了以下工具：
* **CMake** (3.16+)
* **Rust Toolchain** (1.70+, for the `libs/alt_df` shim crate)
* **C++ Compiler** with C++17 (Visual Studio 2022 / Xcode / GCC)

On Linux, CI installs these system packages before configuring:

```
libasound2-dev libfreetype6-dev libgl1-mesa-dev libgtk-3-dev libx11-dev
libxcomposite-dev libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev
libxrender-dev xvfb
```

### 1. Clone Repository / 获取代码

```bash
git clone --recursive https://github.com/bigfnj/Alt-Denoiser.git
cd Alt-Denoiser
```

### 2. Build / 开始构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build (Rust compilation may take time on first run)
cmake --build build --config Release --parallel 4
```

### 3. Choose which models to embed / 选择嵌入的模型

`ALTDENOISER_MODELS` controls which archives are compiled in. It accepts `both` (default), `standard` or `lowlatency`, and CMake refuses to configure with anything else.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DALTDENOISER_MODELS=standard
```

Both models give a VST3 of about 64 MB. `standard` alone gives about 28 MB. A build carrying one archive hides the Model row in the editor and falls back to what it has if a saved session asks for the other, so a session written by a full build still opens in a slim one.

`ALTDENOISER_MODELS` 决定编译进插件的模型：`both`（默认）、`standard` 或 `lowlatency`。两个模型的 VST3 约 64 MB，仅 `standard` 约 28 MB。只含一个模型的构建会隐藏界面上的 Model 一行。

### 4. Artifacts / 获取产物

After the build, the plugin files will be located at:
编译完成后，你可以在以下目录找到插件文件：

* **Windows / macOS / Linux**:
  `build/AltDenoiserPlugin_artefacts/Release/`

### 5. Verification gate / 验证

`scripts/gate.sh` is the single verification entry point, and CI runs the same script, so a local pass and a CI pass mean the same thing.

```bash
bash scripts/gate.sh build
```

It builds Release, runs the offline harness (`tests/OfflineTests.cpp`) across five geometries that have historically behaved differently, runs `pluginval` at strictness level 5, and checks the packaging shape and that `LICENSE` is present. If `PLUGINVAL` is not set to a pluginval binary, the validation stage is skipped and the run reports `GATE PASSED (DEGRADED)` rather than a clean pass.

The harness can also be driven directly with a block size and a sample rate:

```bash
./build/AltDenoiserTests_artefacts/Release/AltDenoiserTests 512 44100
```

pluginval validates VST3 well-formedness, not audio correctness: it passed this plugin at strictness 5 with every defect in `BACKLOG.md` still present. The offline harness exists because of that, and it asserts on rendered samples.

`scripts/gate.sh` 是唯一的验证入口，CI 运行的是同一个脚本。它会构建、在五种几何配置下运行离线测试、运行 pluginval（strictness 5），并检查打包结构与 `LICENSE`。未设置 `PLUGINVAL` 时会明确报告为降级通过。

---

## Internals / 实现说明

* Inference runs on a dedicated worker thread, not on the audio callback. The audio thread only accumulates whole 480-sample hops, hands them over, and copies results back. When the worker is late the plugin splices in the latency-aligned dry signal rather than digital silence, and an offline bounce waits for the worker instead of falling back.
* `libs/alt_df` is a small Rust staticlib that replaces libDF's own C API. That API could not report failure: `df_create` returned a `Box` and never NULL, and both it and `df_process_frame` ended in `.expect()`, so a corrupt model archive panicked across `extern "C"` and aborted the host. Every entry point in the shim returns a status and catches panics, so a bad archive degrades to bypass and a failed frame emits dry audio. It also builds `deep_filter` with `default-features = false`, which keeps libDF's own embedded copy of a DeepFilterNet3 archive out of the binary.
* Bypass **does not save CPU**. The model keeps running while bypassed, and that is the point: nothing is ever flushed, the FIFOs never leave their target fill, and so un-bypassing cannot splice stale audio into the output. The bypassed signal is the original per-channel input at host rate, delayed by exactly the reported latency, crossfaded over 10 ms.

推理在独立的工作线程上运行，音频回调只负责按 480 采样的帧收发数据。`libs/alt_df` 是替代 libDF C API 的 Rust 垫片：所有入口都返回状态码并捕获 panic，因此损坏的模型只会退化为直通，而不会让宿主崩溃。Bypass **不会节省 CPU**，模型仍在运行，这正是取消 bypass 时不会带出陈旧音频的原因。

---

## Installation / 安装路径

Copy the generated plugin bundle (`.vst3` / `.component` / `.lv2`) to your system's plugin directory:
将生成的插件文件复制到你系统的对应目录中：

| Format | Windows | macOS | Linux |
| :--- | :--- | :--- | :--- |
| **VST3** | `C:\Program Files\Common Files\VST3` | `/Library/Audio/Plug-Ins/VST3` | `~/.vst3` |
| **AU** | N/A | `/Library/Audio/Plug-Ins/Components` | N/A |
| **LV2** | N/A | N/A | `~/.lv2` |

---

## Modifications / 修改说明

This is a modified fork of [Altinus/Alt-Denoiser](https://github.com/Altinus/Alt-Denoiser).
Modifications began on 2026-09-08 and are ongoing. They include removing VST2 and
Standalone, moving DeepFilterNet inference off the audio callback, replacing libDF's
C API with a non-panicking Rust shim, deriving the reported latency from the loaded
model, adding a latency-compensated bypass and a model selector, and a number of audio
correctness fixes. See `BACKLOG.md` for the itemised list and `git log` for dates, as
required by AGPLv3 section 5(a).

本项目是 [Altinus/Alt-Denoiser](https://github.com/Altinus/Alt-Denoiser) 的修改版分支，
自 2026-09-08 起持续修改。详见 `BACKLOG.md` 与 `git log`。

## License / 开源协议

This project is licensed under the **GNU Affero General Public License v3.0**.
The full text is in [LICENSE](LICENSE).
本项目遵循 **AGPLv3** 协议，完整文本见 [LICENSE](LICENSE)。

The upstream README stated GPLv3. That is incorrect for the JUCE version this
project builds against: the pinned JUCE 8 submodule is dual-licensed AGPLv3 or
commercial, and a JUCE 8 derivative cannot be distributed under plain GPLv3 with
the AGPLv3 section 13 obligation dropped. Anyone wanting to ship this under
different terms needs a commercial JUCE licence.

Component licences:

| Component | Licence |
| :--- | :--- |
| [JUCE](https://github.com/juce-framework/JUCE) 8 | AGPLv3 or commercial |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | MIT / Apache-2.0 |
| [Resampler](https://github.com/niswegmann/Resampler) | CC0-1.0 (public domain) |

## Credits / 致谢

* **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)**: noise reduction AI models and Rust library.
* **[JUCE](https://github.com/juce-framework/JUCE)**: Cross-platform audio plugin framework.
* **[Resampler](https://github.com/niswegmann/Resampler)**: 4-point Catmull-Rom sample rate conversion.
* **gemini**: For patiently explaining why my code wouldn't compile.
