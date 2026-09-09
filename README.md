# Alt Denoiser

![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-brightgreen)
![Format](https://img.shields.io/badge/Format-VST3%20%7C%20AU%20%7C%20LV2-blue)
![License](https://img.shields.io/badge/License-AGPLv3-red)

**Alt Denoiser** is a real-time AI audio noise suppression plugin based on **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)**. It wraps the open-source DeepFilterNet model into a ready-to-use audio plugin, supporting Windows, macOS, and Linux platforms in VST3, AU (macOS), and LV2 (Linux) formats.

**Alt Denoiser** 是一个基于 **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** 的实时 AI 语音降噪插件。它将开源的实时语音增强模型打包成一个开箱即用的音频插件，支持 Windows、macOS、Linux 平台，以及 VST3、AU (macOS)、LV2 (Linux) 多种格式。

---

## Features / 特性

* **Sample Rate Independent**: Supports any host sample rate with automatic high-quality resampling.
    * **采样率无关**：支持任意宿主采样率，自动重采样。
* **Simple Interface**: Single knob for attenuation control + Input/Output RMS metering.
    * **简单界面**：一个旋钮调节最大衰减量 + 输入/输出 RMS 电平表。
* **Embedded Model**: DeepFilterNet3 models are bundled within the plugin; no external downloads required.
    * **模型嵌入**：DeepFilterNet3 模型已打包进插件，无需额外下载。

## Interface / 界面

<img width="685" height="472" alt="Alt Denoiser UI" src="https://github.com/user-attachments/assets/e71a463d-8c70-4efd-b598-79a90e1edc9f" />

## Download / 下载

visit the **[Releases](https://github.com/Altinus/Alt-Denoiser/releases)** page to download the latest version.
*Note: Currently, pre-compiled binaries are available for Windows only*

前往 **[Releases](https://github.com/Altinus/Alt-Denoiser/releases)** 页面下载最新版本。
*注意：目前仅提供 Windows 版本的预编译文件*

---

## Build Instructions / 构建指南

If you want to build from source, please follow these steps.
如果您想自己编译代码，请参考以下步骤。

### 0. Prerequisites / 环境准备

Ensure you have the following installed:
你需要确保安装了以下工具：
* **CMake** (3.15+)
* **Rust Toolchain**
* **C++ Compiler** (Visual Studio 2022 / Xcode / GCC)

### 1. Clone Repository / 获取代码


```bash
git clone --recursive https://github.com/Altinus/Alt-Denoiser.git
cd Alt-Denoiser
```

### 2. Build / 开始构建


```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build (Rust compilation may take time on first run)
cmake --build build --config Release --parallel 4
```

### 3. Artifacts / 获取产物

After the build, the plugin files will be located at:
编译完成后，你可以在以下目录找到插件文件：

* **Windows / macOS / Linux**: 
  `build/AltDenoiser_artefacts/Release/`

---

## Installation / 安装路径

Copy the generated plugin files (`.vst3` / `.component`) to your system's plugin directory:
将生成的插件文件复制到你系统的对应目录中：

| Format | Windows | macOS | Linux |
| :--- | :--- | :--- | :--- |
| **VST3** | `C:\Program Files\Common Files\VST3` | `/Library/Audio/Plug-Ins/VST3` | `~/.vst3` |
| **AU** | N/A | `/Library/Audio/Plug-Ins/Components` | N/A |
| **LV2** | N/A | N/A | `~/.lv2` |

---

## Modifications / 修改说明

This is a modified fork of [Altinus/Alt-Denoiser](https://github.com/Altinus/Alt-Denoiser).
Modifications began on 2026-09-08 and are ongoing. They include removing VST2,
moving DeepFilterNet inference off the audio callback, and a number of audio
correctness fixes. See `BACKLOG.md` for the itemised list and `git log` for
dates, as required by AGPLv3 section 5(a).

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
* **[Resampler](https://github.com/niswegmann/Resampler)**: High-quality sample rate conversion library.  
* **gemini**: For patiently explaining why my code wouldn't compile xd😂😂
