cmake_minimum_required(VERSION 3.16)
project(AltDenoiser VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compiler options
if(MSVC)
    add_compile_options(/utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-fPIC)
endif()

# path settings
set(JUCE_DIR          ${CMAKE_SOURCE_DIR}/modules/JUCE)
set(RESAMPLER_DIR     ${CMAKE_SOURCE_DIR}/modules/Resampler)
set(DFNET_DIR         ${CMAKE_SOURCE_DIR}/modules/DeepFilterNet)
set(LIBS_DIR          ${CMAKE_SOURCE_DIR}/libs)
set(LIBS_INCLUDE_DIR  ${LIBS_DIR}/Include) # alt_df.h
set(ALT_DF_DIR        ${LIBS_DIR}/alt_df)
set(TARGET_RELEASE_DIR ${ALT_DF_DIR}/target/release)

find_program(CARGO_CMD cargo REQUIRED)
if(WIN32)
    set(RUST_LIB_NAME "alt_df.lib")
else()
    set(RUST_LIB_NAME "libalt_df.a")
endif()

# We build our own shim crate instead of libDF's `capi` feature. Two reasons.
#
# Correctness: libDF's C API cannot report failure. df_create returned a Box and
# never NULL, and both it and df_process_frame ended in .expect(), so a corrupt
# model archive panicked across extern "C" and aborted the host (BACKLOG.md
# C1/C2). libs/alt_df returns a status from every entry point.
#
# Size: `capi` transitively enables `default-model`, which compiles an entire
# DeepFilterNet3 archive into the static library on top of the copy the plugin
# already carries as JUCE BinaryData. alt_df sets default-features = false.
#
# It REPLACES rather than joins libDF here. Linking both staticlibs would put
# two copies of libstd on the link line, which MSVC reports as LNK4006/LNK2005.
#
# --target-dir is passed explicitly rather than relying on the default. Without
# it, a CARGO_TARGET_DIR in the environment or a [build] target-dir in any
# .cargo/config.toml above this directory sends cargo's output elsewhere while
# IMPORTED_LOCATION below keeps pointing here. The build then succeeds and
# silently links whatever stale library happens to be at this path. Both are
# common on CI and on machines that centralise build output.
add_custom_target(build_alt_df ALL
    COMMAND ${CARGO_CMD} build --release
            --manifest-path ${ALT_DF_DIR}/Cargo.toml
            --target-dir ${ALT_DF_DIR}/target
    WORKING_DIRECTORY ${ALT_DF_DIR}
    BYPRODUCTS ${TARGET_RELEASE_DIR}/${RUST_LIB_NAME}
    COMMENT "###### Building alt_df (DeepFilterNet shim) with Cargo... ######"
    VERBATIM
)

# STATIC
add_library(df STATIC IMPORTED GLOBAL)
add_dependencies(df build_alt_df)

if(WIN32)
    # Windows
    set_target_properties(df PROPERTIES IMPORTED_LOCATION "${TARGET_RELEASE_DIR}/${RUST_LIB_NAME}")
    set(RUST_SYSTEM_LIBS
        Bcrypt.lib
        Userenv.lib
        Ntdll.lib
        Advapi32.lib
    )
elseif(APPLE)
    # macOS
    set_target_properties(df PROPERTIES IMPORTED_LOCATION "${TARGET_RELEASE_DIR}/${RUST_LIB_NAME}")
    find_library(SECURITY_FW Security)
    find_library(COREFOUNDATION_FW CoreFoundation)
    find_library(SYSTEMCONFIGURATION_FW SystemConfiguration)
    
    set(RUST_SYSTEM_LIBS
        ${SECURITY_FW}
        ${COREFOUNDATION_FW}
        ${SYSTEMCONFIGURATION_FW}
        resolv
    )
else()
    # Linux
    set_target_properties(df PROPERTIES IMPORTED_LOCATION "${TARGET_RELEASE_DIR}/${RUST_LIB_NAME}")
    set(RUST_SYSTEM_LIBS
        pthread
        dl
        m
    )
endif()

# check if Juce submodule exists
if(NOT EXISTS "${JUCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "JUCE submodule not found! Please run: git submodule update --init --recursive")
endif()

add_subdirectory(${JUCE_DIR})

# binary data
#
# 7b: which archives get embedded is a build option, because the low-latency one
# is not a small addition. Measured uncompressed: standard 8.59 MB against
# low-latency 39.0 MB, and the ONNX graphs behind them are 256 vs 512
# emb_hidden_dim and 2 vs 3 df_num_layers. "Low latency" names its 960-sample
# algorithmic delay against the standard model's 1920; it is the BIGGER and
# SLOWER network, which is the opposite of what the name suggests.
#
# Both by default, so the plugin can show the trade-off rather than describe it.
set(ALTDENOISER_MODELS "both" CACHE STRING
    "Which model archives to embed: both, standard, lowlatency")
set_property(CACHE ALTDENOISER_MODELS PROPERTY STRINGS both standard lowlatency)

set(MODEL_SOURCES "")
set(ALTDENOISER_HAS_STANDARD 0)
set(ALTDENOISER_HAS_LL 0)

if(ALTDENOISER_MODELS STREQUAL "both" OR ALTDENOISER_MODELS STREQUAL "standard")
    list(APPEND MODEL_SOURCES ${DFNET_DIR}/models/DeepFilterNet3_onnx.tar.gz)
    set(ALTDENOISER_HAS_STANDARD 1)
endif()
if(ALTDENOISER_MODELS STREQUAL "both" OR ALTDENOISER_MODELS STREQUAL "lowlatency")
    list(APPEND MODEL_SOURCES ${DFNET_DIR}/models/DeepFilterNet3_ll_onnx.tar.gz)
    set(ALTDENOISER_HAS_LL 1)
endif()

if(NOT MODEL_SOURCES)
    message(FATAL_ERROR
        "ALTDENOISER_MODELS was '${ALTDENOISER_MODELS}'. It must be one of: both, standard, lowlatency.")
endif()

message(STATUS "Alt Denoiser: embedding models -> ${ALTDENOISER_MODELS}")

juce_add_binary_data(AltDenoiserAssets
    NAMESPACE AltDenoiserBinaryData
    SOURCES
        ${MODEL_SOURCES}
)

# format
# VST2 removed: Steinberg stopped issuing VST2 SDK licences in October 2018, so
# distributing VST2 builds is a legal risk regardless of whether they compile.
# The SDK was previously supplied by a third-party mirror submodule.
#
# Standalone removed: it is not distributed. It cost build time on all three CI
# platforms and produced a ~29 MB artefact (BACKLOG.md PKG1) that no release
# ever shipped, so nothing downstream consumed it.
#
# What ships: VST3 everywhere, AU on macOS, LV2 on Linux.
set(PLUGIN_FORMATS VST3)
if(APPLE)
    list(APPEND PLUGIN_FORMATS AU)
elseif(UNIX AND NOT APPLE)
    list(APPEND PLUGIN_FORMATS LV2)
endif()

juce_add_plugin(AltDenoiserPlugin
    PRODUCT_NAME        "Alt Denoiser"
    COMPANY_NAME        "Altinus"
    VERSION             1.0.0
    PLUGIN_MANUFACTURER_CODE "AltA" 
    PLUGIN_CODE              "AltD"
    FORMATS             ${PLUGIN_FORMATS}
)

target_compile_definitions(AltDenoiserPlugin PUBLIC
    ALTDENOISER_HAS_STANDARD_MODEL=${ALTDENOISER_HAS_STANDARD}
    ALTDENOISER_HAS_LL_MODEL=${ALTDENOISER_HAS_LL}
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JucePlugin_LV2URI="https://github.com/Altinus/Alt-Denoiser"
)

# /bigogj for MSVC to avoid "fatal error C1128: number of sections exceeded object file format limit: compile with /bigobj"
if(MSVC)
    target_compile_options(AltDenoiserAssets PRIVATE /bigobj)
    target_compile_options(AltDenoiserPlugin PRIVATE /bigobj)
endif()

target_sources(AltDenoiserPlugin PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/DeepFilterNetProcessor.cpp
    Source/DeepFilterNetProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h
    ${RESAMPLER_DIR}/Resampler.hpp
)

target_include_directories(AltDenoiserPlugin PRIVATE
    ${LIBS_INCLUDE_DIR}
    ${RESAMPLER_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

target_link_libraries(AltDenoiserPlugin PRIVATE
    df
    AltDenoiserAssets
    ${RUST_SYSTEM_LIBS}
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_audio_devices
    juce::juce_core
    juce::juce_audio_basics
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
)
add_dependencies(AltDenoiserPlugin build_alt_df)

# ---------------------------------------------------------------------------
# Offline render harness.
#
# pluginval passes this plugin at strictness 5 with every BACKLOG.md defect
# present, because it validates VST3 well-formedness rather than audio
# correctness. This target drives AltDenoiserProcessor directly and asserts on
# the rendered samples. See tests/OfflineTests.cpp.
# ---------------------------------------------------------------------------
juce_add_console_app(AltDenoiserTests
    PRODUCT_NAME "AltDenoiserTests"
)

target_sources(AltDenoiserTests PRIVATE
    tests/OfflineTests.cpp
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/DeepFilterNetProcessor.cpp
)

target_compile_definitions(AltDenoiserTests PRIVATE
    ALTDENOISER_HAS_STANDARD_MODEL=${ALTDENOISER_HAS_STANDARD}
    ALTDENOISER_HAS_LL_MODEL=${ALTDENOISER_HAS_LL}
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_STANDALONE_APPLICATION=1
)

if(MSVC)
    target_compile_options(AltDenoiserTests PRIVATE /bigobj)
endif()

target_include_directories(AltDenoiserTests PRIVATE
    ${LIBS_INCLUDE_DIR}
    ${RESAMPLER_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

target_link_libraries(AltDenoiserTests PRIVATE
    df
    AltDenoiserAssets
    ${RUST_SYSTEM_LIBS}
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_audio_devices
    juce::juce_core
    juce::juce_audio_basics
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
)
add_dependencies(AltDenoiserTests build_alt_df)
