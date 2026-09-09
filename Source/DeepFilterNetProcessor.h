#pragma once
#include <cstdint>
#include <vector>
#include "df.h"

class DeepFilterNetProcessor {
public:
    DeepFilterNetProcessor(uint32_t sampleRate = 48000);
    ~DeepFilterNetProcessor();

    bool initialize(); 
    void setAttenLim(float limitDB);
    
    void processFrame(const float* input, float* output);
    bool isReady() const { return state != nullptr; }

    /** Hop size the loaded model actually expects, in samples at 48 kHz.
        Returns 0 when no model is loaded.

        M3: df_process_frame builds its ndarray views with from_shape_ptr using
        the model's own hop_size and performs no bounds check, and the only
        guard inside libDF is a debug_assert compiled out in release. Passing
        buffers sized to a different hop is therefore a silent heap overrun, so
        the caller must size its frames from this rather than assuming 480.
    */
    size_t getFrameLength() const { return state != nullptr ? df_get_frame_length(state) : 0; }

private:
    DFState* state = nullptr;
    uint32_t sampleRate = 48000;
};