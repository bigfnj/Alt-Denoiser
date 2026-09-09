#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include "PluginProcessor.h"

class ModernLookAndFeel : public juce::LookAndFeel_V4
{
public:
    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
        float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
        juce::Slider&) override
    {

        auto radius = (float)juce::jmin(width / 2, height / 2) - 20.0f; 
        auto center = juce::Point<float>((float)x + (float)width * 0.5f, (float)y + (float)height * 0.5f);
        const float trackWidth = 14.0f;

        // track
        juce::Path backgroundArc;
        backgroundArc.addArc(center.x - radius, center.y - radius, radius * 2.0f, radius * 2.0f,
                             rotaryStartAngle, rotaryEndAngle, true);

        g.setColour(juce::Colour(0xff1a1a1a));
        g.strokePath(backgroundArc, juce::PathStrokeType(trackWidth, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        // fill
        auto angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);
        juce::Path valueArc;
        valueArc.addArc(center.x - radius, center.y - radius, radius * 2.0f, radius * 2.0f,
                        rotaryStartAngle, angle, true);

        g.setColour(juce::Colour(0xffff9900));
        g.strokePath(valueArc, juce::PathStrokeType(trackWidth, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    juce::Label* createSliderTextBox(juce::Slider& slider) override
    {
        auto* l = LookAndFeel_V4::createSliderTextBox(slider);
        l->setColour(juce::Label::outlineColourId, juce::Colours::transparentBlack);
        l->setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
        l->setColour(juce::Label::textColourId, juce::Colours::lightgrey);
        l->setFont(juce::FontOptions(14.0f));
        return l;
    }
};

/** Sample-peak meter with frame-rate-independent ballistics.

    Replaces a design that decayed by a fixed factor per timer tick. That was
    -1.94 dB per tick, so -116 dB/s at 60 Hz, five to ten times faster than any
    metering standard, on a timer JUCE documents as imprecise to 10-20 ms at
    exactly this interval (M9). The bar and the numeric readout also decayed at
    different rates from separate state, so they disagreed after every transient
    (M11), and the readout decremented without a floor (L3).

    Now one attack event feeds two values in the dB domain. levelDb drives the
    bar and falls at a standard rate; holdDb is a peak hold that drives the
    readout and falls at the same rate once its hold expires. The invariant
    holdDb >= levelDb therefore always holds, and the two converge.
*/
class DbMeter : public juce::Component
{
public:
    explicit DbMeter(bool isInputMode)
        : isInput(isInputMode), caption(isInputMode ? "IN" : "OUT") {}

    static constexpr float kFloorDb      = -100.0f;
    static constexpr float kMinDb        =  -60.0f;  // bottom of the drawn scale
    static constexpr float kMaxDb        =    6.0f;  // top of the drawn scale
    static constexpr float kFallDbPerSec =   20.0f;  // standard digital peak meter
    static constexpr double kHoldSeconds =    1.5;

    /** @param newPeak    linear sample peak since the last call, 0 if none
        @param dtSeconds  real elapsed time, not an assumed frame interval
    */
    void update(float newPeak, double dtSeconds)
    {
        // Clamped so a stalled message thread or a debugger pause cannot slam
        // the meter to the floor in a single step.
        const float dt = (float) juce::jlimit(0.0, 0.2, dtSeconds);
        const float fall = kFallDbPerSec * dt;

        levelDb = juce::jmax(kFloorDb, levelDb - fall);   // L3: clamped, not unbounded

        if (holdRemaining > 0.0f) holdRemaining -= dt;
        else                      holdDb = juce::jmax(kFloorDb, holdDb - fall);

        if (newPeak > 0.0f)
        {
            const float peakDb = juce::Decibels::gainToDecibels(newPeak, kFloorDb);
            if (peakDb > levelDb) levelDb = peakDb;              // instantaneous attack
            if (peakDb > holdDb) { holdDb = peakDb; holdRemaining = (float) kHoldSeconds; }
        }

        holdDb = juce::jmax(holdDb, levelDb);    // M11: the invariant, made explicit

        // L1: repaint only when the rendered result actually differs. In silence
        // both values are pinned at the floor, so the repaint count drops to zero
        // rather than redrawing an identical image 60 times a second forever.
        const int newBarY = juce::roundToInt(mapDbToY(levelDb));
        const juce::String newReadout = (holdDb <= -90.0f) ? juce::String("-inf")
                                                           : juce::String(holdDb, 1);
        if (newBarY != paintedBarY || newReadout != paintedReadout)
        {
            paintedBarY = newBarY;
            paintedReadout = newReadout;
            ++repaintRequests;
            repaint();
        }
    }

    // Test accessors.
    float getLevelDb() const noexcept { return levelDb; }
    float getHoldDb()  const noexcept { return holdDb; }
    int   getPaintCount() const noexcept { return paintCount; }

    /** Counts repaint REQUESTS, not paints.

        A headless test never pumps a message loop, so paint() is never invoked
        and a paint counter cannot move whether the gate works or not. Counting
        the decision is the only thing an offline test can actually assert.
    */
    int   getRepaintRequests() const noexcept { return repaintRequests; }

    void resized() override
    {
        // L2: everything static for a given size is built here rather than
        // rebuilt every frame. paint() previously constructed a Path, a
        // ColourGradient and six Strings per meter per frame, roughly 1000
        // allocations a second across the two meters.
        auto bounds = getLocalBounds().toFloat();
        auto meterArea = bounds.withTrimmedTop(kTopTextHeight)
                               .withTrimmedBottom(kBottomTextHeight)
                               .reduced(14.0f, 0);

        if (isInput) { barRect = meterArea.removeFromLeft(12.0f);  tickArea = meterArea.withTrimmedLeft(6.0f); }
        else         { barRect = meterArea.removeFromRight(12.0f); tickArea = meterArea.withTrimmedRight(6.0f); }

        clipPath.clear();
        clipPath.addRoundedRectangle(barRect, 2.0f);

        y0dB       = mapDbToY(0.0f);
        yMinus6dB  = mapDbToY(-6.0f);
        yMinus24dB = mapDbToY(-24.0f);

        gradient = juce::ColourGradient(juce::Colours::yellow, 0, yMinus6dB,
                                        juce::Colours::green,  0, yMinus24dB, false);

        ticks.clear();
        for (float t : { 0.0f, -6.0f, -12.0f, -24.0f, -48.0f })
        {
            const float y = mapDbToY(t);
            if (y >= barRect.getY() && y <= barRect.getBottom())
                ticks.push_back({ y, juce::String((int) t) });
        }

        topTextRect = juce::Rectangle<float>(0, 0, 60, kTopTextHeight)
                        .withCentre({ barRect.getCentreX(), kTopTextHeight * 0.5f });
        botTextRect = juce::Rectangle<float>(0, 0, 60, kBottomTextHeight)
                        .withCentre({ barRect.getCentreX(), bounds.getBottom() - kBottomTextHeight * 0.5f });

        paintedBarY = juce::roundToInt(mapDbToY(levelDb));
    }

    void paint(juce::Graphics& g) override
    {
        ++paintCount;

        g.setColour(juce::Colour(0xff181818));
        g.fillRoundedRectangle(barRect, 2.0f);

        const float yBottom  = barRect.getBottom();
        const float yCurrent = juce::jlimit(barRect.getY(), yBottom, mapDbToY(levelDb));

        g.saveState();
        g.reduceClipRegion(clipPath);

        const float greenTop = juce::jmax(yCurrent, yMinus24dB);
        if (greenTop < yBottom)
        {
            g.setColour(juce::Colours::green);
            g.fillRect(barRect.withTop(greenTop).withBottom(yBottom));
        }
        const float gradTop = juce::jmax(yCurrent, yMinus6dB);
        if (gradTop < yMinus24dB)
        {
            g.setGradientFill(gradient);
            g.fillRect(barRect.withTop(gradTop).withBottom(yMinus24dB));
        }
        const float yellowTop = juce::jmax(yCurrent, y0dB);
        if (yellowTop < yMinus6dB)
        {
            g.setColour(juce::Colours::yellow);
            g.fillRect(barRect.withTop(yellowTop).withBottom(yMinus6dB));
        }
        if (yCurrent < y0dB)
        {
            // Reachable now that the source is sample peak rather than RMS. A
            // full-scale sine is -3.01 dB RMS and could never light this (L4).
            g.setColour(juce::Colours::red);
            g.fillRect(barRect.withTop(yCurrent).withBottom(y0dB));
        }
        g.restoreState();

        g.setColour(juce::Colours::white);
        g.setFont(juce::FontOptions(12.0f));
        g.drawFittedText(paintedReadout, topTextRect.toNearestInt(), juce::Justification::centred, 1);

        g.setFont(juce::FontOptions(10.0f));
        g.setColour(juce::Colours::grey);
        g.drawFittedText(caption, botTextRect.toNearestInt(), juce::Justification::centred, 1);

        const auto just = isInput ? juce::Justification::centredLeft : juce::Justification::centredRight;
        for (const auto& tick : ticks)
        {
            g.setColour(juce::Colours::white.withAlpha(0.3f));
            g.fillRect(barRect.getX(), tick.y, barRect.getWidth(), 1.0f);
            g.setColour(juce::Colours::grey);
            g.drawText(tick.label, tickArea.withY(tick.y - 5.0f).withHeight(10.0f), just, false);
        }
    }

private:
    struct Tick { float y; juce::String label; };

    static constexpr float kTopTextHeight    = 18.0f;
    static constexpr float kBottomTextHeight = 18.0f;

    float mapDbToY(float db) const
    {
        return juce::jmap(db, kMinDb, kMaxDb, barRect.getBottom(), barRect.getY());
    }

    const bool isInput;
    const juce::String caption;

    float levelDb = kFloorDb;
    float holdDb = kFloorDb;
    float holdRemaining = 0.0f;

    // Precomputed in resized().
    juce::Rectangle<float> barRect, tickArea, topTextRect, botTextRect;
    juce::Path clipPath;
    juce::ColourGradient gradient;
    std::vector<Tick> ticks;
    float y0dB = 0.0f, yMinus6dB = 0.0f, yMinus24dB = 0.0f;

    int paintedBarY = -1;
    juce::String paintedReadout;
    int paintCount = 0;
    int repaintRequests = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DbMeter)
};

class AltDenoiserEditor : public juce::AudioProcessorEditor, public juce::Timer
{
public:
    AltDenoiserEditor(AltDenoiserProcessor&, juce::AudioProcessorValueTreeState&);
    ~AltDenoiserEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    AltDenoiserProcessor& audioProcessor;
    juce::AudioProcessorValueTreeState& apvts;
    
    ModernLookAndFeel modernLook;

    juce::Slider attenSlider;
    juce::Label attenLabel;
    juce::TextButton aboutButton { "i" };

    DbMeter inputMeter { true };  // true = IN mode
    DbMeter outputMeter { false }; // false = OUT mode

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attenAttachment;

    double lastTimerSeconds = 0.0;   // for the real-dt meter ballistics

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AltDenoiserEditor)
};