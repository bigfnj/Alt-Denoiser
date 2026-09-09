#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace
{
    // Height of the model strip, and therefore of the window's growth. A build
    // with one archive has nothing to choose between, so it gets the info line
    // only.
    constexpr int kStripBoth = 80;
    constexpr int kStripSlim = 44;

    /** What each model costs, for the info line.

        Both figures are MEASURED, not estimated, and both are asserted on every
        gate run by "model keeps up with realtime" in tests/OfflineTests.cpp. If
        they drift, that test's printed detail is the source to copy from.

        The latency is exact rather than measured: it is the derived figure,
        1920 and 960 samples at the model's own 48 kHz, and it scales with the
        host rate so the millisecond value holds at every rate.

        Note which way round these go. "Low latency" names the delay, not the
        cost: it is the bigger network (emb_hidden_dim 512 against 256,
        df_num_layers 3 against 2) and takes roughly three times the CPU.
    */
    struct ModelCost { const char* latency; const char* cpu; };

    ModelCost costOf (DfnModel which)
    {
        switch (which)
        {
            case DfnModel::Standard:   return { "40 ms", "9%" };
            case DfnModel::LowLatency: return { "20 ms", "27%" };
        }
        return { "40 ms", "9%" };
    }
}

bool AltDenoiserEditor::hasAChoice()
{
    return DeepFilterNetProcessor::isModelAvailable (DfnModel::Standard)
        && DeepFilterNetProcessor::isModelAvailable (DfnModel::LowLatency);
}

AltDenoiserEditor::AltDenoiserEditor(AltDenoiserProcessor& p, juce::AudioProcessorValueTreeState& vts)
    : AudioProcessorEditor(&p), audioProcessor(p), apvts(vts)
{
    setLookAndFeel(&modernLook);

    attenSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    attenSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 80, 25); 
    attenSlider.setTextValueSuffix(" dB");
    addAndMakeVisible(attenSlider);

    attenLabel.setText("Reduction", juce::dontSendNotification);
    attenLabel.setJustificationType(juce::Justification::centred);
    attenLabel.setFont(juce::Font(juce::FontOptions(13.0f).withStyle("bold")));
    attenLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(attenLabel);

    attenAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "atten_lim", attenSlider
    );

    addAndMakeVisible(aboutButton);
    aboutButton.setColour(juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
    aboutButton.setColour(juce::TextButton::textColourOffId, juce::Colours::grey);
    aboutButton.onClick = [this] { juce::AlertWindow::showMessageBoxAsync(
            juce::AlertWindow::InfoIcon,
            "About",
            // Taken from the build rather than typed. It said "v1.0" while the
            // last release was v1.0.1, and a hardcoded version in an About box
            // drifts the moment anyone tags. JucePlugin_VersionString is not
            // defined for the console test target, which compiles this file too.
           #ifdef JucePlugin_VersionString
            "Alt Denoiser v" JucePlugin_VersionString "\n"
           #else
            "Alt Denoiser\n"
           #endif
            "By Altinus\n\n"
            "Credits:\n"
            "DeepFilterNet (Rikorose)\n"
            "Resampler (niswegmann)\n"
            "JUCE Framework",
            "ok" 
            ); };


    // 7c: the model strip.
    //
    // ModernLookAndFeel overrides only drawRotarySlider and createSliderTextBox,
    // so a stock ComboBox renders with LookAndFeel_V4's light defaults against
    // the 0xff1e1e1e background. Per-component setColour is the existing
    // workaround for exactly that, as already done for aboutButton above.
    for (int i = 0; i < kNumDfnModels; ++i)
        modelBox.addItem(DeepFilterNetProcessor::getModelDisplayName(static_cast<DfnModel>(i)),
                         i + 1);   // ComboBox ids are 1-based and must match the choice order

    modelBox.setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff2a2a2a));
    modelBox.setColour(juce::ComboBox::textColourId,       juce::Colours::lightgrey);
    modelBox.setColour(juce::ComboBox::outlineColourId,    juce::Colour(0xff3c3c3c));
    modelBox.setColour(juce::ComboBox::arrowColourId,      juce::Colours::grey);
    modelBox.setColour(juce::PopupMenu::backgroundColourId,          juce::Colour(0xff2a2a2a));
    modelBox.setColour(juce::PopupMenu::textColourId,                juce::Colours::lightgrey);
    modelBox.setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colour(0xff3d6ea5));

    modelLabel.setText("Model", juce::dontSendNotification);
    modelLabel.setJustificationType(juce::Justification::centredRight);
    modelLabel.setFont(juce::Font(juce::FontOptions(13.0f).withStyle("bold")));
    modelLabel.setColour(juce::Label::textColourId, juce::Colours::grey);

    modelInfo.setJustificationType(juce::Justification::centred);
    modelInfo.setFont(juce::Font(juce::FontOptions(11.0f)));
    modelInfo.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(modelInfo);

    // A slim build has one archive, so the row would be a control that cannot
    // change anything. The info line stays, since what is compiled in and what
    // it costs is still worth stating.
    if (hasAChoice())
    {
        addAndMakeVisible(modelLabel);
        addAndMakeVisible(modelBox);

        modelAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
            apvts, "model", modelBox
        );

        modelBox.onChange = [this] { updateModelInfo(); };
    }

    updateModelInfo();

    addAndMakeVisible(inputMeter);
    addAndMakeVisible(outputMeter);

    startTimerHz(60);
    setSize(460, 320 + (hasAChoice() ? kStripBoth : kStripSlim)); 
}

AltDenoiserEditor::~AltDenoiserEditor()
{
    setLookAndFeel(nullptr);
    stopTimer();
}

void AltDenoiserEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff1e1e1e));
    g.setColour(juce::Colour(0xffdddddd)); 
    g.setFont(juce::Font(juce::FontOptions(25.0f).withStyle("bold")));
    g.drawFittedText("Alt Denoiser", getLocalBounds().removeFromTop(40), juce::Justification::centred, 1);
}

void AltDenoiserEditor::updateModelInfo()
{
    // Describes the SELECTED model, not the loaded one, so the info line answers
    // "what would this buy me" while the user is still deciding.
    const int selected = hasAChoice() ? juce::jmax(0, modelBox.getSelectedItemIndex())
                                      : (int) DeepFilterNetProcessor::getFirstAvailableModel();
    const auto which = static_cast<DfnModel>(juce::jlimit(0, kNumDfnModels - 1, selected));
    const auto cost = costOf(which);

    juce::String text;
    text << cost.latency << " latency, about " << cost.cpu << " of one core.";

    if (hasAChoice())
        text << "  Takes effect when the plugin is reloaded.";

    modelInfo.setText(text, juce::dontSendNotification);
}

void AltDenoiserEditor::resized()
{
    auto area = getLocalBounds();

    // The model strip is reserved FIRST, so everything above keeps the position
    // it had before the strip existed rather than sliding down by half of it.
    auto strip = area.removeFromBottom(hasAChoice() ? kStripBoth : kStripSlim);

    // 1. meter
    inputMeter.setBounds(20, 50, 70, area.getHeight() - 80); 
    outputMeter.setBounds(getWidth() - 90, 50, 70, area.getHeight() - 80);

    // 2. knob and label
    const int knobComponentSize = 200;     
    auto knobBounds = juce::Rectangle<int>(
        area.getCentreX() - knobComponentSize / 2, 
        area.getCentreY() - knobComponentSize / 2 + 10,
        knobComponentSize, 
        knobComponentSize
    );
    attenSlider.setBounds(knobBounds);

    attenLabel.setBounds(
        knobBounds.getX(), 
        knobBounds.getY() + knobBounds.getHeight() + 5,
        knobBounds.getWidth(), 
        20
    );

    // 3.info
    aboutButton.setBounds(getWidth() - 30, 10, 20, 20);

    // 4. model strip
    strip.removeFromBottom(8);

    if (hasAChoice())
    {
        auto row = strip.removeFromTop(26).withSizeKeepingCentre(280, 26);
        modelLabel.setBounds(row.removeFromLeft(56));
        row.removeFromLeft(8);
        modelBox.setBounds(row);
        strip.removeFromTop(4);
    }

    modelInfo.setBounds(strip.reduced(16, 0));
}

void AltDenoiserEditor::timerCallback()
{
    // M9: measure the interval rather than assuming 1/60 s. JUCE documents the
    // timer as imprecise to 10-20 ms, which at a 16.7 ms nominal interval is
    // 60-120% of the interval itself.
    const double now = juce::Time::getMillisecondCounterHiRes() * 0.001;
    const double dt = (lastTimerSeconds > 0.0) ? (now - lastTimerSeconds) : 0.0;
    lastTimerSeconds = now;

    // takeAndReset, not load: the probe holds a running peak, so consuming it
    // is what makes the UI window exactly one frame with nothing dropped.
    inputMeter.update(audioProcessor.inputLevel.takeAndReset(), dt);
    outputMeter.update(audioProcessor.outputLevel.takeAndReset(), dt);
}