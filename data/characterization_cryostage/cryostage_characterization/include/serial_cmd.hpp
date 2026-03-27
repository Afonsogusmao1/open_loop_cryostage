#pragma once

#include <Arduino.h>

#include "characterization.hpp"
#include "control.hpp"
#include "tc_manager.hpp"

class SerialCmdParser
{
public:
    SerialCmdParser(PeltierTempController &ctrl,
                    TcManager &tc,
                    CharacterizationController &char_ctrl,
                    bool &stream_on);

    // Call often from loop().
    void poll();

private:
    PeltierTempController &ctrl_;
    TcManager &tc_;
    CharacterizationController &char_;
    bool &stream_on_;

    char buf_[200];
    size_t n_ = 0;

    void handleLine_(const String &line);
    bool handleCharCommand_(const String &line, const String &lower);
    bool manualCommandsLocked_() const;

    static float parseScalar_(String text);
    static bool parseBoolToken_(String text, bool &out);
    static bool parseSequence_(const String &text, float *values, size_t &count);
};
