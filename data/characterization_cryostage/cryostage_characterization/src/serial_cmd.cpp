#include "serial_cmd.hpp"

SerialCmdParser::SerialCmdParser(PeltierTempController &ctrl,
                                 TcManager &tc,
                                 CharacterizationController &char_ctrl,
                                 bool &stream_on)
    : ctrl_(ctrl),
      tc_(tc),
      char_(char_ctrl),
      stream_on_(stream_on)
{
    buf_[0] = '\0';
}

float SerialCmdParser::parseScalar_(String text)
{
    text.trim();
    text.replace(',', '.');
    return text.toFloat();
}

bool SerialCmdParser::parseBoolToken_(String text, bool &out)
{
    text.trim();
    text.toLowerCase();

    if (text == "1" || text == "on" || text == "true")
    {
        out = true;
        return true;
    }
    if (text == "0" || text == "off" || text == "false")
    {
        out = false;
        return true;
    }
    return false;
}

bool SerialCmdParser::parseSequence_(const String &text, float *values, size_t &count)
{
    count = 0;
    String token;

    auto flush_token = [&](void) -> bool
    {
        token.trim();
        if (token.length() == 0)
            return true;
        if (count >= CharacterizationController::MAX_SEQUENCE_LEN)
            return false;
        values[count++] = parseScalar_(token);
        token = "";
        return true;
    };

    for (size_t i = 0; i < (size_t)text.length(); ++i)
    {
        const char c = text[(unsigned int)i];
        if (c == ',' || c == ';' || c == ' ' || c == '\t')
        {
            if (!flush_token())
                return false;
        }
        else
        {
            token += c;
        }
    }

    return flush_token() && count > 0;
}

void SerialCmdParser::poll()
{
    while (Serial.available())
    {
        const char c = (char)Serial.read();
        if (c == '\r')
            continue;

        if (c == '\n')
        {
            buf_[n_] = '\0';
            n_ = 0;

            String line(buf_);
            line.trim();
            if (line.length() == 0)
                continue;

            handleLine_(line);
        }
        else
        {
            if (n_ < sizeof(buf_) - 1)
                buf_[n_++] = c;
            else
                n_ = 0; // overflow protection
        }
    }
}

bool SerialCmdParser::manualCommandsLocked_() const
{
    return char_.blocksManualCommands();
}

bool SerialCmdParser::handleCharCommand_(const String &line, const String &lower)
{
    const uint32_t now_ms = millis();

    if (lower == "char" || lower == "char help")
    {
        char_.printHelp();
        char_.printConfig();
        return true;
    }

    if (lower == "char status")
    {
        char_.printStatus(now_ms);
        return true;
    }

    if (lower == "char config")
    {
        char_.printConfig();
        return true;
    }

    if (lower == "char start")
    {
        if (char_.start(now_ms))
        {
            ctrl_.printStatus(stream_on_);
            char_.printStatus(now_ms);
        }
        return true;
    }

    if (lower == "char pause")
    {
        if (!char_.pause(now_ms))
            Serial.println("[ERR] characterization is not running");
        char_.printStatus(now_ms);
        return true;
    }

    if (lower == "char resume")
    {
        if (!char_.resume(now_ms))
            Serial.println("[ERR] characterization is not paused");
        char_.printStatus(now_ms);
        return true;
    }

    if (lower == "char stop")
    {
        char_.stop(now_ms);
        ctrl_.printStatus(stream_on_);
        char_.printStatus(now_ms);
        return true;
    }

    if (lower == "char abort")
    {
        char_.abort(now_ms, "user_abort");
        ctrl_.printStatus(stream_on_);
        char_.printStatus(now_ms);
        return true;
    }

    if (lower == "char seq")
    {
        char_.printConfig();
        return true;
    }

    if (lower.startsWith("char seq "))
    {
        if (char_.blocksManualCommands())
        {
            Serial.println("[ERR] stop the active characterization before changing the sequence");
            return true;
        }

        float values[CharacterizationController::MAX_SEQUENCE_LEN];
        size_t count = 0;
        if (!parseSequence_(line.substring(9), values, count))
        {
            Serial.println("[ERR] char seq <v1,v2,...>");
            return true;
        }

        if (!char_.setSequence(values, count))
        {
            Serial.println("[ERR] invalid characterization sequence");
            return true;
        }

        Serial.printf("[OK] characterization sequence updated (%d steps)\r\n", (int)count);
        char_.printConfig();
        return true;
    }

    auto reject_if_busy = [&]() -> bool
    {
        if (char_.blocksManualCommands())
        {
            Serial.println("[ERR] stop the active characterization before changing characterization settings");
            return true;
        }
        return false;
    };

    if (lower.startsWith("char tol "))
    {
        if (reject_if_busy())
            return true;
        char_.setTolerance(parseScalar_(line.substring(9)));
        char_.printConfig();
        return true;
    }

    if (lower.startsWith("char dwell "))
    {
        if (reject_if_busy())
            return true;
        char_.setStabilizationDwell((uint32_t)max(0.0f, parseScalar_(line.substring(11))));
        char_.printConfig();
        return true;
    }

    if (lower.startsWith("char postdwell "))
    {
        if (reject_if_busy())
            return true;
        char_.setDwellAfterStable((uint32_t)max(0.0f, parseScalar_(line.substring(15))));
        char_.printConfig();
        return true;
    }

    if (lower.startsWith("char maxhold "))
    {
        if (reject_if_busy())
            return true;
        char_.setMaxHold((uint32_t)max(0.0f, parseScalar_(line.substring(13))));
        char_.printConfig();
        return true;
    }

    if (lower.startsWith("char continue_timeout "))
    {
        if (reject_if_busy())
            return true;
        bool on = false;
        if (!parseBoolToken_(line.substring(22), on))
        {
            Serial.println("[ERR] char continue_timeout <0|1|on|off>");
            return true;
        }
        char_.setContinueOnTimeout(on);
        char_.printConfig();
        return true;
    }

    if (lower.startsWith("char slope "))
    {
        if (reject_if_busy())
            return true;
        char_.setSlopeLimit(parseScalar_(line.substring(11)));
        char_.printConfig();
        return true;
    }

    if (lower.startsWith("char slope_window "))
    {
        if (reject_if_busy())
            return true;
        char_.setSlopeWindow((uint32_t)max(0.0f, parseScalar_(line.substring(18))));
        char_.printConfig();
        return true;
    }

    Serial.println("[ERR] unknown characterization command (try 'char help')");
    return true;
}

void SerialCmdParser::handleLine_(const String &line)
{
    String lower(line);
    lower.toLowerCase();

    // --- Help ---
    if (lower == "help" || lower == "?")
    {
        ctrl_.printHelp();
        char_.printHelp();
        Serial.println("[INFO] Thermocouple calibration coefficients are fixed in firmware.");
        Serial.println("       Use: status");
        return;
    }

    // --- Status ---
    if (lower == "status")
    {
        ctrl_.printStatus(stream_on_);
        char_.printStatus(millis());
        tc_.printStatus();
        return;
    }

    // --- Characterization ---
    if (lower.startsWith("char"))
    {
        handleCharCommand_(line, lower);
        return;
    }

    auto reject_manual_if_needed = [&]() -> bool
    {
        if (manualCommandsLocked_())
        {
            Serial.println("[ERR] characterization active; use char pause/resume/stop/abort");
            return true;
        }
        return false;
    };

    // --- Setpoint ---
    if (lower.startsWith("setpoint ") || lower.startsWith("set "))
    {
        if (reject_manual_if_needed())
            return;

        const int offset = lower.startsWith("setpoint ") ? 9 : 4;
        const float sp = parseScalar_(line.substring(offset));
        ctrl_.cmdSetpoint(sp);
        Serial.printf("[OK] setpoint=%.2f\r\n", (float)ctrl_.setpoint());
        ctrl_.printStatus(stream_on_);
        return;
    }

    // --- Arm / Disarm ---
    if (lower == "arm")
    {
        if (reject_manual_if_needed())
            return;

        if (ctrl_.cmdArm())
        {
            Serial.println("[OK] ARMED");
            ctrl_.printStatus(stream_on_);
        }
        return;
    }
    if (lower == "disarm")
    {
        if (reject_manual_if_needed())
            return;

        ctrl_.cmdDisarm();
        Serial.println("[OK] DISARMED");
        ctrl_.printStatus(stream_on_);
        return;
    }
    if (lower.startsWith("arm "))
    {
        if (reject_manual_if_needed())
            return;

        bool on = false;
        if (!parseBoolToken_(line.substring(4), on))
        {
            Serial.println("[ERR] arm <0|1|on|off>");
            return;
        }
        if (on)
        {
            if (ctrl_.cmdArm())
                Serial.println("[OK] ARMED");
        }
        else
        {
            ctrl_.cmdDisarm();
            Serial.println("[OK] DISARMED");
        }
        ctrl_.printStatus(stream_on_);
        return;
    }

    // --- PID enable/disable ---
    if (lower.startsWith("pid "))
    {
        if (reject_manual_if_needed())
            return;

        bool on = false;
        if (!parseBoolToken_(line.substring(4), on))
        {
            Serial.println("[ERR] pid <0|1|on|off>");
            return;
        }
        ctrl_.cmdPidEnable(on);
        Serial.printf("[OK] PID %s\r\n", on ? "enabled" : "disabled");
        ctrl_.printStatus(stream_on_);
        return;
    }

    // --- Stream on/off ---
    if (lower.startsWith("stream "))
    {
        bool on = false;
        if (!parseBoolToken_(line.substring(7), on))
        {
            Serial.println("[ERR] stream <0|1|on|off>");
            return;
        }
        stream_on_ = on;
        Serial.printf("[OK] stream=%d\r\n", stream_on_ ? 1 : 0);
        return;
    }
    if (lower == "stream")
    {
        Serial.println("[ERR] stream <0|1|on|off>");
        return;
    }

    // --- PID tuning ---
    if (lower.startsWith("kp "))
    {
        if (reject_manual_if_needed())
            return;
        ctrl_.cmdSetKp(parseScalar_(line.substring(3)));
        Serial.printf("[OK] kp=%.6f\r\n", (double)ctrl_.kp());
        ctrl_.printStatus(stream_on_);
        return;
    }
    if (lower.startsWith("ki "))
    {
        if (reject_manual_if_needed())
            return;
        ctrl_.cmdSetKi(parseScalar_(line.substring(3)));
        Serial.printf("[OK] ki=%.6f\r\n", (double)ctrl_.ki());
        ctrl_.printStatus(stream_on_);
        return;
    }
    if (lower.startsWith("kd "))
    {
        if (reject_manual_if_needed())
            return;
        ctrl_.cmdSetKd(parseScalar_(line.substring(3)));
        Serial.printf("[OK] kd=%.6f\r\n", (double)ctrl_.kd());
        ctrl_.printStatus(stream_on_);
        return;
    }

    // --- Manual power ---
    if (lower == "off")
    {
        if (reject_manual_if_needed())
            return;

        ctrl_.cmdOff();
        Serial.println("[OK] output=0 (OFF) and PID disabled");
        ctrl_.printStatus(stream_on_);
        return;
    }
    if (lower.startsWith("power "))
    {
        if (reject_manual_if_needed())
            return;

        const float p = parseScalar_(line.substring(6));
        ctrl_.cmdSetManualPower(p);
        Serial.printf("[OK] power_cmd=%.1f\r\n", (double)ctrl_.powerCmd());
        ctrl_.printStatus(stream_on_);
        return;
    }

    Serial.println("[ERR] unknown command (try 'help')");
}
