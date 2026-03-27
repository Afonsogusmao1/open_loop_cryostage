#pragma once

#include <Arduino.h>

#include "control.hpp"

class CharacterizationController
{
public:
    static constexpr size_t MAX_SEQUENCE_LEN = 16;
    static constexpr size_t HISTORY_CAPACITY = 420; // 1 Hz filtered samples.

    enum class State : uint8_t
    {
        IDLE = 0,
        STARTING,
        MOVING_TO_SETPOINT,
        STABILIZING,
        DWELLING,
        STEP_COMPLETE,
        PAUSED,
        FINISHED,
        ABORTED,
        FAULT,
    };

    struct Config
    {
        float tolerance_c = 0.5f;
        uint32_t stabilization_dwell_s = 300;
        uint32_t dwell_after_stable_s = 300;
        uint32_t max_hold_s = 900;
        bool continue_on_timeout = true;
        float slope_limit_c_per_min = 0.05f; // <= 0 disables the slope check.
        uint32_t slope_window_s = 60;
        float ema_alpha = 0.10f;
    };

    explicit CharacterizationController(PeltierTempController &ctrl, const Config &cfg);

    void begin();
    void update(uint32_t now_ms, float dt_s);

    bool start(uint32_t now_ms);
    bool pause(uint32_t now_ms);
    bool resume(uint32_t now_ms);
    void stop(uint32_t now_ms);
    void abort(uint32_t now_ms, const char *reason = "user_abort");

    bool setSequence(const float *values, size_t count);
    void restoreDefaultSequence();

    void setTolerance(float v);
    void setStabilizationDwell(uint32_t seconds);
    void setDwellAfterStable(uint32_t seconds);
    void setMaxHold(uint32_t seconds);
    void setContinueOnTimeout(bool on);
    void setSlopeLimit(float v);
    void setSlopeWindow(uint32_t seconds);

    const Config &config() const { return cfg_; }

    bool enabled() const;
    bool blocksManualCommands() const;
    bool faultActive() const { return state_ == State::FAULT; }

    State state() const { return state_; }
    const char *stateName() const;

    size_t sequenceLen() const { return sequence_len_; }
    float sequenceAt(size_t idx) const;

    int stepIndex() const { return step_index_; }
    int stepNumber() const { return step_index_ >= 0 ? step_index_ + 1 : 0; }
    int totalSteps() const { return (int)sequence_len_; }

    float currentTarget() const { return current_target_; }
    bool stabilized() const { return step_stabilized_; }
    bool timeoutFlag() const { return step_timeout_flag_; }
    bool inBand() const { return in_band_; }

    bool filteredTempValid() const { return filtered_valid_; }
    float filteredTemp() const { return filtered_temp_; }
    float slopeCPerMin() const { return slope_c_per_min_; }
    bool slopeCriterionMet() const;

    uint32_t stepElapsedMs(uint32_t now_ms) const;
    uint32_t totalElapsedMs(uint32_t now_ms) const;

    const char *faultMessage() const { return fault_message_; }

    void printHelp() const;
    void printStatus(uint32_t now_ms) const;
    void printConfig() const;

private:
    PeltierTempController &ctrl_;
    Config cfg_;

    State state_ = State::IDLE;
    State resume_state_ = State::IDLE;

    float sequence_[MAX_SEQUENCE_LEN] = {};
    size_t sequence_len_ = 0;

    int step_index_ = -1;
    float current_target_ = NAN;

    bool step_stabilized_ = false;
    bool step_timeout_flag_ = false;
    bool in_band_ = false;

    uint32_t run_started_ms_ = 0;
    uint32_t step_started_ms_ = 0;
    uint32_t band_entered_ms_ = 0;
    uint32_t dwell_started_ms_ = 0;
    uint32_t pause_started_ms_ = 0;
    uint32_t terminal_ms_ = 0;

    bool filtered_valid_ = false;
    float filtered_temp_ = NAN;
    float slope_c_per_min_ = NAN;

    uint32_t history_ts_ms_[HISTORY_CAPACITY] = {};
    float history_temp_[HISTORY_CAPACITY] = {};
    size_t history_count_ = 0;
    size_t history_head_ = 0;
    uint32_t next_history_ms_ = 0;

    char fault_message_[48] = "none";

    void clearRunState_();
    void clearHistory_();
    void safeShutdown_();
    void enterFault_(uint32_t now_ms, const char *reason);
    void enterAborted_(uint32_t now_ms, const char *reason);
    void finish_(uint32_t now_ms);
    void startStep_(size_t idx, uint32_t now_ms);
    void advanceStep_(uint32_t now_ms);
    bool checkTimeout_(uint32_t now_ms);

    void updateFilteredTemp_();
    void pushHistory_(uint32_t now_ms);
    void updateSlope_();
    bool isWithinBand_() const;
    uint32_t elapsedSince_(uint32_t start_ms, uint32_t now_ms) const;
    void shiftTimersForPause_(uint32_t paused_ms);
};
