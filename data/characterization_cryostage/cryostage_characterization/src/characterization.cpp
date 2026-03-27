#include "characterization.hpp"

#include <math.h>
#include <stdio.h>
#include <string.h>

CharacterizationController::CharacterizationController(PeltierTempController &ctrl, const Config &cfg)
    : ctrl_(ctrl),
      cfg_(cfg)
{
}

void CharacterizationController::begin()
{
    restoreDefaultSequence();
    clearRunState_();
}

void CharacterizationController::clearHistory_()
{
    history_count_ = 0;
    history_head_ = 0;
    next_history_ms_ = 0;
    slope_c_per_min_ = NAN;
    filtered_valid_ = false;
    filtered_temp_ = NAN;
}

void CharacterizationController::clearRunState_()
{
    state_ = State::IDLE;
    resume_state_ = State::IDLE;
    step_index_ = -1;
    current_target_ = NAN;
    step_stabilized_ = false;
    step_timeout_flag_ = false;
    in_band_ = false;
    run_started_ms_ = 0;
    step_started_ms_ = 0;
    band_entered_ms_ = 0;
    dwell_started_ms_ = 0;
    pause_started_ms_ = 0;
    terminal_ms_ = 0;
    strncpy(fault_message_, "none", sizeof(fault_message_) - 1);
    fault_message_[sizeof(fault_message_) - 1] = '\0';
    clearHistory_();
}

bool CharacterizationController::enabled() const
{
    switch (state_)
    {
    case State::STARTING:
    case State::MOVING_TO_SETPOINT:
    case State::STABILIZING:
    case State::DWELLING:
    case State::STEP_COMPLETE:
    case State::PAUSED:
        return true;
    default:
        return false;
    }
}

bool CharacterizationController::blocksManualCommands() const
{
    return enabled();
}

const char *CharacterizationController::stateName() const
{
    switch (state_)
    {
    case State::IDLE:
        return "IDLE";
    case State::STARTING:
        return "STARTING";
    case State::MOVING_TO_SETPOINT:
        return "MOVING_TO_SETPOINT";
    case State::STABILIZING:
        return "STABILIZING";
    case State::DWELLING:
        return "DWELLING";
    case State::STEP_COMPLETE:
        return "STEP_COMPLETE";
    case State::PAUSED:
        return "PAUSED";
    case State::FINISHED:
        return "FINISHED";
    case State::ABORTED:
        return "ABORTED";
    case State::FAULT:
        return "FAULT";
    default:
        return "UNKNOWN";
    }
}

float CharacterizationController::sequenceAt(size_t idx) const
{
    if (idx >= sequence_len_)
        return NAN;
    return sequence_[idx];
}

void CharacterizationController::restoreDefaultSequence()
{
    const float defaults[] = {5.0f, 0.0f, -5.0f, -10.0f, -15.0f, -18.0f};
    setSequence(defaults, sizeof(defaults) / sizeof(defaults[0]));
}

bool CharacterizationController::setSequence(const float *values, size_t count)
{
    if (!values || count == 0 || count > MAX_SEQUENCE_LEN)
        return false;

    for (size_t i = 0; i < count; ++i)
        sequence_[i] = values[i];

    sequence_len_ = count;
    return true;
}

void CharacterizationController::setTolerance(float v)
{
    cfg_.tolerance_c = max(0.01f, v);
}

void CharacterizationController::setStabilizationDwell(uint32_t seconds)
{
    cfg_.stabilization_dwell_s = seconds;
}

void CharacterizationController::setDwellAfterStable(uint32_t seconds)
{
    cfg_.dwell_after_stable_s = seconds;
}

void CharacterizationController::setMaxHold(uint32_t seconds)
{
    cfg_.max_hold_s = seconds;
}

void CharacterizationController::setContinueOnTimeout(bool on)
{
    cfg_.continue_on_timeout = on;
}

void CharacterizationController::setSlopeLimit(float v)
{
    cfg_.slope_limit_c_per_min = v;
}

void CharacterizationController::setSlopeWindow(uint32_t seconds)
{
    cfg_.slope_window_s = seconds;
}

bool CharacterizationController::start(uint32_t now_ms)
{
    if (enabled())
    {
        Serial.println("[ERR] characterization already active");
        return false;
    }
    if (sequence_len_ == 0)
    {
        Serial.println("[ERR] characterization sequence is empty");
        return false;
    }

    clearRunState_();
    run_started_ms_ = now_ms;
    step_index_ = 0;
    current_target_ = sequence_[0];
    step_started_ms_ = now_ms;
    state_ = State::STARTING;

    ctrl_.cmdSetpoint(current_target_);

    if (!ctrl_.armed() && !ctrl_.cmdArm())
    {
        enterFault_(now_ms, "arm_failed");
        return false;
    }

    ctrl_.cmdPidEnable(true);
    Serial.printf("[OK] characterization started (%d steps)\r\n", (int)sequence_len_);
    return true;
}

bool CharacterizationController::pause(uint32_t now_ms)
{
    if (!enabled() || state_ == State::PAUSED)
        return false;

    resume_state_ = state_;
    pause_started_ms_ = now_ms;
    state_ = State::PAUSED;
    Serial.println("[OK] characterization paused");
    return true;
}

bool CharacterizationController::resume(uint32_t now_ms)
{
    if (state_ != State::PAUSED)
        return false;

    const uint32_t paused_ms = elapsedSince_(pause_started_ms_, now_ms);
    shiftTimersForPause_(paused_ms);
    pause_started_ms_ = 0;
    state_ = resume_state_;
    resume_state_ = State::IDLE;
    Serial.println("[OK] characterization resumed");
    return true;
}

void CharacterizationController::stop(uint32_t now_ms)
{
    enterAborted_(now_ms, "stopped_by_user");
}

void CharacterizationController::abort(uint32_t now_ms, const char *reason)
{
    enterAborted_(now_ms, reason ? reason : "user_abort");
}

void CharacterizationController::safeShutdown_()
{
    ctrl_.cmdDisarm();
}

void CharacterizationController::enterFault_(uint32_t now_ms, const char *reason)
{
    safeShutdown_();
    state_ = State::FAULT;
    terminal_ms_ = now_ms;
    pause_started_ms_ = 0;

    strncpy(fault_message_, reason ? reason : "fault", sizeof(fault_message_) - 1);
    fault_message_[sizeof(fault_message_) - 1] = '\0';

    Serial.printf("[ERR] characterization fault: %s\r\n", fault_message_);
}

void CharacterizationController::enterAborted_(uint32_t now_ms, const char *reason)
{
    safeShutdown_();
    state_ = State::ABORTED;
    terminal_ms_ = now_ms;
    pause_started_ms_ = 0;
    strncpy(fault_message_, "none", sizeof(fault_message_) - 1);
    fault_message_[sizeof(fault_message_) - 1] = '\0';

    Serial.printf("[WARN] characterization aborted: %s\r\n", reason ? reason : "user_abort");
}

void CharacterizationController::finish_(uint32_t now_ms)
{
    safeShutdown_();
    state_ = State::FINISHED;
    terminal_ms_ = now_ms;
    pause_started_ms_ = 0;
    strncpy(fault_message_, "none", sizeof(fault_message_) - 1);
    fault_message_[sizeof(fault_message_) - 1] = '\0';

    Serial.println("[OK] characterization finished");
}

void CharacterizationController::startStep_(size_t idx, uint32_t now_ms)
{
    if (idx >= sequence_len_)
    {
        finish_(now_ms);
        return;
    }

    step_index_ = (int)idx;
    current_target_ = sequence_[idx];
    step_started_ms_ = now_ms;
    band_entered_ms_ = 0;
    dwell_started_ms_ = 0;
    step_stabilized_ = false;
    step_timeout_flag_ = false;
    in_band_ = false;

    ctrl_.cmdSetpoint(current_target_);
    state_ = State::MOVING_TO_SETPOINT;

    Serial.printf("[OK] characterization step %d/%d target=%.2f\r\n",
                  stepNumber(),
                  totalSteps(),
                  current_target_);
}

void CharacterizationController::advanceStep_(uint32_t now_ms)
{
    const size_t next_idx = (size_t)(step_index_ + 1);
    if (next_idx >= sequence_len_)
        finish_(now_ms);
    else
        startStep_(next_idx, now_ms);
}

bool CharacterizationController::checkTimeout_(uint32_t now_ms)
{
    if (cfg_.max_hold_s == 0 || step_started_ms_ == 0 || step_timeout_flag_)
        return false;

    if (stepElapsedMs(now_ms) < (uint32_t)(cfg_.max_hold_s * 1000UL))
        return false;

    step_timeout_flag_ = true;
    step_stabilized_ = false;

    if (cfg_.continue_on_timeout)
    {
        state_ = State::STEP_COMPLETE;
        Serial.printf("[WARN] characterization timeout at step %d/%d\r\n", stepNumber(), totalSteps());
    }
    else
    {
        enterAborted_(now_ms, "step_timeout");
    }
    return true;
}

void CharacterizationController::updateFilteredTemp_()
{
    if (!ctrl_.sensorOk())
        return;

    // The characterization workflow uses a filtered temperature trace for
    // stabilization logic and telemetry. The PID itself stays on ctrl_.tRaw().
    const float raw = ctrl_.tRaw();
    if (!isfinite(raw))
        return;

    if (!filtered_valid_)
    {
        filtered_temp_ = raw;
        filtered_valid_ = true;
        return;
    }

    const float alpha = constrain(cfg_.ema_alpha, 0.001f, 1.0f);
    filtered_temp_ = filtered_temp_ + alpha * (raw - filtered_temp_);
}

void CharacterizationController::pushHistory_(uint32_t now_ms)
{
    if (!filtered_valid_)
        return;

    if (next_history_ms_ != 0 && now_ms < next_history_ms_)
        return;

    history_ts_ms_[history_head_] = now_ms;
    history_temp_[history_head_] = filtered_temp_;
    history_head_ = (history_head_ + 1) % HISTORY_CAPACITY;
    if (history_count_ < HISTORY_CAPACITY)
        history_count_++;

    next_history_ms_ = now_ms + 1000UL;
    updateSlope_();
}

void CharacterizationController::updateSlope_()
{
    slope_c_per_min_ = NAN;

    if (cfg_.slope_limit_c_per_min <= 0.0f || cfg_.slope_window_s == 0 || history_count_ < 2)
        return;

    const size_t newest_idx = (history_head_ + HISTORY_CAPACITY - 1) % HISTORY_CAPACITY;
    const uint32_t newest_ts = history_ts_ms_[newest_idx];
    const float newest_temp = history_temp_[newest_idx];
    const uint32_t window_ms = cfg_.slope_window_s * 1000UL;

    for (size_t i = 1; i < history_count_; ++i)
    {
        const size_t idx = (newest_idx + HISTORY_CAPACITY - i) % HISTORY_CAPACITY;
        const uint32_t ref_ts = history_ts_ms_[idx];
        if (newest_ts < ref_ts)
            continue;

        const uint32_t age_ms = newest_ts - ref_ts;
        if (age_ms < window_ms)
            continue;

        const float dt_min = age_ms / 60000.0f;
        if (dt_min <= 0.0f)
            return;

        slope_c_per_min_ = (newest_temp - history_temp_[idx]) / dt_min;
        return;
    }
}

bool CharacterizationController::isWithinBand_() const
{
    if (!filtered_valid_ || !isfinite(current_target_))
        return false;

    return fabsf(filtered_temp_ - current_target_) <= cfg_.tolerance_c;
}

bool CharacterizationController::slopeCriterionMet() const
{
    if (cfg_.slope_limit_c_per_min <= 0.0f || cfg_.slope_window_s == 0)
        return true;

    return isfinite(slope_c_per_min_) && fabsf(slope_c_per_min_) <= cfg_.slope_limit_c_per_min;
}

uint32_t CharacterizationController::elapsedSince_(uint32_t start_ms, uint32_t now_ms) const
{
    if (start_ms == 0 || now_ms < start_ms)
        return 0;
    return now_ms - start_ms;
}

void CharacterizationController::shiftTimersForPause_(uint32_t paused_ms)
{
    if (run_started_ms_ != 0)
        run_started_ms_ += paused_ms;
    if (step_started_ms_ != 0)
        step_started_ms_ += paused_ms;
    if (band_entered_ms_ != 0)
        band_entered_ms_ += paused_ms;
    if (dwell_started_ms_ != 0)
        dwell_started_ms_ += paused_ms;
}

uint32_t CharacterizationController::stepElapsedMs(uint32_t now_ms) const
{
    if (step_started_ms_ == 0)
        return 0;

    uint32_t ref_ms = now_ms;
    if (state_ == State::PAUSED && pause_started_ms_ != 0)
        ref_ms = pause_started_ms_;
    else if (!enabled() && terminal_ms_ != 0)
        ref_ms = terminal_ms_;

    return elapsedSince_(step_started_ms_, ref_ms);
}

uint32_t CharacterizationController::totalElapsedMs(uint32_t now_ms) const
{
    if (run_started_ms_ == 0)
        return 0;

    uint32_t ref_ms = now_ms;
    if (state_ == State::PAUSED && pause_started_ms_ != 0)
        ref_ms = pause_started_ms_;
    else if (!enabled() && terminal_ms_ != 0)
        ref_ms = terminal_ms_;

    return elapsedSince_(run_started_ms_, ref_ms);
}

void CharacterizationController::update(uint32_t now_ms, float dt_s)
{
    (void)dt_s;

    updateFilteredTemp_();
    pushHistory_(now_ms);

    if (ctrl_.sensorInhibit() && enabled())
    {
        char reason[48];
        if (ctrl_.lastFault() != 0)
            snprintf(reason, sizeof(reason), "%s_0x%02X", ctrl_.faultReason(), ctrl_.lastFault());
        else
            snprintf(reason, sizeof(reason), "%s", ctrl_.faultReason());
        enterFault_(now_ms, reason);
        return;
    }

    switch (state_)
    {
    case State::IDLE:
    case State::FINISHED:
    case State::ABORTED:
    case State::FAULT:
    case State::PAUSED:
        return;

    case State::STARTING:
        if (!ctrl_.armed() && !ctrl_.cmdArm())
        {
            enterFault_(now_ms, "arm_failed");
            return;
        }
        ctrl_.cmdPidEnable(true);
        ctrl_.cmdSetpoint(current_target_);
        state_ = State::MOVING_TO_SETPOINT;
        return;

    case State::STEP_COMPLETE:
        advanceStep_(now_ms);
        return;

    case State::MOVING_TO_SETPOINT:
        if (checkTimeout_(now_ms))
            return;

        in_band_ = isWithinBand_();
        if (in_band_)
        {
            band_entered_ms_ = now_ms;
            state_ = State::STABILIZING;
        }
        return;

    case State::STABILIZING:
        if (checkTimeout_(now_ms))
            return;

        if (!isWithinBand_())
        {
            in_band_ = false;
            band_entered_ms_ = 0;
            state_ = State::MOVING_TO_SETPOINT;
            return;
        }

        in_band_ = true;
        if (elapsedSince_(band_entered_ms_, now_ms) >= (uint32_t)(cfg_.stabilization_dwell_s * 1000UL) &&
            slopeCriterionMet())
        {
            step_stabilized_ = true;
            dwell_started_ms_ = now_ms;
            state_ = State::DWELLING;
        }
        return;

    case State::DWELLING:
        if (checkTimeout_(now_ms))
            return;

        if (!isWithinBand_())
        {
            in_band_ = false;
            step_stabilized_ = false;
            band_entered_ms_ = 0;
            dwell_started_ms_ = 0;
            state_ = State::MOVING_TO_SETPOINT;
            return;
        }

        in_band_ = true;
        if (cfg_.dwell_after_stable_s == 0 ||
            elapsedSince_(dwell_started_ms_, now_ms) >= (uint32_t)(cfg_.dwell_after_stable_s * 1000UL))
        {
            state_ = State::STEP_COMPLETE;
        }
        return;
    }
}

void CharacterizationController::printHelp() const
{
    Serial.println("Characterization commands:");
    Serial.println("  char start                     -> arm, enable PID, and run the profile");
    Serial.println("  char pause                     -> keep PID on, freeze progression");
    Serial.println("  char resume                    -> resume the paused sequence");
    Serial.println("  char stop                      -> safe stop (PID off, power=0, disarm)");
    Serial.println("  char abort                     -> immediate safe abort");
    Serial.println("  char status                    -> print characterization state");
    Serial.println("  char config                    -> print configuration and sequence");
    Serial.println("  char seq 5,0,-5,-10,-15,-18    -> set the setpoint sequence");
    Serial.println("  char tol 0.5                   -> set the stabilization band in degC");
    Serial.println("  char dwell 300                 -> time inside band before stabilized");
    Serial.println("  char postdwell 300             -> dwell time after stabilization");
    Serial.println("  char maxhold 900               -> maximum allowed step duration");
    Serial.println("  char continue_timeout 0|1      -> continue or abort on timeout");
    Serial.println("  char slope 0.05                -> slope limit in degC/min (0 disables)");
    Serial.println("  char slope_window 60           -> slope averaging window in seconds");
}

void CharacterizationController::printStatus(uint32_t now_ms) const
{
    Serial.printf(
        "[CHAR] active=%d | state=%s | step=%d/%d | target=%.2f | filtered=%.3f | step_elapsed_s=%.1f | total_elapsed_s=%.1f | stabilized=%d | in_band=%d | timeout=%d | slope=%.4f | slope_ok=%d | fault=%d | fault_msg=%s\r\n",
        enabled() ? 1 : 0,
        stateName(),
        stepNumber(),
        totalSteps(),
        current_target_,
        filtered_valid_ ? filtered_temp_ : NAN,
        stepElapsedMs(now_ms) / 1000.0f,
        totalElapsedMs(now_ms) / 1000.0f,
        step_stabilized_ ? 1 : 0,
        in_band_ ? 1 : 0,
        step_timeout_flag_ ? 1 : 0,
        slope_c_per_min_,
        slopeCriterionMet() ? 1 : 0,
        faultActive() ? 1 : 0,
        fault_message_);
}

void CharacterizationController::printConfig() const
{
    Serial.printf(
        "[CHARCFG] tol=%.3f | dwell_s=%lu | postdwell_s=%lu | maxhold_s=%lu | continue_timeout=%d | slope=%.4f | slope_window_s=%lu\r\n",
        cfg_.tolerance_c,
        (unsigned long)cfg_.stabilization_dwell_s,
        (unsigned long)cfg_.dwell_after_stable_s,
        (unsigned long)cfg_.max_hold_s,
        cfg_.continue_on_timeout ? 1 : 0,
        cfg_.slope_limit_c_per_min,
        (unsigned long)cfg_.slope_window_s);

    Serial.print("[CHARSEQ] ");
    for (size_t i = 0; i < sequence_len_; ++i)
    {
        if (i)
            Serial.print(',');
        Serial.print(sequence_[i], 2);
    }
    Serial.println();
}
