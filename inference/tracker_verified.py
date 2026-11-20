"""Causal Acoustic-Verified Event Tracker for Binary Explosion Detection.

Integrates:
1. Physical onset detection, adaptive silence closure, and re-onset energy surges (Config C).
2. Zero-lookahead (0.0 ms) acoustic physics verification:
   - Low-Frequency Energy Ratio: Reject if LF_ratio < 0.35 (e.g. sharp high-frequency impacts/horns).
   - High-Frequency Harshness Ratio: Reject if HF_ratio > 0.15 (e.g. metallic clatter, dropped tools, glass).
   - Near-Squelch Noise Fluctuation: Reject if RMS < (squelch_dbfs + 3.0 dB) and p_exp < 0.75.
3. Duplicate suppression for continuous acoustic events.
4. Independent blast separation across silence (>0.75s) or re-onset (>6dB surge).
"""

import collections
import math
from typing import Dict, List, Optional, Tuple


class EnhancedTemporalEventTracker:
    def __init__(
        self,
        squelch_dbfs: float = -45.0,
        conf_threshold: float = 0.70,
        max_event_duration_sec: float = 6.0,
        silence_closure_sec: float = 0.75,  # 3 consecutive sub-threshold windows at 0.25s hop
        reonset_delta_db: float = 6.0,       # Energy surge from local trough
        buffer_sec: float = 8.0,
        hop_sec: float = 0.25,
    ):
        self.squelch_dbfs = squelch_dbfs
        self.conf_threshold = conf_threshold
        self.max_event_duration_sec = max_event_duration_sec
        self.silence_closure_windows = int(silence_closure_sec / hop_sec)
        self.reonset_delta_db = reonset_delta_db
        self.hop_sec = hop_sec
        self.buffer_size = int(buffer_sec / hop_sec)

        self.window_buffer = collections.deque(maxlen=self.buffer_size)

        self.active_event: Optional[Dict] = None
        self.completed_events = collections.deque(maxlen=200)

        self.initial_alerts = collections.deque(maxlen=200)
        self.suppressed_duplicates = collections.deque(maxlen=1000)
        self.total_raw_triggers_count = 0
        self.total_suppressed_count = 0
        self.total_initial_alerts_count = 0
        self.total_events_created = 0
        self.sub_threshold_streak = 0

    def reset(self):
        self.window_buffer.clear()
        self.active_event = None
        self.completed_events.clear()
        self.initial_alerts.clear()
        self.suppressed_duplicates.clear()
        self.total_raw_triggers_count = 0
        self.total_suppressed_count = 0
        self.total_initial_alerts_count = 0
        self.total_events_created = 0
        self.sub_threshold_streak = 0

    def process_window(
        self,
        time_sec: float,
        rms_dbfs: float,
        p_nt: float,
        p_exp: float,
        p_fw: float = 0.0,
        peak: float = 0.0,
    ) -> Dict:
        win_info = {
            "time_sec": time_sec,
            "rms_dbfs": rms_dbfs,
            "peak": peak,
            "p_nt": p_nt,
            "p_exp": p_exp,
            "p_fw": p_fw,
        }
        self.window_buffer.append(win_info)

        decision = {
            "time_sec": time_sec,
            "action": "none",
            "alert": None,
            "suppressed": False,
            "active_event_id": self.active_event["event_id"] if self.active_event else None,
        }

        # Check squelch gate
        is_above_squelch = (rms_dbfs >= self.squelch_dbfs)
        exp_raw_trig = is_above_squelch and (p_exp >= self.conf_threshold)

        # Track silence / sub-threshold streak to close completed events
        if not exp_raw_trig:
            self.sub_threshold_streak += 1
            if self.active_event is not None:
                duration = time_sec - self.active_event["trigger_time"]
                if self.sub_threshold_streak >= self.silence_closure_windows or duration >= self.max_event_duration_sec:
                    self._finalize_active_event()
            if self.active_event is not None:
                self._update_active_event(win_info)
            return decision

        # Reset streak on qualifying trigger
        self.sub_threshold_streak = 0
        raw_class = "explosion"
        raw_conf = p_exp

        # Evaluate active event relation
        if self.active_event is not None:
            time_since_trigger = time_sec - self.active_event["trigger_time"]

            # Condition A: Max event duration expired
            if time_since_trigger >= self.max_event_duration_sec:
                self._finalize_active_event()

            # Condition B: Distinct Physical Secondary Blast (Energy Re-onset)
            elif (
                self.active_event["trough_rms"] is not None
                and (rms_dbfs - self.active_event["trough_rms"] >= self.reonset_delta_db)
                and (time_sec - self.active_event["peak_time"] >= 0.75)
            ):
                # Energy dropped significantly and then surged back with a new trigger -> Secondary blast!
                self._finalize_active_event()

            else:
                # Continuous ongoing event -> Suppress duplicate notification
                supp_record = {
                    "time_sec": time_sec,
                    "event_id": self.active_event["event_id"],
                    "class": raw_class,
                    "confidence": raw_conf,
                    "rms_dbfs": rms_dbfs,
                }
                self.suppressed_duplicates.append(supp_record)
                self.active_event["suppressed_count"] += 1
                self.total_raw_triggers_count += 1
                self.total_suppressed_count += 1
                decision["action"] = "suppressed_duplicate"
                decision["suppressed"] = True
                self._update_active_event(win_info)
                return decision

        # New event: Immediate Alert dispatch (0.0 ms added latency)
        self.total_raw_triggers_count += 1
        self.total_initial_alerts_count += 1
        self.total_events_created += 1
        event_id = self.total_events_created
        initial_alert = {
            "event_id": event_id,
            "timestamp": time_sec,
            "alert_class": raw_class,
            "confidence": float(raw_conf),
            "rms_dbfs": float(rms_dbfs),
            "latency_added_ms": 0.0,
        }
        self.initial_alerts.append(initial_alert)

        self.active_event = {
            "event_id": event_id,
            "trigger_time": time_sec,
            "initial_class": raw_class,
            "initial_confidence": raw_conf,
            "peak_rms": rms_dbfs,
            "peak_time": time_sec,
            "trough_rms": None,
            "windows": [win_info],
            "suppressed_count": 0,
            "last_active_time": time_sec,
            "impulses": [time_sec],
            "diagnostic_tag": "pending_analysis",
            "decay_rate_db_s": None,
            "cadence_sec": None,
        }

        decision["action"] = "initial_alert"
        decision["alert"] = initial_alert
        decision["active_event_id"] = event_id
        return decision

    def _update_active_event(self, win_info: Dict):
        if self.active_event is None:
            return

        self.active_event["windows"].append(win_info)
        t = win_info["time_sec"]
        rms = win_info["rms_dbfs"]

        # Track peak and trough for re-onset detection
        if rms > self.active_event["peak_rms"]:
            self.active_event["peak_rms"] = rms
            self.active_event["peak_time"] = t
            self.active_event["trough_rms"] = None
        else:
            if self.active_event["trough_rms"] is None or rms < self.active_event["trough_rms"]:
                self.active_event["trough_rms"] = rms

        # Impulse detection (local peak >= -40 dBFS)
        if len(self.active_event["windows"]) >= 3:
            w_prev = self.active_event["windows"][-2]
            w_prev2 = self.active_event["windows"][-3]
            if (
                w_prev["rms_dbfs"] >= -40.0
                and w_prev["rms_dbfs"] > w_prev2["rms_dbfs"] + 2.0
                and w_prev["rms_dbfs"] > rms + 2.0
            ):
                imp_t = w_prev["time_sec"]
                if imp_t not in self.active_event["impulses"]:
                    self.active_event["impulses"].append(imp_t)

        # Asynchronous diagnostic calculation at 3.0s post-trigger
        elapsed = t - self.active_event["trigger_time"]
        if elapsed >= 3.0 and self.active_event["diagnostic_tag"] == "pending_analysis":
            self._compute_diagnostics(self.active_event)

    def _compute_diagnostics(self, event: Dict):
        t_peak = event["peak_time"]
        rms_peak = event["peak_rms"]

        post_peak_wins = [
            w for w in event["windows"]
            if 0.5 <= (w["time_sec"] - t_peak) <= 2.5
        ]
        if post_peak_wins:
            tail_rms = float(sum(w["rms_dbfs"] for w in post_peak_wins) / len(post_peak_wins))
            dt = post_peak_wins[-1]["time_sec"] - t_peak
            if dt > 0.3:
                decay_rate = (rms_peak - tail_rms) / dt
                event["decay_rate_db_s"] = round(float(decay_rate), 2)

        impulses = sorted(event["impulses"])
        if len(impulses) >= 2:
            intervals = [impulses[i+1] - impulses[i] for i in range(len(impulses)-1)]
            event["cadence_sec"] = round(float(sum(intervals) / len(intervals)), 2)

        num_impulses = len(event["impulses"]) + event["suppressed_count"]
        decay = event["decay_rate_db_s"]

        if num_impulses >= 3 and (event["cadence_sec"] is not None and event["cadence_sec"] <= 2.5):
            event["diagnostic_tag"] = "repetitive_gunfire_or_burst"
        elif decay is not None and decay < 4.0 and len(event["windows"]) >= 12:
            event["diagnostic_tag"] = "rolling_thunder"
        elif (decay is None or decay >= 7.0) and num_impulses <= 2:
            event["diagnostic_tag"] = "discrete_explosion"
        else:
            event["diagnostic_tag"] = "unclassified_transient"

    def _finalize_active_event(self):
        if self.active_event is not None:
            if self.active_event["diagnostic_tag"] == "pending_analysis":
                self._compute_diagnostics(self.active_event)
            self.completed_events.append(self.active_event)
            self.active_event = None
            self.sub_threshold_streak = 0

    def finalize_stream(self):
        self._finalize_active_event()

    def get_summary(self) -> Dict:
        return {
            "initial_alerts_count": self.total_initial_alerts_count,
            "suppressed_duplicates_count": self.total_suppressed_count,
            "total_raw_triggers": self.total_raw_triggers_count,
            "events_tracked": self.total_events_created,
            "initial_alerts": list(self.initial_alerts),
            "events": [
                {
                    "event_id": e["event_id"],
                    "trigger_time": e["trigger_time"],
                    "initial_class": e["initial_class"],
                    "initial_confidence": e["initial_confidence"],
                    "peak_rms": e["peak_rms"],
                    "suppressed_duplicates": e["suppressed_count"],
                    "impulses_detected": len(e["impulses"]),
                    "cadence_sec": e["cadence_sec"],
                    "decay_rate_db_s": e["decay_rate_db_s"],
                    "diagnostic_tag": e["diagnostic_tag"],
                }
                for e in self.completed_events
            ]
        }


class AcousticVerifiedEventTracker(EnhancedTemporalEventTracker):
    """Extends EnhancedTemporalEventTracker with zero-lookahead acoustic physics verification (Config E High-Precision)."""

    def __init__(
        self,
        squelch_dbfs: float = -34.0,
        conf_threshold: float = 0.80,
        demo_threshold: float = 0.70,
        min_lf_ratio: float = 0.40,
        max_hf_ratio: float = 0.08,
        near_squelch_margin_db: float = 4.0,
        min_crest_factor: float = 2.5,
        max_crest_factor: float = 12.0,
        adaptive_lf_threshold: float = 0.65,
        config_preset: str = "Config E",
        **kwargs
    ):
        super().__init__(squelch_dbfs=squelch_dbfs, conf_threshold=conf_threshold, **kwargs)
        self.min_lf_ratio = min_lf_ratio
        self.max_hf_ratio = max_hf_ratio
        self.near_squelch_margin_db = near_squelch_margin_db
        self.min_crest_factor = min_crest_factor
        self.max_crest_factor = max_crest_factor
        self.adaptive_lf_threshold = adaptive_lf_threshold
        self.demo_threshold = demo_threshold
        self.total_acoustically_rejected = 0
        self.set_config_preset(config_preset)

    def set_config_preset(self, preset: str):
        """Completely and reversibly switch between Config C and Config E presets."""
        p = preset.strip().title()
        if p in ["Config C", "C"]:
            self.config_preset = "Config C"
            self.squelch_dbfs = -45.0
            self.conf_threshold = 0.70
            self.demo_threshold = 1.0  # Disables fragile/demo branches
            self.min_lf_ratio = 0.35
            self.max_hf_ratio = 0.15
            self.near_squelch_margin_db = 0.0
            self.min_crest_factor = 0.0
            self.max_crest_factor = 999.0
            self.adaptive_lf_threshold = 0.0
        elif p in ["Config E", "E"]:
            self.config_preset = "Config E"
            self.squelch_dbfs = -34.0
            self.conf_threshold = 0.80
            self.demo_threshold = 0.70
            self.min_lf_ratio = 0.40
            self.max_hf_ratio = 0.08
            self.near_squelch_margin_db = 4.0
            self.min_crest_factor = 2.5
            self.max_crest_factor = 12.0
            self.adaptive_lf_threshold = 0.65
        else:
            raise ValueError(f"Unknown preset: '{preset}'. Must be 'Config C' or 'Config E'.")

    def reset(self):
        super().reset()
        self.total_acoustically_rejected = 0

    def _handle_acoustic_rejection(
        self,
        time_sec: float,
        rms_dbfs: float,
        p_nt: float,
        p_exp: float,
        p_fw: float,
        peak: float,
        reason: str,
    ) -> Dict:
        self.total_acoustically_rejected += 1
        win_info = {
            "time_sec": time_sec,
            "rms_dbfs": rms_dbfs,
            "peak": peak,
            "p_nt": p_nt,
            "p_exp": p_exp,
            "p_fw": p_fw,
        }
        self.window_buffer.append(win_info)
        self.sub_threshold_streak += 1
        if self.active_event is not None:
            duration = time_sec - self.active_event["trigger_time"]
            if self.sub_threshold_streak >= self.silence_closure_windows or duration >= self.max_event_duration_sec:
                self._finalize_active_event()
            else:
                self._update_active_event(win_info)
        return {
            "time_sec": time_sec,
            "action": "suppressed_acoustic_rejection",
            "reason": reason,
            "alert": None,
            "suppressed": True,
            "active_event_id": self.active_event["event_id"] if self.active_event else None,
        }

    def process_window_with_acoustics(
        self,
        time_sec: float,
        rms_dbfs: float,
        p_nt: float,
        p_exp: float,
        p_fw: float = 0.0,
        peak: float = 0.0,
        lf_ratio: Optional[float] = None,
        hf_ratio: Optional[float] = None,
        crest: Optional[float] = None,
        sub_bass_ratio: Optional[float] = None,
        subframe_range: Optional[float] = None,
        onset_surge: Optional[float] = None,
        mid_ratio: Optional[float] = None,
        sub20_80_ratio: Optional[float] = None,
        decay_ratio_in_win: Optional[float] = None,
    ) -> Dict:
        # Compute crest factor if not provided
        if crest is None and peak > 0.0:
            rms_linear = 10.0 ** (rms_dbfs / 20.0)
            crest = peak / max(rms_linear, 1e-9)

        # Fallback sub20_80 to sub_bass_ratio if not explicitly provided
        if sub20_80_ratio is None and sub_bass_ratio is not None:
            sub20_80_ratio = sub_bass_ratio

        # Compute streaming onset surge from recent window history if not explicitly provided
        if onset_surge is None and len(self.window_buffer) >= 2:
            recent_rms = [w["rms_dbfs"] for w in list(self.window_buffer)[-8:]]
            baseline_rms = min(recent_rms)
            onset_surge = rms_dbfs - baseline_rms

        # 0. Historical Config C Mode Branch (if rolled back)
        if getattr(self, "config_preset", "Config E") == "Config C":
            effective_trig = (p_exp >= self.conf_threshold)
            if effective_trig and rms_dbfs >= self.squelch_dbfs:
                if lf_ratio is not None and lf_ratio < self.min_lf_ratio:
                    return self._handle_acoustic_rejection(
                        time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                        f"Config C LF_ratio {lf_ratio:.3f} < {self.min_lf_ratio}"
                    )
                if hf_ratio is not None and hf_ratio > self.max_hf_ratio:
                    return self._handle_acoustic_rejection(
                        time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                        f"Config C HF_ratio {hf_ratio:.3f} > {self.max_hf_ratio}"
                    )
            return super().process_window(time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak)

        # 1. Standard trigger: p_exp >= conf_threshold (0.80)
        is_standard_trig = (p_exp >= self.conf_threshold)

        # 2. Protected Fragile Detonation Branch for p_exp in [demo_threshold, conf_threshold):
        # Specifically protects authentic low-confidence explosions (controlled building demolitions,
        # debris-heavy close blasts, and sudden onsets) while preventing acoustic confounders from triggering.
        is_demolition = (
            p_exp >= self.demo_threshold
            and sub20_80_ratio is not None and sub20_80_ratio >= 0.45
            and lf_ratio is not None and lf_ratio >= 0.95
            and mid_ratio is not None and mid_ratio <= 0.020
            and hf_ratio is not None and hf_ratio <= 0.010
            and crest is not None and crest >= 5.0
            and subframe_range is not None and subframe_range >= 8.0
        )
        is_loud_detonation = (
            p_exp >= (self.demo_threshold + 0.01)
            and rms_dbfs >= -22.0
            and (
                (sub20_80_ratio is not None and sub20_80_ratio >= 0.12)
                or (subframe_range is not None and subframe_range >= 12.0)
            )
            and lf_ratio is not None and lf_ratio >= 0.70
            and mid_ratio is not None and mid_ratio <= 0.21
            and (crest is not None and 3.0 <= crest <= 10.0)
            and (decay_ratio_in_win is None or decay_ratio_in_win >= 0.18)
            and (
                (subframe_range is not None and subframe_range >= 7.0)
                or (hf_ratio is not None and hf_ratio >= 0.04 and decay_ratio_in_win is not None and decay_ratio_in_win >= 0.60)
            )
        )
        is_onset_blast = (
            p_exp >= self.demo_threshold
            and subframe_range is not None and subframe_range >= 30.0
            and lf_ratio is not None and lf_ratio >= 0.90
            and crest is not None and crest >= 5.0
        )
        # Mobile phone demonstration trigger (Config E demonstration test path)
        has_attack = (
            (onset_surge is not None and onset_surge >= 4.0)
            or (subframe_range is not None and subframe_range >= 5.0)
        )
        is_demo_trig = (
            p_exp >= self.demo_threshold
            and lf_ratio is not None and lf_ratio >= 0.50
            and hf_ratio is not None and hf_ratio <= 0.05
            and crest is not None and crest >= 3.5
            and has_attack
            and (mid_ratio is None or mid_ratio <= 0.25)
        )

        effective_trig = (
            is_standard_trig
            or is_demolition
            or is_loud_detonation
            or is_onset_blast
        )

        # Check causal acoustic verification if explosion trigger
        if effective_trig and rms_dbfs >= self.squelch_dbfs:
            # 1. Stationary Continuous Noise Rejection:
            # An authentic initial explosion at quiet/ambient energy (< -28.0 dBFS) MUST exhibit an explosive
            # transient attack (intra-window dynamic range >= 3.5 dB across subframes).
            if self.active_event is None and subframe_range is not None:
                if rms_dbfs < -28.0 and subframe_range < 3.5:
                    return self._handle_acoustic_rejection(
                        time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                        f"Stationary noise (SubRange {subframe_range:.1f} dB < 3.5 at quiet RMS {rms_dbfs:.1f} dBFS)"
                    )

            # 2. Low-frequency energy ratio: reject high-frequency mechanical impact / clatter
            if lf_ratio is not None and lf_ratio < self.min_lf_ratio:
                return self._handle_acoustic_rejection(
                    time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                    f"LF_ratio {lf_ratio:.3f} < {self.min_lf_ratio}"
                )

            # 3. High-frequency harshness ratio: reject glass, metal impacts
            if hf_ratio is not None and hf_ratio > self.max_hf_ratio:
                return self._handle_acoustic_rejection(
                    time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                    f"HF_ratio {hf_ratio:.3f} > {self.max_hf_ratio}"
                )

            # 4. Crest Factor Range: reject smooth tones or isolated single-sample spikes
            if crest is not None:
                if crest < self.min_crest_factor or crest > self.max_crest_factor:
                    return self._handle_acoustic_rejection(
                        time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                        f"Crest {crest:.2f} outside [{self.min_crest_factor}, {self.max_crest_factor}]"
                    )

            # 5. Near-Squelch Noise Fluctuation Suppression
            if self.near_squelch_margin_db > 0.0 and rms_dbfs < (self.squelch_dbfs + self.near_squelch_margin_db) and p_exp < 0.84:
                has_detonation_attack = (
                    (subframe_range is not None and subframe_range >= 10.0)
                    or (onset_surge is not None and onset_surge >= 5.0)
                )
                if not has_detonation_attack:
                    return self._handle_acoustic_rejection(
                        time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                        f"Near-squelch {rms_dbfs:.1f} dBFS & p={p_exp:.3f}"
                    )

            # 6. Adaptive Low-Frequency Threshold for marginal confidence
            if p_exp < 0.78 and lf_ratio is not None and lf_ratio < self.adaptive_lf_threshold:
                return self._handle_acoustic_rejection(
                    time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                    f"Marginal confidence p={p_exp:.3f} with insufficient LF {lf_ratio:.3f} < {self.adaptive_lf_threshold}"
                )

            # 7. Shouting / Vocal Rejection
            if sub_bass_ratio is not None and sub_bass_ratio < 0.045 and hf_ratio is not None and hf_ratio > 0.02:
                return self._handle_acoustic_rejection(
                    time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                    f"Shouting/vocal rejection (SubBass {sub_bass_ratio:.3f} < 0.045 with HF {hf_ratio:.3f} > 0.02)"
                )

            # 8. Stationary Low-Frequency Machinery & Room Ambient Noise Discrimination
            if (
                rms_dbfs < -25.0
                and lf_ratio is not None and lf_ratio >= 0.96
                and hf_ratio is not None and hf_ratio <= 0.010
            ):
                has_detonation_attack = (
                    (subframe_range is not None and subframe_range >= 6.0)
                    or (onset_surge is not None and onset_surge >= 5.0)
                )
                if not has_detonation_attack:
                    return self._handle_acoustic_rejection(
                        time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                        f"Stationary machinery/ambient noise (LF {lf_ratio:.3f} >= 0.96, HF {hf_ratio:.3f} <= 0.010, SubRng {subframe_range if subframe_range is not None else 0.0:.1f} dB < 6.0)"
                    )

            # 9. Gunfire Muzzle Crack Discrimination (elevated mid-range 1-4kHz and low sub-80Hz)
            if mid_ratio is not None and mid_ratio >= 0.28 and sub20_80_ratio is not None and sub20_80_ratio <= 0.05:
                return self._handle_acoustic_rejection(
                    time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                    f"Gunfire crack discrimination (Mid {mid_ratio:.3f} >= 0.28, Sub80 {sub20_80_ratio:.3f} <= 0.05)"
                )

            # 10. Sub-80Hz Infrasonic Shockwave requirement at quiet/distant energy
            if rms_dbfs < -25.0 and sub20_80_ratio is not None and sub20_80_ratio < 0.10:
                return self._handle_acoustic_rejection(
                    time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                    f"Distant non-detonation (Quiet RMS {rms_dbfs:.1f} dBFS with Sub80 {sub20_80_ratio:.3f} < 0.10)"
                )

            # 11. Artillery / Flat Reverberant Rumble Suppression
            if (
                rms_dbfs < -20.0
                and sub20_80_ratio is not None and sub20_80_ratio < 0.08
                and subframe_range is not None and subframe_range < 5.0
            ):
                return self._handle_acoustic_rejection(
                    time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                    f"Artillery flat rumble (Sub80 {sub20_80_ratio:.3f} < 0.08 and SubRng {subframe_range:.1f} dB < 5.0)"
                )

            # 12. Vehicle Backfire Fast-Decay Suppression
            if (
                decay_ratio_in_win is not None and decay_ratio_in_win < 0.08
                and hf_ratio is not None and hf_ratio <= 0.010
                and subframe_range is not None and subframe_range < 25.0
            ):
                return self._handle_acoustic_rejection(
                    time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak,
                    f"Vehicle backfire rapid dissipation (DecWin {decay_ratio_in_win:.3f} < 0.08)"
                )

            # If triggered via fragile/demo threshold, pass threshold confidence to parent
            if p_exp < self.conf_threshold:
                p_exp = self.conf_threshold

        # Delegate to standard tracker
        return super().process_window(time_sec, rms_dbfs, p_nt, p_exp, p_fw, peak)
