use std::time::{Duration, Instant};

use super::metrics::PrometheusValues;

/// How long to keep the last active phase / last non-zero live speed when a
/// poll lands between counter updates while a request is still running.
pub const PHASE_HOLD_TTL: Duration = Duration::from_millis(2500);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InferencePhase {
    #[default]
    Idle,
    Prefill,
    Generating,
}

#[derive(Debug, Clone, Copy)]
pub struct ThroughputSample {
    pub prompt_tokens_total: u64,
    pub predicted_tokens_total: u64,
    pub requests_processing: u32,
    pub prompt_gauge: f64,
    pub predicted_gauge: f64,
}

impl From<&PrometheusValues> for ThroughputSample {
    fn from(prom: &PrometheusValues) -> Self {
        Self {
            prompt_tokens_total: prom.prompt_tokens_total,
            predicted_tokens_total: prom.predicted_tokens_total,
            requests_processing: prom.requests_processing,
            prompt_gauge: prom.prompt_tokens_per_sec,
            predicted_gauge: prom.predicted_tokens_per_sec,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiveThroughputView {
    pub phase: InferencePhase,
    pub prompt_tokens_per_sec: f64,
    pub generation_tokens_per_sec: f64,
}

impl LiveThroughputView {
    fn idle() -> Self {
        Self {
            phase: InferencePhase::Idle,
            prompt_tokens_per_sec: 0.0,
            generation_tokens_per_sec: 0.0,
        }
    }
}

/// Runtime-only tracker. Never persisted.
#[derive(Debug, Clone, Default)]
pub struct LiveThroughputTracker {
    primed: bool,
    last_prompt_total: u64,
    last_predicted_total: u64,
    last_at: Option<Instant>,
    phase: InferencePhase,
    last_prompt_speed: f64,
    last_prompt_speed_at: Option<Instant>,
    last_gen_speed: f64,
    last_gen_speed_at: Option<Instant>,
}

impl LiveThroughputTracker {
    pub fn observe(&mut self, sample: &ThroughputSample, now: Instant) -> LiveThroughputView {
        let prompt_gauge = finite_positive(sample.prompt_gauge);
        let gen_gauge = finite_positive(sample.predicted_gauge);

        if !self.primed {
            self.prime(
                sample.prompt_tokens_total,
                sample.predicted_tokens_total,
                now,
            );
            return self.view_without_delta(
                sample.requests_processing,
                prompt_gauge,
                gen_gauge,
                now,
            );
        }

        let reset = sample.prompt_tokens_total < self.last_prompt_total
            || sample.predicted_tokens_total < self.last_predicted_total;
        if reset {
            self.prime(
                sample.prompt_tokens_total,
                sample.predicted_tokens_total,
                now,
            );
            self.clear_held_speeds();
            self.set_phase(InferencePhase::Idle, now);
            return self.view_without_delta(
                sample.requests_processing,
                prompt_gauge,
                gen_gauge,
                now,
            );
        }

        let elapsed = self
            .last_at
            .map(|t| now.saturating_duration_since(t))
            .unwrap_or_default();
        let prompt_delta = sample.prompt_tokens_total - self.last_prompt_total;
        let pred_delta = sample.predicted_tokens_total - self.last_predicted_total;
        self.last_prompt_total = sample.prompt_tokens_total;
        self.last_predicted_total = sample.predicted_tokens_total;
        self.last_at = Some(now);

        if sample.requests_processing == 0 {
            self.set_phase(InferencePhase::Idle, now);
            self.clear_held_speeds();
            return LiveThroughputView::idle();
        }

        if pred_delta > 0 {
            self.set_phase(InferencePhase::Generating, now);
        } else if prompt_delta > 0 {
            self.set_phase(InferencePhase::Prefill, now);
        } else if self.phase == InferencePhase::Idle {
            self.infer_phase_from_gauges(prompt_gauge, gen_gauge, now);
        }

        self.speeds_for_phase(
            prompt_gauge,
            gen_gauge,
            prompt_delta,
            pred_delta,
            elapsed,
            now,
        )
    }

    fn prime(&mut self, prompt_total: u64, predicted_total: u64, now: Instant) {
        self.primed = true;
        self.last_prompt_total = prompt_total;
        self.last_predicted_total = predicted_total;
        self.last_at = Some(now);
    }

    fn view_without_delta(
        &mut self,
        requests: u32,
        prompt_gauge: Option<f64>,
        gen_gauge: Option<f64>,
        now: Instant,
    ) -> LiveThroughputView {
        if requests == 0 {
            self.set_phase(InferencePhase::Idle, now);
            self.clear_held_speeds();
            return LiveThroughputView::idle();
        }
        if self.phase == InferencePhase::Idle {
            self.infer_phase_from_gauges(prompt_gauge, gen_gauge, now);
        }
        self.speeds_for_phase(prompt_gauge, gen_gauge, 0, 0, Duration::ZERO, now)
    }

    fn infer_phase_from_gauges(
        &mut self,
        prompt_gauge: Option<f64>,
        gen_gauge: Option<f64>,
        now: Instant,
    ) {
        if gen_gauge.is_some() {
            self.set_phase(InferencePhase::Generating, now);
        } else if prompt_gauge.is_some() {
            self.set_phase(InferencePhase::Prefill, now);
        }
    }

    fn set_phase(&mut self, phase: InferencePhase, _now: Instant) {
        if self.phase != phase {
            match phase {
                InferencePhase::Idle => self.clear_held_speeds(),
                InferencePhase::Prefill => {
                    self.last_gen_speed = 0.0;
                    self.last_gen_speed_at = None;
                }
                InferencePhase::Generating => {
                    self.last_prompt_speed = 0.0;
                    self.last_prompt_speed_at = None;
                }
            }
            self.phase = phase;
        }
    }

    fn speeds_for_phase(
        &mut self,
        prompt_gauge: Option<f64>,
        gen_gauge: Option<f64>,
        prompt_delta: u64,
        pred_delta: u64,
        elapsed: Duration,
        now: Instant,
    ) -> LiveThroughputView {
        let prompt = match self.phase {
            InferencePhase::Prefill => {
                self.resolve_speed(prompt_gauge, prompt_delta, elapsed, true, now)
            }
            _ => 0.0,
        };
        let generation = match self.phase {
            InferencePhase::Generating => {
                self.resolve_speed(gen_gauge, pred_delta, elapsed, false, now)
            }
            _ => 0.0,
        };
        LiveThroughputView {
            phase: self.phase,
            prompt_tokens_per_sec: prompt,
            generation_tokens_per_sec: generation,
        }
    }

    fn resolve_speed(
        &mut self,
        gauge: Option<f64>,
        delta: u64,
        elapsed: Duration,
        prompt: bool,
        now: Instant,
    ) -> f64 {
        if let Some(g) = gauge {
            self.store_speed(prompt, g, now);
            return g;
        }
        if delta > 0 && elapsed > Duration::ZERO {
            let speed = delta as f64 / elapsed.as_secs_f64();
            self.store_speed(prompt, speed, now);
            return speed;
        }
        let (held, held_at) = if prompt {
            (self.last_prompt_speed, self.last_prompt_speed_at)
        } else {
            (self.last_gen_speed, self.last_gen_speed_at)
        };
        if held > 0.0
            && held_at.is_some_and(|at| now.saturating_duration_since(at) <= PHASE_HOLD_TTL)
        {
            return held;
        }
        0.0
    }

    fn store_speed(&mut self, prompt: bool, speed: f64, now: Instant) {
        if prompt {
            self.last_prompt_speed = speed;
            self.last_prompt_speed_at = Some(now);
        } else {
            self.last_gen_speed = speed;
            self.last_gen_speed_at = Some(now);
        }
    }

    fn clear_held_speeds(&mut self) {
        self.last_prompt_speed = 0.0;
        self.last_prompt_speed_at = None;
        self.last_gen_speed = 0.0;
        self.last_gen_speed_at = None;
    }
}

fn finite_positive(value: f64) -> Option<f64> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(
        prompt_total: u64,
        pred_total: u64,
        requests: u32,
        pg: f64,
        gg: f64,
    ) -> ThroughputSample {
        ThroughputSample {
            prompt_tokens_total: prompt_total,
            predicted_tokens_total: pred_total,
            requests_processing: requests,
            prompt_gauge: pg,
            predicted_gauge: gg,
        }
    }

    #[test]
    fn prompt_delta_sets_prefill() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(1000, 0, 1, 0.0, 0.0), t0);
        let view = t.observe(&sample(1500, 0, 1, 0.0, 0.0), t0 + Duration::from_secs(1));
        assert_eq!(view.phase, InferencePhase::Prefill);
        assert!((view.prompt_tokens_per_sec - 500.0).abs() < 1e-9);
        assert_eq!(view.generation_tokens_per_sec, 0.0);
    }

    #[test]
    fn generation_delta_sets_generating() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(1000, 200, 1, 0.0, 0.0), t0);
        let view = t.observe(&sample(1000, 250, 1, 0.0, 0.0), t0 + Duration::from_secs(1));
        assert_eq!(view.phase, InferencePhase::Generating);
        assert_eq!(view.prompt_tokens_per_sec, 0.0);
        assert!((view.generation_tokens_per_sec - 50.0).abs() < 1e-9);
    }

    #[test]
    fn busy_without_delta_keeps_phase_within_ttl() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(1000, 0, 1, 0.0, 0.0), t0);
        t.observe(&sample(1400, 0, 1, 0.0, 0.0), t0 + Duration::from_secs(1));
        let view = t.observe(
            &sample(1400, 0, 1, 0.0, 0.0),
            t0 + Duration::from_millis(2500),
        );
        assert_eq!(view.phase, InferencePhase::Prefill);
        assert!(view.prompt_tokens_per_sec > 0.0);
        assert_eq!(view.generation_tokens_per_sec, 0.0);
    }

    #[test]
    fn requests_zero_goes_idle_immediately() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(1000, 10, 1, 0.0, 0.0), t0);
        t.observe(&sample(1000, 60, 1, 0.0, 40.0), t0 + Duration::from_secs(1));
        let view = t.observe(
            &sample(1000, 60, 0, 40.0, 40.0),
            t0 + Duration::from_millis(1100),
        );
        assert_eq!(view.phase, InferencePhase::Idle);
        assert_eq!(view.prompt_tokens_per_sec, 0.0);
        assert_eq!(view.generation_tokens_per_sec, 0.0);
    }

    #[test]
    fn counter_reset_does_not_invent_speed() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(9000, 4000, 1, 0.0, 0.0), t0);
        t.observe(
            &sample(9500, 4100, 1, 0.0, 0.0),
            t0 + Duration::from_secs(1),
        );
        let view = t.observe(&sample(10, 5, 1, 0.0, 0.0), t0 + Duration::from_secs(2));
        assert_eq!(view.prompt_tokens_per_sec, 0.0);
        assert_eq!(view.generation_tokens_per_sec, 0.0);
        assert_ne!(view.phase, InferencePhase::Prefill);
    }

    #[test]
    fn gauge_zero_with_counter_delta_uses_fallback() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(100, 0, 1, 0.0, 0.0), t0);
        let view = t.observe(
            &sample(350, 0, 1, 0.0, 0.0),
            t0 + Duration::from_millis(500),
        );
        assert_eq!(view.phase, InferencePhase::Prefill);
        assert!((view.prompt_tokens_per_sec - 500.0).abs() < 1e-9);
    }

    #[test]
    fn native_gauge_wins_when_positive() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(100, 0, 1, 0.0, 0.0), t0);
        let view = t.observe(&sample(200, 0, 1, 77.5, 0.0), t0 + Duration::from_secs(1));
        assert_eq!(view.phase, InferencePhase::Prefill);
        assert!((view.prompt_tokens_per_sec - 77.5).abs() < 1e-9);
    }

    #[test]
    fn generating_does_not_show_stale_prompt_speed() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(100, 0, 1, 0.0, 0.0), t0);
        t.observe(&sample(600, 0, 1, 80.0, 0.0), t0 + Duration::from_secs(1));
        let view = t.observe(&sample(600, 40, 1, 80.0, 35.0), t0 + Duration::from_secs(2));
        assert_eq!(view.phase, InferencePhase::Generating);
        assert_eq!(view.prompt_tokens_per_sec, 0.0);
        assert!((view.generation_tokens_per_sec - 35.0).abs() < 1e-9);
    }

    #[test]
    fn prefill_does_not_show_stale_generation_speed() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(100, 50, 1, 0.0, 0.0), t0);
        t.observe(&sample(100, 90, 1, 0.0, 40.0), t0 + Duration::from_secs(1));
        let view = t.observe(&sample(400, 90, 1, 60.0, 40.0), t0 + Duration::from_secs(2));
        assert_eq!(view.phase, InferencePhase::Prefill);
        assert!((view.prompt_tokens_per_sec - 60.0).abs() < 1e-9);
        assert_eq!(view.generation_tokens_per_sec, 0.0);
    }

    #[test]
    fn never_uses_lifetime_average() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(10_000, 5_000, 0, 0.0, 0.0), t0);
        let view = t.observe(
            &sample(10_000, 5_000, 0, 0.0, 0.0),
            t0 + Duration::from_secs(1),
        );
        let lifetime_prompt = 10_000.0 / 8.1;
        let lifetime_gen = 5_000.0 / 88.2;
        assert_eq!(view.phase, InferencePhase::Idle);
        assert_eq!(view.prompt_tokens_per_sec, 0.0);
        assert_eq!(view.generation_tokens_per_sec, 0.0);
        assert!((view.prompt_tokens_per_sec - lifetime_prompt).abs() > 1.0);
        assert!((view.generation_tokens_per_sec - lifetime_gen).abs() > 1.0);
    }

    #[test]
    fn both_deltas_prefer_generating() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(100, 10, 1, 0.0, 0.0), t0);
        let view = t.observe(&sample(400, 40, 1, 0.0, 0.0), t0 + Duration::from_secs(1));
        assert_eq!(view.phase, InferencePhase::Generating);
        assert_eq!(view.prompt_tokens_per_sec, 0.0);
        assert!(view.generation_tokens_per_sec > 0.0);
    }

    #[test]
    fn speed_hold_expires_after_ttl_while_busy() {
        let t0 = Instant::now();
        let mut t = LiveThroughputTracker::default();
        t.observe(&sample(100, 0, 1, 0.0, 0.0), t0);
        t.observe(&sample(200, 0, 1, 90.0, 0.0), t0 + Duration::from_secs(1));
        let held = t.observe(
            &sample(200, 0, 1, 0.0, 0.0),
            t0 + Duration::from_millis(2000),
        );
        assert_eq!(held.phase, InferencePhase::Prefill);
        assert!((held.prompt_tokens_per_sec - 90.0).abs() < 1e-9);
        let expired = t.observe(
            &sample(200, 0, 1, 0.0, 0.0),
            t0 + Duration::from_millis(1_000 + 2500 + 1),
        );
        assert_eq!(expired.prompt_tokens_per_sec, 0.0);
    }
}
