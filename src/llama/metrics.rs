/// Live llama.cpp snapshot. Gauges are `None` when the source endpoint failed
/// this cycle (`0` is a valid idle reading and must not be replaced by averages).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LlamaMetrics {
    /// Instantaneous prompt t/s (`None` = unavailable).
    pub prompt_tokens_per_sec: Option<f64>,
    /// Instantaneous generation t/s (`None` = unavailable).
    pub generation_tokens_per_sec: Option<f64>,
    /// Runtime phase from llama.cpp log lines (`None` = no log evidence yet).
    pub inference_phase: Option<super::live_slots::InferencePhase>,
    /// Whether the prompt tile is a live prefill reading or a retained last value.
    pub prompt_speed_kind: Option<super::live_slots::SpeedKind>,
    /// Whether the generation tile is a live `tg_3s` reading or a retained last value.
    pub generation_speed_kind: Option<super::live_slots::SpeedKind>,
    /// Whole-generation average `tg` from the log (not the live gauge).
    pub generation_tokens_per_sec_avg: Option<f64>,
    /// llama-server session counters from the last successful `/metrics` scrape.
    pub prompt_tokens_total: u64,
    pub predicted_tokens_total: u64,
    pub kv_cache_tokens: Option<u64>,
    pub kv_cache_max: Option<u64>,
    /// Session high-water mark from `llamacpp:n_tokens_max` — not current KV used.
    pub n_tokens_max: Option<u64>,
    pub slots_idle: Option<u32>,
    pub slots_processing: Option<u32>,
    pub requests_processing: Option<u32>,
    pub spec_draft_tokens: Option<u64>,
    pub spec_accepted_tokens: Option<u64>,
    pub spec_drafts: Option<u64>,
    /// `accepted / draft`; `None` when spec metrics absent or draft_tokens == 0.
    pub spec_acceptance_ratio: Option<f64>,
    pub n_busy_slots_per_decode: Option<f64>,
    pub status: String,
    #[serde(skip)]
    live_slots: super::live_slots::LiveSlotTracker,
}

impl LlamaMetrics {
    /// Apply live gauges + session counters from a successful `/metrics` scrape.
    pub fn apply_metrics(&mut self, prom: &PrometheusValues) {
        self.apply_metrics_at(prom, std::time::Instant::now());
    }

    pub fn apply_metrics_at(&mut self, prom: &PrometheusValues, now: std::time::Instant) {
        self.prompt_tokens_total = prom.prompt_tokens_total;
        self.predicted_tokens_total = prom.predicted_tokens_total;
        self.requests_processing = Some(prom.requests_processing);
        self.n_tokens_max = prom.n_tokens_max;
        self.spec_draft_tokens = prom.spec_decode_num_draft_tokens_total;
        self.spec_accepted_tokens = prom.spec_decode_num_accepted_tokens_total;
        self.spec_drafts = prom.spec_decode_num_drafts_total;
        self.spec_acceptance_ratio = prom.speculative_acceptance_ratio();
        self.n_busy_slots_per_decode = prom.n_busy_slots_per_decode;
        // `/metrics` gauges are not authoritative for live UI speed.
        self.publish_live_from_logs(now);
    }

    pub fn apply_log_line(&mut self, line: &str) {
        self.apply_log_line_at(line, std::time::Instant::now());
    }

    pub fn apply_log_line_at(&mut self, line: &str, now: std::time::Instant) {
        if self.live_slots.apply_line(line, now) {
            self.publish_live_from_logs(now);
        }
    }

    /// Recompute live vs last from timestamps (stale `tg_3s` must not stay live).
    pub fn refresh_live(&mut self) {
        self.publish_live_from_logs(std::time::Instant::now());
    }

    fn publish_live_from_logs(&mut self, now: std::time::Instant) {
        if self.live_slots.is_empty() {
            return;
        }
        let view = self.live_slots.dashboard(now);
        self.prompt_tokens_per_sec = view.prompt_tokens_per_sec;
        self.generation_tokens_per_sec = view.generation_tokens_per_sec;
        self.generation_tokens_per_sec_avg = view.generation_tokens_per_sec_avg;
        self.inference_phase = Some(view.phase);
        self.prompt_speed_kind = view.prompt_speed_kind;
        self.generation_speed_kind = view.generation_speed_kind;
    }

    /// `/metrics` failed this cycle: occupancy gauges become unavailable.
    /// Log-derived live/last speeds are kept.
    pub fn clear_metrics_gauges(&mut self) {
        self.requests_processing = None;
    }

    /// Apply live KV/slot occupancy from a successful `/slots` scrape.
    pub fn apply_slots(&mut self, slots: &[serde_json::Value]) {
        let mut idle = 0u32;
        let mut processing = 0u32;
        for slot in slots {
            if slot
                .get("is_processing")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                processing += 1;
            } else {
                idle += 1;
            }
        }
        let (kv_used, kv_max) = slots_kv_usage(slots);
        self.slots_idle = Some(idle);
        self.slots_processing = Some(processing);
        self.kv_cache_tokens = Some(kv_used);
        self.kv_cache_max = Some(kv_max);
    }

    /// `/slots` failed this cycle: live KV/slot gauges become unavailable.
    pub fn clear_slots_gauges(&mut self) {
        self.kv_cache_tokens = None;
        self.kv_cache_max = None;
        self.slots_idle = None;
        self.slots_processing = None;
    }

    pub fn busy_requests(&self) -> u32 {
        self.requests_processing.unwrap_or(0)
    }

    pub fn busy_slots(&self) -> u32 {
        self.slots_processing.unwrap_or(0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct PrometheusValues {
    pub prompt_tokens_per_sec: f64,
    pub predicted_tokens_per_sec: f64,
    /// Session counters converted to u64 at parse time — never keep these as f64.
    pub prompt_tokens_total: u64,
    pub prompt_seconds_total: f64,
    pub predicted_tokens_total: u64,
    pub predicted_seconds_total: f64,
    pub requests_processing: u32,
    /// Present only when llama.cpp exports `llamacpp:prompt_tokens_cached_total`.
    pub prompt_tokens_cached_total: Option<u64>,
    pub n_tokens_max: Option<u64>,
    pub spec_decode_num_draft_tokens_total: Option<u64>,
    pub spec_decode_num_accepted_tokens_total: Option<u64>,
    pub spec_decode_num_drafts_total: Option<u64>,
    pub n_busy_slots_per_decode: Option<f64>,
}

impl PrometheusValues {
    /// Speculative/MTP acceptance: accepted draft tokens / drafted tokens.
    /// `None` when either counter is missing or there were no draft tokens.
    pub fn speculative_acceptance_ratio(&self) -> Option<f64> {
        let draft = self.spec_decode_num_draft_tokens_total?;
        let accepted = self.spec_decode_num_accepted_tokens_total?;
        if draft == 0 {
            return None;
        }
        Some(accepted as f64 / draft as f64)
    }
}

/// Convert a Prometheus number to a counter. Accepts scientific notation via f64.
/// Rejects NaN, infinities, negatives, and values that overflow u64.
pub fn prometheus_f64_to_u64(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let rounded = value.round();
    if rounded > u64::MAX as f64 {
        return None;
    }
    Some(rounded as u64)
}

/// Parse a Prometheus counter. Digit-only strings go through u64 so values
/// above the f64 mantissa (2^53) are not rounded. Scientific notation still
/// uses f64, then [`prometheus_f64_to_u64`].
pub fn parse_prometheus_counter(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if raw.as_bytes().iter().all(|b| b.is_ascii_digit()) {
        return raw.parse::<u64>().ok();
    }
    raw.parse::<f64>().ok().and_then(prometheus_f64_to_u64)
}

/// Parse Prometheus text format and extract the metrics we care about.
/// llama.cpp uses colon-separated names like `llamacpp:prompt_tokens_total`.
pub fn parse_prometheus_metrics(body: &str) -> PrometheusValues {
    let mut vals = PrometheusValues::default();
    for line in body.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let name = match parts.next() {
            Some(n) => n,
            None => continue,
        };
        let raw = match parts.next() {
            Some(v) => v,
            None => continue,
        };
        match name {
            "llamacpp:prompt_tokens_seconds" => {
                if let Ok(v) = raw.parse::<f64>() {
                    vals.prompt_tokens_per_sec = v;
                }
            }
            "llamacpp:predicted_tokens_seconds" => {
                if let Ok(v) = raw.parse::<f64>() {
                    vals.predicted_tokens_per_sec = v;
                }
            }
            "llamacpp:prompt_tokens_total" => {
                vals.prompt_tokens_total = parse_prometheus_counter(raw).unwrap_or(0);
            }
            "llamacpp:prompt_seconds_total" => {
                if let Ok(v) = raw.parse::<f64>() {
                    vals.prompt_seconds_total = v;
                }
            }
            "llamacpp:tokens_predicted_total" => {
                vals.predicted_tokens_total = parse_prometheus_counter(raw).unwrap_or(0);
            }
            "llamacpp:tokens_predicted_seconds_total" => {
                if let Ok(v) = raw.parse::<f64>() {
                    vals.predicted_seconds_total = v;
                }
            }
            "llamacpp:requests_processing" => {
                if let Ok(v) = raw.parse::<f64>()
                    && v.is_finite()
                    && v >= 0.0
                {
                    vals.requests_processing = v.round().min(u32::MAX as f64) as u32;
                }
            }
            "llamacpp:prompt_tokens_cached_total" => {
                vals.prompt_tokens_cached_total = parse_prometheus_counter(raw);
            }
            "llamacpp:n_tokens_max" => vals.n_tokens_max = parse_prometheus_counter(raw),
            "llamacpp:spec_decode_num_draft_tokens_total" => {
                vals.spec_decode_num_draft_tokens_total = parse_prometheus_counter(raw);
            }
            "llamacpp:spec_decode_num_accepted_tokens_total" => {
                vals.spec_decode_num_accepted_tokens_total = parse_prometheus_counter(raw);
            }
            "llamacpp:spec_decode_num_drafts_total" => {
                vals.spec_decode_num_drafts_total = parse_prometheus_counter(raw);
            }
            "llamacpp:n_busy_slots_per_decode" => {
                if let Ok(value) = raw.parse::<f64>()
                    && value.is_finite()
                {
                    vals.n_busy_slots_per_decode = Some(value);
                }
            }
            _ => {}
        }
    }
    vals
}

fn json_u64(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| {
        x.as_u64()
            .or_else(|| x.as_i64().and_then(|i| u64::try_from(i).ok()))
            .or_else(|| x.as_f64().and_then(prometheus_f64_to_u64))
    })
}

fn slot_decoded_tokens(slot: &serde_json::Value) -> u64 {
    if let Some(n) = json_u64(slot, "n_decoded") {
        return n;
    }
    let Some(nt) = slot.get("next_token") else {
        return 0;
    };
    if let Some(n) = json_u64(nt, "n_decoded") {
        return n;
    }
    nt.as_array()
        .and_then(|arr| arr.first())
        .and_then(|first| json_u64(first, "n_decoded"))
        .unwrap_or(0)
}

/// Current KV occupancy for one slot. Prefers `n_past` (older llama.cpp).
/// Newer servers expose `n_prompt_tokens` as `prompt.tokens.size()`, which already
/// includes generated tokens; do not add `n_decoded` on top of it.
fn slot_used_tokens(slot: &serde_json::Value) -> u64 {
    let n_ctx = json_u64(slot, "n_ctx").unwrap_or(u64::MAX);
    if let Some(n_past) = json_u64(slot, "n_past") {
        return n_past.min(n_ctx);
    }
    if let Some(prompt) = json_u64(slot, "n_prompt_tokens") {
        return prompt.min(n_ctx);
    }
    slot_decoded_tokens(slot).min(n_ctx)
}

/// Sum current KV tokens and per-slot `n_ctx` across `/slots`.
/// `llamacpp:n_tokens_max` is a high-water mark and must not be used here.
pub fn slots_kv_usage(slots: &[serde_json::Value]) -> (u64, u64) {
    let mut used = 0u64;
    let mut max = 0u64;
    for slot in slots {
        let n_ctx = json_u64(slot, "n_ctx").unwrap_or(0);
        max = max.saturating_add(n_ctx);
        used = used.saturating_add(slot_used_tokens(slot));
    }
    (used, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_prometheus_metrics() {
        let body = include_str!("../../tests/fixtures/prometheus_metrics.txt");
        let vals = parse_prometheus_metrics(body);

        assert!((vals.prompt_tokens_per_sec - 1234.5).abs() < 0.1);
        assert!((vals.predicted_tokens_per_sec - 56.7).abs() < 0.1);
        assert_eq!(vals.prompt_tokens_total, 10_000);
        assert!((vals.prompt_seconds_total - 8.1).abs() < 0.1);
        assert_eq!(vals.predicted_tokens_total, 5_000);
        assert!((vals.predicted_seconds_total - 88.2).abs() < 0.1);
        assert_eq!(vals.requests_processing, 1);
        // High-water mark is present in the fixture but must not be parsed as live KV.
        assert!(body.contains("llamacpp:n_tokens_max"));
    }

    #[test]
    fn test_parse_prometheus_metrics_empty() {
        let vals = parse_prometheus_metrics("");
        assert_eq!(vals.prompt_tokens_per_sec, 0.0);
        assert_eq!(vals.requests_processing, 0);
    }

    #[test]
    fn test_parse_prometheus_metrics_comments_only() {
        let body = "# HELP llamacpp:prompt_tokens_total Total prompt tokens\n# TYPE llamacpp:prompt_tokens_total counter\n";
        let vals = parse_prometheus_metrics(body);
        assert_eq!(vals.prompt_tokens_total, 0);
    }

    #[test]
    fn slots_kv_usage_uses_n_past_when_present() {
        let slots = vec![serde_json::json!({
            "id": 0,
            "n_ctx": 8192,
            "n_past": 1024,
            "n_decoded": 50,
            "is_processing": true
        })];
        assert_eq!(slots_kv_usage(&slots), (1024, 8192));
    }

    #[test]
    fn slots_kv_usage_uses_prompt_tokens_as_current_occupancy() {
        let slots = vec![serde_json::json!({
            "id": 0,
            "n_ctx": 131072,
            "n_prompt_tokens": 400,
            "is_processing": true,
            "next_token": { "n_decoded": 80 }
        })];
        assert_eq!(slots_kv_usage(&slots), (400, 131072));
    }

    #[test]
    fn slots_kv_usage_drops_after_new_session_unlike_n_tokens_max() {
        let full = vec![serde_json::json!({
            "id": 0,
            "n_ctx": 8192,
            "n_prompt_tokens": 8192,
            "next_token": { "n_decoded": 192 }
        })];
        let new_session = vec![serde_json::json!({
            "id": 0,
            "n_ctx": 8192,
            "n_prompt_tokens": 120,
            "next_token": [{ "n_decoded": 12 }]
        })];
        assert_eq!(slots_kv_usage(&full), (8192, 8192));
        assert_eq!(slots_kv_usage(&new_session), (120, 8192));
    }

    #[test]
    fn slots_kv_usage_idle_slot_without_task_is_empty() {
        let slots = vec![serde_json::json!({
            "id": 0,
            "n_ctx": 65536,
            "speculative": false,
            "is_processing": false
        })];
        assert_eq!(slots_kv_usage(&slots), (0, 65536));
    }

    #[test]
    fn slots_kv_usage_sums_parallel_slots() {
        let slots = vec![
            serde_json::json!({
                "id": 0,
                "n_ctx": 4096,
                "n_prompt_tokens": 100,
                "next_token": { "n_decoded": 20 }
            }),
            serde_json::json!({
                "id": 1,
                "n_ctx": 4096,
                "n_prompt_tokens": 50,
                "n_decoded": 10
            }),
        ];
        assert_eq!(slots_kv_usage(&slots), (150, 8192));
    }

    #[test]
    fn slots_kv_usage_falls_back_to_n_decoded_when_prompt_missing() {
        let slots = vec![serde_json::json!({
            "id": 0,
            "n_ctx": 4096,
            "is_processing": true,
            "next_token": { "n_decoded": 64 }
        })];
        assert_eq!(slots_kv_usage(&slots), (64, 4096));
    }

    fn sample_prom(prompt_tps: f64, gen_tps: f64) -> PrometheusValues {
        PrometheusValues {
            prompt_tokens_per_sec: prompt_tps,
            predicted_tokens_per_sec: gen_tps,
            prompt_tokens_total: 10_000,
            prompt_seconds_total: 8.1,
            predicted_tokens_total: 5_000,
            predicted_seconds_total: 88.2,
            requests_processing: 1,
            ..Default::default()
        }
    }

    fn idle_prom() -> PrometheusValues {
        PrometheusValues {
            prompt_tokens_per_sec: 0.0,
            predicted_tokens_per_sec: 0.0,
            prompt_tokens_total: 10_000,
            prompt_seconds_total: 8.1,
            predicted_tokens_total: 5_000,
            predicted_seconds_total: 88.2,
            requests_processing: 0,
            ..Default::default()
        }
    }

    #[test]
    fn live_prompt_speed_metrics_zero_is_not_lifetime_average() {
        let t0 = std::time::Instant::now();
        let mut m = LlamaMetrics::default();
        m.apply_metrics_at(&idle_prom(), t0);
        assert_eq!(m.prompt_tokens_per_sec, None);
        assert_eq!(m.inference_phase, None);
        let lifetime = 10_000.0 / 8.1;
        assert!(lifetime > 1.0);
    }

    #[test]
    fn live_generation_speed_metrics_zero_is_not_lifetime_average() {
        let t0 = std::time::Instant::now();
        let mut m = LlamaMetrics::default();
        m.apply_metrics_at(&idle_prom(), t0);
        assert_eq!(m.generation_tokens_per_sec, None);
        let lifetime = 5_000.0 / 88.2;
        assert!(lifetime > 1.0);
    }

    #[test]
    fn missing_log_speeds_serialize_as_null() {
        let mut m = LlamaMetrics::default();
        m.apply_metrics(&idle_prom());
        let v = serde_json::to_value(&m).unwrap();
        assert!(v["prompt_tokens_per_sec"].is_null());
        assert!(v["generation_tokens_per_sec"].is_null());
        assert!(v["inference_phase"].is_null());
        m.clear_metrics_gauges();
        let v = serde_json::to_value(&m).unwrap();
        assert!(v["prompt_tokens_per_sec"].is_null());
        assert!(v["generation_tokens_per_sec"].is_null());
        assert!(v["inference_phase"].is_null());
    }

    #[test]
    fn metrics_failure_keeps_session_counters() {
        let mut m = LlamaMetrics::default();
        m.apply_metrics(&sample_prom(42.0, 18.0));
        assert_eq!(m.prompt_tokens_total, 10_000);
        assert_eq!(m.predicted_tokens_total, 5_000);
        m.clear_metrics_gauges();
        assert_eq!(m.requests_processing, None);
        assert_eq!(m.prompt_tokens_total, 10_000);
        assert_eq!(m.predicted_tokens_total, 5_000);
    }

    #[test]
    fn slots_failure_clears_live_kv_not_log_speed() {
        let t0 = std::time::Instant::now();
        let mut m = LlamaMetrics::default();
        m.apply_log_line_at(
            "slot print_timing: id 1 | task 916 | n_gen = 276, tg = 28.66 t/s, tg_3s = 30.85 t/s",
            t0,
        );
        m.apply_metrics_at(&sample_prom(12.0, 8.0), t0);
        m.apply_slots(&[serde_json::json!({
            "id": 0,
            "n_ctx": 8192,
            "n_prompt_tokens": 4096,
            "is_processing": true
        })]);
        assert_eq!(m.kv_cache_tokens, Some(4096));
        m.clear_slots_gauges();
        assert_eq!(m.kv_cache_tokens, None);
        assert_eq!(m.kv_cache_max, None);
        assert_eq!(
            m.inference_phase,
            Some(crate::llama::live_slots::InferencePhase::Generation)
        );
        assert!((m.generation_tokens_per_sec.unwrap() - 30.85).abs() < 1e-9);
    }

    #[test]
    fn metrics_predicted_zero_does_not_override_log_tg_3s() {
        let t0 = std::time::Instant::now();
        let mut m = LlamaMetrics::default();
        m.apply_log_line_at(
            "slot print_timing: id 1 | task 916 | n_gen = 276, tg = 28.66 t/s, tg_3s = 30.85 t/s",
            t0,
        );
        let mut prom = sample_prom(0.0, 0.0);
        prom.predicted_tokens_per_sec = 0.0;
        m.apply_metrics_at(&prom, t0 + std::time::Duration::from_millis(200));
        assert_eq!(
            m.inference_phase,
            Some(crate::llama::live_slots::InferencePhase::Generation)
        );
        assert!((m.generation_tokens_per_sec.unwrap() - 30.85).abs() < 1e-9);
        assert_eq!(
            m.generation_speed_kind,
            Some(crate::llama::live_slots::SpeedKind::Live)
        );
        let json = serde_json::to_value(&m).unwrap();
        assert!((json["generation_tokens_per_sec"].as_f64().unwrap() - 30.85).abs() < 1e-9);
        assert_eq!(json["inference_phase"], "generation");
    }

    #[test]
    fn stale_log_speed_stays_numeric_last_not_metrics_zero() {
        let t0 = std::time::Instant::now();
        let mut m = LlamaMetrics::default();
        m.apply_log_line_at(
            "slot print_timing: id 1 | task 916 | n_gen = 276, tg = 28.66 t/s, tg_3s = 30.85 t/s",
            t0,
        );
        m.apply_metrics_at(
            &idle_prom(),
            t0 + crate::llama::live_slots::LIVE_STALE + std::time::Duration::from_secs(1),
        );
        assert_eq!(
            m.inference_phase,
            Some(crate::llama::live_slots::InferencePhase::Idle)
        );
        assert_eq!(
            m.generation_speed_kind,
            Some(crate::llama::live_slots::SpeedKind::Last)
        );
        assert!((m.generation_tokens_per_sec.unwrap() - 30.85).abs() < 1e-9);
        assert_ne!(m.generation_tokens_per_sec, Some(0.0));
    }

    #[test]
    fn apply_slots_updates_context_from_new_payload() {
        let mut m = LlamaMetrics::default();
        m.apply_slots(&[serde_json::json!({
            "id": 0,
            "n_ctx": 8192,
            "n_prompt_tokens": 8192
        })]);
        m.apply_slots(&[serde_json::json!({
            "id": 0,
            "n_ctx": 8192,
            "n_prompt_tokens": 120
        })]);
        assert_eq!(m.kv_cache_tokens, Some(120));
        assert_eq!(m.kv_cache_max, Some(8192));
        assert_eq!(m.slots_idle, Some(1));
        assert_eq!(m.slots_processing, Some(0));
    }

    fn live_server_prom() -> PrometheusValues {
        parse_prometheus_metrics(include_str!(
            "../../tests/fixtures/prometheus_metrics_live_server.txt"
        ))
    }

    #[test]
    fn live_server_scientific_notation_cached_tokens() {
        assert_eq!(
            live_server_prom().prompt_tokens_cached_total,
            Some(4_657_250)
        );
    }

    #[test]
    fn parse_prompt_tokens_cached_total_scientific_notation() {
        let vals = parse_prometheus_metrics("llamacpp:prompt_tokens_cached_total 4.65725e+06\n");
        assert_eq!(vals.prompt_tokens_cached_total, Some(4_657_250));
    }

    #[test]
    fn live_server_processed_and_predicted_totals() {
        let vals = live_server_prom();
        assert_eq!(vals.prompt_tokens_total, 251_342);
        assert_eq!(vals.predicted_tokens_total, 71_788);
    }

    #[test]
    fn live_server_idle_gauges_are_zero_not_lifetime_average() {
        let vals = live_server_prom();
        assert_eq!(vals.prompt_tokens_per_sec, 0.0);
        assert_eq!(vals.predicted_tokens_per_sec, 0.0);
        assert_eq!(vals.requests_processing, 0);
        let lifetime_prompt = 251_342.0 / 579.864;
        let lifetime_gen = 71_788.0 / 2719.99;
        assert!((vals.prompt_tokens_per_sec - lifetime_prompt).abs() > 1.0);
        assert!((vals.predicted_tokens_per_sec - lifetime_gen).abs() > 1.0);
    }

    #[test]
    fn live_server_n_tokens_max_is_peak_not_kv_used() {
        let vals = live_server_prom();
        assert_eq!(vals.n_tokens_max, Some(113_868));
        let mut m = LlamaMetrics::default();
        m.apply_metrics(&vals);
        m.apply_slots(&[serde_json::json!({
            "id": 0,
            "n_ctx": 180224,
            "n_prompt_tokens": 120,
            "is_processing": false
        })]);
        assert_eq!(m.n_tokens_max, Some(113_868));
        assert_eq!(m.kv_cache_tokens, Some(120));
        assert_eq!(m.kv_cache_max, Some(180_224));
        assert_ne!(m.kv_cache_tokens, m.n_tokens_max);
    }

    #[test]
    fn live_server_speculative_acceptance_ratio() {
        let vals = live_server_prom();
        assert_eq!(vals.spec_decode_num_draft_tokens_total, Some(40_018));
        assert_eq!(vals.spec_decode_num_accepted_tokens_total, Some(31_710));
        assert_eq!(vals.spec_decode_num_drafts_total, Some(40_018));
        let ratio = vals.speculative_acceptance_ratio().unwrap();
        assert!((ratio - 31_710.0 / 40_018.0).abs() < 1e-12);
        assert!((ratio - 0.7924).abs() < 0.0001);
    }

    #[test]
    fn speculative_acceptance_ratio_none_when_no_drafts() {
        let vals = PrometheusValues {
            spec_decode_num_draft_tokens_total: Some(0),
            spec_decode_num_accepted_tokens_total: Some(0),
            ..Default::default()
        };
        assert!(vals.speculative_acceptance_ratio().is_none());
    }

    #[test]
    fn speculative_acceptance_ratio_none_when_metric_absent() {
        assert!(
            PrometheusValues::default()
                .speculative_acceptance_ratio()
                .is_none()
        );
    }

    #[test]
    fn cached_counter_absent_on_legacy_metrics() {
        let body = include_str!("../../tests/fixtures/prometheus_metrics.txt");
        let vals = parse_prometheus_metrics(body);
        assert_eq!(vals.prompt_tokens_cached_total, None);
        assert_eq!(vals.spec_decode_num_draft_tokens_total, None);
    }

    #[test]
    fn prometheus_f64_to_u64_rejects_invalid() {
        assert_eq!(prometheus_f64_to_u64(4.65725e+06), Some(4_657_250));
        assert_eq!(prometheus_f64_to_u64(0.0), Some(0));
        assert_eq!(prometheus_f64_to_u64(-1.0), None);
        assert_eq!(prometheus_f64_to_u64(f64::NAN), None);
        assert_eq!(prometheus_f64_to_u64(f64::INFINITY), None);
        assert_eq!(prometheus_f64_to_u64(f64::NEG_INFINITY), None);
    }

    #[test]
    fn live_server_busy_slots_per_decode_is_diagnostic() {
        let vals = live_server_prom();
        assert!((vals.n_busy_slots_per_decode.unwrap() - 1.05701).abs() < 1e-5);
    }

    #[test]
    fn apply_metrics_copies_peak_and_spec_to_snapshot() {
        let mut m = LlamaMetrics::default();
        m.apply_metrics(&live_server_prom());
        assert_eq!(m.n_tokens_max, Some(113_868));
        assert_eq!(m.spec_draft_tokens, Some(40_018));
        assert_eq!(m.spec_accepted_tokens, Some(31_710));
        assert_eq!(m.spec_drafts, Some(40_018));
        let ratio = m.spec_acceptance_ratio.unwrap();
        assert!((ratio - 31_710.0 / 40_018.0).abs() < 1e-12);
        assert!((m.n_busy_slots_per_decode.unwrap() - 1.05701).abs() < 1e-5);
        assert_eq!(m.prompt_tokens_per_sec, None);
        assert_eq!(m.generation_tokens_per_sec, None);
        assert_eq!(m.inference_phase, None);
    }

    #[test]
    fn apply_metrics_does_not_invent_prefill_from_counter_delta() {
        let t0 = std::time::Instant::now();
        let mut m = LlamaMetrics::default();
        let mut prom = idle_prom();
        prom.requests_processing = 1;
        m.apply_metrics_at(&prom, t0);
        prom.prompt_tokens_total = 10_400;
        m.apply_metrics_at(&prom, t0 + std::time::Duration::from_secs(1));
        assert_eq!(m.inference_phase, None);
        assert_eq!(m.prompt_tokens_per_sec, None);
        assert_eq!(m.generation_tokens_per_sec, None);
    }

    #[test]
    fn parse_prometheus_counter_digit_string_is_u64() {
        assert_eq!(parse_prometheus_counter("0"), Some(0));
        assert_eq!(parse_prometheus_counter("4294967296"), Some(4_294_967_296));
        assert_eq!(
            parse_prometheus_counter("9007199254740993"),
            Some(9_007_199_254_740_993)
        );
        assert_eq!(parse_prometheus_counter("4.65725e+06"), Some(4_657_250));
        assert_eq!(parse_prometheus_counter("-1"), None);
        assert_eq!(parse_prometheus_counter("NaN"), None);
        assert_eq!(parse_prometheus_counter("+Inf"), None);
    }

    #[test]
    fn prometheus_token_counters_above_u32_stay_u64() {
        let body = "\
llamacpp:prompt_tokens_total 5000000000
llamacpp:tokens_predicted_total 3000000000
llamacpp:prompt_tokens_cached_total 12000000000
llamacpp:spec_decode_num_draft_tokens_total 4294967296
llamacpp:spec_decode_num_accepted_tokens_total 5000000000
llamacpp:spec_decode_num_drafts_total 4294967296
";
        let vals = parse_prometheus_metrics(body);
        assert_eq!(vals.prompt_tokens_total, 5_000_000_000);
        assert_eq!(vals.predicted_tokens_total, 3_000_000_000);
        assert_eq!(vals.prompt_tokens_cached_total, Some(12_000_000_000));
        assert_eq!(vals.spec_decode_num_draft_tokens_total, Some(4_294_967_296));
        assert_eq!(
            vals.spec_decode_num_accepted_tokens_total,
            Some(5_000_000_000)
        );
        assert_eq!(vals.spec_decode_num_drafts_total, Some(4_294_967_296));
        assert_ne!(vals.prompt_tokens_total, 0);
        assert_ne!(vals.prompt_tokens_total as i64, i32::MIN as i64);

        let mut m = LlamaMetrics::default();
        m.apply_metrics(&vals);
        assert_eq!(m.prompt_tokens_total, 5_000_000_000);
        assert_eq!(m.predicted_tokens_total, 3_000_000_000);
        assert_eq!(m.spec_draft_tokens, Some(4_294_967_296));
        assert_eq!(m.spec_accepted_tokens, Some(5_000_000_000));
        assert_eq!(m.spec_drafts, Some(4_294_967_296));
        let json = serde_json::to_value(&m).unwrap();
        assert_eq!(json["prompt_tokens_total"], 5_000_000_000_u64);
        assert_eq!(json["predicted_tokens_total"], 3_000_000_000_u64);
        assert_eq!(json["spec_draft_tokens"], 4_294_967_296_u64);
    }

    #[test]
    fn prometheus_u32_overflow_plus_one_does_not_wrap() {
        let body = "llamacpp:prompt_tokens_total 4294967296\n";
        let vals = parse_prometheus_metrics(body);
        assert_eq!(vals.prompt_tokens_total, 4_294_967_296);
        assert_ne!(vals.prompt_tokens_total, 0);
        assert!(vals.prompt_tokens_total > u32::MAX as u64);
    }

    #[test]
    fn prometheus_preserves_integers_beyond_f64_mantissa() {
        let body = "llamacpp:prompt_tokens_total 9007199254740993\n";
        let vals = parse_prometheus_metrics(body);
        assert_eq!(vals.prompt_tokens_total, 9_007_199_254_740_993);
        assert_ne!(
            prometheus_f64_to_u64(9_007_199_254_740_993.0),
            Some(9_007_199_254_740_993)
        );
    }

    #[test]
    fn llama_metrics_large_counters_serde_round_trip() {
        let mut m = LlamaMetrics::default();
        m.prompt_tokens_total = 5_000_000_000;
        m.predicted_tokens_total = 3_000_000_000;
        m.spec_draft_tokens = Some(4_294_967_296);
        m.spec_accepted_tokens = Some(12_000_000_000);
        m.spec_drafts = Some(4_294_967_296);
        let json = serde_json::to_string(&m).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["prompt_tokens_total"].as_u64(), Some(5_000_000_000));
        assert_eq!(v["predicted_tokens_total"].as_u64(), Some(3_000_000_000));
        assert_eq!(v["spec_draft_tokens"].as_u64(), Some(4_294_967_296));
        assert_eq!(v["spec_accepted_tokens"].as_u64(), Some(12_000_000_000));
        assert!(v["prompt_tokens_total"].as_u64().unwrap() > u32::MAX as u64);
        assert!(v["prompt_tokens_total"].as_i64().unwrap() > 0);
    }
}
