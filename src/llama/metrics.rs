/// Live llama.cpp snapshot. Gauges are `None` when the source endpoint failed
/// this cycle (`0` is a valid idle reading and must not be replaced by averages).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LlamaMetrics {
    /// Instantaneous prompt t/s from `/metrics` (`None` = unavailable).
    pub prompt_tokens_per_sec: Option<f64>,
    /// Instantaneous generation t/s from `/metrics` (`None` = unavailable).
    pub generation_tokens_per_sec: Option<f64>,
    /// llama-server session counters from the last successful `/metrics` scrape.
    pub prompt_tokens_total: u64,
    pub predicted_tokens_total: u64,
    pub kv_cache_tokens: Option<u64>,
    pub kv_cache_max: Option<u64>,
    pub slots_idle: Option<u32>,
    pub slots_processing: Option<u32>,
    pub requests_processing: Option<u32>,
    pub status: String,
}

impl LlamaMetrics {
    /// Apply live gauges + session counters from a successful `/metrics` scrape.
    /// A gauge of `0` stays `0` — never substitute lifetime averages.
    pub fn apply_metrics(&mut self, prom: &PrometheusValues) {
        self.prompt_tokens_per_sec = Some(prom.prompt_tokens_per_sec);
        self.generation_tokens_per_sec = Some(prom.predicted_tokens_per_sec);
        self.prompt_tokens_total = prom.prompt_tokens_total as u64;
        self.predicted_tokens_total = prom.predicted_tokens_total as u64;
        self.requests_processing = Some(prom.requests_processing);
    }

    /// `/metrics` failed this cycle: live gauges become unavailable.
    /// Session counters are left as last-known (counter semantics).
    pub fn clear_metrics_gauges(&mut self) {
        self.prompt_tokens_per_sec = None;
        self.generation_tokens_per_sec = None;
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
    pub prompt_tokens_total: f64,
    pub prompt_seconds_total: f64,
    pub predicted_tokens_total: f64,
    pub predicted_seconds_total: f64,
    pub requests_processing: u32,
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
        let value = match parts.next().and_then(|v| v.parse::<f64>().ok()) {
            Some(v) => v,
            None => continue,
        };
        match name {
            "llamacpp:prompt_tokens_seconds" => vals.prompt_tokens_per_sec = value,
            "llamacpp:predicted_tokens_seconds" => vals.predicted_tokens_per_sec = value,
            "llamacpp:prompt_tokens_total" => vals.prompt_tokens_total = value,
            "llamacpp:prompt_seconds_total" => vals.prompt_seconds_total = value,
            "llamacpp:tokens_predicted_total" => vals.predicted_tokens_total = value,
            "llamacpp:tokens_predicted_seconds_total" => vals.predicted_seconds_total = value,
            "llamacpp:requests_processing" => vals.requests_processing = value as u32,
            _ => {}
        }
    }
    vals
}

fn json_u64(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| {
        x.as_u64()
            .or_else(|| x.as_i64().and_then(|i| u64::try_from(i).ok()))
            .or_else(|| x.as_f64().and_then(|f| (f >= 0.0).then_some(f as u64)))
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
        assert!((vals.prompt_tokens_total - 10000.0).abs() < 0.1);
        assert!((vals.prompt_seconds_total - 8.1).abs() < 0.1);
        assert!((vals.predicted_tokens_total - 5000.0).abs() < 0.1);
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
        assert_eq!(vals.prompt_tokens_total, 0.0);
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
            prompt_tokens_total: 10_000.0,
            prompt_seconds_total: 8.1,
            predicted_tokens_total: 5_000.0,
            predicted_seconds_total: 88.2,
            requests_processing: 1,
        }
    }

    #[test]
    fn live_prompt_speed_zero_is_not_lifetime_average() {
        let mut m = LlamaMetrics::default();
        m.apply_metrics(&sample_prom(42.0, 20.0));
        assert_eq!(m.prompt_tokens_per_sec, Some(42.0));
        m.apply_metrics(&sample_prom(0.0, 20.0));
        assert_eq!(m.prompt_tokens_per_sec, Some(0.0));
        let lifetime = 10_000.0 / 8.1;
        assert!((m.prompt_tokens_per_sec.unwrap() - lifetime).abs() > 1.0);
    }

    #[test]
    fn live_generation_speed_zero_is_not_lifetime_average() {
        let mut m = LlamaMetrics::default();
        m.apply_metrics(&sample_prom(10.0, 55.0));
        assert_eq!(m.generation_tokens_per_sec, Some(55.0));
        m.apply_metrics(&sample_prom(10.0, 0.0));
        assert_eq!(m.generation_tokens_per_sec, Some(0.0));
        let lifetime = 5_000.0 / 88.2;
        assert!((m.generation_tokens_per_sec.unwrap() - lifetime).abs() > 1.0);
    }

    #[test]
    fn live_zero_serializes_as_zero_not_null() {
        let mut m = LlamaMetrics::default();
        m.apply_metrics(&sample_prom(0.0, 0.0));
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["prompt_tokens_per_sec"], 0.0);
        assert_eq!(v["generation_tokens_per_sec"], 0.0);
        m.clear_metrics_gauges();
        let v = serde_json::to_value(&m).unwrap();
        assert!(v["prompt_tokens_per_sec"].is_null());
        assert!(v["generation_tokens_per_sec"].is_null());
    }

    #[test]
    fn metrics_failure_clears_live_speed_keeps_session_counters() {
        let mut m = LlamaMetrics::default();
        m.apply_metrics(&sample_prom(42.0, 18.0));
        assert_eq!(m.prompt_tokens_total, 10_000);
        assert_eq!(m.predicted_tokens_total, 5_000);
        m.clear_metrics_gauges();
        assert_eq!(m.prompt_tokens_per_sec, None);
        assert_eq!(m.generation_tokens_per_sec, None);
        assert_eq!(m.requests_processing, None);
        assert_eq!(m.prompt_tokens_total, 10_000);
        assert_eq!(m.predicted_tokens_total, 5_000);
    }

    #[test]
    fn slots_failure_clears_live_kv_not_speed() {
        let mut m = LlamaMetrics::default();
        m.apply_metrics(&sample_prom(12.0, 8.0));
        m.apply_slots(&[serde_json::json!({
            "id": 0,
            "n_ctx": 8192,
            "n_prompt_tokens": 4096,
            "is_processing": true
        })]);
        assert_eq!(m.kv_cache_tokens, Some(4096));
        assert_eq!(m.kv_cache_max, Some(8192));
        assert_eq!(m.slots_processing, Some(1));
        m.clear_slots_gauges();
        assert_eq!(m.kv_cache_tokens, None);
        assert_eq!(m.kv_cache_max, None);
        assert_eq!(m.slots_idle, None);
        assert_eq!(m.slots_processing, None);
        assert_eq!(m.prompt_tokens_per_sec, Some(12.0));
        assert_eq!(m.generation_tokens_per_sec, Some(8.0));
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
}
