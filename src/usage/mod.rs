use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;

/// OpenAI GPT-5.6 Luna rates (USD per 1M tokens). Cached input is billed
/// separately and must not use the full input rate.
pub const LUNA_RATES: ProviderRates = ProviderRates {
    input_per_m: 0.20,
    cached_input_per_m: Some(0.02),
    output_per_m: 1.20,
};

/// Alibaba Cloud Model Studio Qwen3.8-27B, International/Singapore
/// (USD per 1M tokens). Cached input uses Implicit Cache, which matches
/// llama.cpp automatic prefix/context reuse — not Explicit Cache Read.
pub const QWEN_RATES: ProviderRates = ProviderRates {
    input_per_m: 0.50,
    cached_input_per_m: Some(0.10),
    output_per_m: 3.00,
};

/// USD per 1M tokens for one comparison provider.
#[derive(Debug, Clone, Copy)]
pub struct ProviderRates {
    pub input_per_m: f64,
    pub cached_input_per_m: Option<f64>,
    pub output_per_m: f64,
}

impl ProviderRates {
    pub fn effective_cached_input_per_m(self) -> f64 {
        self.cached_input_per_m.unwrap_or(self.input_per_m)
    }
}

const CACHE_DEDUP_WINDOW: Duration = Duration::from_secs(2);
const SAVE_DEBOUNCE: Duration = Duration::from_secs(5);

/// One `/metrics` scrape folded into lifetime totals.
///
/// `prompt` / `predicted` are always present (0 when the names are missing).
/// Optional fields stay `None` when that llama.cpp build does not export them.
#[derive(Debug, Clone, Default)]
pub struct PrometheusUsageSample {
    pub prompt: u64,
    pub predicted: u64,
    pub cached: Option<u64>,
    pub peak_context: Option<u64>,
    pub mtp_draft: Option<u64>,
    pub mtp_accepted: Option<u64>,
}

/// Persisted lifetime counters + Prometheus baselines.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UsageStats {
    /// Lifetime prompt tokens processed (not cache hits). u64 — do not narrow.
    #[serde(default)]
    pub prompt_tokens: u64,
    /// Lifetime generated tokens. u64 — do not narrow.
    #[serde(default)]
    pub predicted_tokens: u64,
    /// Lifetime prompt tokens reused from cache. u64 — do not narrow.
    #[serde(default)]
    pub cached_tokens: u64,
    /// Lifetime high-water mark of `llamacpp:n_tokens_max`. Not a counter.
    #[serde(default)]
    pub peak_context_tokens: u64,
    /// Lifetime speculative/MTP draft tokens. u64 — do not narrow.
    #[serde(default)]
    pub mtp_draft_tokens: u64,
    /// Lifetime speculative/MTP accepted tokens. u64 — do not narrow.
    #[serde(default)]
    pub mtp_accepted_tokens: u64,
    /// Last seen llama-server Prometheus prompt counter (session baseline).
    #[serde(default)]
    pub last_prompt: u64,
    /// Last seen llama-server Prometheus predicted counter (session baseline).
    #[serde(default)]
    pub last_predicted: u64,
    /// Last seen `llamacpp:prompt_tokens_cached_total` (session baseline).
    #[serde(default)]
    pub last_cached: u64,
    /// Last seen `llamacpp:spec_decode_num_draft_tokens_total`.
    #[serde(default)]
    pub last_mtp_draft: u64,
    /// Last seen `llamacpp:spec_decode_num_accepted_tokens_total`.
    #[serde(default)]
    pub last_mtp_accepted: u64,
    /// True once this llama-server session exported `prompt_tokens_cached_total`.
    /// Persisted so monitor restart does not fall back to `parse_cache_n` and
    /// double-count before the first scrape.
    #[serde(default)]
    pub native_cached_counter: bool,
    #[serde(default)]
    pub updated_at: u64,
    /// Runtime-only: skip serializing.
    #[serde(skip)]
    dirty: bool,
    #[serde(skip)]
    last_saved: Option<Instant>,
    #[serde(skip)]
    last_cache_n: Option<(u64, Instant)>,
}

/// Snapshot pushed over WebSocket / returned by API.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UsageSnapshot {
    pub prompt_tokens: u64,
    pub predicted_tokens: u64,
    pub cached_tokens: u64,
    pub total_prompt_tokens: u64,
    pub cache_hit_ratio: f64,
    pub cache_reuse_ratio: f64,
    pub peak_context_tokens: u64,
    pub mtp_draft_tokens: u64,
    pub mtp_accepted_tokens: u64,
    pub mtp_acceptance_ratio: Option<f64>,
    pub saved_luna_usd: f64,
    pub saved_qwen_usd: f64,
    pub rates: UsageRatesInfo,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UsageRatesInfo {
    pub luna_input_per_m: f64,
    pub luna_cached_input_per_m: Option<f64>,
    pub luna_output_per_m: f64,
    pub qwen_input_per_m: f64,
    pub qwen_cached_input_per_m: Option<f64>,
    pub qwen_output_per_m: f64,
    pub label: &'static str,
}

impl UsageStats {
    pub fn snapshot(&self) -> UsageSnapshot {
        let total_prompt_tokens = self.prompt_tokens.saturating_add(self.cached_tokens);
        let cache_reuse_ratio = if total_prompt_tokens > 0 {
            self.cached_tokens as f64 / total_prompt_tokens as f64
        } else {
            0.0
        };
        UsageSnapshot {
            prompt_tokens: self.prompt_tokens,
            predicted_tokens: self.predicted_tokens,
            cached_tokens: self.cached_tokens,
            total_prompt_tokens,
            cache_hit_ratio: cache_reuse_ratio,
            cache_reuse_ratio,
            peak_context_tokens: self.peak_context_tokens,
            mtp_draft_tokens: self.mtp_draft_tokens,
            mtp_accepted_tokens: self.mtp_accepted_tokens,
            mtp_acceptance_ratio: self.mtp_acceptance_ratio(),
            saved_luna_usd: estimated_api_cost(
                self.prompt_tokens,
                self.cached_tokens,
                self.predicted_tokens,
                &LUNA_RATES,
            ),
            saved_qwen_usd: estimated_api_cost(
                self.prompt_tokens,
                self.cached_tokens,
                self.predicted_tokens,
                &QWEN_RATES,
            ),
            rates: UsageRatesInfo {
                luna_input_per_m: LUNA_RATES.input_per_m,
                luna_cached_input_per_m: LUNA_RATES.cached_input_per_m,
                luna_output_per_m: LUNA_RATES.output_per_m,
                qwen_input_per_m: QWEN_RATES.input_per_m,
                qwen_cached_input_per_m: QWEN_RATES.cached_input_per_m,
                qwen_output_per_m: QWEN_RATES.output_per_m,
                label: "GPT-5.6 Luna / Qwen3.8-27B Singapore Implicit Cache",
            },
        }
    }

    /// Speculative/MTP acceptance from lifetime counters.
    /// `None` when there were no draft tokens.
    pub fn mtp_acceptance_ratio(&self) -> Option<f64> {
        if self.mtp_draft_tokens == 0 {
            return None;
        }
        Some(self.mtp_accepted_tokens as f64 / self.mtp_draft_tokens as f64)
    }

    /// Fold a Prometheus sample into lifetime totals.
    ///
    /// If `current >= last` → add delta. If `current < last` (llama-server
    /// restarted) → treat `current` as a fresh session and add it.
    ///
    /// `cached` is `Some` when `llamacpp:prompt_tokens_cached_total` was present.
    /// Once seen, it is the sole source of `cached_tokens` for this session.
    #[cfg(test)]
    pub fn apply_prometheus(&mut self, prompt: u64, predicted: u64, cached: Option<u64>) {
        self.apply_sample(&PrometheusUsageSample {
            prompt,
            predicted,
            cached,
            ..Default::default()
        });
    }

    /// Same delta/reset semantics as [`Self::apply_prometheus`], plus peak
    /// context (high-water) and MTP draft/accepted counters.
    pub fn apply_sample(&mut self, sample: &PrometheusUsageSample) {
        let prompt = sample.prompt;
        let predicted = sample.predicted;
        let cached = sample.cached;

        let prompt_restart = prompt < self.last_prompt;
        let predicted_restart = predicted < self.last_predicted;
        let cached_restart = cached.is_some_and(|c| c < self.last_cached);
        let server_restart = prompt_restart || predicted_restart || cached_restart;

        let prompt_delta = counter_delta(prompt, self.last_prompt);
        let predicted_delta = counter_delta(predicted, self.last_predicted);

        let mut changed = prompt_delta != 0 || predicted_delta != 0;

        if prompt_delta != 0 {
            self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_delta);
        }
        if predicted_delta != 0 {
            self.predicted_tokens = self.predicted_tokens.saturating_add(predicted_delta);
        }
        self.last_prompt = prompt;
        self.last_predicted = predicted;

        if let Some(current) = cached {
            self.native_cached_counter = true;
            let cached_delta = counter_delta(current, self.last_cached);
            if cached_delta != 0 {
                self.cached_tokens = self.cached_tokens.saturating_add(cached_delta);
                changed = true;
            }
            self.last_cached = current;
        } else if server_restart {
            self.native_cached_counter = false;
        }

        if let Some(peak) = sample.peak_context
            && peak > self.peak_context_tokens
        {
            self.peak_context_tokens = peak;
            changed = true;
        }

        if let Some(current) = sample.mtp_draft {
            let draft_delta = counter_delta(current, self.last_mtp_draft);
            if draft_delta != 0 {
                self.mtp_draft_tokens = self.mtp_draft_tokens.saturating_add(draft_delta);
                changed = true;
            }
            self.last_mtp_draft = current;
        }

        if let Some(current) = sample.mtp_accepted {
            let accepted_delta = counter_delta(current, self.last_mtp_accepted);
            if accepted_delta != 0 {
                self.mtp_accepted_tokens = self.mtp_accepted_tokens.saturating_add(accepted_delta);
                changed = true;
            }
            self.last_mtp_accepted = current;
        }

        if changed {
            self.touch();
        }
    }

    /// Add KV prefix-cache hits. Deduplicates identical `cache_n` within 2s
    /// (log + chat proxy may both report the same request).
    /// No-op when the native Prometheus cached counter is authoritative.
    pub fn add_cached(&mut self, n: u64) -> bool {
        if self.native_cached_counter || n == 0 {
            return false;
        }
        let now = Instant::now();
        if let Some((prev_n, prev_at)) = self.last_cache_n
            && prev_n == n
            && now.duration_since(prev_at) < CACHE_DEDUP_WINDOW
        {
            return false;
        }
        self.cached_tokens = self.cached_tokens.saturating_add(n);
        self.last_cache_n = Some((n, now));
        self.touch();
        true
    }

    pub fn reset(&mut self) {
        self.prompt_tokens = 0;
        self.predicted_tokens = 0;
        self.cached_tokens = 0;
        self.peak_context_tokens = 0;
        self.mtp_draft_tokens = 0;
        self.mtp_accepted_tokens = 0;
        // Keep last_* (including last_cached / last_mtp_*) so the next
        // Prometheus sample does not re-add the current llama-server session totals.
        self.last_cache_n = None;
        self.touch();
    }

    fn touch(&mut self) {
        self.dirty = true;
        self.updated_at = now_secs();
    }

    /// Persist if dirty and debounce elapsed (or `force`).
    pub fn maybe_save(&mut self, path: &Path, force: bool) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let due = force
            || self
                .last_saved
                .map(|t| t.elapsed() >= SAVE_DEBOUNCE)
                .unwrap_or(true);
        if !due {
            return Ok(());
        }
        save_usage_stats(path, self)?;
        self.dirty = false;
        self.last_saved = Some(Instant::now());
        Ok(())
    }
}

/// Process-counter delta: add the increase, or the fresh session total after a reset.
fn counter_delta(current: u64, previous: u64) -> u64 {
    if current >= previous {
        current - previous
    } else {
        current
    }
}

pub fn estimated_api_cost(prompt: u64, cached: u64, predicted: u64, rates: &ProviderRates) -> f64 {
    let cached_rate = rates.effective_cached_input_per_m();
    prompt as f64 / 1_000_000.0 * rates.input_per_m
        + cached as f64 / 1_000_000.0 * cached_rate
        + predicted as f64 / 1_000_000.0 * rates.output_per_m
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn load_usage_stats(path: &Path) -> UsageStats {
    if path.exists()
        && let Ok(contents) = std::fs::read_to_string(path)
        && let Ok(mut s) = serde_json::from_str::<UsageStats>(&contents)
    {
        s.dirty = false;
        s.last_saved = Some(Instant::now());
        return s;
    }
    UsageStats::default()
}

pub fn save_usage_stats(path: &Path, stats: &UsageStats) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(stats)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Extract `"cache_n": <int>` from a llama-server log line or SSE chunk.
pub fn parse_cache_n(text: &str) -> Option<u64> {
    // Prefer timings.cache_n style / JSON field.
    const KEY: &str = "\"cache_n\"";
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find(KEY) {
        let start = search_from + rel + KEY.len();
        let rest = text[start..].trim_start();
        let rest = rest.strip_prefix(':')?.trim_start();
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if end == 0 {
            search_from = start;
            continue;
        }
        if let Ok(n) = rest[..end].parse::<u64>() {
            return Some(n);
        }
        search_from = start;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn apply_prometheus_delta() {
        let mut s = UsageStats::default();
        s.apply_prometheus(100, 50, None);
        assert_eq!(s.prompt_tokens, 100);
        assert_eq!(s.predicted_tokens, 50);
        assert_eq!(s.last_prompt, 100);
        assert_eq!(s.last_predicted, 50);

        s.apply_prometheus(150, 80, None);
        assert_eq!(s.prompt_tokens, 150);
        assert_eq!(s.predicted_tokens, 80);
    }

    #[test]
    fn skipped_prometheus_sample_does_not_reset_lifetime() {
        let mut s = UsageStats::default();
        s.apply_prometheus(1000, 400, None);
        // /metrics failure: poller must not call apply_prometheus
        assert_eq!(s.prompt_tokens, 1000);
        assert_eq!(s.predicted_tokens, 400);
        assert_eq!(s.last_prompt, 1000);
        s.apply_prometheus(1100, 450, None);
        assert_eq!(s.prompt_tokens, 1100);
        assert_eq!(s.predicted_tokens, 450);
    }

    #[test]
    fn apply_prometheus_server_restart() {
        let mut s = UsageStats::default();
        s.apply_prometheus(1000, 500, None);
        // llama-server restarted → counters drop
        s.apply_prometheus(10, 5, None);
        assert_eq!(s.prompt_tokens, 1010);
        assert_eq!(s.predicted_tokens, 505);
        assert_eq!(s.last_prompt, 10);
        assert_eq!(s.last_predicted, 5);
    }

    #[test]
    fn apply_prometheus_no_double_after_load() {
        let mut s = UsageStats {
            prompt_tokens: 5000,
            predicted_tokens: 2000,
            last_prompt: 1000,
            last_predicted: 400,
            ..Default::default()
        };
        // Same sample as last_* → no change
        s.apply_prometheus(1000, 400, None);
        assert_eq!(s.prompt_tokens, 5000);
        assert_eq!(s.predicted_tokens, 2000);
        // New delta
        s.apply_prometheus(1100, 450, None);
        assert_eq!(s.prompt_tokens, 5100);
        assert_eq!(s.predicted_tokens, 2050);
    }

    #[test]
    fn reset_keeps_baseline() {
        let mut s = UsageStats::default();
        s.apply_prometheus(100, 50, None);
        s.add_cached(20);
        s.reset();
        assert_eq!(s.prompt_tokens, 0);
        assert_eq!(s.predicted_tokens, 0);
        assert_eq!(s.cached_tokens, 0);
        assert_eq!(s.last_prompt, 100);
        assert_eq!(s.last_predicted, 50);
        // Same counters again → no re-add
        s.apply_prometheus(100, 50, None);
        assert_eq!(s.prompt_tokens, 0);
    }

    #[test]
    fn compute_saved_one_million_each() {
        let luna = estimated_api_cost(1_000_000, 0, 1_000_000, &LUNA_RATES);
        let qwen = estimated_api_cost(1_000_000, 0, 1_000_000, &QWEN_RATES);
        assert!((luna - 1.40).abs() < 1e-9);
        assert!((qwen - 3.50).abs() < 1e-9);
    }

    #[test]
    fn snapshot_includes_cache_in_savings() {
        let s = UsageStats {
            prompt_tokens: 500_000,
            cached_tokens: 500_000,
            predicted_tokens: 0,
            ..Default::default()
        };
        let snap = s.snapshot();
        // 500k processed at $0.20/M + 500k cached at $0.02/M
        assert!((snap.saved_luna_usd - 0.11).abs() < 1e-9);
        assert!((snap.cache_hit_ratio - 0.5).abs() < 1e-9);
        assert!((snap.cache_reuse_ratio - 0.5).abs() < 1e-9);
        assert_eq!(snap.total_prompt_tokens, 1_000_000);
    }

    #[test]
    fn cache_dedup_within_window() {
        let mut s = UsageStats::default();
        assert!(s.add_cached(42));
        assert!(!s.add_cached(42));
        assert_eq!(s.cached_tokens, 42);
        assert!(s.add_cached(43));
        assert_eq!(s.cached_tokens, 85);
    }

    #[test]
    fn parse_cache_n_from_json_line() {
        let line = r#"slot update_slots: id  0 | task 1 | prompt processing progress, n_past = 128, n_tokens = 32, progress = 0.25, cache_n = ignored"#;
        // JSON style
        let json = r#"{"timings":{"prompt_n":100,"cache_n":256,"predicted_n":50}}"#;
        assert_eq!(parse_cache_n(json), Some(256));
        assert_eq!(parse_cache_n(line), None);
    }

    #[test]
    fn test_parse_cache_n_from_sse() {
        let body = "data: {\"choices\":[]}\n\ndata: {\"timings\":{\"cache_n\":128,\"prompt_n\":20}}\n\ndata: [DONE]\n";
        let mut last = None;
        for line in body.lines() {
            let data = line.strip_prefix("data:").unwrap_or(line).trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            if let Some(n) = parse_cache_n(data) {
                last = Some(n);
            }
        }
        assert_eq!(last, Some(128));
    }

    #[test]
    fn load_save_roundtrip() {
        let dir = std::env::temp_dir().join(format!("llama-monitor-usage-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("usage-stats.json");

        let mut s = UsageStats::default();
        s.apply_prometheus(10000, 5000, None);
        s.add_cached(100);
        save_usage_stats(&path, &s).unwrap();

        let loaded = load_usage_stats(&path);
        assert_eq!(loaded.prompt_tokens, 10000);
        assert_eq!(loaded.predicted_tokens, 5000);
        assert_eq!(loaded.cached_tokens, 100);
        assert_eq!(loaded.last_prompt, 10000);
        assert_eq!(loaded.last_predicted, 5000);

        // First sample after restart with same counters must not double
        let mut loaded = loaded;
        loaded.apply_prometheus(10000, 5000, None);
        assert_eq!(loaded.prompt_tokens, 10000);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn maybe_save_debounce() {
        let dir =
            std::env::temp_dir().join(format!("llama-monitor-usage-db-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("usage-stats.json");

        let mut s = UsageStats::default();
        s.apply_prometheus(1, 1, None);
        s.maybe_save(&path, true).unwrap();
        assert!(!s.dirty);

        // Corrupt file then force-save again
        {
            let mut f = std::fs::File::create(&path).unwrap();
            write!(f, "broken").unwrap();
        }
        s.apply_prometheus(2, 2, None);
        s.maybe_save(&path, true).unwrap();
        let loaded = load_usage_stats(&path);
        assert_eq!(loaded.prompt_tokens, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_server_totals_and_cache_reuse_ratio() {
        let mut s = UsageStats::default();
        s.apply_prometheus(251_342, 71_788, Some(4_657_250));
        assert_eq!(s.prompt_tokens, 251_342);
        assert_eq!(s.cached_tokens, 4_657_250);
        assert_eq!(s.predicted_tokens, 71_788);
        let snap = s.snapshot();
        assert_eq!(snap.total_prompt_tokens, 4_908_592);
        assert!((snap.cache_reuse_ratio - 4_657_250.0 / 4_908_592.0).abs() < 1e-12);
        assert!((snap.cache_reuse_ratio - 0.9488).abs() < 0.0001);
        assert!((snap.cache_hit_ratio - snap.cache_reuse_ratio).abs() < 1e-12);
    }

    #[test]
    fn native_cached_counter_delta() {
        let mut s = UsageStats::default();
        s.apply_prometheus(100, 10, Some(1_000));
        assert_eq!(s.cached_tokens, 1_000);
        s.apply_prometheus(120, 20, Some(1_250));
        assert_eq!(s.cached_tokens, 1_250);
        assert_eq!(s.last_cached, 1_250);
        assert!(s.native_cached_counter);
    }

    #[test]
    fn native_cached_counter_server_restart_no_underflow() {
        let mut s = UsageStats::default();
        s.apply_prometheus(1_000, 500, Some(4_000));
        s.apply_prometheus(10, 5, Some(80));
        assert_eq!(s.prompt_tokens, 1_010);
        assert_eq!(s.predicted_tokens, 505);
        assert_eq!(s.cached_tokens, 4_080);
        assert_eq!(s.last_cached, 80);
    }

    #[test]
    fn native_cached_delta_when_prompt_unchanged() {
        let mut s = UsageStats::default();
        s.apply_prometheus(100, 50, Some(10));
        s.apply_prometheus(100, 50, Some(40));
        assert_eq!(s.prompt_tokens, 100);
        assert_eq!(s.predicted_tokens, 50);
        assert_eq!(s.cached_tokens, 40);
    }

    #[test]
    fn native_cached_counter_blocks_parse_cache_n() {
        let mut s = UsageStats::default();
        s.apply_prometheus(100, 10, Some(500));
        assert!(!s.add_cached(500));
        assert!(!s.add_cached(12));
        assert_eq!(s.cached_tokens, 500);
    }

    #[test]
    fn legacy_parse_cache_n_when_native_counter_absent() {
        let mut s = UsageStats::default();
        s.apply_prometheus(100, 10, None);
        assert!(!s.native_cached_counter);
        assert!(s.add_cached(256));
        assert_eq!(s.cached_tokens, 256);
        s.apply_prometheus(110, 12, None);
        assert_eq!(s.cached_tokens, 256);
        assert!(!s.native_cached_counter);
    }

    #[test]
    fn native_cached_survives_monitor_reload() {
        let dir =
            std::env::temp_dir().join(format!("llama-monitor-usage-cached-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("usage-stats.json");

        let mut s = UsageStats::default();
        s.apply_prometheus(251_342, 71_788, Some(4_657_250));
        save_usage_stats(&path, &s).unwrap();

        let mut loaded = load_usage_stats(&path);
        assert_eq!(loaded.cached_tokens, 4_657_250);
        assert_eq!(loaded.last_cached, 4_657_250);
        assert!(loaded.native_cached_counter);
        loaded.apply_prometheus(251_342, 71_788, Some(4_657_250));
        assert_eq!(loaded.cached_tokens, 4_657_250);
        assert!(!loaded.add_cached(99));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_keeps_native_cached_baseline() {
        let mut s = UsageStats::default();
        s.apply_prometheus(100, 50, Some(20));
        s.reset();
        assert_eq!(s.cached_tokens, 0);
        assert_eq!(s.last_cached, 20);
        assert!(s.native_cached_counter);
        s.apply_prometheus(100, 50, Some(20));
        assert_eq!(s.cached_tokens, 0);
        assert!(!s.add_cached(7));
    }

    #[test]
    fn estimated_api_cost_cached_falls_back_to_input_rate() {
        let rates = ProviderRates {
            input_per_m: 2.0,
            cached_input_per_m: None,
            output_per_m: 4.0,
        };
        let cost = estimated_api_cost(1_000_000, 1_000_000, 1_000_000, &rates);
        assert!((cost - 8.0).abs() < 1e-9);
    }

    #[test]
    fn estimated_api_cost_uses_cached_input_rate_when_present() {
        let rates = ProviderRates {
            input_per_m: 2.0,
            cached_input_per_m: Some(0.2),
            output_per_m: 4.0,
        };
        let cost = estimated_api_cost(1_000_000, 1_000_000, 1_000_000, &rates);
        assert!((cost - 6.2).abs() < 1e-9);
    }

    #[test]
    fn saved_equals_estimated_api_cost_without_energy() {
        let s = UsageStats {
            prompt_tokens: 251_342,
            cached_tokens: 4_657_250,
            predicted_tokens: 71_788,
            ..Default::default()
        };
        let snap = s.snapshot();
        let luna = estimated_api_cost(
            s.prompt_tokens,
            s.cached_tokens,
            s.predicted_tokens,
            &LUNA_RATES,
        );
        let qwen = estimated_api_cost(
            s.prompt_tokens,
            s.cached_tokens,
            s.predicted_tokens,
            &QWEN_RATES,
        );
        assert!((snap.saved_luna_usd - luna).abs() < 1e-12);
        assert!((snap.saved_qwen_usd - qwen).abs() < 1e-12);
        assert_eq!(LUNA_RATES.cached_input_per_m, Some(0.02));
        assert_eq!(QWEN_RATES.cached_input_per_m, Some(0.10));
    }

    #[test]
    fn live_server_fixture_saved_uses_official_cached_input_rates() {
        let s = UsageStats {
            prompt_tokens: 251_342,
            cached_tokens: 4_657_250,
            predicted_tokens: 71_788,
            ..Default::default()
        };
        let snap = s.snapshot();
        let luna = 251_342.0 / 1_000_000.0 * 0.20
            + 4_657_250.0 / 1_000_000.0 * 0.02
            + 71_788.0 / 1_000_000.0 * 1.20;
        let qwen = 251_342.0 / 1_000_000.0 * 0.50
            + 4_657_250.0 / 1_000_000.0 * 0.10
            + 71_788.0 / 1_000_000.0 * 3.00;
        assert!((luna - 0.229559_f64).abs() < 1e-6);
        assert!((qwen - 0.806760_f64).abs() < 1e-6);
        assert!((snap.saved_luna_usd - luna).abs() < 1e-12);
        assert!((snap.saved_qwen_usd - qwen).abs() < 1e-12);
        // UI rounds to cents: $0.23 / $0.81
        assert_eq!(format!("${:.2}", snap.saved_luna_usd), "$0.23");
        assert_eq!(format!("${:.2}", snap.saved_qwen_usd), "$0.81");
        let billed_cached_as_input = 251_342.0 / 1_000_000.0 * 0.20
            + 4_657_250.0 / 1_000_000.0 * 0.20
            + 71_788.0 / 1_000_000.0 * 1.20;
        assert!(snap.saved_luna_usd < billed_cached_as_input - 0.5);
    }

    #[test]
    fn large_token_counters_total_prompt_and_cost_are_u64() {
        let mut s = UsageStats::default();
        s.apply_prometheus(5_000_000_000, 3_000_000_000, Some(12_000_000_000));
        assert_eq!(s.prompt_tokens, 5_000_000_000);
        assert_eq!(s.cached_tokens, 12_000_000_000);
        assert_eq!(s.predicted_tokens, 3_000_000_000);
        let snap = s.snapshot();
        assert_eq!(snap.total_prompt_tokens, 17_000_000_000);
        assert_eq!(snap.prompt_tokens, 5_000_000_000);
        assert_eq!(snap.cached_tokens, 12_000_000_000);
        assert_eq!(snap.predicted_tokens, 3_000_000_000);
        assert!(snap.total_prompt_tokens > u32::MAX as u64);

        let luna = estimated_api_cost(5_000_000_000, 12_000_000_000, 3_000_000_000, &LUNA_RATES);
        // Convert to f64 only for the cost formula; counters stay u64.
        let expected = 5_000_000_000.0 / 1_000_000.0 * 0.20
            + 12_000_000_000.0 / 1_000_000.0 * 0.02
            + 3_000_000_000.0 / 1_000_000.0 * 1.20;
        assert!((luna - expected).abs() < 1e-9);
        assert!((luna - 4_840.0).abs() < 1e-9);
        assert!((snap.saved_luna_usd - luna).abs() < 1e-12);
    }

    #[test]
    fn token_counter_just_above_u32_does_not_wrap() {
        let mut s = UsageStats::default();
        s.apply_prometheus(4_294_967_296, 4_294_967_296, Some(4_294_967_296));
        assert_eq!(s.prompt_tokens, 4_294_967_296);
        assert_eq!(s.predicted_tokens, 4_294_967_296);
        assert_eq!(s.cached_tokens, 4_294_967_296);
        assert_ne!(s.prompt_tokens, 0);
        assert!(s.prompt_tokens as i64 > 0);
        let snap = s.snapshot();
        assert_eq!(snap.total_prompt_tokens, 8_589_934_592);
        let cost = estimated_api_cost(4_294_967_296, 4_294_967_296, 4_294_967_296, &LUNA_RATES);
        let expected = 4_294_967_296.0 / 1_000_000.0 * (0.20 + 0.02 + 1.20);
        assert!((cost - expected).abs() < 1e-6);
        assert!(cost > 0.0);
    }

    #[test]
    fn large_counters_serde_and_persistence_round_trip() {
        let dir =
            std::env::temp_dir().join(format!("llama-monitor-usage-u64-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("usage-stats.json");

        let mut s = UsageStats::default();
        s.apply_prometheus(5_000_000_000, 3_000_000_000, Some(12_000_000_000));
        let json = serde_json::to_string(&s).unwrap();
        let decoded: UsageStats = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.prompt_tokens, 5_000_000_000);
        assert_eq!(decoded.cached_tokens, 12_000_000_000);
        assert_eq!(decoded.predicted_tokens, 3_000_000_000);
        assert_eq!(decoded.last_prompt, 5_000_000_000);
        assert_eq!(decoded.last_cached, 12_000_000_000);

        save_usage_stats(&path, &s).unwrap();
        let loaded = load_usage_stats(&path);
        assert_eq!(loaded.prompt_tokens, 5_000_000_000);
        assert_eq!(loaded.cached_tokens, 12_000_000_000);
        assert_eq!(loaded.predicted_tokens, 3_000_000_000);
        assert_eq!(loaded.snapshot().total_prompt_tokens, 17_000_000_000);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn saturating_add_does_not_panic_near_u64_max() {
        let mut s = UsageStats {
            prompt_tokens: u64::MAX - 10,
            cached_tokens: u64::MAX - 5,
            ..Default::default()
        };
        s.apply_prometheus(100, 0, None);
        assert_eq!(s.prompt_tokens, u64::MAX);
        s.add_cached(50);
        assert_eq!(s.cached_tokens, u64::MAX);
    }

    fn usage_sample(
        cached: Option<u64>,
        peak: Option<u64>,
        draft: Option<u64>,
        accepted: Option<u64>,
    ) -> PrometheusUsageSample {
        PrometheusUsageSample {
            cached,
            peak_context: peak,
            mtp_draft: draft,
            mtp_accepted: accepted,
            ..Default::default()
        }
    }

    // A
    #[test]
    fn cached_native_counter_increments_by_delta() {
        let mut s = UsageStats {
            cached_tokens: 1000,
            last_cached: 1000,
            native_cached_counter: true,
            ..Default::default()
        };
        s.apply_sample(&usage_sample(Some(1300), None, None, None));
        assert_eq!(s.cached_tokens, 1300);
        assert_eq!(s.last_cached, 1300);
    }

    // B
    #[test]
    fn cached_native_counter_reset_adds_current() {
        let mut s = UsageStats {
            cached_tokens: 1000,
            last_cached: 1000,
            native_cached_counter: true,
            ..Default::default()
        };
        s.apply_sample(&usage_sample(Some(200), None, None, None));
        assert_eq!(s.cached_tokens, 1200);
        assert_eq!(s.last_cached, 200);
    }

    // C
    #[test]
    fn native_cached_counter_wins_over_legacy_parse_cache_n() {
        let mut s = UsageStats::default();
        s.apply_sample(&usage_sample(Some(926_031), None, None, None));
        assert!(!s.add_cached(926_031));
        assert!(!s.add_cached(12_345));
        assert_eq!(s.cached_tokens, 926_031);
    }

    // D
    #[test]
    fn scientific_notation_cached_total_applies_as_u64() {
        let parsed = crate::llama::metrics::parse_prometheus_metrics(
            "llamacpp:prompt_tokens_cached_total 4.65725e+06\n",
        );
        assert_eq!(parsed.prompt_tokens_cached_total, Some(4_657_250));
        let mut s = UsageStats::default();
        s.apply_sample(&usage_sample(
            parsed.prompt_tokens_cached_total,
            None,
            None,
            None,
        ));
        assert_eq!(s.cached_tokens, 4_657_250);
    }

    // E
    #[test]
    fn peak_context_persists_high_water_mark() {
        let mut s = UsageStats {
            peak_context_tokens: 58_818,
            ..Default::default()
        };
        s.apply_sample(&usage_sample(None, Some(0), None, None));
        assert_eq!(s.peak_context_tokens, 58_818);
        s.apply_sample(&usage_sample(None, Some(42_000), None, None));
        assert_eq!(s.peak_context_tokens, 58_818);
        s.apply_sample(&usage_sample(None, Some(91_000), None, None));
        assert_eq!(s.peak_context_tokens, 91_000);
    }

    // F
    #[test]
    fn mtp_normal_delta() {
        let mut s = UsageStats {
            mtp_draft_tokens: 1000,
            mtp_accepted_tokens: 800,
            last_mtp_draft: 1000,
            last_mtp_accepted: 800,
            ..Default::default()
        };
        s.apply_sample(&usage_sample(None, None, Some(1200), Some(950)));
        assert_eq!(s.mtp_draft_tokens, 1200);
        assert_eq!(s.mtp_accepted_tokens, 950);
        assert_eq!(s.last_mtp_draft, 1200);
        assert_eq!(s.last_mtp_accepted, 950);
    }

    // G
    #[test]
    fn mtp_reset_after_llama_restart() {
        let mut s = UsageStats {
            mtp_draft_tokens: 3198,
            mtp_accepted_tokens: 2644,
            last_mtp_draft: 3198,
            last_mtp_accepted: 2644,
            ..Default::default()
        };
        s.apply_sample(&usage_sample(None, None, Some(100), Some(80)));
        assert_eq!(s.mtp_draft_tokens, 3298);
        assert_eq!(s.mtp_accepted_tokens, 2724);
        assert_eq!(s.last_mtp_draft, 100);
        assert_eq!(s.last_mtp_accepted, 80);
    }

    // H
    #[test]
    fn lifetime_mtp_percentage() {
        let s = UsageStats {
            mtp_accepted_tokens: 2644,
            mtp_draft_tokens: 3198,
            ..Default::default()
        };
        let ratio = s.mtp_acceptance_ratio().unwrap();
        assert!((ratio - 2644.0 / 3198.0).abs() < 1e-12);
        assert!((ratio * 100.0 - 82.6767).abs() < 0.0001);
        let snap = s.snapshot();
        assert_eq!(snap.mtp_accepted_tokens, 2644);
        assert_eq!(snap.mtp_draft_tokens, 3198);
        assert!((snap.mtp_acceptance_ratio.unwrap() - ratio).abs() < 1e-12);
    }

    #[test]
    fn lifetime_mtp_percentage_none_when_no_drafts() {
        let s = UsageStats {
            mtp_accepted_tokens: 0,
            mtp_draft_tokens: 0,
            ..Default::default()
        };
        assert!(s.mtp_acceptance_ratio().is_none());
        assert!(s.snapshot().mtp_acceptance_ratio.is_none());
    }

    // I
    #[test]
    fn old_usage_json_without_new_fields_loads() {
        let json = r#"{
            "prompt_tokens": 100,
            "predicted_tokens": 50,
            "cached_tokens": 20,
            "last_prompt": 100,
            "last_predicted": 50,
            "updated_at": 1
        }"#;
        let s: UsageStats = serde_json::from_str(json).unwrap();
        assert_eq!(s.prompt_tokens, 100);
        assert_eq!(s.predicted_tokens, 50);
        assert_eq!(s.cached_tokens, 20);
        assert_eq!(s.peak_context_tokens, 0);
        assert_eq!(s.mtp_draft_tokens, 0);
        assert_eq!(s.mtp_accepted_tokens, 0);
        assert_eq!(s.last_mtp_draft, 0);
        assert_eq!(s.last_mtp_accepted, 0);
        assert!(!s.native_cached_counter);
        assert_eq!(s.last_cached, 0);
    }

    // J
    #[test]
    fn monitor_restart_without_llama_restart_does_not_double_count() {
        let dir = std::env::temp_dir().join(format!(
            "llama-monitor-usage-no-double-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("usage-stats.json");

        let mut s = UsageStats::default();
        s.apply_sample(&PrometheusUsageSample {
            prompt: 806_400,
            predicted: 386_100,
            cached: Some(926_031),
            peak_context: Some(58_818),
            mtp_draft: Some(3198),
            mtp_accepted: Some(2644),
        });
        save_usage_stats(&path, &s).unwrap();

        let mut loaded = load_usage_stats(&path);
        assert_eq!(loaded.cached_tokens, 926_031);
        assert_eq!(loaded.last_cached, 926_031);
        assert_eq!(loaded.prompt_tokens, 806_400);
        assert_eq!(loaded.last_prompt, 806_400);
        assert_eq!(loaded.predicted_tokens, 386_100);
        assert_eq!(loaded.last_predicted, 386_100);
        assert_eq!(loaded.peak_context_tokens, 58_818);
        assert_eq!(loaded.mtp_draft_tokens, 3198);
        assert_eq!(loaded.mtp_accepted_tokens, 2644);
        assert_eq!(loaded.last_mtp_draft, 3198);
        assert_eq!(loaded.last_mtp_accepted, 2644);

        loaded.apply_sample(&PrometheusUsageSample {
            prompt: 806_400,
            predicted: 386_100,
            cached: Some(926_031),
            peak_context: Some(58_818),
            mtp_draft: Some(3198),
            mtp_accepted: Some(2644),
        });
        assert_eq!(loaded.cached_tokens, 926_031);
        assert_eq!(loaded.prompt_tokens, 806_400);
        assert_eq!(loaded.predicted_tokens, 386_100);
        assert_eq!(loaded.peak_context_tokens, 58_818);
        assert_eq!(loaded.mtp_draft_tokens, 3198);
        assert_eq!(loaded.mtp_accepted_tokens, 2644);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // K
    #[test]
    fn llama_restart_continues_cached_lifetime() {
        let mut s = UsageStats {
            cached_tokens: 926_031,
            last_cached: 926_031,
            native_cached_counter: true,
            peak_context_tokens: 58_818,
            mtp_draft_tokens: 3198,
            mtp_accepted_tokens: 2644,
            last_mtp_draft: 3198,
            last_mtp_accepted: 2644,
            ..Default::default()
        };
        s.apply_sample(&PrometheusUsageSample {
            cached: Some(0),
            peak_context: Some(0),
            mtp_draft: Some(0),
            mtp_accepted: Some(0),
            ..Default::default()
        });
        assert_eq!(s.cached_tokens, 926_031);
        assert_eq!(s.peak_context_tokens, 58_818);
        assert_eq!(s.mtp_draft_tokens, 3198);
        assert_eq!(s.mtp_accepted_tokens, 2644);

        s.apply_sample(&PrometheusUsageSample {
            cached: Some(300_000),
            peak_context: Some(42_000),
            mtp_draft: Some(100),
            mtp_accepted: Some(80),
            ..Default::default()
        });
        assert_eq!(s.cached_tokens, 1_226_031);
        assert_eq!(s.peak_context_tokens, 58_818);
        assert_eq!(s.mtp_draft_tokens, 3298);
        assert_eq!(s.mtp_accepted_tokens, 2724);
    }

    // L
    #[test]
    fn lifetime_peak_and_mtp_above_u32_round_trip() {
        let dir = std::env::temp_dir().join(format!(
            "llama-monitor-usage-peak-mtp-u64-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("usage-stats.json");

        let mut s = UsageStats::default();
        s.apply_sample(&PrometheusUsageSample {
            prompt: 5_000_000_000,
            predicted: 3_000_000_000,
            cached: Some(12_000_000_000),
            peak_context: Some(4_294_967_296),
            mtp_draft: Some(4_294_967_296),
            mtp_accepted: Some(5_000_000_000),
        });
        assert_eq!(s.prompt_tokens, 5_000_000_000);
        assert_eq!(s.cached_tokens, 12_000_000_000);
        assert_eq!(s.predicted_tokens, 3_000_000_000);
        assert_eq!(s.peak_context_tokens, 4_294_967_296);
        assert_eq!(s.mtp_draft_tokens, 4_294_967_296);
        assert_eq!(s.mtp_accepted_tokens, 5_000_000_000);

        save_usage_stats(&path, &s).unwrap();
        let loaded = load_usage_stats(&path);
        assert_eq!(loaded.peak_context_tokens, 4_294_967_296);
        assert_eq!(loaded.mtp_draft_tokens, 4_294_967_296);
        assert_eq!(loaded.mtp_accepted_tokens, 5_000_000_000);
        let snap = loaded.snapshot();
        assert_eq!(snap.peak_context_tokens, 4_294_967_296);
        assert_eq!(snap.mtp_draft_tokens, 4_294_967_296);
        assert_eq!(snap.mtp_accepted_tokens, 5_000_000_000);
        let json = serde_json::to_value(&snap).unwrap();
        assert_eq!(json["peak_context_tokens"].as_u64(), Some(4_294_967_296));
        assert_eq!(json["mtp_draft_tokens"].as_u64(), Some(4_294_967_296));
        assert_eq!(json["mtp_accepted_tokens"].as_u64(), Some(5_000_000_000));
        assert!(json["peak_context_tokens"].as_u64().unwrap() > u32::MAX as u64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reset_clears_peak_and_mtp_keeps_baselines() {
        let mut s = UsageStats::default();
        s.apply_sample(&PrometheusUsageSample {
            prompt: 100,
            predicted: 50,
            cached: Some(20),
            peak_context: Some(58_818),
            mtp_draft: Some(3198),
            mtp_accepted: Some(2644),
        });
        s.reset();
        assert_eq!(s.peak_context_tokens, 0);
        assert_eq!(s.mtp_draft_tokens, 0);
        assert_eq!(s.mtp_accepted_tokens, 0);
        assert_eq!(s.last_mtp_draft, 3198);
        assert_eq!(s.last_mtp_accepted, 2644);
        s.apply_sample(&PrometheusUsageSample {
            prompt: 100,
            predicted: 50,
            cached: Some(20),
            peak_context: Some(58_818),
            mtp_draft: Some(3198),
            mtp_accepted: Some(2644),
        });
        assert_eq!(s.mtp_draft_tokens, 0);
        assert_eq!(s.mtp_accepted_tokens, 0);
        assert_eq!(s.peak_context_tokens, 58_818);
    }
}
