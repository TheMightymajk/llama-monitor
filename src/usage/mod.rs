use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;

/// OpenAI GPT-5.6 Luna rates (USD per 1M tokens), Sep 2026.
pub const LUNA_INPUT_PER_M: f64 = 0.20;
pub const LUNA_OUTPUT_PER_M: f64 = 1.20;

/// Alibaba Model Studio Qwen3.8-27B Beijing rates (USD per 1M tokens), Sep 2026.
pub const QWEN_INPUT_PER_M: f64 = 0.424;
pub const QWEN_OUTPUT_PER_M: f64 = 1.696;

const CACHE_DEDUP_WINDOW: Duration = Duration::from_secs(2);
const SAVE_DEBOUNCE: Duration = Duration::from_secs(5);

/// Persisted lifetime counters + Prometheus baselines.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UsageStats {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub predicted_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
    /// Last seen llama-server Prometheus prompt counter (session baseline).
    #[serde(default)]
    pub last_prompt: u64,
    /// Last seen llama-server Prometheus predicted counter (session baseline).
    #[serde(default)]
    pub last_predicted: u64,
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
    pub cache_hit_ratio: f64,
    pub saved_luna_usd: f64,
    pub saved_qwen_usd: f64,
    pub rates: UsageRatesInfo,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UsageRatesInfo {
    pub luna_input_per_m: f64,
    pub luna_output_per_m: f64,
    pub qwen_input_per_m: f64,
    pub qwen_output_per_m: f64,
    pub label: &'static str,
}

impl UsageStats {
    pub fn snapshot(&self) -> UsageSnapshot {
        let prompt_equiv = self.prompt_tokens.saturating_add(self.cached_tokens);
        let denom = prompt_equiv;
        let cache_hit_ratio = if denom > 0 {
            self.cached_tokens as f64 / denom as f64
        } else {
            0.0
        };
        UsageSnapshot {
            prompt_tokens: self.prompt_tokens,
            predicted_tokens: self.predicted_tokens,
            cached_tokens: self.cached_tokens,
            cache_hit_ratio,
            saved_luna_usd: compute_saved(
                prompt_equiv,
                self.predicted_tokens,
                LUNA_INPUT_PER_M,
                LUNA_OUTPUT_PER_M,
            ),
            saved_qwen_usd: compute_saved(
                prompt_equiv,
                self.predicted_tokens,
                QWEN_INPUT_PER_M,
                QWEN_OUTPUT_PER_M,
            ),
            rates: UsageRatesInfo {
                luna_input_per_m: LUNA_INPUT_PER_M,
                luna_output_per_m: LUNA_OUTPUT_PER_M,
                qwen_input_per_m: QWEN_INPUT_PER_M,
                qwen_output_per_m: QWEN_OUTPUT_PER_M,
                label: "Sep 2026 · GPT-5.6 Luna / Qwen3.8-27B Alibaba Beijing",
            },
        }
    }

    /// Fold a Prometheus sample into lifetime totals.
    ///
    /// If `current >= last` → add delta. If `current < last` (llama-server
    /// restarted) → treat `current` as a fresh session and add it.
    pub fn apply_prometheus(&mut self, prompt: u64, predicted: u64) {
        let prompt_delta = if prompt >= self.last_prompt {
            prompt - self.last_prompt
        } else {
            prompt
        };
        let predicted_delta = if predicted >= self.last_predicted {
            predicted - self.last_predicted
        } else {
            predicted
        };

        if prompt_delta == 0 && predicted_delta == 0 {
            self.last_prompt = prompt;
            self.last_predicted = predicted;
            return;
        }

        self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_delta);
        self.predicted_tokens = self.predicted_tokens.saturating_add(predicted_delta);
        self.last_prompt = prompt;
        self.last_predicted = predicted;
        self.touch();
    }

    /// Add KV prefix-cache hits. Deduplicates identical `cache_n` within 2s
    /// (log + chat proxy may both report the same request).
    pub fn add_cached(&mut self, n: u64) -> bool {
        if n == 0 {
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
        // Keep last_* so the next Prometheus sample does not re-add the
        // current llama-server session totals.
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

pub fn compute_saved(
    prompt_equiv: u64,
    predicted: u64,
    input_per_m: f64,
    output_per_m: f64,
) -> f64 {
    (prompt_equiv as f64) * input_per_m / 1_000_000.0
        + (predicted as f64) * output_per_m / 1_000_000.0
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
        s.apply_prometheus(100, 50);
        assert_eq!(s.prompt_tokens, 100);
        assert_eq!(s.predicted_tokens, 50);
        assert_eq!(s.last_prompt, 100);
        assert_eq!(s.last_predicted, 50);

        s.apply_prometheus(150, 80);
        assert_eq!(s.prompt_tokens, 150);
        assert_eq!(s.predicted_tokens, 80);
    }

    #[test]
    fn apply_prometheus_server_restart() {
        let mut s = UsageStats::default();
        s.apply_prometheus(1000, 500);
        // llama-server restarted → counters drop
        s.apply_prometheus(10, 5);
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
        s.apply_prometheus(1000, 400);
        assert_eq!(s.prompt_tokens, 5000);
        assert_eq!(s.predicted_tokens, 2000);
        // New delta
        s.apply_prometheus(1100, 450);
        assert_eq!(s.prompt_tokens, 5100);
        assert_eq!(s.predicted_tokens, 2050);
    }

    #[test]
    fn reset_keeps_baseline() {
        let mut s = UsageStats::default();
        s.apply_prometheus(100, 50);
        s.add_cached(20);
        s.reset();
        assert_eq!(s.prompt_tokens, 0);
        assert_eq!(s.predicted_tokens, 0);
        assert_eq!(s.cached_tokens, 0);
        assert_eq!(s.last_prompt, 100);
        assert_eq!(s.last_predicted, 50);
        // Same counters again → no re-add
        s.apply_prometheus(100, 50);
        assert_eq!(s.prompt_tokens, 0);
    }

    #[test]
    fn compute_saved_one_million_each() {
        let luna = compute_saved(1_000_000, 1_000_000, LUNA_INPUT_PER_M, LUNA_OUTPUT_PER_M);
        let qwen = compute_saved(1_000_000, 1_000_000, QWEN_INPUT_PER_M, QWEN_OUTPUT_PER_M);
        assert!((luna - 1.40).abs() < 1e-9);
        assert!((qwen - 2.12).abs() < 1e-9);
    }

    #[test]
    fn snapshot_includes_cache_in_savings() {
        let mut s = UsageStats::default();
        s.prompt_tokens = 500_000;
        s.cached_tokens = 500_000;
        s.predicted_tokens = 0;
        let snap = s.snapshot();
        assert!((snap.saved_luna_usd - 0.20).abs() < 1e-9);
        assert!((snap.cache_hit_ratio - 0.5).abs() < 1e-9);
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
        s.apply_prometheus(10000, 5000);
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
        loaded.apply_prometheus(10000, 5000);
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
        s.apply_prometheus(1, 1);
        s.maybe_save(&path, true).unwrap();
        assert!(!s.dirty);

        // Corrupt file then force-save again
        {
            let mut f = std::fs::File::create(&path).unwrap();
            write!(f, "broken").unwrap();
        }
        s.apply_prometheus(2, 2);
        s.maybe_save(&path, true).unwrap();
        let loaded = load_usage_stats(&path);
        assert_eq!(loaded.prompt_tokens, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
