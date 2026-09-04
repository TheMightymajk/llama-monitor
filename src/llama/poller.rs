use std::time::Duration;

use crate::state::AppState;

use super::metrics::parse_prometheus_metrics;
use super::running_model::{
    HealthStickiness, ModelDiscoveryPartial, merge_running_model, parse_props_json,
    parse_v1_models_json,
};

const LLAMA_POLL_INTERVAL: Duration = Duration::from_secs(1);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(3);
const HEALTH_FAIL_CLEAR_THRESHOLD: u32 = 3;

pub async fn llama_metrics_poller(state: AppState) {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .unwrap();

    let mut health_stick = HealthStickiness::new(HEALTH_FAIL_CLEAR_THRESHOLD);

    loop {
        // Determine port: from config if started via UI, else default 8080
        let port = state
            .server_config
            .lock()
            .unwrap()
            .as_ref()
            .map(|c| c.port)
            .unwrap_or(8080);

        let base = format!("http://127.0.0.1:{port}");

        // Poll /health first to detect if any server is reachable
        let server_reachable = if let Ok(resp) = client.get(format!("{base}/health")).send().await {
            if let Ok(body) = resp.text().await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    let mut m = state.llama_metrics.lock().unwrap();
                    m.status = json
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };
        *state.llama_reachable.lock().unwrap() = server_reachable;

        if health_stick.on_health_result(server_reachable) {
            {
                let mut m = state.llama_metrics.lock().unwrap();
                *m = super::metrics::LlamaMetrics::default();
            }
            {
                let mut rm = state.running_model.lock().unwrap();
                rm.clear();
            }
        }

        if !server_reachable {
            tokio::time::sleep(LLAMA_POLL_INTERVAL).await;
            continue;
        }

        // Poll /metrics — failures must not abort the rest of the loop
        if let Ok(resp) = client.get(format!("{base}/metrics")).send().await
            && let Ok(body) = resp.text().await
        {
            let prom = parse_prometheus_metrics(&body);

            let prompt_tps = if prom.prompt_tokens_per_sec > 0.0 {
                prom.prompt_tokens_per_sec
            } else if prom.prompt_seconds_total > 0.0 {
                prom.prompt_tokens_total / prom.prompt_seconds_total
            } else {
                0.0
            };

            let gen_tps = if prom.predicted_tokens_per_sec > 0.0 {
                prom.predicted_tokens_per_sec
            } else if prom.predicted_seconds_total > 0.0 {
                prom.predicted_tokens_total / prom.predicted_seconds_total
            } else {
                0.0
            };

            let prompt_total = prom.prompt_tokens_total as u64;
            let predicted_total = prom.predicted_tokens_total as u64;

            {
                let mut m = state.llama_metrics.lock().unwrap();
                m.prompt_tokens_per_sec = prompt_tps;
                m.generation_tokens_per_sec = gen_tps;
                m.prompt_tokens_total = prompt_total;
                m.predicted_tokens_total = predicted_total;
                m.kv_cache_tokens = prom.n_tokens_max;
                m.requests_processing = prom.requests_processing;
            }

            {
                let mut usage = state.usage.lock().unwrap();
                usage.apply_prometheus(prompt_total, predicted_total);
                let _ = usage.maybe_save(&state.usage_path, false);
            }
        }

        // Poll /slots — get per-slot processing state + total context
        if let Ok(resp) = client.get(format!("{base}/slots")).send().await
            && let Ok(body) = resp.text().await
            && let Ok(slots) = serde_json::from_str::<Vec<serde_json::Value>>(&body)
        {
            let mut idle = 0u32;
            let mut processing = 0u32;
            let num_slots = slots.len() as u64;
            let mut per_slot_ctx = 0u64;
            for slot in &slots {
                if slot
                    .get("is_processing")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    processing += 1;
                } else {
                    idle += 1;
                }
                if let Some(n) = slot.get("n_ctx").and_then(|v| v.as_u64()) {
                    per_slot_ctx = n;
                }
            }
            let mut m = state.llama_metrics.lock().unwrap();
            m.slots_idle = idle;
            m.slots_processing = processing;
            if per_slot_ctx > 0 {
                m.kv_cache_max = per_slot_ctx * num_slots;
            }
        }

        // Discover running model: /props then /v1/models (errors are non-fatal)
        refresh_running_model(&client, &base, &state).await;

        tokio::time::sleep(LLAMA_POLL_INTERVAL).await;
    }
}

async fn refresh_running_model(client: &reqwest::Client, base: &str, state: &AppState) {
    let mut props_partial: Option<ModelDiscoveryPartial> = None;
    let mut models_partial: Option<ModelDiscoveryPartial> = None;

    match client.get(format!("{base}/props")).send().await {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(body) = resp.text().await {
                match parse_props_json(&body) {
                    Ok(p) => props_partial = Some(p),
                    Err(e) => eprintln!("[warn] /props parse: {e}"),
                }
            }
        }
        Ok(_) => {}
        Err(e) => eprintln!("[warn] /props request: {e}"),
    }

    match client.get(format!("{base}/v1/models")).send().await {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(body) = resp.text().await {
                match parse_v1_models_json(&body) {
                    Ok(p) => models_partial = Some(p),
                    Err(e) => eprintln!("[warn] /v1/models parse: {e}"),
                }
            }
        }
        Ok(_) => {}
        Err(e) => eprintln!("[warn] /v1/models request: {e}"),
    }

    // If neither live endpoint worked, keep last good RunningModelInfo (sticky).
    if props_partial.is_none() && models_partial.is_none() {
        return;
    }

    let process = state.server_config.lock().unwrap().clone();

    // Preset fallback only when live sources left gaps — still recorded with source=preset.
    let (preset_name, preset_path, preset_ctx) = {
        let ui = state.ui_settings.lock().unwrap();
        let presets = state.presets.lock().unwrap();
        let preset = presets.iter().find(|p| p.id == ui.preset_id);
        (
            preset.map(|p| p.name.clone()),
            preset.map(|p| p.model_path.clone()),
            preset.map(|p| p.context_size),
        )
    };

    let info = merge_running_model(
        props_partial.as_ref(),
        models_partial.as_ref(),
        process.as_ref(),
        preset_name.as_deref(),
        preset_path.as_deref(),
        preset_ctx,
    );

    *state.running_model.lock().unwrap() = info;
}
