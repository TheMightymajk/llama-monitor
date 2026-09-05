use std::time::Duration;

use crate::state::AppState;

use super::metrics::{parse_prometheus_metrics, prometheus_f64_to_u64};
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
            {
                let mut m = state.llama_metrics.lock().unwrap();
                m.clear_metrics_gauges();
                m.clear_slots_gauges();
                m.status.clear();
            }
            tokio::time::sleep(LLAMA_POLL_INTERVAL).await;
            continue;
        }

        // Poll /metrics — failures must not abort the rest of the loop
        let metrics_ok = if let Ok(resp) = client.get(format!("{base}/metrics")).send().await
            && resp.status().is_success()
            && let Ok(body) = resp.text().await
        {
            let prom = parse_prometheus_metrics(&body);
            {
                let mut m = state.llama_metrics.lock().unwrap();
                m.apply_metrics(&prom);
            }
            {
                let mut usage = state.usage.lock().unwrap();
                usage.apply_prometheus(
                    prometheus_f64_to_u64(prom.prompt_tokens_total).unwrap_or(0),
                    prometheus_f64_to_u64(prom.predicted_tokens_total).unwrap_or(0),
                    prom.prompt_tokens_cached_total,
                );
                let _ = usage.maybe_save(&state.usage_path, false);
            }
            true
        } else {
            false
        };
        if !metrics_ok {
            state.llama_metrics.lock().unwrap().clear_metrics_gauges();
        }

        // Poll /slots — live KV occupancy + slot busy flags
        let slots_ok = if let Ok(resp) = client.get(format!("{base}/slots")).send().await
            && resp.status().is_success()
            && let Ok(body) = resp.text().await
            && let Ok(slots) = serde_json::from_str::<Vec<serde_json::Value>>(&body)
        {
            state.llama_metrics.lock().unwrap().apply_slots(&slots);
            true
        } else {
            false
        };
        if !slots_ok {
            state.llama_metrics.lock().unwrap().clear_slots_gauges();
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
