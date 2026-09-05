use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use warp::Filter;
use warp::ws::{Message, Ws};

use crate::logs::LogSourceInfo;
use crate::state::AppState;

const WS_PUSH_INTERVAL: Duration = Duration::from_millis(500);

pub fn ws_route(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let ws_state = state;
    warp::path("ws").and(warp::ws()).map(move |ws: Ws| {
        let state = ws_state.clone();
        ws.on_upgrade(move |socket| {
            let state = state.clone();
            async move {
                let (mut ws_tx, mut ws_rx) = socket.split();

                let update_task = tokio::spawn(async move {
                    let mut interval = tokio::time::interval(WS_PUSH_INTERVAL);
                    loop {
                        interval.tick().await;
                        let json = {
                            let gpu = state.gpu_metrics.lock().unwrap().clone();
                            let llama = state.llama_metrics.lock().unwrap().clone();
                            let logs = state.log_buffer.lock().unwrap().snapshot();
                            let log_source = state.log_source.lock().unwrap().clone();
                            let running = *state.server_running.lock().unwrap();
                            let started_at = *state.server_started_at.lock().unwrap();
                            let usage = state.usage.lock().unwrap().snapshot();
                            let running_model = state.running_model.lock().unwrap().clone();
                            let energy = state
                                .energy
                                .lock()
                                .unwrap()
                                .snapshot(std::time::Instant::now(), chrono::Local::now());
                            build_ws_payload(
                                &gpu,
                                &llama,
                                &logs,
                                &log_source,
                                running,
                                started_at,
                                &usage,
                                &running_model,
                                &energy,
                            )
                            .to_string()
                        };
                        if ws_tx.send(Message::text(&json)).await.is_err() {
                            break;
                        }
                    }
                });

                while let Some(_msg) = ws_rx.next().await {}
                update_task.abort();
            }
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_ws_payload(
    gpu: &Option<std::collections::BTreeMap<String, crate::gpu::GpuMetrics>>,
    llama: &crate::llama::metrics::LlamaMetrics,
    logs: &[String],
    log_source: &LogSourceInfo,
    server_running: bool,
    server_started_at: Option<u64>,
    usage: &crate::usage::UsageSnapshot,
    running_model: &crate::llama::running_model::RunningModelInfo,
    energy: &crate::energy::EnergySnapshot,
) -> serde_json::Value {
    serde_json::json!({
        "gpu": gpu,
        "llama": llama,
        "logs": logs,
        "log_source": log_source,
        "server_running": server_running,
        "server_started_at": server_started_at,
        "usage": usage,
        "running_model": running_model,
        "energy": energy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::metrics::LlamaMetrics;
    use crate::logs::LogSourceInfo;

    fn empty_usage() -> crate::usage::UsageSnapshot {
        crate::usage::UsageStats::default().snapshot()
    }

    fn empty_running_model() -> crate::llama::running_model::RunningModelInfo {
        crate::llama::running_model::RunningModelInfo::default()
    }

    fn empty_energy() -> crate::energy::EnergySnapshot {
        crate::energy::EnergyState::new_default()
            .snapshot(std::time::Instant::now(), chrono::Local::now())
    }

    #[test]
    fn ws_payload_includes_started_at_when_running() {
        let payload = build_ws_payload(
            &None,
            &LlamaMetrics::default(),
            &[],
            &LogSourceInfo::default(),
            true,
            Some(1_700_000_000),
            &empty_usage(),
            &empty_running_model(),
            &empty_energy(),
        );
        assert_eq!(payload["server_running"], true);
        assert_eq!(payload["server_started_at"], 1_700_000_000);
        assert!(payload.get("log_source").is_some());
        assert_eq!(payload["log_source"]["kind"], "none");
        assert_eq!(payload["running_model"]["detected"], false);
        assert_eq!(payload["energy"]["currency"], "PLN");
        assert_eq!(payload["energy"]["available"], false);
        assert!(payload["gpu"].is_null());
        assert!(payload["llama"]["prompt_tokens_per_sec"].is_null());
        assert!(payload["llama"]["kv_cache_tokens"].is_null());
        assert!(payload["llama"]["generation_tokens_per_sec"].is_null());
    }

    #[test]
    fn ws_payload_null_started_at_when_stopped() {
        let payload = build_ws_payload(
            &None,
            &LlamaMetrics::default(),
            &["line".into()],
            &LogSourceInfo::default(),
            false,
            None,
            &empty_usage(),
            &empty_running_model(),
            &empty_energy(),
        );
        assert_eq!(payload["server_running"], false);
        assert!(payload["server_started_at"].is_null());
        assert_eq!(payload["logs"][0], "line");
    }

    #[test]
    fn ws_payload_exposes_lifetime_and_mtp_diagnostics() {
        let mut usage = crate::usage::UsageStats::default();
        usage.apply_prometheus(251_342, 71_788, Some(4_657_250));
        let snap = usage.snapshot();
        let mut llama = LlamaMetrics::default();
        llama.apply_metrics(&crate::llama::metrics::parse_prometheus_metrics(
            include_str!("../../tests/fixtures/prometheus_metrics_live_server.txt"),
        ));

        let payload = build_ws_payload(
            &None,
            &llama,
            &[],
            &LogSourceInfo::default(),
            true,
            Some(1),
            &snap,
            &empty_running_model(),
            &empty_energy(),
        );
        assert_eq!(payload["usage"]["prompt_tokens"], 251_342);
        assert_eq!(payload["usage"]["cached_tokens"], 4_657_250);
        assert_eq!(payload["usage"]["total_prompt_tokens"], 4_908_592);
        assert_eq!(payload["usage"]["predicted_tokens"], 71_788);
        assert!(payload["usage"]["cache_reuse_ratio"].as_f64().unwrap() > 0.94);
        assert_eq!(payload["llama"]["n_tokens_max"], 113_868);
        assert_eq!(payload["llama"]["spec_accepted_tokens"], 31_710);
        assert_eq!(payload["llama"]["spec_draft_tokens"], 40_018);
        assert!(payload["llama"]["spec_acceptance_ratio"].as_f64().unwrap() > 0.79);
        assert_eq!(payload["llama"]["prompt_tokens_per_sec"], 0.0);
        assert!(payload["llama"]["kv_cache_tokens"].is_null());
    }
}
