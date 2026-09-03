use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use warp::Filter;
use warp::ws::{Message, Ws};

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
                            let logs: Vec<String> =
                                state.server_logs.lock().unwrap().iter().cloned().collect();
                            let running = *state.server_running.lock().unwrap();
                            let started_at = *state.server_started_at.lock().unwrap();
                            build_ws_payload(&gpu, &llama, &logs, running, started_at).to_string()
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

pub fn build_ws_payload(
    gpu: &std::collections::BTreeMap<String, crate::gpu::GpuMetrics>,
    llama: &crate::llama::metrics::LlamaMetrics,
    logs: &[String],
    server_running: bool,
    server_started_at: Option<u64>,
) -> serde_json::Value {
    serde_json::json!({
        "gpu": gpu,
        "llama": llama,
        "logs": logs,
        "server_running": server_running,
        "server_started_at": server_started_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::metrics::LlamaMetrics;
    use std::collections::BTreeMap;

    #[test]
    fn ws_payload_includes_started_at_when_running() {
        let payload = build_ws_payload(
            &BTreeMap::new(),
            &LlamaMetrics::default(),
            &[],
            true,
            Some(1_700_000_000),
        );
        assert_eq!(payload["server_running"], true);
        assert_eq!(payload["server_started_at"], 1_700_000_000);
        assert!(payload.get("gpu").is_some());
        assert!(payload.get("llama").is_some());
        assert!(payload.get("logs").is_some());
    }

    #[test]
    fn ws_payload_null_started_at_when_stopped() {
        let payload = build_ws_payload(
            &BTreeMap::new(),
            &LlamaMetrics::default(),
            &["line".into()],
            false,
            None,
        );
        assert_eq!(payload["server_running"], false);
        assert!(payload["server_started_at"].is_null());
        assert_eq!(payload["logs"][0], "line");
    }
}
