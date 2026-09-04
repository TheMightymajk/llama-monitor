mod cli;
mod config;
#[allow(dead_code)]
mod energy;
mod gpu;
mod llama;
mod logs;
mod models;
mod presets;
mod state;
mod usage;
mod web;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const GPU_POLL_INTERVAL: Duration = Duration::from_millis(500);
const ENERGY_SAVE_INTERVAL: Duration = Duration::from_secs(30);
const SAVE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
const SERVER_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

#[tokio::main]
async fn main() -> Result<()> {
    let args = cli::AppArgs::parse();
    let app_config = Arc::new(config::AppConfig::from_args(args));

    // Load presets from disk (or defaults)
    let initial_presets = presets::load_presets(&app_config.presets_file);
    println!(
        "[info] Loaded {} presets from {}",
        initial_presets.len(),
        app_config.presets_file.display()
    );

    // Load GPU environment config
    let mut gpu_env = gpu::env::load_gpu_env(&app_config.gpu_env_file);

    // CLI overrides take precedence
    if let Some(ref arch) = app_config.gpu_arch_override {
        gpu_env.arch = arch.clone();
    }
    if let Some(ref devices) = app_config.gpu_devices_override {
        gpu_env.devices = devices.clone();
    }

    // Auto-detect GPUs and log results
    if let Some(detected) = gpu::env::detect_gpus() {
        println!(
            "[info] Detected {}x {} GPU(s)",
            detected.count, detected.arch
        );
        // If arch is "auto" and devices is empty, suggest detected values
        if gpu_env.arch == "auto" && gpu_env.devices.is_empty() {
            gpu_env.devices = gpu::env::device_list_for_count(detected.count);
        }
    }

    println!(
        "[info] GPU env: arch={}, devices={}",
        gpu_env.arch,
        if gpu_env.devices.is_empty() {
            "all"
        } else {
            &gpu_env.devices
        }
    );

    // Load UI settings from disk (or defaults)
    let mut ui_settings = state::load_ui_settings(&app_config.ui_settings_file);

    // Seed UI external log from CLI when UI has none (persisted UI still wins later).
    if ui_settings.external_log_file.trim().is_empty()
        && let Some(ref cli_log) = app_config.external_log_file
    {
        ui_settings.external_log_file = cli_log.to_string_lossy().into_owned();
    }

    let external_log_path = state::AppState::resolve_external_log_path(
        &ui_settings,
        app_config.external_log_file.as_deref(),
    );
    if let Some(ref p) = external_log_path {
        println!("[info] External log file: {}", p.display());
    }

    // Load lifetime usage counters
    let usage = usage::load_usage_stats(&app_config.usage_stats_file);
    println!(
        "[info] Usage stats: {} prompt / {} predicted / {} cached tokens from {}",
        usage.prompt_tokens,
        usage.predicted_tokens,
        usage.cached_tokens,
        app_config.usage_stats_file.display()
    );

    let (mut energy_state, energy_warning) =
        energy::load_energy_state(&app_config.energy_stats_file);
    energy_state.set_save_warning(energy_warning.clone());
    if let Some(warning) = energy_warning {
        eprintln!("[warn] {warning}");
    }
    let local_now = chrono::Local::now();
    println!(
        "[info] Energy history: {} (timezone {}, local date {})",
        app_config.energy_stats_file.display(),
        local_now.offset(),
        local_now.date_naive()
    );

    let state = state::AppState::new(
        initial_presets,
        app_config.presets_file.clone(),
        app_config.models_dir.clone(),
        gpu_env,
        app_config.gpu_env_file.clone(),
        ui_settings,
        app_config.ui_settings_file.clone(),
        usage,
        app_config.usage_stats_file.clone(),
        energy_state,
        app_config.energy_stats_file.clone(),
        external_log_path,
        app_config.external_log_file.clone(),
    );

    if let Some(ref dir) = app_config.models_dir {
        let count = state.discovered_models.lock().unwrap().len();
        println!("[info] Discovered {count} models in {}", dir.display());
    }

    // Detect and start GPU poller
    let backend = gpu::detect_backend(&app_config.gpu_backend);
    let ingest_enabled = Arc::new(AtomicBool::new(true));
    {
        let gpu = state.gpu_metrics.clone();
        let llama_metrics = state.llama_metrics.clone();
        let llama_reachable = state.llama_reachable.clone();
        let energy = state.energy.clone();
        let ingest_enabled = ingest_enabled.clone();
        thread::spawn(move || {
            // Runtime lock order: gpu_metrics -> llama_metrics/health -> energy.
            loop {
                if !ingest_enabled.load(Ordering::SeqCst) {
                    thread::sleep(GPU_POLL_INTERVAL);
                    continue;
                }
                match backend.read_metrics() {
                    Ok(metrics) => {
                        let samples: Vec<_> = metrics
                            .iter()
                            .map(|(id, metrics)| energy::GpuPowerSample {
                                id: id.clone(),
                                power_w: metrics.power_consumption,
                                utilization: metrics.load as f32,
                            })
                            .collect();
                        *gpu.lock().unwrap() = metrics;
                        let busy = {
                            let llama = llama_metrics.lock().unwrap();
                            let health_ok = *llama_reachable.lock().unwrap();
                            energy::BusyFlags {
                                requests_processing: llama.requests_processing,
                                slots_processing: llama.slots_processing,
                                health_ok,
                            }
                        };
                        energy.lock().unwrap().ingest(
                            &samples,
                            busy,
                            Instant::now(),
                            chrono::Local::now(),
                        );
                    }
                    Err(e) => eprintln!("[error] GPU metrics: {e}"),
                }
                thread::sleep(GPU_POLL_INTERVAL);
            }
        });
    }

    let energy_save_task = {
        let energy = state.energy.clone();
        let path = state.energy_path.clone();
        let save_gate = state.energy_save_gate.clone();
        let ingest_enabled = ingest_enabled.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(ENERGY_SAVE_INTERVAL);
            loop {
                interval.tick().await;
                if !ingest_enabled.load(Ordering::SeqCst) {
                    break;
                }
                if let Err(error) = energy::save_with_gate(&energy, &path, &save_gate).await {
                    eprintln!("[warn] Energy history save failed: {error}");
                }
            }
        })
    };

    // Llama metrics poller
    {
        let s = state.clone();
        tokio::spawn(async move { llama::poller::llama_metrics_poller(s).await });
    }

    // External log file follower (tail -F)
    {
        let s = state.clone();
        tokio::spawn(async move { logs::external_log_poller(s).await });
    }

    let port = app_config.port;
    let host = config::parse_bind_ip(&app_config.host);
    let routes = web::build_routes(state.clone(), app_config);

    println!("[info] Llama Monitor running on http://{host}:{port}");
    let shutdown_state = state.clone();
    let shutdown_ingest_enabled = ingest_enabled.clone();
    let shutdown_save_gate = state.energy_save_gate.clone();
    let (_, server) = warp::serve(routes).bind_with_graceful_shutdown((host, port), async move {
        wait_for_shutdown_signal().await;
        println!("[info] Shutdown requested; saving energy history");

        shutdown_ingest_enabled.store(false, Ordering::SeqCst);
        shutdown_state
            .energy
            .lock()
            .unwrap()
            .set_ingest_enabled(false);
        energy_save_task.abort();

        let gate_available =
            tokio::time::timeout(SAVE_SHUTDOWN_TIMEOUT, shutdown_save_gate.lock()).await;
        match gate_available {
            Ok(guard) => {
                drop(guard);
                if let Err(error) = energy::force_save(
                    &shutdown_state.energy,
                    &shutdown_state.energy_path,
                    &shutdown_save_gate,
                )
                .await
                {
                    eprintln!("[error] Final energy history save failed: {error}");
                }
            }
            Err(_) => {
                eprintln!(
                    "[error] Timed out waiting for an in-flight energy save; skipping final save"
                );
            }
        }
        println!("[info] Energy shutdown sequence complete");
    });
    match tokio::time::timeout(SERVER_DRAIN_TIMEOUT, server).await {
        Ok(()) => {}
        Err(_) => {
            eprintln!("[warn] Timed out waiting for HTTP server drain; exiting");
        }
    }
    std::process::exit(0);
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                if let Err(error) = result {
                    eprintln!("[warn] SIGINT handler failed: {error}");
                }
            }
            _ = terminate.recv() => {}
        }
    }

    #[cfg(not(unix))]
    if let Err(error) = tokio::signal::ctrl_c().await {
        eprintln!("[warn] shutdown signal handler failed: {error}");
    }
}
