mod cli;
mod config;
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
use std::thread;
use std::time::Duration;

const GPU_POLL_INTERVAL: Duration = Duration::from_millis(500);

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
        external_log_path,
        app_config.external_log_file.clone(),
    );

    if let Some(ref dir) = app_config.models_dir {
        let count = state.discovered_models.lock().unwrap().len();
        println!("[info] Discovered {count} models in {}", dir.display());
    }

    // Detect and start GPU poller
    let backend = gpu::detect_backend(&app_config.gpu_backend);
    {
        let gpu = state.gpu_metrics.clone();
        thread::spawn(move || {
            loop {
                match backend.read_metrics() {
                    Ok(m) => *gpu.lock().unwrap() = m,
                    Err(e) => eprintln!("[error] GPU metrics: {e}"),
                }
                thread::sleep(GPU_POLL_INTERVAL);
            }
        });
    }

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
    let routes = web::build_routes(state, app_config);

    println!("[info] Llama Monitor running on http://{host}:{port}");
    warp::serve(routes).run((host, port)).await;

    Ok(())
}
