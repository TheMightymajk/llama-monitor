use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::energy::{EnergyState, SaveGate};
use crate::gpu::GpuMetrics;
use crate::gpu::env::GpuEnv;
use crate::llama::metrics::LlamaMetrics;
use crate::llama::running_model::RunningModelInfo;
use crate::llama::server::ServerConfig;
use crate::logs::{
    LogBuffer, LogSourceInfo, LogSourceKind, LogSourceStatus, MAX_LOG_LINES, expand_tilde,
};
use crate::models::DiscoveredModel;
use crate::presets::ModelPreset;
use crate::usage::UsageStats;

/// Persisted UI control-bar settings (survives page reload).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UiSettings {
    #[serde(default)]
    pub preset_id: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub llama_server_path: String,
    #[serde(default)]
    pub llama_server_cwd: String,
    #[serde(default)]
    pub models_dir: String,
    /// Optional external llama-server log file (`~/` expanded). When set, Logs
    /// follow this file instead of the managed process stdout/stderr.
    #[serde(default)]
    pub external_log_file: String,
}

fn default_port() -> u16 {
    8080
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            preset_id: String::new(),
            port: 8080,
            llama_server_path: String::new(),
            llama_server_cwd: String::new(),
            models_dir: String::new(),
            external_log_file: String::new(),
        }
    }
}

pub fn load_ui_settings(path: &Path) -> UiSettings {
    if path.exists()
        && let Ok(contents) = std::fs::read_to_string(path)
        && let Ok(s) = serde_json::from_str::<UiSettings>(&contents)
    {
        return s;
    }
    UiSettings::default()
}

pub fn save_ui_settings(path: &Path, settings: &UiSettings) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(settings)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Clone)]
pub struct AppState {
    pub gpu_metrics: Arc<Mutex<BTreeMap<String, GpuMetrics>>>,
    pub llama_metrics: Arc<Mutex<LlamaMetrics>>,
    pub llama_reachable: Arc<Mutex<bool>>,
    pub energy: Arc<Mutex<EnergyState>>,
    pub energy_path: PathBuf,
    pub energy_save_gate: SaveGate,
    pub running_model: Arc<Mutex<RunningModelInfo>>,
    pub log_buffer: Arc<Mutex<LogBuffer>>,
    pub log_source: Arc<Mutex<LogSourceInfo>>,
    /// Effective external log path (UI overrides CLI when non-empty).
    pub external_log_path: Arc<Mutex<Option<PathBuf>>>,
    /// CLI fallback path when UI field is empty.
    pub cli_external_log_path: Option<PathBuf>,
    pub server_child: Arc<tokio::sync::Mutex<Option<tokio::process::Child>>>,
    pub server_running: Arc<Mutex<bool>>,
    pub server_started_at: Arc<Mutex<Option<u64>>>,
    pub server_config: Arc<Mutex<Option<ServerConfig>>>,
    pub llama_poll_notify: Arc<tokio::sync::Notify>,
    pub presets: Arc<Mutex<Vec<ModelPreset>>>,
    pub presets_path: PathBuf,
    pub discovered_models: Arc<Mutex<Vec<DiscoveredModel>>>,
    pub models_dir: Option<PathBuf>,
    pub gpu_env: Arc<Mutex<GpuEnv>>,
    pub gpu_env_path: PathBuf,
    pub ui_settings: Arc<Mutex<UiSettings>>,
    pub ui_settings_path: PathBuf,
    pub usage: Arc<Mutex<UsageStats>>,
    pub usage_path: PathBuf,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        presets: Vec<ModelPreset>,
        presets_path: PathBuf,
        models_dir: Option<PathBuf>,
        gpu_env: GpuEnv,
        gpu_env_path: PathBuf,
        ui_settings: UiSettings,
        ui_settings_path: PathBuf,
        usage: UsageStats,
        usage_path: PathBuf,
        energy: EnergyState,
        energy_path: PathBuf,
        external_log_path: Option<PathBuf>,
        cli_external_log_path: Option<PathBuf>,
    ) -> Self {
        let discovered = models_dir
            .as_ref()
            .and_then(|dir| crate::models::scan_models_dir(dir).ok())
            .unwrap_or_default();

        let log_source = if external_log_path.is_some() {
            LogSourceInfo {
                kind: LogSourceKind::ExternalFile,
                status: LogSourceStatus::WaitingForFile,
                file_name: external_log_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string()),
                error: None,
                line_count: 0,
            }
        } else {
            LogSourceInfo::default()
        };

        Self {
            gpu_metrics: Arc::new(Mutex::new(BTreeMap::new())),
            llama_metrics: Arc::new(Mutex::new(LlamaMetrics::default())),
            llama_reachable: Arc::new(Mutex::new(false)),
            energy: Arc::new(Mutex::new(energy)),
            energy_path,
            energy_save_gate: Arc::new(tokio::sync::Mutex::new(())),
            running_model: Arc::new(Mutex::new(RunningModelInfo::default())),
            log_buffer: Arc::new(Mutex::new(LogBuffer::new(MAX_LOG_LINES))),
            log_source: Arc::new(Mutex::new(log_source)),
            external_log_path: Arc::new(Mutex::new(external_log_path)),
            cli_external_log_path,
            server_child: Arc::new(tokio::sync::Mutex::new(None)),
            server_running: Arc::new(Mutex::new(false)),
            server_started_at: Arc::new(Mutex::new(None)),
            server_config: Arc::new(Mutex::new(None)),
            llama_poll_notify: Arc::new(tokio::sync::Notify::new()),
            presets: Arc::new(Mutex::new(presets)),
            presets_path,
            discovered_models: Arc::new(Mutex::new(discovered)),
            models_dir,
            gpu_env: Arc::new(Mutex::new(gpu_env)),
            gpu_env_path,
            ui_settings: Arc::new(Mutex::new(ui_settings)),
            ui_settings_path,
            usage: Arc::new(Mutex::new(usage)),
            usage_path,
        }
    }

    /// Whether Logs should follow an external file (excludes managed process output).
    pub fn using_external_logs(&self) -> bool {
        self.external_log_path.lock().unwrap().is_some()
    }

    pub fn push_log(&self, line: String) {
        // Cache hits from managed-process lines (and still useful if mixed tooling).
        if let Some(n) = crate::usage::parse_cache_n(&line) {
            let mut usage = self.usage.lock().unwrap();
            if usage.add_cached(n) {
                let _ = usage.maybe_save(&self.usage_path, false);
            }
        }
        // Do not mix managed stdout into the buffer when an external file is configured.
        if self.using_external_logs() {
            return;
        }
        let mut logs = self.log_buffer.lock().unwrap();
        logs.push_line(line);
        let mut src = self.log_source.lock().unwrap();
        src.kind = LogSourceKind::ManagedProcess;
        src.status = LogSourceStatus::Connected;
        src.file_name = None;
        src.error = None;
        src.line_count = logs.len();
    }

    pub fn clear_log_view(&self) {
        self.log_buffer.lock().unwrap().clear();
        let mut src = self.log_source.lock().unwrap();
        src.line_count = 0;
    }

    /// Resolve effective external log path: non-empty UI setting wins, else CLI.
    pub fn resolve_external_log_path(ui: &UiSettings, cli: Option<&Path>) -> Option<PathBuf> {
        let from_ui = ui.external_log_file.trim();
        if !from_ui.is_empty() {
            let p = expand_tilde(from_ui);
            if !p.as_os_str().is_empty() {
                return Some(p);
            }
        }
        cli.map(|p| p.to_path_buf())
            .filter(|p| !p.as_os_str().is_empty())
    }
}
