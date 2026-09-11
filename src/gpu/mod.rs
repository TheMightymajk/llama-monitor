pub mod amdgpu;
pub mod dummy;
pub mod env;
pub mod nvidia;
pub mod rocm;

pub use amdgpu::RepeatErrorLimiter;

use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// How a watt reading was obtained. Average is not the same as instantaneous draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerKind {
    #[default]
    Current,
    Average,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GpuMetrics {
    pub temp: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_edge: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_junction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_memory: Option<f32>,
    pub load: u32,
    /// Instantaneous or average watts; `None` when the tool omitted power.
    pub power_consumption: Option<f32>,
    pub power_kind: PowerKind,
    pub power_limit: u32,
    pub vram_used: u64,
    pub vram_total: u64,
    pub sclk_mhz: u32,
    pub mclk_mhz: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fan_rpm: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fan_percent: Option<u32>,
}

/// Successful poll replaces live GPU gauges. A failed poll must not keep the
/// previous sample: `None` means "unavailable", not "no GPUs".
pub fn apply_gpu_poll_result(
    store: &mut Option<BTreeMap<String, GpuMetrics>>,
    result: Result<BTreeMap<String, GpuMetrics>>,
) -> Result<BTreeMap<String, GpuMetrics>> {
    match result {
        Ok(metrics) => {
            *store = Some(metrics.clone());
            Ok(metrics)
        }
        Err(e) => {
            *store = None;
            Err(e)
        }
    }
}

pub trait GpuBackend: Send + Sync + 'static {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>>;
    #[allow(dead_code)]
    fn name(&self) -> &str;
}

/// Polls several backends and merges their metrics into one map, so machines
/// with a mix of vendors (e.g. AMD + NVIDIA cards) report every GPU. A failure
/// in one backend is logged and skipped without hiding the others.
pub struct MultiBackend {
    backends: Vec<Arc<dyn GpuBackend>>,
}

impl GpuBackend for MultiBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        let mut all = BTreeMap::new();
        let mut any_ok = false;
        let mut last_err: Option<anyhow::Error> = None;
        for backend in &self.backends {
            match backend.read_metrics() {
                Ok(metrics) => {
                    any_ok = true;
                    all.extend(metrics);
                }
                Err(e) => {
                    eprintln!("[error] GPU metrics ({}): {e}", backend.name());
                    last_err = Some(e);
                }
            }
        }
        if !any_ok && let Some(e) = last_err {
            return Err(e);
        }
        Ok(all)
    }

    fn name(&self) -> &str {
        "multi"
    }
}

pub fn detect_backend(force: &str, devices: &str) -> Arc<dyn GpuBackend> {
    detect_backend_ex(force, Path::new("/sys"), command_exists, devices)
}

/// Select a telemetry backend. `vulkan` is an alias of canonical `amdgpu`.
/// Explicit `amdgpu` never inspects or launches external processes.
pub fn detect_backend_ex(
    force: &str,
    sysfs_root: &Path,
    command_exists: fn(&str) -> bool,
    devices: &str,
) -> Arc<dyn GpuBackend> {
    match canonicalize_gpu_backend(force) {
        "rocm" => Arc::new(rocm::RocmBackend),
        "nvidia" => Arc::new(nvidia::NvidiaBackend),
        "none" => Arc::new(dummy::DummyBackend),
        "amdgpu" => Arc::new(amdgpu::AmdgpuBackend::with_sysfs_and_devices(
            sysfs_root, devices,
        )),
        _ => {
            let mut backends: Vec<Arc<dyn GpuBackend>> = Vec::new();
            if command_exists("nvidia-smi") {
                backends.push(Arc::new(nvidia::NvidiaBackend));
            }
            if command_exists("rocm-smi") {
                backends.push(Arc::new(rocm::RocmBackend));
            } else if backends.is_empty() && amdgpu::amdgpu_sysfs_available(sysfs_root) {
                backends.push(Arc::new(amdgpu::AmdgpuBackend::with_sysfs_and_devices(
                    sysfs_root, devices,
                )));
            }
            match backends.len() {
                0 => {
                    eprintln!(
                        "[warn] No GPU telemetry backend found (nvidia-smi / rocm-smi / AMDGPU sysfs)"
                    );
                    Arc::new(dummy::DummyBackend)
                }
                1 => backends.into_iter().next().unwrap(),
                _ => Arc::new(MultiBackend { backends }),
            }
        }
    }
}

pub fn canonicalize_gpu_backend(force: &str) -> &str {
    match force {
        "vulkan" => "amdgpu",
        other => other,
    }
}

fn command_exists(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubBackend {
        name: &'static str,
        cards: Vec<&'static str>,
        fail: bool,
    }

    impl GpuBackend for StubBackend {
        fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
            if self.fail {
                anyhow::bail!("stub failure");
            }
            Ok(self
                .cards
                .iter()
                .map(|c| {
                    (
                        c.to_string(),
                        GpuMetrics {
                            power_consumption: Some(0.0),
                            ..GpuMetrics::default()
                        },
                    )
                })
                .collect())
        }

        fn name(&self) -> &str {
            self.name
        }
    }

    #[test]
    fn multi_backend_merges_all_vendors() {
        let multi = MultiBackend {
            backends: vec![
                Arc::new(StubBackend {
                    name: "rocm",
                    cards: vec!["card0", "card1"],
                    fail: false,
                }),
                Arc::new(StubBackend {
                    name: "nvidia",
                    cards: vec!["GPU0 NVIDIA"],
                    fail: false,
                }),
            ],
        };
        let metrics = multi.read_metrics().unwrap();
        assert_eq!(metrics.len(), 3);
        assert!(metrics.contains_key("card0"));
        assert!(metrics.contains_key("card1"));
        assert!(metrics.contains_key("GPU0 NVIDIA"));
    }

    #[test]
    fn multi_backend_skips_failing_backend() {
        let multi = MultiBackend {
            backends: vec![
                Arc::new(StubBackend {
                    name: "rocm",
                    cards: vec!["card0"],
                    fail: true,
                }),
                Arc::new(StubBackend {
                    name: "nvidia",
                    cards: vec!["GPU0 NVIDIA"],
                    fail: false,
                }),
            ],
        };
        let metrics = multi.read_metrics().unwrap();
        assert_eq!(metrics.len(), 1);
        assert!(metrics.contains_key("GPU0 NVIDIA"));
    }

    #[test]
    fn multi_backend_all_fail_is_error() {
        let multi = MultiBackend {
            backends: vec![Arc::new(StubBackend {
                name: "rocm",
                cards: vec!["card0"],
                fail: true,
            })],
        };
        assert!(multi.read_metrics().is_err());
    }

    #[test]
    fn gpu_poll_error_clears_previous_sample() {
        let sample = BTreeMap::from([(
            "card0".to_string(),
            GpuMetrics {
                temp: 72.0,
                load: 97,
                power_consumption: Some(280.0),
                power_kind: PowerKind::Current,
                power_limit: 300,
                vram_used: 28000,
                vram_total: 32000,
                sclk_mhz: 2000,
                mclk_mhz: 1000,
                ..GpuMetrics::default()
            },
        )]);
        let mut store = Some(sample.clone());
        apply_gpu_poll_result(&mut store, Ok(sample)).unwrap();
        assert!(store.is_some());
        let err = apply_gpu_poll_result(&mut store, Err(anyhow::anyhow!("rocm-smi failed")));
        assert!(err.is_err());
        assert!(store.is_none());
    }

    fn no_commands(_: &str) -> bool {
        false
    }

    fn panic_on_command(cmd: &str) -> bool {
        panic!("explicit amdgpu must not launch {cmd}");
    }

    fn only_rocm(cmd: &str) -> bool {
        cmd == "rocm-smi"
    }

    fn only_nvidia(cmd: &str) -> bool {
        cmd == "nvidia-smi"
    }

    fn write_amd_sysfs(root: &Path) {
        let device = root.join("class/drm/card0/device");
        std::fs::create_dir_all(&device).unwrap();
        std::fs::write(device.join("vendor"), "0x1002\n").unwrap();
        std::fs::write(device.join("gpu_busy_percent"), "1\n").unwrap();
    }

    #[test]
    fn auto_selects_amdgpu_when_rocm_smi_missing() {
        let root =
            std::env::temp_dir().join(format!("llama-monitor-auto-amd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_amd_sysfs(&root);
        let backend = detect_backend_ex("auto", &root, no_commands, "");
        assert_eq!(backend.name(), "amdgpu");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn explicit_amdgpu_never_runs_external_process() {
        let root =
            std::env::temp_dir().join(format!("llama-monitor-explicit-amd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_amd_sysfs(&root);
        let backend = detect_backend_ex("amdgpu", &root, panic_on_command, "");
        assert_eq!(backend.name(), "amdgpu");
        assert!(backend.read_metrics().is_ok());
        let vulkan = detect_backend_ex("vulkan", &root, panic_on_command, "");
        assert_eq!(vulkan.name(), "amdgpu");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn auto_prefers_rocm_when_tool_exists() {
        let root =
            std::env::temp_dir().join(format!("llama-monitor-auto-rocm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_amd_sysfs(&root);
        let backend = detect_backend_ex("auto", &root, only_rocm, "");
        assert_eq!(backend.name(), "rocm");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn auto_prefers_nvidia_when_nvidia_smi_exists() {
        let root =
            std::env::temp_dir().join(format!("llama-monitor-auto-nv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_amd_sysfs(&root);
        let backend = detect_backend_ex("auto", &root, only_nvidia, "");
        assert_eq!(backend.name(), "nvidia");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn canonicalize_vulkan_alias() {
        assert_eq!(canonicalize_gpu_backend("vulkan"), "amdgpu");
        assert_eq!(canonicalize_gpu_backend("amdgpu"), "amdgpu");
    }
}
