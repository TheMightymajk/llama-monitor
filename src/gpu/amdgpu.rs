//! AMDGPU sysfs telemetry (Vulkan/RADV compatible; no ROCm / rocm-smi).

use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{GpuBackend, GpuMetrics, PowerKind};

const AMD_VENDOR: u16 = 0x1002;
const MAX_HWMON_SENSORS: u32 = 16;

/// AMD GPU discovered under a sysfs tree (`/sys` in production).
#[derive(Debug, Clone)]
pub struct AmdGpuDevice {
    pub display_name: String,
    pub bdf: Option<String>,
    pub card_dir: PathBuf,
    pub device_dir: PathBuf,
    pub hwmon_dir: Option<PathBuf>,
}

pub struct AmdgpuBackend {
    sysfs_root: PathBuf,
    device_filter: Vec<usize>,
    devices: Mutex<Vec<AmdGpuDevice>>,
    errors: OnceKindLog,
}

impl AmdgpuBackend {
    #[cfg(test)]
    pub fn with_sysfs(sysfs_root: impl Into<PathBuf>) -> Self {
        Self::with_sysfs_and_devices(sysfs_root, "")
    }

    pub fn with_sysfs_and_devices(sysfs_root: impl Into<PathBuf>, devices: &str) -> Self {
        let sysfs_root = sysfs_root.into();
        let device_filter = parse_device_filter(devices);
        let discovered = discover_amd_gpus(&sysfs_root);
        let devices = apply_device_filter(discovered, &device_filter);
        Self {
            sysfs_root,
            device_filter,
            devices: Mutex::new(devices),
            errors: OnceKindLog::new(),
        }
    }

    fn snapshot_devices(&self) -> Vec<AmdGpuDevice> {
        let mut guard = self.devices.lock().unwrap();
        let lost = guard.iter().any(|d| !d.device_dir.exists());
        if lost {
            let discovered = discover_amd_gpus(&self.sysfs_root);
            *guard = apply_device_filter(discovered, &self.device_filter);
        }
        guard.clone()
    }

    fn read_trimmed(&self, path: &Path) -> Option<String> {
        match fs::read_to_string(path) {
            Ok(raw) => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => None,
            Err(err) => {
                self.errors.emit(
                    &path.display().to_string(),
                    &err.kind().to_string(),
                    &format!("AMDGPU sysfs skip {}: {err}", path.display()),
                );
                None
            }
        }
    }

    fn read_metrics_for(&self, device: &AmdGpuDevice) -> GpuMetrics {
        let mut metrics = GpuMetrics::default();

        if let Some(raw) = self.read_trimmed(&device.device_dir.join("gpu_busy_percent"))
            && let Some(load) = parse_busy_percent(&raw)
        {
            metrics.load = load;
        }

        if let Some(raw) = self.read_trimmed(&device.device_dir.join("mem_info_vram_used"))
            && let Some(bytes) = parse_u64(&raw)
        {
            metrics.vram_used = bytes_to_mib(bytes);
        }
        if let Some(raw) = self.read_trimmed(&device.device_dir.join("mem_info_vram_total"))
            && let Some(bytes) = parse_u64(&raw)
        {
            metrics.vram_total = bytes_to_mib(bytes);
        }

        if let Some(raw) = self.read_trimmed(&device.device_dir.join("pp_dpm_sclk"))
            && let Some(mhz) = parse_pp_dpm_mhz(&raw)
        {
            metrics.sclk_mhz = mhz;
        }
        if let Some(raw) = self.read_trimmed(&device.device_dir.join("pp_dpm_mclk"))
            && let Some(mhz) = parse_pp_dpm_mhz(&raw)
        {
            metrics.mclk_mhz = mhz;
        }

        if let Some(hwmon) = &device.hwmon_dir {
            self.read_hwmon(hwmon, &mut metrics);
        }

        metrics.temp = metrics
            .temp_junction
            .or(metrics.temp_edge)
            .or(metrics.temp_memory)
            .unwrap_or(0.0);

        metrics
    }

    fn read_hwmon(&self, hwmon: &Path, metrics: &mut GpuMetrics) {
        for idx in 1..=MAX_HWMON_SENSORS {
            let label_path = hwmon.join(format!("temp{idx}_label"));
            let input_path = hwmon.join(format!("temp{idx}_input"));
            let label = match self.read_trimmed(&label_path) {
                Some(label) => label,
                None => continue,
            };
            let Some(kind) = classify_temp_label(&label) else {
                continue;
            };
            let Some(raw) = self.read_trimmed(&input_path) else {
                continue;
            };
            let Some(celsius) = parse_f64(&raw).and_then(millidegrees_to_celsius) else {
                continue;
            };
            match kind {
                TempKind::Edge => metrics.temp_edge = Some(celsius),
                TempKind::Junction => metrics.temp_junction = Some(celsius),
                TempKind::Memory => metrics.temp_memory = Some(celsius),
            }
        }

        if let Some(raw) = self.read_trimmed(&hwmon.join("power1_average"))
            && let Some(watts) = power_to_watts(parse_f64(&raw).unwrap_or(f64::NAN))
        {
            metrics.power_consumption = Some(watts);
            metrics.power_kind = PowerKind::Average;
        } else if let Some(raw) = self.read_trimmed(&hwmon.join("power1_input"))
            && let Some(watts) = power_to_watts(parse_f64(&raw).unwrap_or(f64::NAN))
        {
            metrics.power_consumption = Some(watts);
            metrics.power_kind = PowerKind::Current;
        }

        if let Some(raw) = self.read_trimmed(&hwmon.join("power1_cap"))
            && let Some(watts) = power_to_watts(parse_f64(&raw).unwrap_or(f64::NAN))
        {
            metrics.power_limit = watts.round() as u32;
        }

        if let Some(raw) = self.read_trimmed(&hwmon.join("fan1_input"))
            && let Some(rpm) = parse_u32(&raw)
        {
            metrics.fan_rpm = Some(rpm);
        }
        let pwm = self
            .read_trimmed(&hwmon.join("pwm1"))
            .and_then(|raw| parse_f64(&raw));
        let pwm_max = self
            .read_trimmed(&hwmon.join("pwm1_max"))
            .and_then(|raw| parse_f64(&raw));
        if let (Some(pwm), Some(pwm_max)) = (pwm, pwm_max)
            && pwm.is_finite()
            && pwm_max.is_finite()
            && pwm_max > 0.0
            && pwm >= 0.0
        {
            let pct = (pwm / pwm_max * 100.0).round();
            if pct.is_finite() && (0.0..=100.0).contains(&pct) {
                metrics.fan_percent = Some(pct as u32);
            }
        }

        if metrics.sclk_mhz == 0
            && let Some(mhz) = self.read_labeled_freq(hwmon, &["sclk", "gfx"])
        {
            metrics.sclk_mhz = mhz;
        }
        if metrics.mclk_mhz == 0
            && let Some(mhz) = self.read_labeled_freq(hwmon, &["mclk", "mem"])
        {
            metrics.mclk_mhz = mhz;
        }
    }

    fn read_labeled_freq(&self, hwmon: &Path, wanted: &[&str]) -> Option<u32> {
        for idx in 1..=MAX_HWMON_SENSORS {
            let Some(label) = self.read_trimmed(&hwmon.join(format!("freq{idx}_label"))) else {
                continue;
            };
            let needle = label.trim().to_ascii_lowercase();
            if !wanted.iter().any(|w| needle.contains(w)) {
                continue;
            }
            let Some(raw) = self.read_trimmed(&hwmon.join(format!("freq{idx}_input"))) else {
                continue;
            };
            if let Some(mhz) = parse_f64(&raw).and_then(freq_to_mhz) {
                return Some(mhz);
            }
        }
        None
    }
}

impl GpuBackend for AmdgpuBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        let devices = self.snapshot_devices();
        let mut metrics = BTreeMap::new();
        for device in devices {
            metrics.insert(device.display_name.clone(), self.read_metrics_for(&device));
        }
        Ok(metrics)
    }

    fn name(&self) -> &str {
        "amdgpu"
    }
}

/// Scan `{sysfs_root}/class/drm/cardN/device/` for AMD GPUs. Does not spawn processes.
pub fn discover_amd_gpus(sysfs_root: &Path) -> Vec<AmdGpuDevice> {
    let drm = sysfs_root.join("class/drm");
    let entries = match fs::read_dir(&drm) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };

    let mut seen_paths = HashSet::new();
    let mut devices = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !is_drm_card_name(name) {
            continue;
        }
        let card_dir = entry.path();
        let device_link = card_dir.join("device");
        let vendor_raw = match fs::read_to_string(device_link.join("vendor")) {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        let Some(vendor) = parse_hex_id(&vendor_raw) else {
            continue;
        };
        if vendor != AMD_VENDOR {
            continue;
        }

        let device_dir = fs::canonicalize(&device_link).unwrap_or(device_link);
        if !seen_paths.insert(device_dir.clone()) {
            continue;
        }

        let bdf = read_pci_bdf(&device_dir);
        let product = fs::read_to_string(device_dir.join("product_name"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let display_name = match (product, bdf.as_deref()) {
            (Some(product), Some(bdf)) => format!("{product} ({bdf})"),
            (Some(product), None) => product,
            (None, Some(bdf)) => format!("AMD GPU ({bdf})"),
            (None, None) => name.to_string(),
        };

        devices.push(AmdGpuDevice {
            display_name,
            bdf,
            card_dir,
            hwmon_dir: find_amdgpu_hwmon(&device_dir),
            device_dir,
        });
    }

    devices.sort_by(|a, b| match (&a.bdf, &b.bdf) {
        (Some(left), Some(right)) => left.cmp(right),
        _ => a.card_dir.cmp(&b.card_dir),
    });
    devices
}

pub fn amdgpu_sysfs_available(sysfs_root: &Path) -> bool {
    !discover_amd_gpus(sysfs_root).is_empty()
}

fn find_amdgpu_hwmon(device_dir: &Path) -> Option<PathBuf> {
    let hwmon_root = device_dir.join("hwmon");
    let entries = fs::read_dir(&hwmon_root).ok()?;
    let mut matches: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|path| {
            fs::read_to_string(path.join("name"))
                .map(|name| name.trim().eq_ignore_ascii_case("amdgpu"))
                .unwrap_or(false)
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

fn read_pci_bdf(device_dir: &Path) -> Option<String> {
    if let Ok(uevent) = fs::read_to_string(device_dir.join("uevent")) {
        for line in uevent.lines() {
            if let Some(slot) = line.strip_prefix("PCI_SLOT_NAME=") {
                let slot = slot.trim();
                if is_pci_bdf(slot) {
                    return Some(slot.to_string());
                }
            }
        }
    }
    for component in device_dir.iter().rev() {
        if let Some(name) = component.to_str()
            && is_pci_bdf(name)
        {
            return Some(name.to_string());
        }
    }
    None
}

fn is_drm_card_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("card") else {
        return false;
    };
    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
}

fn is_pci_bdf(name: &str) -> bool {
    let parts: Vec<&str> = name.split(':').collect();
    if parts.len() != 3 {
        return false;
    }
    let (domain, bus, rest) = (parts[0], parts[1], parts[2]);
    let Some((slot, func)) = rest.split_once('.') else {
        return false;
    };
    domain.len() == 4
        && bus.len() == 2
        && slot.len() == 2
        && func.len() == 1
        && domain.chars().all(|c| c.is_ascii_hexdigit())
        && bus.chars().all(|c| c.is_ascii_hexdigit())
        && slot.chars().all(|c| c.is_ascii_hexdigit())
        && func.chars().all(|c| c.is_ascii_hexdigit())
}

fn parse_device_filter(devices: &str) -> Vec<usize> {
    let trimmed = devices.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut indices: Vec<usize> = trimmed
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn apply_device_filter(devices: Vec<AmdGpuDevice>, filter: &[usize]) -> Vec<AmdGpuDevice> {
    if filter.is_empty() {
        return devices;
    }
    devices
        .into_iter()
        .enumerate()
        .filter_map(|(idx, device)| filter.contains(&idx).then_some(device))
        .collect()
}

#[derive(Clone, Copy)]
enum TempKind {
    Edge,
    Junction,
    Memory,
}

fn classify_temp_label(label: &str) -> Option<TempKind> {
    match label.trim().to_ascii_lowercase().as_str() {
        "edge" => Some(TempKind::Edge),
        "junction" | "hotspot" => Some(TempKind::Junction),
        "mem" | "memory" => Some(TempKind::Memory),
        _ => None,
    }
}

pub fn millidegrees_to_celsius(raw: f64) -> Option<f32> {
    if !raw.is_finite() {
        return None;
    }
    let celsius = raw / 1000.0;
    if !celsius.is_finite() || !(-50.0..=200.0).contains(&celsius) {
        return None;
    }
    Some(celsius as f32)
}

pub fn power_to_watts(raw: f64) -> Option<f32> {
    if !raw.is_finite() || raw < 0.0 {
        return None;
    }
    let watts = if raw >= 1_000_000.0 {
        raw / 1_000_000.0
    } else if raw >= 10_000.0 {
        raw / 1_000.0
    } else {
        raw
    };
    if !watts.is_finite() || !(0.0..=2_000.0).contains(&watts) {
        return None;
    }
    Some(watts as f32)
}

pub fn parse_pp_dpm_mhz(text: &str) -> Option<u32> {
    let line = text
        .lines()
        .find(|line| line.contains('*'))
        .or_else(|| text.lines().rev().find(|line| !line.trim().is_empty()))?;
    parse_mhz_token(line)
}

fn parse_mhz_token(line: &str) -> Option<u32> {
    let lower = line.to_ascii_lowercase();
    let idx = lower.find("mhz")?;
    let prefix = line[..idx].trim_end();
    let number = prefix
        .rsplit(|c: char| !c.is_ascii_digit() && c != '.')
        .next()?;
    let mhz: f64 = number.parse().ok()?;
    if !mhz.is_finite() || !(0.0..=10_000.0).contains(&mhz) {
        return None;
    }
    Some(mhz.round() as u32)
}

fn freq_to_mhz(raw: f64) -> Option<u32> {
    if !raw.is_finite() || raw < 0.0 {
        return None;
    }
    let mhz = if raw >= 1_000_000.0 {
        raw / 1_000_000.0
    } else if raw >= 1_000.0 {
        raw / 1_000.0
    } else {
        raw
    };
    if !mhz.is_finite() || !(0.0..=10_000.0).contains(&mhz) {
        return None;
    }
    Some(mhz.round() as u32)
}

fn parse_busy_percent(raw: &str) -> Option<u32> {
    let value: f64 = parse_f64(raw)?;
    if !value.is_finite() || !(0.0..=100.0).contains(&value) {
        return None;
    }
    Some(value.round() as u32)
}

fn bytes_to_mib(bytes: u64) -> u64 {
    bytes / 1024 / 1024
}

fn parse_hex_id(raw: &str) -> Option<u16> {
    let trimmed = raw.trim().trim_start_matches("0x").trim_start_matches("0X");
    u16::from_str_radix(trimmed, 16).ok()
}

fn parse_f64(raw: &str) -> Option<f64> {
    raw.trim().parse().ok()
}

fn parse_u64(raw: &str) -> Option<u64> {
    raw.trim().parse().ok()
}

fn parse_u32(raw: &str) -> Option<u32> {
    raw.trim().parse().ok()
}

/// Log a sysfs problem once per path, and again only if the error kind changes.
struct OnceKindLog {
    seen: Mutex<HashMap<String, String>>,
}

impl OnceKindLog {
    fn new() -> Self {
        Self {
            seen: Mutex::new(HashMap::new()),
        }
    }

    fn emit(&self, key: &str, kind: &str, message: &str) {
        let mut seen = self.seen.lock().unwrap();
        if seen.get(key).map(String::as_str) == Some(kind) {
            return;
        }
        seen.insert(key.to_string(), kind.to_string());
        eprintln!("[warn] {message}");
    }
}

/// Rate-limit identical poller errors (used by the GPU poller thread).
pub struct RepeatErrorLimiter {
    last: Option<(String, Instant)>,
    interval: Duration,
}

impl RepeatErrorLimiter {
    pub fn new(interval: Duration) -> Self {
        Self {
            last: None,
            interval,
        }
    }

    pub fn should_log(&mut self, msg: &str, now: Instant) -> bool {
        match &self.last {
            None => {
                self.last = Some((msg.to_string(), now));
                true
            }
            Some((prev, at)) => {
                if prev != msg || now.saturating_duration_since(*at) >= self.interval {
                    self.last = Some((msg.to_string(), now));
                    true
                } else {
                    false
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::{BusyFlags, EnergyState, GpuPowerSample};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static SEQ: AtomicU64 = AtomicU64::new(0);

    struct TempSysfs {
        root: PathBuf,
    }

    impl TempSysfs {
        fn new() -> Self {
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir()
                .join(format!("llama-monitor-amdgpu-{}-{seq}", std::process::id()));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(root.join("class/drm")).unwrap();
            Self { root }
        }

        fn put(&self, rel: &str, contents: &str) {
            let path = self.root.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, contents).unwrap();
        }

        fn populate_r9700(&self, card: &str) {
            let base = format!("class/drm/{card}/device");
            self.put(&format!("{base}/vendor"), "0x1002\n");
            self.put(
                &format!("{base}/uevent"),
                "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:03:00.0\nPCI_ID=1002:7590\n",
            );
            self.put(&format!("{base}/product_name"), "AMD Radeon AI PRO R9700\n");
            self.put(&format!("{base}/gpu_busy_percent"), "42\n");
            self.put(&format!("{base}/mem_info_vram_used"), "16106127360\n");
            self.put(&format!("{base}/mem_info_vram_total"), "34359738368\n");
            self.put(&format!("{base}/mem_info_gtt_used"), "1024\n");
            self.put(&format!("{base}/mem_info_gtt_total"), "2048\n");
            self.put(&format!("{base}/pp_dpm_sclk"), "0: 500Mhz \n1: 2979Mhz *\n");
            self.put(&format!("{base}/pp_dpm_mclk"), "0: 456Mhz \n1: 1500Mhz *\n");
            self.put(&format!("{base}/hwmon/hwmon0/name"), "nvme\n");
            self.put(&format!("{base}/hwmon/hwmon2/name"), "amdgpu\n");
            self.put(&format!("{base}/hwmon/hwmon2/temp1_label"), "edge\n");
            self.put(&format!("{base}/hwmon/hwmon2/temp1_input"), "85000\n");
            self.put(&format!("{base}/hwmon/hwmon2/temp2_label"), "junction\n");
            self.put(&format!("{base}/hwmon/hwmon2/temp2_input"), "102000\n");
            self.put(&format!("{base}/hwmon/hwmon2/temp3_label"), "mem\n");
            self.put(&format!("{base}/hwmon/hwmon2/temp3_input"), "90000\n");
            self.put(
                &format!("{base}/hwmon/hwmon2/power1_average"),
                "299000000\n",
            );
            self.put(&format!("{base}/hwmon/hwmon2/power1_cap"), "400000000\n");
            self.put(&format!("{base}/hwmon/hwmon2/fan1_input"), "1200\n");
            self.put(&format!("{base}/hwmon/hwmon2/pwm1"), "128\n");
            self.put(&format!("{base}/hwmon/hwmon2/pwm1_max"), "255\n");
        }
    }

    impl Drop for TempSysfs {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn first_card(metrics: &BTreeMap<String, GpuMetrics>) -> GpuMetrics {
        metrics.values().next().expect("expected one GPU").clone()
    }

    #[test]
    fn millidegrees_102000_is_102c() {
        assert!((millidegrees_to_celsius(102_000.0).unwrap() - 102.0).abs() < f32::EPSILON);
    }

    #[test]
    fn power_microwatts_to_watts() {
        assert!((power_to_watts(299_000_000.0).unwrap() - 299.0).abs() < 0.01);
    }

    #[test]
    fn power_rejects_negative_and_non_finite() {
        assert!(power_to_watts(-1.0).is_none());
        assert!(power_to_watts(f64::NAN).is_none());
        assert!(power_to_watts(f64::INFINITY).is_none());
        assert!(power_to_watts(5_000_000_000.0).is_none());
    }

    #[test]
    fn parse_starred_sclk_line() {
        assert_eq!(
            parse_pp_dpm_mhz("0: 500Mhz \n1: 2979Mhz *\n").unwrap(),
            2979
        );
    }

    #[test]
    fn discovers_one_amd_card() {
        let sys = TempSysfs::new();
        sys.populate_r9700("card1");
        let devices = discover_amd_gpus(&sys.root);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].bdf.as_deref(), Some("0000:03:00.0"));
        assert!(devices[0].display_name.contains("R9700"));
        assert!(devices[0].hwmon_dir.as_ref().unwrap().ends_with("hwmon2"));
    }

    #[test]
    fn skips_non_amd_vendor() {
        let sys = TempSysfs::new();
        sys.put("class/drm/card0/device/vendor", "0x10de\n");
        sys.put("class/drm/card0/device/gpu_busy_percent", "90\n");
        assert!(discover_amd_gpus(&sys.root).is_empty());
    }

    #[test]
    fn ignores_drm_connectors_named_like_card() {
        let sys = TempSysfs::new();
        sys.populate_r9700("card0");
        sys.put("class/drm/card0-DP-1/device/vendor", "0x1002\n");
        assert_eq!(discover_amd_gpus(&sys.root).len(), 1);
    }

    #[test]
    fn picks_hwmon_named_amdgpu() {
        let sys = TempSysfs::new();
        sys.populate_r9700("card0");
        let hwmon = discover_amd_gpus(&sys.root)[0].hwmon_dir.clone().unwrap();
        let name = fs::read_to_string(hwmon.join("name")).unwrap();
        assert!(name.trim().eq_ignore_ascii_case("amdgpu"));
    }

    #[test]
    fn maps_junction_edge_and_memory_temps() {
        let sys = TempSysfs::new();
        sys.populate_r9700("card0");
        let metrics = AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap();
        let card = first_card(&metrics);
        assert!((card.temp_edge.unwrap() - 85.0).abs() < 0.1);
        assert!((card.temp_junction.unwrap() - 102.0).abs() < 0.1);
        assert!((card.temp_memory.unwrap() - 90.0).abs() < 0.1);
        assert!((card.temp - 102.0).abs() < 0.1);
    }

    #[test]
    fn hotspot_label_maps_to_junction() {
        let sys = TempSysfs::new();
        sys.put("class/drm/card0/device/vendor", "0x1002\n");
        sys.put("class/drm/card0/device/hwmon/hwmon1/name", "amdgpu\n");
        sys.put(
            "class/drm/card0/device/hwmon/hwmon1/temp1_label",
            "HOTSPOT\n",
        );
        sys.put(
            "class/drm/card0/device/hwmon/hwmon1/temp1_input",
            "110000\n",
        );
        let card = first_card(&AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap());
        assert!((card.temp_junction.unwrap() - 110.0).abs() < 0.1);
        assert!((card.temp - 110.0).abs() < 0.1);
    }

    #[test]
    fn prefers_edge_when_junction_missing() {
        let sys = TempSysfs::new();
        sys.put("class/drm/card0/device/vendor", "0x1002\n");
        sys.put("class/drm/card0/device/hwmon/hwmon1/name", "amdgpu\n");
        sys.put("class/drm/card0/device/hwmon/hwmon1/temp1_label", "edge\n");
        sys.put("class/drm/card0/device/hwmon/hwmon1/temp1_input", "77000\n");
        let card = first_card(&AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap());
        assert!((card.temp - 77.0).abs() < 0.1);
    }

    #[test]
    fn reads_vram_bytes_as_mib_without_mixing_gtt() {
        let sys = TempSysfs::new();
        sys.populate_r9700("card0");
        let card = first_card(&AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap());
        assert_eq!(card.vram_used, 15360);
        assert_eq!(card.vram_total, 32768);
    }

    #[test]
    fn reads_gpu_busy_percent() {
        let sys = TempSysfs::new();
        sys.populate_r9700("card0");
        let card = first_card(&AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap());
        assert_eq!(card.load, 42);
    }

    #[test]
    fn missing_optional_fan_does_not_fail_read() {
        let sys = TempSysfs::new();
        sys.put("class/drm/card0/device/vendor", "0x1002\n");
        sys.put("class/drm/card0/device/gpu_busy_percent", "8\n");
        sys.put("class/drm/card0/device/hwmon/hwmon1/name", "amdgpu\n");
        sys.put(
            "class/drm/card0/device/hwmon/hwmon1/power1_average",
            "50000000\n",
        );
        let metrics = AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap();
        assert_eq!(metrics.len(), 1);
        let card = first_card(&metrics);
        assert_eq!(card.load, 8);
        assert!(card.fan_rpm.is_none());
        assert!(card.fan_percent.is_none());
        assert!((card.power_consumption.unwrap() - 50.0).abs() < 0.1);
    }

    #[test]
    fn corrupted_numbers_do_not_panic() {
        let sys = TempSysfs::new();
        sys.put("class/drm/card0/device/vendor", "0x1002\n");
        sys.put("class/drm/card0/device/gpu_busy_percent", "n/a\n");
        sys.put("class/drm/card0/device/mem_info_vram_used", "oops\n");
        sys.put("class/drm/card0/device/mem_info_vram_total", "\n");
        sys.put("class/drm/card0/device/pp_dpm_sclk", "garbage\n");
        sys.put("class/drm/card0/device/hwmon/hwmon1/name", "amdgpu\n");
        sys.put("class/drm/card0/device/hwmon/hwmon1/temp1_label", "edge\n");
        sys.put(
            "class/drm/card0/device/hwmon/hwmon1/temp1_input",
            "not-a-temp\n",
        );
        sys.put(
            "class/drm/card0/device/hwmon/hwmon1/power1_average",
            "NaN\n",
        );
        let metrics = AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap();
        let card = first_card(&metrics);
        assert_eq!(card.load, 0);
        assert_eq!(card.vram_used, 0);
        assert_eq!(card.sclk_mhz, 0);
        assert!(card.power_consumption.is_none());
        assert_eq!(card.temp, 0.0);
    }

    #[test]
    fn deduplicates_same_physical_device() {
        let sys = TempSysfs::new();
        let real = sys.root.join("devices/0000:03:00.0");
        fs::create_dir_all(&real).unwrap();
        fs::write(real.join("vendor"), "0x1002\n").unwrap();
        fs::write(real.join("gpu_busy_percent"), "11\n").unwrap();
        fs::write(real.join("uevent"), "PCI_SLOT_NAME=0000:03:00.0\n").unwrap();
        for card in ["card0", "card2"] {
            let card_dir = sys.root.join(format!("class/drm/{card}"));
            fs::create_dir_all(&card_dir).unwrap();
            std::os::unix::fs::symlink(&real, card_dir.join("device")).unwrap();
        }
        assert_eq!(discover_amd_gpus(&sys.root).len(), 1);
    }

    #[test]
    fn gpu_devices_filter_keeps_selected_indices() {
        let sys = TempSysfs::new();
        sys.put("class/drm/card0/device/vendor", "0x1002\n");
        sys.put(
            "class/drm/card0/device/uevent",
            "PCI_SLOT_NAME=0000:03:00.0\n",
        );
        sys.put("class/drm/card1/device/vendor", "0x1002\n");
        sys.put(
            "class/drm/card1/device/uevent",
            "PCI_SLOT_NAME=0000:04:00.0\n",
        );
        let backend = AmdgpuBackend::with_sysfs_and_devices(&sys.root, "1");
        let metrics = backend.read_metrics().unwrap();
        assert_eq!(metrics.len(), 1);
        assert!(metrics.keys().next().unwrap().contains("0000:04:00.0"));
    }

    #[test]
    fn power_feeds_existing_energy_integrator() {
        let sys = TempSysfs::new();
        sys.populate_r9700("card0");
        let metrics = AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap();
        let card = first_card(&metrics);
        let power = card.power_consumption.expect("power from sysfs");
        assert!((power - 299.0).abs() < 0.1);
        assert_eq!(card.power_kind, PowerKind::Average);

        let mut energy = EnergyState::new_default();
        energy.apply_settings(1.0, 20.0).unwrap();
        let t0 = Instant::now();
        let local = chrono::Local::now();
        let sample = GpuPowerSample {
            id: "gpu0".into(),
            power_w: power,
            utilization: card.load as f32,
        };
        let busy = BusyFlags {
            requests_processing: 1,
            slots_processing: 1,
            health_ok: true,
        };
        energy.ingest(&[sample], busy, t0, local);
        energy.ingest(
            &[GpuPowerSample {
                id: "gpu0".into(),
                power_w: power,
                utilization: card.load as f32,
            }],
            busy,
            t0 + Duration::from_millis(500),
            local,
        );
        let snap = energy.snapshot(t0 + Duration::from_millis(500), local);
        let expected = f64::from(power) * 0.5 / 3_600_000.0;
        assert!((snap.lifetime.energy_kwh - expected).abs() < 1e-9);
        assert!(snap.lifetime.inference_energy_kwh > 0.0);
        assert!(snap.lifetime.inference_cost_pln > 0.0);
    }

    #[test]
    fn reads_fan_rpm_and_percent() {
        let sys = TempSysfs::new();
        sys.populate_r9700("card0");
        let card = first_card(&AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap());
        assert_eq!(card.fan_rpm, Some(1200));
        assert_eq!(card.fan_percent, Some(50));
        assert_eq!(card.sclk_mhz, 2979);
        assert_eq!(card.mclk_mhz, 1500);
    }

    #[test]
    fn freq_input_is_sclk_fallback() {
        let sys = TempSysfs::new();
        sys.put("class/drm/card0/device/vendor", "0x1002\n");
        sys.put("class/drm/card0/device/hwmon/hwmon1/name", "amdgpu\n");
        sys.put("class/drm/card0/device/hwmon/hwmon1/freq1_label", "sclk\n");
        sys.put(
            "class/drm/card0/device/hwmon/hwmon1/freq1_input",
            "2100000000\n",
        );
        let card = first_card(&AmdgpuBackend::with_sysfs(&sys.root).read_metrics().unwrap());
        assert_eq!(card.sclk_mhz, 2100);
    }

    #[test]
    fn repeat_error_limiter_logs_on_kind_or_interval_change() {
        let mut limiter = RepeatErrorLimiter::new(Duration::from_secs(30));
        let t0 = Instant::now();
        assert!(limiter.should_log("a", t0));
        assert!(!limiter.should_log("a", t0 + Duration::from_millis(500)));
        assert!(limiter.should_log("b", t0 + Duration::from_millis(600)));
        assert!(limiter.should_log("b", t0 + Duration::from_secs(31)));
    }
}
