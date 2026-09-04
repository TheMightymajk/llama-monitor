use chrono::{DateTime, Days, Local, NaiveDate};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_SAMPLE_GAP: Duration = Duration::from_secs(5);
const AVAILABILITY_WINDOW: Duration = Duration::from_secs(3);
const KWH_DIVISOR: f64 = 3_600_000.0;

pub struct GpuPowerSample {
    pub id: String,
    pub power_w: f32,
    pub utilization: f32,
}

#[derive(Clone, Copy)]
pub struct BusyFlags {
    pub requests_processing: u32,
    pub slots_processing: u32,
    pub health_ok: bool,
}

#[derive(Clone, Default, Deserialize, Serialize)]
pub struct EnergyTotals {
    pub energy_kwh: f64,
    pub inference_energy_kwh: f64,
    pub cost_pln: f64,
    pub inference_cost_pln: f64,
}

#[derive(Clone, Serialize)]
pub struct EnergySnapshot {
    pub available: bool,
    pub telemetry_stale: bool,
    pub save_warning: Option<String>,
    pub currency: String,
    pub price_per_kwh: f64,
    pub inference_util_threshold: f32,
    pub timezone_label: String,
    pub session: EnergyTotals,
    pub today: EnergyTotals,
    pub last_7_days: EnergyTotals,
    pub lifetime: EnergyTotals,
    pub first_measurement_at: Option<DateTime<Local>>,
    pub last_measurement_at: Option<DateTime<Local>>,
}

pub struct EnergyState {
    ingest_enabled: bool,
    load_warning: Option<String>,
    save_warning: Option<String>,
    price_per_kwh: f64,
    price_changed_at: DateTime<Local>,
    inference_util_threshold: f32,
    bases: HashMap<String, GpuBase>,
    last_valid_power_instant: Option<Instant>,
    session: EnergyTotals,
    lifetime: EnergyTotals,
    daily: BTreeMap<NaiveDate, EnergyTotals>,
    first_measurement_at: Option<DateTime<Local>>,
    last_measurement_at: Option<DateTime<Local>>,
}

struct GpuBase {
    power_w: f64,
    instant: Instant,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct EnergyStore {
    schema_version: u32,
    currency: String,
    price_per_kwh: f64,
    price_changed_at: DateTime<Local>,
    inference_util_threshold: f32,
    lifetime_energy_kwh: f64,
    lifetime_inference_energy_kwh: f64,
    lifetime_cost_pln: f64,
    lifetime_inference_cost_pln: f64,
    first_measurement_at: Option<DateTime<Local>>,
    last_measurement_at: Option<DateTime<Local>>,
    timezone_label: String,
    daily: Vec<DailyEnergy>,
}

#[derive(Clone, Deserialize, Serialize)]
struct DailyEnergy {
    date: NaiveDate,
    energy_kwh: f64,
    inference_energy_kwh: f64,
    cost_pln: f64,
    inference_cost_pln: f64,
}

pub fn default_energy_path() -> PathBuf {
    dirs::state_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".local/state")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llama-monitor")
        .join("energy.json")
}

pub fn load_energy_state(path: &Path) -> (EnergyState, Option<String>) {
    if !path.exists() {
        return (EnergyState::new_default(), None);
    }

    let contents = match std::fs::read(path) {
        Ok(contents) => contents,
        Err(error) => {
            return backup_corrupt_store(
                path,
                &format!("Energy history could not be read: {error}"),
            );
        }
    };
    match serde_json::from_slice::<EnergyStore>(&contents) {
        Ok(store) if store.schema_version == 1 => (EnergyState::from_store(store), None),
        Ok(_) => backup_corrupt_store(path, "Energy history has an unsupported schema"),
        Err(error) => backup_corrupt_store(path, &format!("Energy history is corrupt: {error}")),
    }
}

fn backup_corrupt_store(path: &Path, warning: &str) -> (EnergyState, Option<String>) {
    let timestamp = Local::now().format("%Y%m%dT%H%M%S");
    backup_corrupt_store_at(path, warning, &timestamp.to_string())
}

fn backup_corrupt_store_at(
    path: &Path,
    warning: &str,
    timestamp: &str,
) -> (EnergyState, Option<String>) {
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "energy.json".into());
    let backup_result = (|| -> io::Result<PathBuf> {
        let mut suffix = 0_u32;
        let (backup, mut backup_file) = loop {
            let suffix_label = if suffix == 0 {
                String::new()
            } else {
                format!("-{suffix}")
            };
            let candidate =
                path.with_file_name(format!("{file_name}.corrupt-{timestamp}{suffix_label}"));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(file) => break (candidate, file),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    suffix = suffix.checked_add(1).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::AlreadyExists, "backup suffix exhausted")
                    })?;
                }
                Err(error) => return Err(error),
            }
        };

        let copy_result = (|| {
            let mut source = File::open(path)?;
            io::copy(&mut source, &mut backup_file)?;
            backup_file.flush()?;
            std::fs::remove_file(path)
        })();
        if let Err(error) = copy_result {
            drop(backup_file);
            let _ = std::fs::remove_file(&backup);
            return Err(error);
        }
        Ok(backup)
    })();

    let backup_warning = match backup_result {
        Ok(_) => format!("{warning}; a backup was created and fresh history was started"),
        Err(error) => format!("{warning}; the original file was preserved: {error}"),
    };
    (EnergyState::new_default(), Some(backup_warning))
}

pub fn save_energy_store(path: &Path, store: &EnergyStore) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Could not create energy history directory: {error}"))?;
    }
    let temp_path = path.with_extension("json.tmp");
    let result = (|| {
        let json = serde_json::to_vec_pretty(store)
            .map_err(|error| format!("Could not encode energy history: {error}"))?;
        let mut file = File::create(&temp_path)
            .map_err(|error| format!("Could not create energy history temporary file: {error}"))?;
        file.write_all(&json)
            .map_err(|error| format!("Could not write energy history: {error}"))?;
        file.flush()
            .map_err(|error| format!("Could not flush energy history: {error}"))?;
        std::fs::rename(&temp_path, path)
            .map_err(|error| format!("Could not replace energy history: {error}"))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp_path);
    }
    result
}

pub type SaveGate = Arc<tokio::sync::Mutex<()>>;

pub async fn save_with_gate(
    energy: &Arc<Mutex<EnergyState>>,
    path: &Path,
    save_gate: &SaveGate,
) -> Result<(), String> {
    let _save_guard = save_gate.lock().await;
    let store = energy.lock().unwrap().persistable_store();
    let result = save_energy_store(path, &store);
    let mut energy = energy.lock().unwrap();
    energy.save_warning = result.as_ref().err().cloned();
    result
}

pub async fn force_save(
    energy: &Arc<Mutex<EnergyState>>,
    path: &Path,
    save_gate: &SaveGate,
) -> Result<(), String> {
    save_with_gate(energy, path, save_gate).await
}

impl EnergyState {
    pub fn new_default() -> Self {
        Self {
            ingest_enabled: true,
            load_warning: None,
            save_warning: None,
            price_per_kwh: 1.0,
            price_changed_at: Local::now(),
            inference_util_threshold: 20.0,
            bases: HashMap::new(),
            last_valid_power_instant: None,
            session: EnergyTotals::default(),
            lifetime: EnergyTotals::default(),
            daily: BTreeMap::new(),
            first_measurement_at: None,
            last_measurement_at: None,
        }
    }

    pub fn apply_settings(
        &mut self,
        price_per_kwh: f64,
        inference_util_threshold: f32,
    ) -> Result<(), String> {
        if !price_per_kwh.is_finite() || price_per_kwh < 0.0 {
            return Err("Price per kWh must be finite and non-negative".to_string());
        }
        if !inference_util_threshold.is_finite()
            || !(0.0..=100.0).contains(&inference_util_threshold)
        {
            return Err("Inference utilization threshold must be between 0 and 100".to_string());
        }
        if self.price_per_kwh != price_per_kwh {
            self.price_changed_at = Local::now();
        }
        self.price_per_kwh = price_per_kwh;
        self.inference_util_threshold = inference_util_threshold;
        Ok(())
    }

    pub fn persistable_store(&self) -> EnergyStore {
        let cutoff = Local::now().date_naive().checked_sub_days(Days::new(29));
        let daily = self
            .daily
            .iter()
            .filter(|(date, _)| cutoff.is_none_or(|cutoff| **date >= cutoff))
            .map(|(date, totals)| DailyEnergy {
                date: *date,
                energy_kwh: totals.energy_kwh,
                inference_energy_kwh: totals.inference_energy_kwh,
                cost_pln: totals.cost_pln,
                inference_cost_pln: totals.inference_cost_pln,
            })
            .collect();
        EnergyStore {
            schema_version: 1,
            currency: "PLN".to_string(),
            price_per_kwh: self.price_per_kwh,
            price_changed_at: self.price_changed_at,
            inference_util_threshold: self.inference_util_threshold,
            lifetime_energy_kwh: self.lifetime.energy_kwh,
            lifetime_inference_energy_kwh: self.lifetime.inference_energy_kwh,
            lifetime_cost_pln: self.lifetime.cost_pln,
            lifetime_inference_cost_pln: self.lifetime.inference_cost_pln,
            first_measurement_at: self.first_measurement_at,
            last_measurement_at: self.last_measurement_at,
            timezone_label: Local::now().offset().to_string(),
            daily,
        }
    }

    fn from_store(store: EnergyStore) -> Self {
        let daily = store
            .daily
            .into_iter()
            .map(|record| {
                (
                    record.date,
                    EnergyTotals {
                        energy_kwh: record.energy_kwh,
                        inference_energy_kwh: record.inference_energy_kwh,
                        cost_pln: record.cost_pln,
                        inference_cost_pln: record.inference_cost_pln,
                    },
                )
            })
            .collect();
        Self {
            ingest_enabled: true,
            load_warning: None,
            save_warning: None,
            price_per_kwh: store.price_per_kwh,
            price_changed_at: store.price_changed_at,
            inference_util_threshold: store.inference_util_threshold,
            bases: HashMap::new(),
            last_valid_power_instant: None,
            session: EnergyTotals::default(),
            lifetime: EnergyTotals {
                energy_kwh: store.lifetime_energy_kwh,
                inference_energy_kwh: store.lifetime_inference_energy_kwh,
                cost_pln: store.lifetime_cost_pln,
                inference_cost_pln: store.lifetime_inference_cost_pln,
            },
            daily,
            first_measurement_at: store.first_measurement_at,
            last_measurement_at: store.last_measurement_at,
        }
    }

    pub fn ingest(
        &mut self,
        gpus: &[GpuPowerSample],
        busy: BusyFlags,
        now_instant: Instant,
        now_local: DateTime<Local>,
    ) {
        if !self.ingest_enabled {
            return;
        }

        let reported_ids: HashSet<&str> = gpus.iter().map(|gpu| gpu.id.as_str()).collect();
        self.bases
            .retain(|id, _| reported_ids.contains(id.as_str()));

        let mut delta_kwh = 0.0_f64;
        let mut accepted_valid_sample = false;

        for gpu in gpus {
            let power_w = f64::from(gpu.power_w);
            if !power_w.is_finite() || power_w < 0.0 {
                self.bases.remove(&gpu.id);
                continue;
            }

            accepted_valid_sample = true;
            if let Some(previous) = self.bases.get(&gpu.id) {
                match now_instant.checked_duration_since(previous.instant) {
                    Some(dt) if !dt.is_zero() && dt <= MAX_SAMPLE_GAP => {
                        let average_power_w = (previous.power_w + power_w) / 2.0;
                        delta_kwh += average_power_w * dt.as_secs_f64() / KWH_DIVISOR;
                    }
                    _ => {}
                }
            }

            self.bases.insert(
                gpu.id.clone(),
                GpuBase {
                    power_w,
                    instant: now_instant,
                },
            );
        }

        if accepted_valid_sample {
            self.last_valid_power_instant = Some(now_instant);
            self.first_measurement_at.get_or_insert(now_local);
            self.last_measurement_at = Some(now_local);
        }

        self.prune_daily(now_local);

        if delta_kwh == 0.0 {
            return;
        }

        let is_inference = busy.requests_processing > 0
            || busy.slots_processing > 0
            || (busy.health_ok
                && gpus
                    .iter()
                    .any(|gpu| gpu.utilization >= self.inference_util_threshold));
        let delta_cost = delta_kwh * self.price_per_kwh;

        Self::add_delta(&mut self.session, delta_kwh, delta_cost, is_inference);
        Self::add_delta(&mut self.lifetime, delta_kwh, delta_cost, is_inference);
        Self::add_delta(
            self.daily.entry(now_local.date_naive()).or_default(),
            delta_kwh,
            delta_cost,
            is_inference,
        );

        self.prune_daily(now_local);
    }

    fn prune_daily(&mut self, now_local: DateTime<Local>) {
        let Some(cutoff) = now_local.date_naive().checked_sub_days(Days::new(29)) else {
            return;
        };
        self.daily.retain(|date, _| *date >= cutoff);
    }

    pub fn snapshot(&self, now_instant: Instant, now_local: DateTime<Local>) -> EnergySnapshot {
        let available = self.last_valid_power_instant.is_some_and(|last| {
            now_instant
                .checked_duration_since(last)
                .is_some_and(|elapsed| elapsed <= AVAILABILITY_WINDOW)
        });
        let today = self
            .daily
            .get(&now_local.date_naive())
            .cloned()
            .unwrap_or_default();
        let mut last_7_days = EnergyTotals::default();
        let today_date = now_local.date_naive();
        for days_ago in 0..7 {
            if let Some(date) = today_date.checked_sub_days(Days::new(days_ago))
                && let Some(totals) = self.daily.get(&date)
            {
                last_7_days.add_assign(totals);
            }
        }

        EnergySnapshot {
            available,
            telemetry_stale: !available,
            save_warning: self
                .save_warning
                .clone()
                .or_else(|| self.load_warning.clone()),
            currency: "PLN".to_string(),
            price_per_kwh: self.price_per_kwh,
            inference_util_threshold: self.inference_util_threshold,
            timezone_label: now_local.offset().to_string(),
            session: self.session.clone(),
            today,
            last_7_days,
            lifetime: self.lifetime.clone(),
            first_measurement_at: self.first_measurement_at,
            last_measurement_at: self.last_measurement_at,
        }
    }

    pub fn reset_lifetime(&mut self) {
        self.session = EnergyTotals::default();
        self.lifetime = EnergyTotals::default();
        self.daily.clear();
        self.bases.clear();
        self.last_valid_power_instant = None;
        self.first_measurement_at = None;
        self.last_measurement_at = None;
    }

    pub fn set_ingest_enabled(&mut self, enabled: bool) {
        self.ingest_enabled = enabled;
    }

    pub fn set_load_warning(&mut self, warning: Option<String>) {
        self.load_warning = warning;
    }

    fn add_delta(totals: &mut EnergyTotals, delta_kwh: f64, delta_cost: f64, is_inference: bool) {
        totals.energy_kwh += delta_kwh;
        totals.cost_pln += delta_cost;
        if is_inference {
            totals.inference_energy_kwh += delta_kwh;
            totals.inference_cost_pln += delta_cost;
        }
    }
}

impl EnergyTotals {
    fn add_assign(&mut self, other: &Self) {
        self.energy_kwh += other.energy_kwh;
        self.inference_energy_kwh += other.inference_energy_kwh;
        self.cost_pln += other.cost_pln;
        self.inference_cost_pln += other.inference_cost_pln;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static TEMP_DIR_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn tempfile_dir() -> PathBuf {
        let sequence = TEMP_DIR_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "llama-monitor-energy-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample(id: &str, power: f32, util: f32) -> GpuPowerSample {
        GpuPowerSample {
            id: id.to_string(),
            power_w: power,
            utilization: util,
        }
    }

    fn idle_busy() -> BusyFlags {
        BusyFlags {
            requests_processing: 0,
            slots_processing: 0,
            health_ok: true,
        }
    }

    #[test]
    fn trapezoid_single_gpu_500ms() {
        let mut e = EnergyState::new_default();
        e.apply_settings(1.0, 20.0).unwrap();
        let t0 = Instant::now();
        let local = chrono::Local::now();
        e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, local);
        // 100W for 0.5s = 100 * 0.5 / 3_600_000 kWh
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_millis(500),
            local,
        );
        let s = e.snapshot(t0 + Duration::from_millis(500), local);
        let expected = 100.0_f64 * 0.5 / 3_600_000.0;
        assert!((s.lifetime.energy_kwh - expected).abs() < 1e-12);
        assert!((s.lifetime.cost_pln - expected * 1.0).abs() < 1e-12);
        assert_eq!(s.lifetime.inference_energy_kwh, 0.0);
    }

    #[test]
    fn first_sample_adds_no_energy() {
        let mut e = EnergyState::new_default();
        let t0 = Instant::now();
        let local = chrono::Local::now();
        e.ingest(&[sample("0", 200.0, 50.0)], idle_busy(), t0, local);
        assert_eq!(e.snapshot(t0, local).lifetime.energy_kwh, 0.0);
    }

    #[test]
    fn gap_over_5s_resets_base_no_energy() {
        let mut e = EnergyState::new_default();
        let t0 = Instant::now();
        let local = chrono::Local::now();
        e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, local);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_secs(6),
            local,
        );
        assert_eq!(e.snapshot(t0, local).lifetime.energy_kwh, 0.0);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_secs(6) + Duration::from_millis(500),
            local,
        );
        let expected = 100.0_f64 * 0.5 / 3_600_000.0;
        assert!((e.snapshot(t0, local).lifetime.energy_kwh - expected).abs() < 1e-12);
    }

    #[test]
    fn one_gpu_missing_does_not_block_other() {
        let mut e = EnergyState::new_default();
        let t0 = Instant::now();
        let local = chrono::Local::now();
        e.ingest(
            &[sample("0", 100.0, 0.0), sample("1", 50.0, 0.0)],
            idle_busy(),
            t0,
            local,
        );
        // Only GPU 0 present on second tick
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_millis(500),
            local,
        );
        let expected = 100.0_f64 * 0.5 / 3_600_000.0;
        assert!((e.snapshot(t0, local).lifetime.energy_kwh - expected).abs() < 1e-12);
    }

    #[test]
    fn inference_when_requests_processing() {
        let mut e = EnergyState::new_default();
        let t0 = Instant::now();
        let local = chrono::Local::now();
        let busy = BusyFlags {
            requests_processing: 1,
            slots_processing: 0,
            health_ok: true,
        };
        e.ingest(&[sample("0", 100.0, 0.0)], busy, t0, local);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            busy,
            t0 + Duration::from_millis(500),
            local,
        );
        let s = e.snapshot(t0, local);
        assert!(s.lifetime.inference_energy_kwh > 0.0);
        assert_eq!(s.lifetime.inference_energy_kwh, s.lifetime.energy_kwh);
    }

    #[test]
    fn prune_drops_daily_buckets_older_than_30_calendar_days() {
        let mut e = EnergyState::new_default();
        let t0 = Instant::now();
        let today = Local::now().date_naive();
        let old_date = today
            .checked_sub_days(Days::new(35))
            .expect("valid old date");
        let old_local = Local
            .from_local_datetime(&old_date.and_hms_opt(12, 0, 0).unwrap())
            .single()
            .expect("valid local datetime");
        let today_local = Local
            .from_local_datetime(&today.and_hms_opt(12, 0, 0).unwrap())
            .single()
            .expect("valid local datetime");

        e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, old_local);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_millis(500),
            old_local,
        );
        assert!(e.snapshot(t0, old_local).today.energy_kwh > 0.0);

        e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, today_local);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_millis(500),
            today_local,
        );

        assert_eq!(e.snapshot(t0, old_local).today.energy_kwh, 0.0);
        assert!(e.snapshot(t0, today_local).today.energy_kwh > 0.0);
    }

    #[test]
    fn prune_drops_old_daily_buckets_on_zero_energy_ingest() {
        let mut e = EnergyState::new_default();
        let t0 = Instant::now();
        let today = Local::now().date_naive();
        let old_date = today
            .checked_sub_days(Days::new(35))
            .expect("valid old date");
        let old_local = Local
            .from_local_datetime(&old_date.and_hms_opt(12, 0, 0).unwrap())
            .single()
            .expect("valid local datetime");
        let today_local = Local
            .from_local_datetime(&today.and_hms_opt(12, 0, 0).unwrap())
            .single()
            .expect("valid local datetime");

        e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, old_local);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_millis(500),
            old_local,
        );
        assert!(e.snapshot(t0, old_local).today.energy_kwh > 0.0);

        let lifetime_before = e.snapshot(t0, today_local).lifetime.energy_kwh;
        assert!(lifetime_before > 0.0);

        // First sample on today adds no energy but must still prune stale daily buckets.
        e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, today_local);
        let snap = e.snapshot(t0, today_local);
        assert_eq!(snap.lifetime.energy_kwh, lifetime_before);
        assert_eq!(snap.today.energy_kwh, 0.0);
        assert_eq!(e.snapshot(t0, old_local).today.energy_kwh, 0.0);
    }

    #[test]
    fn util_fallback_any_gpu_not_average() {
        let mut e = EnergyState::new_default();
        e.apply_settings(1.0, 20.0).unwrap();
        let t0 = Instant::now();
        let local = chrono::Local::now();
        let busy = BusyFlags {
            requests_processing: 0,
            slots_processing: 0,
            health_ok: true,
        };
        // GPU0 util 80, GPU1 util 0 — average would be 40 but we use ANY >= 20
        e.ingest(
            &[sample("0", 100.0, 80.0), sample("1", 10.0, 0.0)],
            busy,
            t0,
            local,
        );
        e.ingest(
            &[sample("0", 100.0, 80.0), sample("1", 10.0, 0.0)],
            busy,
            t0 + Duration::from_millis(500),
            local,
        );
        assert!(e.snapshot(t0, local).lifetime.inference_energy_kwh > 0.0);
    }

    #[test]
    fn lifetime_survives_save_reload() {
        let dir = tempfile_dir();
        let path = dir.join("energy.json");
        let mut e = EnergyState::new_default();
        let t0 = Instant::now();
        let local = Local::now();
        e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, local);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_millis(500),
            local,
        );
        let expected = 100.0_f64 * 0.5 / KWH_DIVISOR;

        save_energy_store(&path, &e.persistable_store()).unwrap();
        let (e2, warning) = load_energy_state(&path);

        assert!(warning.is_none());
        assert!((e2.snapshot(t0, local).lifetime.energy_kwh - expected).abs() < 1e-12);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tariff_change_does_not_reprice_history() {
        let mut e = EnergyState::new_default();
        e.apply_settings(1.20, 20.0).unwrap();
        let t0 = Instant::now();
        let local = Local::now();
        e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, local);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_millis(500),
            local,
        );
        let first_delta = 100.0_f64 * 0.5 / KWH_DIVISOR;
        let cost_before = e.snapshot(t0, local).lifetime.cost_pln;

        e.apply_settings(1.40, 20.0).unwrap();
        assert!((e.snapshot(t0, local).lifetime.cost_pln - cost_before).abs() < 1e-12);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_secs(1),
            local,
        );

        let expected_cost = first_delta * 1.20 + first_delta * 1.40;
        assert!((e.snapshot(t0, local).lifetime.cost_pln - expected_cost).abs() < 1e-12);
    }

    #[test]
    fn settings_validation_rejects_invalid_values_without_mutation() {
        let mut e = EnergyState::new_default();

        assert!(e.apply_settings(f64::NAN, 20.0).is_err());
        assert!(e.apply_settings(-0.01, 20.0).is_err());
        assert!(e.apply_settings(1.0, f32::INFINITY).is_err());
        assert!(e.apply_settings(1.0, 100.01).is_err());

        let snapshot = e.snapshot(Instant::now(), Local::now());
        assert_eq!(snapshot.price_per_kwh, 1.0);
        assert_eq!(snapshot.inference_util_threshold, 20.0);
    }

    #[test]
    fn snapshot_serializes_with_public_websocket_shape() {
        let snapshot = EnergyState::new_default().snapshot(Instant::now(), Local::now());

        let json = serde_json::to_value(snapshot).unwrap();

        assert_eq!(json["available"], false);
        assert_eq!(json["telemetry_stale"], true);
        assert_eq!(json["currency"], "PLN");
        assert!(json["timezone_label"].is_string());
        assert!(json["session"].is_object());
        assert!(json["today"].is_object());
        assert!(json["last_7_days"].is_object());
        assert!(json["lifetime"].is_object());
        assert!(json.get("first_measurement_at").is_some());
        assert!(json.get("last_measurement_at").is_some());
    }

    #[test]
    fn threshold_change_does_not_update_price_changed_at() {
        let mut e = EnergyState::new_default();
        let changed_at = e.persistable_store().price_changed_at;

        e.apply_settings(1.0, 30.0).unwrap();

        assert_eq!(e.persistable_store().price_changed_at, changed_at);
    }

    #[test]
    fn reset_clears_bases_next_sample_no_energy() {
        let mut e = EnergyState::new_default();
        let t0 = Instant::now();
        let local = Local::now();
        e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, local);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_millis(500),
            local,
        );
        assert!(e.snapshot(t0, local).lifetime.energy_kwh > 0.0);

        e.reset_lifetime();

        let reset = e.snapshot(t0, local);
        assert_eq!(reset.lifetime.energy_kwh, 0.0);
        assert_eq!(reset.session.energy_kwh, 0.0);
        assert!(!reset.available);
        e.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            t0 + Duration::from_secs(1),
            local,
        );
        assert_eq!(e.snapshot(t0, local).lifetime.energy_kwh, 0.0);
    }

    #[test]
    fn corrupt_json_backed_up() {
        let dir = tempfile_dir();
        let path = dir.join("energy.json");
        std::fs::write(&path, "{not json").unwrap();

        let (_e, warning) = load_energy_state(&path);

        assert!(warning.is_some());
        assert!(dir.read_dir().unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("corrupt")
        }));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn corrupt_backup_never_overwrites_existing_candidate() {
        let dir = tempfile_dir();
        let path = dir.join("energy.json");
        let existing_backup = dir.join("energy.json.corrupt-fixed");
        std::fs::write(&path, b"corrupt source").unwrap();
        std::fs::write(&existing_backup, b"existing backup").unwrap();

        let (_e, warning) = backup_corrupt_store_at(&path, "corrupt", "fixed");

        assert!(warning.unwrap().contains("backup was created"));
        assert_eq!(std::fs::read(&existing_backup).unwrap(), b"existing backup");
        assert_eq!(
            std::fs::read(dir.join("energy.json.corrupt-fixed-1")).unwrap(),
            b"corrupt source"
        );
        assert!(!path.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn non_utf8_store_is_backed_up() {
        let dir = tempfile_dir();
        let path = dir.join("energy.json");
        std::fs::write(&path, [0xff, 0xfe]).unwrap();

        let (_e, warning) = load_energy_state(&path);

        assert!(warning.is_some());
        assert!(!path.exists());
        assert!(dir.read_dir().unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("corrupt")
        }));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_save_preserves_existing_store() {
        let dir = tempfile_dir();
        let path = dir.join("energy.json");
        let e = EnergyState::new_default();
        save_energy_store(&path, &e.persistable_store()).unwrap();
        let original = std::fs::read(&path).unwrap();
        std::fs::create_dir(path.with_extension("json.tmp")).unwrap();

        let error = save_energy_store(&path, &e.persistable_store()).unwrap_err();

        assert!(error.contains("temporary file"));
        assert_eq!(std::fs::read(&path).unwrap(), original);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn disabled_ingest_does_not_accept_samples() {
        let mut energy = EnergyState::new_default();
        let now = Instant::now();
        let local = Local::now();

        energy.set_ingest_enabled(false);
        energy.ingest(&[sample("0", 100.0, 50.0)], idle_busy(), now, local);

        let snapshot = energy.snapshot(now, local);
        assert!(!snapshot.available);
        assert!(snapshot.first_measurement_at.is_none());
    }

    #[tokio::test]
    async fn two_sequential_force_saves_leave_valid_json() {
        let dir = tempfile_dir();
        let path = dir.join("energy.json");
        let energy = std::sync::Arc::new(std::sync::Mutex::new(EnergyState::new_default()));
        let save_gate = std::sync::Arc::new(tokio::sync::Mutex::new(()));

        force_save(&energy, &path, &save_gate).await.unwrap();
        force_save(&energy, &path, &save_gate).await.unwrap();

        let bytes = std::fs::read(&path).unwrap();
        serde_json::from_slice::<EnergyStore>(&bytes).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn load_warning_survives_successful_save() {
        let dir = tempfile_dir();
        let path = dir.join("energy.json");
        std::fs::write(&path, "{not json").unwrap();
        let (mut loaded, warning) = load_energy_state(&path);
        loaded.set_load_warning(warning.clone());
        let energy = Arc::new(Mutex::new(loaded));
        let save_gate = Arc::new(tokio::sync::Mutex::new(()));

        save_with_gate(&energy, &path, &save_gate).await.unwrap();

        assert_eq!(
            energy
                .lock()
                .unwrap()
                .snapshot(Instant::now(), Local::now())
                .save_warning,
            warning
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn persistable_store_is_independent_of_later_ingest() {
        let dir = tempfile_dir();
        let path = dir.join("energy.json");
        let mut energy = EnergyState::new_default();
        let now = Instant::now();
        let local = Local::now();
        let store = energy.persistable_store();

        energy.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), now, local);
        energy.ingest(
            &[sample("0", 100.0, 0.0)],
            idle_busy(),
            now + Duration::from_millis(500),
            local,
        );
        save_energy_store(&path, &store).unwrap();

        let (loaded, warning) = load_energy_state(&path);
        assert!(warning.is_none());
        assert_eq!(loaded.snapshot(now, local).lifetime.energy_kwh, 0.0);
        assert!(energy.snapshot(now, local).lifetime.energy_kwh > 0.0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
