use chrono::{DateTime, Days, Local, NaiveDate};
use std::collections::{BTreeMap, HashMap, HashSet};
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

#[derive(Clone, Default)]
pub struct EnergyTotals {
    pub energy_kwh: f64,
    pub inference_energy_kwh: f64,
    pub cost_pln: f64,
    pub inference_cost_pln: f64,
}

pub struct EnergySnapshot {
    pub available: bool,
    pub telemetry_stale: bool,
    pub currency: String,
    pub price_per_kwh: f64,
    pub inference_util_threshold: f32,
    pub session: EnergyTotals,
    pub today: EnergyTotals,
    pub last_7_days: EnergyTotals,
    pub lifetime: EnergyTotals,
    pub first_measurement_at: Option<DateTime<Local>>,
    pub last_measurement_at: Option<DateTime<Local>>,
}

pub struct EnergyState {
    price_per_kwh: f64,
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

impl EnergyState {
    pub fn new_default() -> Self {
        Self {
            price_per_kwh: 1.0,
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

    pub fn set_price_per_kwh(&mut self, price_per_kwh: f64) {
        self.price_per_kwh = price_per_kwh;
    }

    pub fn set_inference_util_threshold(&mut self, threshold: f32) {
        self.inference_util_threshold = threshold;
    }

    pub fn ingest(
        &mut self,
        gpus: &[GpuPowerSample],
        busy: BusyFlags,
        now_instant: Instant,
        now_local: DateTime<Local>,
    ) {
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
            currency: "PLN".to_string(),
            price_per_kwh: self.price_per_kwh,
            inference_util_threshold: self.inference_util_threshold,
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
        self.first_measurement_at = None;
        self.last_measurement_at = None;
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
    use std::time::Duration;

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
        e.set_price_per_kwh(1.0);
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
        let old_date = today.checked_sub_days(Days::new(35)).expect("valid old date");
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
    fn util_fallback_any_gpu_not_average() {
        let mut e = EnergyState::new_default();
        e.set_inference_util_threshold(20.0);
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
}
