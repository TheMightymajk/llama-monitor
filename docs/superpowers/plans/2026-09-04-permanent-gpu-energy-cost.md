# Permanent GPU Energy Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist GPU energy (total + inference) and PLN cost in `energy.json`, show a Lifetime **GPU Energy Cost** card, with tariff UI and confirmed lifetime reset — without changing Luna/Qwen USD savings.

**Architecture:** New `src/energy/` module owns trapezoidal per-GPU integration (`f64`, `Instant` Δt), `EnergyStore` persistence, WS snapshot, and save serialization. GPU poller snapshots metrics then calls `ingest` (no disk). Periodic/force/shutdown saves clone under short lock then write atomically outside it. Luna/Qwen/`usage-stats.json` untouched.

**Tech Stack:** Rust, tokio, warp, serde_json, chrono (local DateTime), dirs, existing GPU poller + LlamaMetrics.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-04-permanent-gpu-energy-cost-design.md`
- Path: `~/.local/state/llama-monitor/energy.json` via `dirs::state_dir()` with fallback `home/.local/state` (macOS `state_dir` is `None`)
- Integration/cost/accumulators: **`f64` only** (power may enter as `f32`)
- Δt from **`std::time::Instant` only**; local DateTime for daily buckets / timestamps
- Skip Δt ≤ 0 or Δt > 5s; reset that GPU’s base to current sample
- Inference classification priority: `requests_processing > 0` → `slots_processing > 0` → health OK && any GPU util ≥ threshold → total only
- Main card: `lifetime.inference_cost_pln` + `lifetime.inference_energy_kwh` only
- Settings sole source: `energy.json` via `PUT /api/energy/settings`
- Reset: `POST /api/energy/reset-lifetime` with `{ "confirm": true }`; clears lifetime + daily + session + **per-GPU bases**
- Do not change Luna/Qwen formulas or token Reset
- Lock order: `gpu_metrics` → `llama_metrics` / health → `energy`; never hold energy during disk write
- Shutdown: stop ingest → stop ticker → await in-flight save (bounded) → final snapshot → final save → HTTP exit; **no ingest after final snapshot**
- CI: `cargo fmt --check`, `clippy -D warnings`, `test`, `build --release`

## File map

| File | Responsibility |
|------|----------------|
| `src/energy/mod.rs` | Types, ingest, snapshot, load/save, settings, reset, tests |
| `src/config.rs` | `energy_stats_file` path |
| `src/state.rs` | `energy`, `energy_path`, `llama_reachable` |
| `src/main.rs` | Load energy, wire poller ingest, save ticker, SIGINT/SIGTERM shutdown |
| `src/llama/poller.rs` | Write `llama_reachable` each health poll |
| `src/web/ws.rs` | Push `energy` snapshot |
| `src/web/api.rs` | Energy settings + reset endpoints |
| `static/index.html` | Card + config fields + details DOM shell |
| `static/app.js` | Formatters, WS apply, settings, confirm reset |
| `static/style.css` | Minimal card/details styles if needed |
| `README.md` / `AGENTS.md` | Document energy.json + APIs |
| `Cargo.toml` | Add `chrono` |

---

### Task 1: Energy module core — types, trapezoid ingest, in-memory tests

**Files:**
- Create: `src/energy/mod.rs`
- Modify: `src/main.rs` (add `mod energy;`)
- Modify: `Cargo.toml` (add `chrono = { version = "0.4", default-features = false, features = ["clock", "std", "serde"] }`)

**Interfaces:**
- Produces:
  - `pub struct GpuPowerSample { pub id: String, pub power_w: f32, pub utilization: f32 }`
  - `pub struct BusyFlags { pub requests_processing: u32, pub slots_processing: u32, pub health_ok: bool }`
  - `pub struct EnergyState` with `ingest(&mut self, gpus: &[GpuPowerSample], busy: BusyFlags, now_instant: Instant, now_local: DateTime<Local>)`
  - `pub fn snapshot(&self, now_instant: Instant, now_local: DateTime<Local>) -> EnergySnapshot`
  - `pub fn reset_lifetime(&mut self)` — zeros lifetime/daily/session + clears GPU bases
  - Accumulators and costs as `f64`

- [ ] **Step 1: Add chrono dependency and empty module**

```toml
# Cargo.toml
chrono = { version = "0.4", default-features = false, features = ["clock", "std", "serde"] }
```

```rust
// src/main.rs — with other mods
mod energy;
```

Create `src/energy/mod.rs` with skeleton structs and `#[cfg(test)] mod tests {}`.

- [ ] **Step 2: Write failing tests for trapezoid + gap + multi-GPU**

```rust
#[test]
fn trapezoid_single_gpu_500ms() {
    let mut e = EnergyState::new_default();
    e.set_price_per_kwh(1.0);
    let t0 = Instant::now();
    let local = chrono::Local::now();
    e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, local);
    // 100W for 0.5s = 100 * 0.5 / 3_600_000 kWh
    e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0 + Duration::from_millis(500), local);
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
    e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0 + Duration::from_secs(6), local);
    assert_eq!(e.snapshot(t0, local).lifetime.energy_kwh, 0.0);
    e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0 + Duration::from_secs(6) + Duration::from_millis(500), local);
    let expected = 100.0_f64 * 0.5 / 3_600_000.0;
    assert!((e.snapshot(t0, local).lifetime.energy_kwh - expected).abs() < 1e-12);
}

#[test]
fn one_gpu_missing_does_not_block_other() {
    let mut e = EnergyState::new_default();
    let t0 = Instant::now();
    let local = chrono::Local::now();
    e.ingest(&[sample("0", 100.0, 0.0), sample("1", 50.0, 0.0)], idle_busy(), t0, local);
    // Only GPU 0 present on second tick
    e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0 + Duration::from_millis(500), local);
    let expected = 100.0_f64 * 0.5 / 3_600_000.0;
    assert!((e.snapshot(t0, local).lifetime.energy_kwh - expected).abs() < 1e-12);
}

#[test]
fn inference_when_requests_processing() {
    let mut e = EnergyState::new_default();
    let t0 = Instant::now();
    let local = chrono::Local::now();
    let busy = BusyFlags { requests_processing: 1, slots_processing: 0, health_ok: true };
    e.ingest(&[sample("0", 100.0, 0.0)], busy, t0, local);
    e.ingest(&[sample("0", 100.0, 0.0)], busy, t0 + Duration::from_millis(500), local);
    let s = e.snapshot(t0, local);
    assert!(s.lifetime.inference_energy_kwh > 0.0);
    assert_eq!(s.lifetime.inference_energy_kwh, s.lifetime.energy_kwh);
}

#[test]
fn util_fallback_any_gpu_not_average() {
    let mut e = EnergyState::new_default();
    e.set_inference_util_threshold(20.0);
    let t0 = Instant::now();
    let local = chrono::Local::now();
    let busy = BusyFlags { requests_processing: 0, slots_processing: 0, health_ok: true };
    // GPU0 util 80, GPU1 util 0 — average would be 40 but we use ANY >= 20
    e.ingest(&[sample("0", 100.0, 80.0), sample("1", 10.0, 0.0)], busy, t0, local);
    e.ingest(&[sample("0", 100.0, 80.0), sample("1", 10.0, 0.0)], busy, t0 + Duration::from_millis(500), local);
    assert!(e.snapshot(t0, local).lifetime.inference_energy_kwh > 0.0);
}
```

Helper: `fn sample(id, power, util) -> GpuPowerSample`, `fn idle_busy() -> BusyFlags`.

- [ ] **Step 3: Run tests — expect FAIL**

Run: `cargo test -- energy::tests::trapezoid_single_gpu_500ms --nocapture`  
Expected: compile error or FAIL (module incomplete)

- [ ] **Step 4: Implement ingest + snapshot (RAM only, no disk yet)**

Key rules in `ingest`:
- Convert power to `f64`; reject NaN/inf/negative
- Per-GPU trapezoid; sum deltas
- Classify with end-sample busy flags
- `delta_cost = delta_kwh * price_per_kwh` (f64)
- Update session + lifetime + today’s daily bucket (local date of end sample)
- Prune daily to 30 days
- Track `last_valid_power_instant` for availability (3s hysteresis in `snapshot`)
- Disappeared GPU ids: remove bases

- [ ] **Step 5: Run tests — expect PASS**

Run: `cargo test energy:: -- --nocapture`  
Expected: PASS for Task 1 tests

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/energy/mod.rs src/main.rs
git commit -m "Add energy module with f64 trapezoid GPU ingest."
```

---

### Task 2: Persistence — load/save, corrupt backup, tariff, reset bases

**Files:**
- Modify: `src/energy/mod.rs`
- Modify: `src/config.rs` (energy file path)

**Interfaces:**
- Produces:
  - `pub fn default_energy_path() -> PathBuf`
  - `pub fn load_energy_state(path: &Path) -> (EnergyState, Option<String> /* load warning */)`
  - `pub fn save_energy_store(path: &Path, store: &EnergyStore) -> Result<(), String>`
  - `EnergyState::apply_settings(price, threshold) -> Result<(), String>`
  - `EnergyState::persistable_store(&self) -> EnergyStore`
  - `EnergyState::reset_lifetime(&mut self)` clears bases

- [ ] **Step 1: Write failing persistence / tariff / reset tests**

```rust
#[test]
fn lifetime_survives_save_reload() {
    let dir = tempfile_dir();
    let path = dir.join("energy.json");
    let mut e = EnergyState::new_default();
    // ... ingest some energy ...
    save_energy_store(&path, &e.persistable_store()).unwrap();
    let (e2, warn) = load_energy_state(&path);
    assert!(warn.is_none());
    assert!((e2.snapshot(...).lifetime.energy_kwh - expected).abs() < 1e-12);
}

#[test]
fn tariff_change_does_not_reprice_history() {
    let mut e = EnergyState::new_default();
    e.set_price_per_kwh(1.20);
    // ingest delta D at 1.20
    let cost_before = e.snapshot(...).lifetime.cost_pln;
    e.apply_settings(1.40, 20.0).unwrap();
    assert!((e.snapshot(...).lifetime.cost_pln - cost_before).abs() < 1e-12);
    // ingest another equal energy delta — cost increases by D*1.40
}

#[test]
fn reset_clears_bases_next_sample_no_energy() {
    let mut e = EnergyState::new_default();
    let t0 = Instant::now();
    let local = chrono::Local::now();
    e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0, local);
    e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0 + Duration::from_millis(500), local);
    assert!(e.snapshot(t0, local).lifetime.energy_kwh > 0.0);
    e.reset_lifetime();
    assert_eq!(e.snapshot(t0, local).lifetime.energy_kwh, 0.0);
    assert_eq!(e.snapshot(t0, local).session.energy_kwh, 0.0);
    // Next sample only sets base
    e.ingest(&[sample("0", 100.0, 0.0)], idle_busy(), t0 + Duration::from_secs(1), local);
    assert_eq!(e.snapshot(t0, local).lifetime.energy_kwh, 0.0);
}

#[test]
fn corrupt_json_backed_up() {
    let dir = tempfile_dir();
    let path = dir.join("energy.json");
    std::fs::write(&path, "{not json").unwrap();
    let (_e, warn) = load_energy_state(&path);
    assert!(warn.is_some());
    assert!(dir.read_dir().unwrap().any(|e| e.unwrap().file_name().to_string_lossy().contains("corrupt")));
}
```

Use `std::env::temp_dir().join(unique)` — no `tempfile` crate required unless already present.

- [ ] **Step 2: Run — expect FAIL**

Run: `cargo test energy::tests::lifetime_survives_save_reload -- --nocapture`

- [ ] **Step 3: Implement `EnergyStore` serde, atomic save, load, path helper**

```rust
pub fn default_energy_path() -> PathBuf {
    dirs::state_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/state")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llama-monitor")
        .join("energy.json")
}
```

Atomic save: same-dir temp → flush → rename; on failure keep old file; set `save_warning`.

Corrupt: backup `energy.json.corrupt-<YYYYMMDDTHHMMSS>` without clobber; only then fresh state.

`reset_lifetime`: zero counters, clear `daily`, clear `gpu_bases` HashMap, clear session, clear first/last measurement timestamps in store (or keep first null).

`apply_settings`: validate finite ≥0 price and 0..=100 threshold; update `price_changed_at` only when price changes.

- [ ] **Step 4: Add `energy_stats_file` to `AppConfig`**

In `config.rs` `from_args`, set:

```rust
energy_stats_file: {
    dirs::state_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/state")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llama-monitor")
        .join("energy.json")
},
```

Update config tests’ `AppArgs` construction if they construct full structs (energy path not needed on AppArgs).

- [ ] **Step 5: Run energy tests — PASS**

Run: `cargo test energy:: -- --nocapture`

- [ ] **Step 6: Commit**

```bash
git add src/energy/mod.rs src/config.rs
git commit -m "Persist energy.json with atomic save and lifetime reset."
```

---

### Task 3: Wire AppState, GPU poller ingest, save ticker, shutdown

**Files:**
- Modify: `src/state.rs`
- Modify: `src/main.rs`
- Modify: `src/llama/poller.rs`
- Modify: `src/energy/mod.rs` (ingest gate / `stop_ingest` flag + save mutex helper)

**Interfaces:**
- Produces:
  - `AppState.energy: Arc<Mutex<EnergyState>>`
  - `AppState.energy_path: PathBuf`
  - `AppState.llama_reachable: Arc<Mutex<bool>>`
  - `EnergyState::set_ingest_enabled(&mut self, bool)` / check at start of `ingest`
  - `energy::SaveGate` or `tokio::sync::Mutex<()>` ensuring one save at a time
  - `energy::force_save(state, path, save_gate) -> ...` clones store under lock then writes outside

- [ ] **Step 1: Add `llama_reachable` + energy fields to `AppState::new`**

Update all `AppState::new` call sites (main + any tests).

In `llama/poller.rs` after health result:

```rust
*state.llama_reachable.lock().unwrap() = server_reachable;
```

- [ ] **Step 2: Rewrite GPU poller loop to snapshot then ingest**

```rust
thread::spawn(move || {
    let save_gate = Arc::clone(&energy_save_gate);
    loop {
        if !ingest_enabled.load(Ordering::SeqCst) {
            thread::sleep(GPU_POLL_INTERVAL);
            continue;
        }
        match backend.read_metrics() {
            Ok(m) => {
                let samples: Vec<_> = m.iter().map(|(id, g)| GpuPowerSample {
                    id: id.clone(),
                    power_w: g.power_consumption,
                    utilization: g.load as f32,
                }).collect();
                *gpu.lock().unwrap() = m;
                // Drop gpu lock before reading llama / energy
                let (busy, health) = {
                    let llama = llama_metrics.lock().unwrap();
                    let health = *llama_reachable.lock().unwrap();
                    (BusyFlags {
                        requests_processing: llama.requests_processing,
                        slots_processing: llama.slots_processing,
                        health_ok: health,
                    }, health)
                };
                let _ = health; // used inside BusyFlags
                let now_i = Instant::now();
                let now_l = chrono::Local::now();
                energy.lock().unwrap().ingest(&samples, busy, now_i, now_l);
            }
            Err(e) => eprintln!("[error] GPU metrics: {e}"),
        }
        thread::sleep(GPU_POLL_INTERVAL);
    }
});
```

Document lock order in a short comment above the loop.

Note: `GpuMetrics.load` is already utilization %.

- [ ] **Step 3: Periodic save every 30s (tokio task)**

```rust
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(30));
    loop {
        interval.tick().await;
        if !ingest_enabled.load(Ordering::SeqCst) { break; }
        energy::save_with_gate(&energy, &path, &save_gate).await;
    }
});
```

`save_with_gate`: acquire save mutex; clone `persistable_store()` under energy lock; drop energy; write file; on success clear `save_warning` under energy lock.

- [ ] **Step 4: Graceful shutdown SIGINT/SIGTERM**

Use `tokio::signal` (unix: `signal(SignalKind::terminate())` + ctrl_c). On signal:

1. `ingest_enabled.store(false)`
2. Abort/stop save ticker (flag or JoinHandle abort)
3. `tokio::time::timeout(Duration::from_secs(2), save_gate.lock())` then final `save_with_gate`
4. Exit process (or stop warp — if warp has no graceful shutdown helper, `std::process::exit(0)` after save is acceptable with log; prefer documenting bounded exit)

Replace bare `warp::serve(...).run(...).await` with select between server and signal.

Log at startup: energy path, timezone label/offset, local date.

- [ ] **Step 5: Unit/architecture test for save not holding lock during slow write**

```rust
#[test]
fn save_releases_energy_lock_before_disk() {
    // Use a custom save path on a slow mock: after cloning store, energy mutex can be locked by ingest
    // Simpler assertion: document + test that save_energy_store is called with owned EnergyStore
    // and EnergyState::ingest can run on another thread while save_energy_store sleeps on a pipe.
}
```

Minimal version: spawn thread that holds a barrier inside a patched save; assert ingest completes within 100ms while “disk” sleeps 500ms — only if practical. Otherwise test `persistable_store` clone independence: mutate after clone, saved bytes unchanged.

Also test: two sequential `force_save` leave valid JSON.

- [ ] **Step 6: Run tests + clippy focus**

Run: `cargo test && cargo clippy --all-targets -- -D warnings`

- [ ] **Step 7: Commit**

```bash
git add src/state.rs src/main.rs src/llama/poller.rs src/energy/mod.rs src/config.rs
git commit -m "Wire GPU energy ingest, save ticker, and SIGTERM shutdown."
```

---

### Task 4: WebSocket + HTTP APIs

**Files:**
- Modify: `src/web/ws.rs`
- Modify: `src/web/api.rs`
- Modify: `src/energy/mod.rs` (`EnergySnapshot` serde shape matching spec)

**Interfaces:**
- Consumes: `EnergyState::snapshot`, `apply_settings`, `reset_lifetime`, `save_with_gate`
- Produces: WS field `energy`; `PUT /api/energy/settings`; `POST /api/energy/reset-lifetime`

- [ ] **Step 1: Align `EnergySnapshot` JSON exactly with spec** (`available`, `telemetry_stale`, `session`/`today`/`last_7_days`/`lifetime`, etc.)

- [ ] **Step 2: Extend `build_ws_payload` with `energy: EnergySnapshot`**

Update existing ws tests to pass a default snapshot.

- [ ] **Step 3: Implement API handlers**

```rust
// PUT /api/energy/settings
// body: EnergySettingsBody { price_per_kwh: f64, inference_util_threshold: f64 }
// validate → apply_settings → force_save → 200 {ok:true} or 400

// POST /api/energy/reset-lifetime
// body: { confirm: bool }
// if !confirm → 400
// reset_lifetime → force_save → 200
```

Wire into `api_routes`.

- [ ] **Step 4: Tests for validation / confirm**

Prefer unit tests on `apply_settings` / reset; optional warp filter tests if pattern exists — otherwise API covered by validation unit tests.

- [ ] **Step 5: Commit**

```bash
git add src/web/ws.rs src/web/api.rs src/energy/mod.rs
git commit -m "Expose energy snapshot on WebSocket and settings/reset APIs."
```

---

### Task 5: Frontend Lifetime card + config + reset confirm

**Files:**
- Modify: `static/index.html`
- Modify: `static/app.js`
- Modify: `static/style.css`

- [ ] **Step 1: HTML — card after Qwen**

```html
<details class="metric-card energy-card" id="energy-card">
  <summary>
    <div class="metric-label">GPU Energy Cost</div>
    <div class="metric-value amber" id="e-inference-cost">—</div>
    <div class="metric-sub"><span id="e-inference-energy">—</span> · GPU energy only</div>
    <div class="metric-sub" id="e-tariff">— PLN/kWh</div>
  </summary>
  <div class="energy-details" id="energy-details">
    <!-- static rows with ids: e-sess-*, e-today-*, e-7d-*, e-life-inf-*, e-life-total-*, e-first, e-tz, e-warn -->
    <button type="button" class="btn-sm btn-preset" id="btn-energy-reset">Reset lifetime energy</button>
  </div>
</details>
```

Ensure `#lifetime` CSS grid still flows (may need `energy-card` to span or fit as another metric-card).

Configuration modal fields:

```html
<label>Electricity price (PLN/kWh)
  <input type="number" id="set-energy-price" min="0" step="0.01" value="1">
</label>
<label>Inference util fallback threshold (%)
  <input type="number" id="set-energy-util" min="0" max="100" step="1" value="20">
</label>
```

- [ ] **Step 2: JS formatters + `applyEnergy(e)`**

```javascript
function formatEnergyKwh(kwh) {
  if (!Number.isFinite(kwh)) return '—';
  if (kwh > 0 && kwh < 0.01) return (kwh * 1000).toFixed(1) + ' Wh';
  return kwh.toFixed(3) + ' kWh';
}
function formatPln(v) {
  if (!Number.isFinite(v)) return '—';
  if (v > 0 && v < 0.01) return '<0.01 PLN';
  return v.toFixed(2) + ' PLN';
}
```

`applyEnergy`: only `textContent` on existing nodes; never replace `<details>` innerHTML. Main values from `e.lifetime.inference_*`. Show `save_warning` if set. If `first_measurement_at` null and lifetime zeros → `—`.

Load energy settings on config open via `GET` — if no GET endpoint, use last WS snapshot fields to populate inputs; on saveConfig also:

```javascript
fetch('/api/energy/settings', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    price_per_kwh: parseFloat($('set-energy-price').value),
    inference_util_threshold: parseFloat($('set-energy-util').value),
  }),
});
```

Optional: add `GET /api/energy/settings` returning current tariff/threshold for config modal — if added, document in README; otherwise populate from WS.

- [ ] **Step 3: Reset confirm**

Reuse existing confirm modal with the exact English copy from the spec. On OK:

```javascript
fetch('/api/energy/reset-lifetime', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ confirm: true }),
});
```

- [ ] **Step 4: Wire `applyWsPayload` → `applyEnergy(d.energy)`**

- [ ] **Step 5: Manual smoke (dev)**

Run: `cargo run -- --gpu-backend none` then open UI — card shows `—` or zeros; tariff visible.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/app.js static/style.css
git commit -m "Add GPU Energy Cost card and energy settings UI."
```

---

### Task 6: Docs + full verification

**Files:**
- Modify: `README.md`, `AGENTS.md`

- [ ] **Step 1: Document energy.json path, APIs, inference vs total, default 1 PLN/kWh**

- [ ] **Step 2: Run full CI suite**

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add README.md AGENTS.md
git commit -m "Document permanent GPU energy cost feature."
```

---

## Plan self-review

| Spec requirement | Task |
|------------------|------|
| f64 trapezoid per GPU, 5s gap, Instant Δt | 1 |
| Inference classification + util ANY GPU | 1 |
| energy.json schema, atomic save, corrupt backup | 2 |
| Tariff no reprice; settings API source of truth | 2, 4, 5 |
| Reset clears bases + session + daily | 2, 4, 5 |
| Poller snapshot locks; ingest no disk | 3 |
| 30s save; single flight; shutdown order | 3 |
| WS snapshot shape; available hysteresis | 1, 4 |
| UI card + details textContent; confirm copy | 5 |
| Luna/Qwen unchanged | Global / Task 5 |
| README | 6 |

No TBD steps. Types consistent: `EnergyState`, `EnergyStore`, `EnergySnapshot`, `BusyFlags`, `GpuPowerSample`.
