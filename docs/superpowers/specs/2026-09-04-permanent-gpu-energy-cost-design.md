# Permanent GPU Energy Cost — Design Spec

**Date:** 2026-09-04  
**Status:** Approved for implementation planning  
**Branch context:** Llama Monitor fork — extend Lifetime dashboard with permanent GPU energy cost

## Goal

Add a permanent **GPU Energy Cost** metric card next to existing Lifetime Luna/Qwen savings cards. Cost is real electricity spend for **inference** energy in PLN, integrated from GPU power telemetry, persisted across monitor / llama-server / OS / model restarts.

Do **not** change Luna/Qwen USD calculation or display. Do **not** compute a net Savings (USD − PLN). Do **not** introduce FX rates or new model cloud tariffs.

## Non-goals

- Net savings tile (cloud estimate minus GPU cost)
- Per-GPU cost breakdown in the main UI (internal per-GPU bases only)
- Resetting token / Luna / Qwen counters via energy APIs
- Subtracting GPU cost from existing Lifetime USD figures
- Counting CPU / system PSU / non-GPU energy

---

## Architecture

New module `src/energy/mod.rs` owns:

- In-memory accumulators and per-GPU sample bases
- Persistable state (`EnergyStore`)
- Tariff / util-threshold settings (sole source of truth in `energy.json`)
- Snapshot for WebSocket
- Load / atomic save / corrupt backup
- Validation for settings and reset

**Integration points:**

| Component | Change |
|-----------|--------|
| `src/main.rs` | Load energy store; spawn save ticker; wire shutdown force-save; log timezone; pass energy into GPU poll path |
| `src/state.rs` | `energy: Arc<Mutex<EnergyState>>`, `energy_path` |
| GPU poller (~500 ms) | Snapshot GPU + llama busy flags under short locks → `energy.ingest(...)` with no disk I/O |
| `src/web/ws.rs` | Include `energy` snapshot |
| `src/web/api.rs` | `PUT /api/energy/settings`, `POST /api/energy/reset-lifetime` |
| Configuration modal | Price/kWh + inference util threshold |
| Lifetime panel UI | New metric card + details |

Lock order (documented, must not invert):

1. `gpu_metrics`
2. `llama_metrics` / related busy flags (`server_running` / health as already stored)
3. `energy`

Never hold (1) or (2) while calling into disk. Never hold `energy` during file write/rename.

---

## Persistence

**Path:** `~/.local/state/llama-monitor/energy.json`  
Create parent directory if missing.

### Schema (`schema_version: 1`)

```json
{
  "schema_version": 1,
  "currency": "PLN",
  "price_per_kwh": 1.0,
  "price_changed_at": "2026-09-04T18:00:00+02:00",
  "inference_util_threshold": 20.0,
  "lifetime_energy_kwh": 0.0,
  "lifetime_inference_energy_kwh": 0.0,
  "lifetime_cost_pln": 0.0,
  "lifetime_inference_cost_pln": 0.0,
  "first_measurement_at": null,
  "last_measurement_at": null,
  "timezone_label": "Europe/Warsaw",
  "daily": [
    {
      "date": "2026-09-04",
      "energy_kwh": 0.0,
      "inference_energy_kwh": 0.0,
      "cost_pln": 0.0,
      "inference_cost_pln": 0.0
    }
  ]
}
```

Notes:

- `timezone_label`: IANA name when obtainable; otherwise omit guessing — store/log only what the OS provides (e.g. offset). Do not invent `Europe/Warsaw` if unknown.
- Token counters are **not** stored here (`usage-stats.json` remains sole token store).
- Daily retention: max 30 local calendar days; prune on day change and before save.

### Atomic save

1. Write complete JSON to a temp file in the **same directory** as `energy.json`
2. `flush`
3. `rename` temp → `energy.json`
4. On failure: leave previous `energy.json` intact; remove leftover temp if safe; set `save_warning`
5. At most one save in flight; force-save (tariff change, reset, shutdown) serializes behind / awaits the in-flight save

### Save cadence

- Every 30 seconds (periodic)
- Immediately after successful tariff/threshold change
- Immediately after lifetime reset
- On graceful shutdown (SIGINT / SIGTERM)

`energy.ingest` **never** touches disk. Periodic/force save: clone snapshot under short `energy` lock, then write outside the lock.

### Corrupt JSON

1. Attempt backup to `energy.json.corrupt-<YYYYMMDDTHHMMSS>` (no colons); never overwrite an existing backup name
2. Only after successful backup (or if original unreadable and backup of bytes succeeded): start fresh defaults
3. If backup fails: **do not** delete/overwrite the original file; keep running with in-memory defaults and warn
4. Surface clear warning that a new energy history was started (or that load failed)

### `save_warning`

- Short, safe user-facing string
- No unnecessary system paths/PII dumps
- Cleared only after the next successful save
- Frontend shows as warning, not inference failure

---

## Time model

Two clocks, never mixed for Δt:

| Clock | Use |
|-------|-----|
| `std::time::Instant` | Δt, trapezoid integration, 5s gap detection, telemetry stale window |
| Local `DateTime` (chrono local / equivalent) | Daily bucket date, `first_measurement_at` / `last_measurement_at` ISO strings, UI timezone label |

Wall-clock jumps and DST must not change integrated energy. Startup log prints the timezone label/offset actually used and today’s local date.

---

## Integration algorithm

### Numeric precision

Power telemetry may enter the module as `f32`, but all integration, energy, cost, and persisted accumulator calculations use `f64`. Convert each valid power sample to `f64` before averaging or multiplying by `dt`. Permanent counters must not accumulate in `f32`.

### Per-GPU base (RAM only)

For each GPU id currently reporting:

- `prev_power_w: Option<f64>`
- `prev_instant: Option<Instant>`

When a GPU disappears from telemetry: drop or invalidate its base. On reappearance: first sample establishes base only (no energy).

### Sample ingest (no disk)

Inputs (already snapshotted outside locks):

- Per-GPU: power_w, utilization %
- Llama: `requests_processing`, any slot `is_processing`, health OK
- `now_instant`, `now_local`

For **each** GPU independently:

1. If current power invalid (NaN, ±inf, negative) → do not integrate; if somehow later valid, treat as new base. Invalid current sample does not update base with garbage.
2. If no previous base → set base to current valid sample; no energy
3. `dt = now_instant - prev_instant`
4. If `dt <= 0` or `dt > 5s` or previous power invalid → **no energy**; set current valid sample as new base
5. Else trapezoid:

```
average_power_w = (previous_power_w + current_power_w) / 2
delta_kwh = average_power_w * dt_seconds / 3_600_000
```

6. Sum `delta_kwh` across GPUs that produced a valid delta this tick
7. Classify interval using **end sample** busy state (priority):

   1. `requests_processing > 0` → inference  
   2. else any slot `is_processing == true` → inference  
   3. else health OK **and any** GPU `utilization >= inference_util_threshold` → inference  
   4. else total only  

8. Apply summed delta:

   - Always: lifetime/session/daily **total** energy + `delta * price_per_kwh` → total cost  
   - If inference: also lifetime/session/daily **inference** energy + cost  

9. Update `last_measurement_at` (local) when any valid power sample accepted; set `first_measurement_at` on first ever accepted sample (persisted)

Daily bucket: local calendar date of the **end** sample. Intervals ~500 ms do not need midnight splitting. Missing days in Last-7 read as zero. Do not rewrite historical day keys on timezone change.

### First sample after process start

Establishes Instant + power bases only. Does **not** charge energy for wall time since persisted `last_measurement_at`.

### After telemetry gap / ROCm blip

Same as dt>5s or missing GPU: new base, no backfill across the gap. Never pair pre-outage sample with post-outage sample.

### Availability

- Track Instant of last **valid** power sample (any GPU)
- `available = true` if a valid sample occurred within the last **3 seconds**
- Single failed ROCm read must not flicker `available` to false
- When stale: `available=false`, `telemetry_stale=true`; lifetime totals remain in snapshot
- Lifetime values always returned even if `available=false`

---

## Cost / tariff

- Default `price_per_kwh = 1.0`, `currency = "PLN"`
- Sole settings source: `energy.json` (not `ui-settings.json`)
- On each valid energy delta: `delta_cost = delta_kwh * current_price_per_kwh` added to the appropriate lifetime/session/daily cost fields
- Changing tariff updates `price_changed_at` and applies only to **future** deltas; historical costs never recomputed
- Changing util threshold does **not** update `price_changed_at`

### Settings API

`PUT /api/energy/settings`

```json
{ "price_per_kwh": 1.0, "inference_util_threshold": 20 }
```

Validation:

- `price_per_kwh`: finite, `>= 0`
- `inference_util_threshold`: finite, `0..=100`
- Reject NaN / inf / out of range with 400
- Atomic force-save immediately on success

### Reset API

`POST /api/energy/reset-lifetime`

```json
{ "confirm": true }
```

- Missing/false `confirm` → 400
- Resets: lifetime totals, daily history, **and session** energy/cost (documented behavior for UI consistency)
- Reset lifetime also clears all per-GPU integration bases. The first valid GPU sample following reset establishes a new baseline and does not add energy. This prevents an interval spanning the reset operation from being charged into the fresh lifetime/session counters.
- Does not reset tokens / Luna / Qwen
- Does not stop GPU poller or llama-server
- Immediate atomic force-save
- Existing Lifetime **Reset** button unchanged (tokens only; must not touch energy)

Confirm UI copy:

> Reset lifetime GPU energy statistics? This permanently removes all stored GPU energy and cost history. Token statistics will not be affected. This action cannot be undone.

Explicit list in confirm UI of what is removed: lifetime GPU energy, lifetime inference energy, lifetime GPU costs, daily energy history.

---

## WebSocket snapshot

```json
{
  "available": true,
  "telemetry_stale": false,
  "currency": "PLN",
  "price_per_kwh": 1.0,
  "inference_util_threshold": 20,
  "timezone_label": "Europe/Warsaw",
  "save_warning": null,
  "session": {
    "energy_kwh": 0.0,
    "inference_energy_kwh": 0.0,
    "cost_pln": 0.0,
    "inference_cost_pln": 0.0
  },
  "today": { "energy_kwh": 0.0, "inference_energy_kwh": 0.0, "cost_pln": 0.0, "inference_cost_pln": 0.0 },
  "last_7_days": { "energy_kwh": 0.0, "inference_energy_kwh": 0.0, "cost_pln": 0.0, "inference_cost_pln": 0.0 },
  "lifetime": { "energy_kwh": 0.0, "inference_energy_kwh": 0.0, "cost_pln": 0.0, "inference_cost_pln": 0.0 },
  "first_measurement_at": null,
  "last_measurement_at": null
}
```

- **Today:** current local date bucket  
- **Last 7 days:** today + six previous local days (missing = 0)  
- Session starts at 0 on every monitor process start  

---

## UI

### Lifetime panel card (primary)

Place after Saved vs Luna / Saved vs Qwen:

- Title: **GPU Energy Cost**
- Value: `lifetime.inference_cost_pln` (PLN, 2 decimals; `<0.01 PLN` for tiny positive; `—` if no measurements ever)
- Sub: formatted `lifetime.inference_energy_kwh` + **GPU energy only**
- Always show current tariff clearly (e.g. `1.00 PLN/kWh`) so users know to configure it

### Energy formatting

- Energy &lt; 0.01 kWh → show Wh (e.g. `7.2 Wh`)
- Otherwise kWh with up to 3 decimal places
- PLN: 2 decimals; tiny cost → `<0.01 PLN`
- Never show NaN / Infinity

### Details (`<details>` on the card)

Build DOM once; update via `textContent` only — **never** recreate via `innerHTML` on each WS tick (preserves open state).

Show:

- Session inference cost / energy  
- Today / Last 7 days cost + energy  
- Lifetime inference cost + energy  
- **Total monitored GPU energy cost** + total kWh (`lifetime.cost_pln` / `lifetime.energy_kwh`) with label *Total GPU Energy — includes idle and other GPU workloads*  
- Current tariff  
- `first_measurement_at`  
- Aggregation timezone label  
- `save_warning` if present  
- **Reset lifetime energy** (not a primary dashboard button) → confirm modal with irreversible copy  

Session metrics auto-zero on monitor restart; no Reset session button.

### Configuration (gear)

- Electricity price (PLN/kWh), default 1.00  
- Inference util fallback threshold (%), default 20  
Saves via `PUT /api/energy/settings` only.

---

## Graceful shutdown

Handle at least **SIGINT** and **SIGTERM** (scripts stop the monitor with SIGTERM).

On shutdown:

1. Signal the GPU energy ingest path to stop.
2. Stop the periodic save ticker.
3. Wait for any in-flight save to finish, with a bounded timeout.
4. Take the final `EnergyStore` snapshot.
5. Perform the final atomic save.
6. Shut down HTTP and exit.

No ingest may occur after the final snapshot is taken. Final save failure must be logged but must not block process exit indefinitely.

---

## Testing

Use temp directories only (never `$HOME` fixtures).

Required cases:

1. Lifetime survives process restart (load → ingest → save → reload)  
2. First sample after restart does not charge offline gap  
3. Tariff 1.20 → 1.40 does not reprice historical cost; new deltas use 1.40  
4. Token Reset does not clear energy  
5. Reset lifetime requires `confirm: true`; clears lifetime + daily + session + per-GPU bases; first sample after reset adds no energy; tokens untouched  
6. Trapezoid per GPU; one GPU missing sample does not discard others’ deltas  
7. dt > 5s resets base without energy; subsequent samples work  
8. Corrupt JSON → backup + new history warning path  
9. Save every 30s does not double-count (ingest independent of save)  
10. Concurrency/architecture: ingest proceeds while a slow save holds no energy lock during disk I/O; two serialized force-saves leave valid JSON  
11. `available` hysteresis (~3s) vs single failed sample  
12. Today / last-7 local-date aggregation with missing days as zero  

---

## Dependencies

Likely add `chrono` (local DateTime, formatting). Prefer minimal extras; no heavy filesystem watchers. Timezone IANA name: best-effort via chrono/`iana-time-zone` or OS APIs if already acceptable; otherwise log offset only.

---

## Implementation order (for planning)

1. `EnergyState` + file I/O + unit tests (trapezoid, gaps, tariff, corrupt, daily)  
2. Wire AppState + GPU poller ingest + save ticker + shutdown  
3. WS snapshot + API settings/reset  
4. Frontend card, details, config fields, confirm modal  
5. README / AGENTS notes  

---

## Spec self-review notes

- Markdown is unescaped plain CommonMark (headings, bold, tables render normally)
- Integration / cost / persisted accumulators are `f64` (power may enter as `f32`)
- Lifetime reset clears per-GPU bases; first post-reset sample is baseline-only
- Shutdown stops ingest before final snapshot/save; no ingest after final snapshot
- No TBD placeholders remaining for core behavior
- Savings net formula explicitly rejected — Luna/Qwen unchanged
- Session on lifetime-reset: **zeroed** (documented)
- Settings live only in `energy.json`
- Instant vs local DateTime split is mandatory
- Disk I/O never under `energy` lock during write body

Ambiguity resolved: busy flags come from existing `LlamaMetrics` / slots polling already in AppState — energy module does not poll HTTP itself. Use `slots_processing > 0` as the AppState equivalent of “any slot `is_processing`”.
