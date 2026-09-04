# Task 3 Report: Runtime Energy Wiring

## Status

Completed on branch `ui/glass-look`.

## Implementation

### Runtime state

- Added shared energy state and persistence path to `AppState`.
- Added `llama_reachable`, initialized to false and refreshed after every `/health` result.
- Loaded persisted energy state during startup and logged its path, UTC offset, and local date.
- Preserved load/save warnings in `EnergyState` for later API/UI exposure.

### GPU ingestion

- Converted each successful GPU backend read into owned `GpuPowerSample` snapshots.
- Uses `GpuMetrics.load` directly as utilization percent.
- Reads llama request/slot activity and health before calling energy ingest.
- Documents and follows the `gpu_metrics -> llama_metrics/health -> energy` lock order.
- Added an atomic poll-loop gate plus an `EnergyState` ingest gate. The state gate rejects a
  backend read that started before shutdown, preventing ingest after the final snapshot.

### Persistence and shutdown

- Added a single async save gate shared by periodic and final saves.
- Added a 30-second save task that snapshots `EnergyStore` under the energy mutex, releases the
  mutex, and performs the atomic disk write afterward.
- Save failures set `save_warning`; successful saves clear it.
- Replaced the unbounded Warp server run with graceful shutdown on SIGINT or SIGTERM.
- Shutdown disables ingest, aborts the periodic ticker, waits up to two seconds for the save gate,
  takes and atomically saves the final snapshot, then lets HTTP exit.
- If an existing save does not release the gate within two seconds, shutdown logs the timeout and
  skips a racing final write so an older snapshot cannot overwrite newer data.

## Tests

Added tests for:

- Disabled ingest rejecting telemetry without changing availability or timestamps.
- Two sequential gated force-saves leaving valid JSON.
- A persistable snapshot remaining independent of later ingest, demonstrating that disk writes
  consume owned data rather than holding the energy mutex.

Final verification:

```text
cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt -- --check
```

Result: 104 tests passed; Clippy and rustfmt passed. `git diff --check` also passed, and IDE
diagnostics reported no errors in modified files.

## Scope and concerns

- Existing untracked `.idea/` files were not modified or staged.
- Signal-driven shutdown was compile- and lint-verified but not process-tested to avoid writing to
  the developer's real platform state directory.

## Review fix: bounded process exit

**Finding:** After the energy shutdown sequence, `server.await` could hang indefinitely on open
`/ws` WebSocket connections.

**Fix:** Wrap server drain in a 2-second timeout (`SERVER_DRAIN_TIMEOUT`), log a warning on timeout,
then call `std::process::exit(0)` so SIGTERM/SIGINT always terminate within a bounded window.

Verification:

```text
cargo test && cargo clippy --all-targets -- -D warnings
```

Result: 104 tests passed; Clippy passed.

## Final review fix: persistent load warnings

**Finding:** Tokio's first interval tick completes immediately, so the startup save could clear a
corrupt-load warning before an HTTP or WebSocket client observed it.

**Fix:**

- Consume the interval's immediate first tick before entering the periodic save loop.
- Store load-origin warnings separately from transient save failures.
- Continue exposing both through the existing WebSocket `save_warning` field, prioritizing an
  active save failure and returning to the sticky load warning after a successful save.
- Added a regression test covering corrupt load, successful gated save, and snapshot warning
  persistence.

Verification:

```text
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

Result: 110 tests passed; Clippy and rustfmt passed.
