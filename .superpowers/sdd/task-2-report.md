# Task 2 Report: Energy Persistence

## Status

Completed on branch `ui/glass-look`.

Commit: `d3ff49f Persist energy.json with atomic save and lifetime reset.`

## Implementation

### `src/energy/mod.rs`

- Added serde-backed schema version 1 persistence through public `EnergyStore`.
- Added `default_energy_path()` using the state directory, home fallback, and local fallback required by the brief.
- Added `load_energy_state(path)`:
  - Missing files start with defaults and no warning.
  - Valid stores restore settings, lifetime totals, daily buckets, and measurement timestamps.
  - Session totals, GPU integration bases, and telemetry availability remain process-local and start empty.
  - Malformed JSON, invalid UTF-8, and unsupported schema versions are moved to a non-clobbering timestamped `.corrupt-YYYYMMDDTHHMMSS[-N]` backup.
  - Backup failures preserve the original file and return a warning.
- Added `save_energy_store(path, store)`:
  - Creates parent directories.
  - Serializes pretty JSON into a same-directory `.tmp` file.
  - Explicitly flushes before rename.
  - Removes a leftover temporary file after failure where safe.
  - Returns a user-facing error so Task 3 can set `save_warning`.
- Added `EnergyState::persistable_store()`:
  - Copies persisted settings, lifetime totals, daily history, and measurement timestamps.
  - Prunes persisted daily history to the latest 30 local calendar days.
  - Stores the actual local UTC offset as the timezone label instead of guessing an IANA zone.
- Added `EnergyState::apply_settings(price, threshold)`:
  - Rejects non-finite or negative prices.
  - Rejects non-finite thresholds outside `0..=100`.
  - Does not mutate settings on validation failure.
  - Updates `price_changed_at` only when the tariff changes.
  - Never reprices historical cost totals.
- Updated `reset_lifetime()` to clear lifetime, session, daily history, GPU bases, first/last measurement timestamps, and `last_valid_power_instant`.

### `src/config.rs`

- Added `AppConfig.energy_stats_file`.
- Resolved it to `<state_dir>/llama-monitor/energy.json` with the required home and current-directory fallbacks.
- Added a default configuration assertion for the path.

## TDD Evidence

The first persistence test was run before implementation and failed because `EnergyStore`, `persistable_store`, `save_energy_store`, and `load_energy_state` did not exist.

The config path test was run before implementation and failed because `AppConfig.energy_stats_file` did not exist.

The invalid UTF-8 backup regression test was run before its fix and failed because the original file remained in place. Loading was then changed to read bytes and route all unreadable/corrupt stores through the backup path.

Added energy tests cover:

- Lifetime save/reload round trip.
- Tariff changes affecting future deltas only.
- Settings validation and no partial mutation.
- Threshold-only changes preserving `price_changed_at`.
- Lifetime reset clearing bases and telemetry availability.
- Corrupt JSON backup.
- Invalid UTF-8 backup.
- Failed atomic save preserving the existing store.

## Verification

Final verification command:

```text
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test && cargo build --release && git diff --check
```

Result:

- Formatting: passed.
- Clippy with warnings denied: passed.
- Tests: 100 passed, 0 failed.
- Release build: passed.
- Diff whitespace check: passed.
- IDE diagnostics for both modified source files: none.

## Scope and Follow-up

- No AppState, poller, API, WebSocket, or UI wiring was added.
- `save_energy_store` returns save failures; Task 3's save gate is responsible for storing and clearing `EnergyState.save_warning` around actual save attempts, as described by the implementation plan.
- Existing untracked `.idea/` files were not modified or committed.

## Important Review Fixes

- Replaced corrupt-backup `exists()` plus `rename` handling with atomic destination reservation via `OpenOptions::create_new`.
- Corrupt bytes are copied into the reserved backup and flushed before the original is removed.
- Existing backup candidates are never overwritten; collisions advance to the next numeric suffix.
- Copy, flush, or removal failures clean up the newly reserved backup and preserve the original store.
- Added `corrupt_backup_never_overwrites_existing_candidate` as a red-green regression test.
- Removed the public `set_price_per_kwh` and `set_inference_util_threshold` validation bypasses; tests now use `apply_settings`.

Review-fix verification:

```text
cargo test energy::tests
cargo fmt && cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

Result:

- Energy tests: 17 passed, 0 failed.
- Full tests: 101 passed, 0 failed.
- Formatting and Clippy with warnings denied: passed.
