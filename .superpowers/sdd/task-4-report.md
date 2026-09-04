# Task 4 Report: Energy WebSocket and HTTP APIs

## Status

Completed on branch `ui/glass-look`.

## Implementation

- Made `EnergySnapshot` serializable with the approved WebSocket shape, including availability,
  stale telemetry, tariff, timezone label, session/today/last-7/lifetime totals, warnings, and
  measurement timestamps.
- Added the energy snapshot to every WebSocket payload.
- Added `PUT /api/energy/settings` with finite/range validation, in-memory application, immediate
  force-save, HTTP 400 validation responses, and HTTP 500 persistence responses.
- Added `POST /api/energy/reset-lifetime`; missing or false confirmation is rejected with HTTP 400,
  while confirmed resets immediately force-save.
- Moved the existing save gate into `AppState` so API, periodic, and shutdown saves all serialize
  through the same gate.

## Tests

- Added snapshot serialization and WebSocket payload coverage.
- Added Warp filter tests for invalid settings, successful settings persistence, rejected reset,
  and confirmed reset persistence.
- Full verification passed:
  - `cargo fmt -- --check`
  - `cargo clippy --all-targets -- -D warnings`
  - `cargo test` — 109 passed
  - `cargo build --release`
  - `git diff --check`

## Concerns

- `timezone_label` uses the local UTC offset because the runtime does not currently resolve an IANA
  timezone name; this is explicitly allowed by the approved design.
- Existing untracked `.idea/` files were not modified or staged.
