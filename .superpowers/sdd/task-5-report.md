# Task 5 Report: Frontend Energy Card and Controls

## Status

Completed on branch `ui/glass-look`.

## Implementation

- Added the expandable **GPU Energy Cost** card after the unchanged Luna and Qwen cards.
- The card shows lifetime inference cost and energy, the current tariff, session/today/last-7-day
  summaries, lifetime inference and total GPU energy, first measurement, timezone, and save warning.
- Energy updates only change existing nodes with `textContent`, preserving the `<details>` open state.
- Added finite-safe Wh/kWh and PLN formatters, including tiny-positive-value formatting and the
  no-measurement placeholder state.
- Added energy tariff and inference-utilization threshold fields to Configuration, populated from
  the latest WebSocket snapshot and saved through `PUT /api/energy/settings`.
- Added the confirmed lifetime reset flow with the exact irreversible warning and explicit list of
  removed energy history categories.

## Tests

- `node --check static/app.js`
- Formatter edge checks for Wh, kWh, PLN, NaN, and Infinity
- HTML duplicate-ID check — 166 unique IDs
- `cargo test` — 109 passed
- `git diff --check`
- Live server asset smoke on an alternate port with `--gpu-backend none`

## Concerns

- Port 7778 was already occupied, so the live smoke used port 7790.
- The IDE browser failed to retain a usable test tab; the live server and served assets were
  verified, but interactive visual inspection was not completed.
- Existing untracked `.idea/` files were not modified or staged.
