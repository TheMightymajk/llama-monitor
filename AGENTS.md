# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

Llama Monitor is a single-binary web dashboard (Rust backend + embedded vanilla JS frontend) for
managing llama.cpp `llama-server` processes with real-time GPU monitoring (AMD ROCm / NVIDIA).
The frontend has no build step: files in `static/` are compiled into the binary via `include_str!`
in `src/web/static_assets.rs`.

## Commands

```bash
cargo build --release          # release binary at target/release/llama-monitor
cargo run                      # dev run (default port 7778)
cargo test                     # unit tests (inside src/, fixtures in tests/fixtures/)
cargo fmt -- --check           # CI enforces rustfmt
cargo clippy -- -D warnings    # CI enforces zero warnings
```

CI (`.github/workflows/ci.yml`) runs, in order: `fmt --check`, `clippy -D warnings`, `test`,
`build --release`. All four must pass before merging.

## Architecture

```
src/
  main.rs              wiring: CLI -> AppState, GPU poller thread, llama poller task, warp server
  cli.rs / config.rs   clap args -> resolved AppConfig
  state.rs             AppState: all shared state in Arc<Mutex<...>>
  gpu/                 GpuBackend trait + rocm.rs (rocm-smi JSON), nvidia.rs (nvidia-smi CSV),
                       env.rs (arch table + auto-detect), dummy.rs (no-op), MultiBackend (mix of vendors)
  llama/               server.rs (subprocess start/stop), metrics.rs (Prometheus parser),
                       poller.rs (async /health, /metrics, /slots polling)
  presets/             ModelPreset CRUD, persisted to ~/.config/llama-monitor/presets.json
  usage/               Lifetime token counters + $ savings, persisted to usage-stats.json
  energy/              GPU power trapezoid integration, inference vs total, energy.json
  logs/                LogBuffer + external file follow (tail -F); ManagedProcess | ExternalFile | None
  models/              GGUF discovery in a configured directory
  web/                 warp routes: api.rs (REST + file browser + chat proxy),
                       ws.rs (WebSocket push), static_assets.rs (embedded frontend)
static/                index.html, style.css, app.js, manifest.json, sw.js, icon.svg
tests/fixtures/        sample tool outputs used by unit tests
```

Data flow: GPU poller (500 ms, OS thread) and llama poller (1 s, tokio task) write into
`AppState`; a WebSocket (500 ms) pushes snapshots to the browser.

## Conventions & gotchas

- **No frontend build step.** Edit `static/*.js|css|html` directly; a rebuild of the Rust crate
  is required for changes to take effect. Verify by running `cargo run` and opening the UI.
- **Shared state is `Arc<Mutex<T>>` in `AppState`** (std `Mutex` for synchronous data,
  `tokio::sync::Mutex` for the child process). Do not introduce new global state; add fields to
  `AppState`. Keep lock scopes small; the GPU poller runs on a plain thread.
- **GPU tool output parsing is fixture-driven.** `tests/fixtures/` contains real samples of
  `rocm-smi` JSON, `nvidia-smi` CSV, and Prometheus text format. Parsers (`gpu/rocm.rs`,
  `gpu/nvidia.rs`, `llama/metrics.rs`) must stay tolerant of new fields; add fixtures when
  changing parsing logic.
- **Config precedence:** CLI flags < persisted UI settings (`~/.config/llama-monitor/`) —
  `ui-settings.json`, `gpu-env.json`, `presets.json`, `usage-stats.json`. Writes are atomic (tmp file + rename);
  keep it that way.
- **GPU energy:** Persisted to `~/.local/state/llama-monitor/energy.json` (not under config dir).
  Default tariff 1 PLN/kWh; inference = active llama requests/slots or any GPU util ≥ threshold (default 20%).
  Total energy includes idle/non-inference draw. Tariff changes do not reprice stored history. APIs:
  `PUT /api/energy/settings`, `POST /api/energy/reset-lifetime` (`confirm: true`); snapshot on WebSocket as `energy`.
- **Cross-platform target: Linux + macOS** (x86_64 and aarch64, see release workflow).
  Avoid Linux-only syscalls; `which` is used for command detection.
- **External binaries are optional.** The app must start and run fine when `llama-server`,
  `rocm-smi`, or `nvidia-smi` are missing (dummy GPU backend, logged warnings).
- **API changes** require updating both `src/web/api.rs` and the corresponding calls in
  `static/app.js` (and `README.md` API table).
- Code style: rustfmt defaults, `anyhow::Result` for error handling, English comments.

## Testing notes

- Unit tests live in `#[cfg(test)]` modules next to the code (see `gpu/mod.rs` for a pattern
  using stub backends).
- Integration with real GPU tools / a live llama-server is manual; there is no CI coverage for it.
- When touching `llama/server.rs` or `web/api.rs`, run the app locally and exercise start/stop
  and the chat proxy if a `llama-server` binary is available.
