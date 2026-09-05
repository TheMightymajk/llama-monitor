# Briefing: live vs lifetime metryki llama-monitor

Dokument dla agenta, który ma napisać **konkretny prompt naprawczy** (nie ogólne „popraw monitoring”).
Repo: `llama-monitor`. Stan kodu: branch `ui/glass-look`, HEAD po commicie `4b68c8b`.

**Nie ma żywego llama-server na `127.0.0.1:8080` w momencie spisywania** — przykłady `/metrics` i `/slots` pochodzą z fixture’ów repo + aktualnego `tools/server` llama.cpp (README + `server-context.cpp`). Przed pisaniem promptu warto zaciągnąć 1–2 surowe odpowiedzi z maszyny użytkownika.

---

## 1. Co już naprawione — nie kazać Cursorowi robić tego drugi raz

Dwa commity z 4–5 IX 2026:

| Commit | Co zmienił |
| --- | --- |
| `b0a524a` *Treat llama.cpp speed and KV occupancy as live gauges.* | Usunięty fallback `prompt_tokens_total / prompt_seconds_total` (i analog dla generation), gdy gauge `t/s == 0`. KV **used** nie pochodzi z `llamacpp:n_tokens_max`. `/metrics` i `/slots` failują niezależnie i czyszczą tylko swoje live-gauge. |
| `4b68c8b` *Render unavailable live gauges as em dash and keep 0 t/s as zero.* | Frontend: `null` → `—`, prawdziwe `0` → `0.0 t/s`. Hero nie mówi już, że to lifetime average. |

**Stary, usunięty kod pollera** (dla kontekstu — tego już nie ma):

```text
prompt_tps = if gauge > 0 { gauge } else if seconds_total > 0 { tokens_total / seconds_total } else { 0 }
kv_cache_tokens = prom.n_tokens_max          // high-water mark, NIE bieżące KV
kv_cache_max    = n_ctx * num_slots          // capacity OK, ale used było złe
```

Aktualne testy, które to blokują:

- `llama::metrics::tests::live_prompt_speed_zero_is_not_lifetime_average`
- `llama::metrics::tests::live_generation_speed_zero_is_not_lifetime_average`
- `llama::metrics::tests::live_zero_serializes_as_zero_not_null`
- `llama::metrics::tests::slots_kv_usage_drops_after_new_session_unlike_n_tokens_max`
- `llama::metrics::tests::slots_kv_usage_idle_slot_without_task_is_empty`

---

## 2. Pliki źródłowe (minimum 5)

| Plik | Rola |
| --- | --- |
| `src/llama/poller.rs` | Co 1 s: `/health` → `/metrics` → `/slots` → `/props` + `/v1/models`. Timeout 3 s. Port z `server_config` albo 8080. |
| `src/llama/metrics.rs` | Parser Prometheus + `LlamaMetrics` + `slots_kv_usage`. **Główna logika live gauges.** |
| `src/usage/mod.rs` | Lifetime: `prompt_tokens`, `predicted_tokens`, `cached_tokens`, `cache_hit_ratio`. Parser `parse_cache_n`. |
| `src/web/ws.rs` | Co 500 ms pcha `{ gpu, llama, usage, energy, running_model, logs, ... }`. |
| `static/index.html` + `static/app.js` | Kafle Inference / Lifetime / Hero. Frontend **bez build step** — zmiana JS wymaga `cargo` rebuild (`include_str!`). |

Powiązane, niekoniecznie do wklejenia w całości:

- `src/state.rs` — `AppState.llama_metrics`, `usage`; `push_log` woła `parse_cache_n`.
- `src/logs/mod.rs` — follow zewnętrznego logu też woła `parse_cache_n`.
- `src/web/api.rs` — proxy `POST /api/chat` skanuje SSE pod `"cache_n"`.
- `src/llama/server.rs` — start procesu z `--metrics` (endpoint `/slots` w nowym llama.cpp jest **on by default**, `--no-slots` wyłącza).
- `src/energy/mod.rs` — inference energy gdy `requests_processing > 0` **lub** `slots_processing > 0` **lub** GPU util ≥ próg. Czyta `LlamaMetrics::busy_*`.
- `tests/fixtures/prometheus_metrics.txt` — jedyny fixture `/metrics`. **Brak fixture `/slots`.**

Drzewo istotnego kodu:

```text
src/llama/{poller.rs,metrics.rs,server.rs,running_model.rs}
src/{state.rs,usage/mod.rs,logs/mod.rs,energy/mod.rs,main.rs}
src/web/{ws.rs,api.rs}
static/{index.html,app.js}
tests/fixtures/prometheus_metrics.txt
```

---

## 3. Semantyka: live / session / lifetime / per-request

Źródło prawdy to **llama-server**, nie llama-monitor. Monitor tylko mapuje.

### 3.1 GET `/metrics` (Prometheus, wymaga `--metrics`)

llama.cpp (aktualny README):

| Metryka | Typ llama.cpp | Znaczenie | Co robi monitor |
| --- | --- | --- | --- |
| `llamacpp:prompt_tokens_seconds` | Gauge | Throughput promptu (w C++: `processed / t_prompt_processing` **bieżącego okna**, `0` gdy `n_prompt_tokens_processed == 0`) | **Live** → `llama.prompt_tokens_per_sec` (`Option<f64>`). `0` zostaje `0`. |
| `llamacpp:predicted_tokens_seconds` | Gauge | Analog generation | **Live** → `llama.generation_tokens_per_sec`. |
| `llamacpp:prompt_tokens_total` | Counter | Tokeny promptu **przetworzone** w tej sesji procesu (bez cache hits) | **Session counter.** Delta → lifetime `usage.prompt_tokens`. Pole `llama.prompt_tokens_total` jest last-known session, UI kafel Inference go **nie pokazuje**. |
| `llamacpp:tokens_predicted_total` | Counter | Tokeny wygenerowane w sesji procesu | Delta → `usage.predicted_tokens`. |
| `llamacpp:prompt_seconds_total` / `tokens_predicted_seconds_total` | Counter | Czas sesji | Parser nadal je czyta (`PrometheusValues`), **UI i live t/s ich nie używają**. Zostawić jako pułapkę — łatwo znów zrobić lifetime average. |
| `llamacpp:n_tokens_max` | Counter (high-water mark) | Historyczne **maksimum** `n_tokens` od startu procesu | **Nie parsowane.** Nie wolno wrzucać do kafelka Context. |
| `llamacpp:requests_processing` | Gauge | Ile slotów/requestów w toku | **Live** → `llama.requests_processing` → Hero **Requests**. |

**Nie ma w `/metrics` żadnego cache-hit countera.** LCP reuse nie zwiększa `prompt_tokens_total`.

### 3.2 GET `/slots` (JSON array; w nowym llama.cpp włączone, wyłączenie: `--no-slots`)

Aktualny `server_slot::to_json()` (gdy slot ma `task` albo `task_prev`):

```json
{
  "id": 0,
  "n_ctx": 65536,
  "speculative": false,
  "is_processing": true,
  "id_task": 135,
  "n_prompt_tokens": 400,
  "n_prompt_tokens_processed": 32,
  "n_prompt_tokens_cache": 368,
  "next_token": { "has_next_token": true, "has_new_line": false, "n_remain": -1, "n_decoded": 80 }
}
```

Idle **bez** taska — tylko `id`, `n_ctx`, `speculative`, `is_processing: false`.

| Pole | Semantyka | Monitor |
| --- | --- | --- |
| `n_ctx` | Capacity slotu | Suma → `kv_cache_max` |
| `n_past` | Starsze llama.cpp: bieżące KV | Preferowane w `slot_used_tokens` |
| `n_prompt_tokens` | `prompt.tokens.size()` — **cały** aktualny ciąg (prompt+wygenerowane), nie high-water | Fallback used |
| `n_decoded` / `next_token.n_decoded` | Tokeny tej generacji | Ostatni fallback used; **nie dodawać** do `n_prompt_tokens` |
| `n_prompt_tokens_cache` | Cache hits **tego requestu** (LCP) | **IGNOROWANE** |
| `n_prompt_tokens_processed` | Ile promptu faktycznie policzono w tym requeście | **IGNOROWANE** |
| `is_processing` | Busy flag | `slots_idle` / `slots_processing` |

**Trzy różne liczby KV, które UI dzisiaj zlewa w jeden kafel:**

1. **Bieżące użycie slotu** — `n_past` / `n_prompt_tokens` (po requestcie z `task_prev` zostaje w cache — to jest occupancy pod LCP, nie „0 bo idle”).
2. **Capacity** — `n_ctx` (× liczba slotów).
3. **Historyczne maksimum** — `llamacpp:n_tokens_max` z `/metrics`. Świadomie odrzucone.

Wzór llama.cpp dla **per-request context usage** z `timings`:

```text
context_tokens = prompt_n + cache_n + predicted_n
```

To **nie** jest to samo co `n_tokens_max` ani to samo co `n_ctx`.

### 3.3 GET `/health`

```json
{"status": "ok"}
```

503 podczas ładowania modelu: `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`.

Poller zapisuje `llama.status` ze stringa `status`. Przy 3 kolejnych failach czyści `LlamaMetrics` i `RunningModelInfo`.

### 3.4 Per-request: `timings` / OpenAI `usage` (chat completions)

Z dokumentacji llama-server (ostatni chunk SSE / final JSON):

```json
"timings": {
  "cache_n": 236,
  "prompt_n": 1,
  "prompt_ms": 30.958,
  "prompt_per_second": 32.30,
  "predicted_n": 35,
  "predicted_ms": 661.064,
  "predicted_per_second": 52.94
}
```

Oraz:

```json
"usage": {
  "prompt_tokens": 44,
  "completion_tokens": 48,
  "prompt_tokens_details": { "cached_tokens": 0 }
}
```

Monitor czyta **tylko** JSON-owe `"cache_n": <int>` (z logów, SSE proxy, zewnętrznego pliku). **Nie czyta** `prompt_tokens_details.cached_tokens`. **Nie czyta** tekstowego `cache_n = 123` z logów.

---

## 4. Modele stanu (aktualne po poprawkach)

### 4.1 Live snapshot — `LlamaMetrics` (`src/llama/metrics.rs`)

```rust
pub struct LlamaMetrics {
    pub prompt_tokens_per_sec: Option<f64>,     // None = scrape fail; Some(0.0) = idle
    pub generation_tokens_per_sec: Option<f64>,
    pub prompt_tokens_total: u64,               // last-known session; nie gauge
    pub predicted_tokens_total: u64,
    pub kv_cache_tokens: Option<u64>,           // used z /slots
    pub kv_cache_max: Option<u64>,              // capacity = sum(n_ctx)
    pub slots_idle: Option<u32>,
    pub slots_processing: Option<u32>,
    pub requests_processing: Option<u32>,
    pub status: String,                         // z /health
}
```

Kontrakt:

- `apply_metrics`: wstawia gauge 1:1, w tym `0`.
- `clear_metrics_gauges`: speed + `requests_processing` → `None`; **nie zeruje** session counters.
- `apply_slots` / `clear_slots_gauges`: analogicznie dla KV i slot busy.
- Serde: `Some(0.0)` → JSON `0.0`, `None` → `null`.

### 4.2 Lifetime — `UsageStats` / `UsageSnapshot` (`src/usage/mod.rs`)

```text
usage.prompt_tokens      += delta(llamacpp:prompt_tokens_total)     // processed, bez cache
usage.predicted_tokens   += delta(llamacpp:tokens_predicted_total)
usage.cached_tokens      += parse_cache_n(...)                      // TYLKO JSON "cache_n"
cache_hit_ratio          = cached / (prompt + cached)
saved_*_usd              = (prompt+cached)*input_rate + predicted*output_rate
```

Restart llama-server (`current < last_*`): dolicza `current` jako nową sesję. Reset UI zostawia `last_*`, żeby nie doliczyć sesji drugi raz.

Dedup cache: identyczne `n` w oknie 2 s (log + chat proxy mogą raportować ten sam request).

### 4.3 WebSocket (`src/web/ws.rs`)

```json
{
  "llama": { "prompt_tokens_per_sec": 0.0, "generation_tokens_per_sec": 0.0, "kv_cache_tokens": 120, "kv_cache_max": 8192, "slots_idle": 1, "slots_processing": 0, "requests_processing": 0, "status": "ok", "...": "session totals" },
  "usage": { "prompt_tokens": 10000, "predicted_tokens": 5000, "cached_tokens": 0, "cache_hit_ratio": 0.0, "saved_luna_usd": 0.0, "saved_qwen_usd": 0.0, "rates": {} },
  "energy": {},
  "running_model": {},
  "gpu": {},
  "server_running": true
}
```

Kafelki **nie** wołają REST po te wartości — tylko WS.

---

## 5. Poller — aktualna pętla (`src/llama/poller.rs`)

Kolejność w cyklu 1 s:

1. `GET /health` → `status`, `llama_reachable`. Fail → clear live gauges, sleep, continue (nie czyść lifetime usage).
2. `GET /metrics` success → `apply_metrics` + `usage.apply_prometheus`. Fail → `clear_metrics_gauges` (nie wołać `apply_prometheus` z zerami).
3. `GET /slots` success → `apply_slots`. Fail (404 gdy `--no-slots`, zły JSON, timeout) → `clear_slots_gauges`. UI Context/Slots → `—`.
4. `/props` + `/v1/models` → karta modelu, **nie** inference tiles.

Logi **nie są** źródłem speed/KV. Są źródłem `cached_tokens` (jeśli linia ma JSON `"cache_n"`).

`slot_used_tokens` (kolejność):

1. `n_past` (clamp do `n_ctx`)
2. else `n_prompt_tokens` (clamp; **nie** dodawaj `n_decoded`)
3. else `n_decoded` / `next_token.n_decoded`

---

## 6. Frontend — kafle (`static/index.html`, `static/app.js`)

### Inference (live)

| Kafel | DOM | Pole WS | Formatter |
| --- | --- | --- | --- |
| Hero Prompt / Generation | `#hero-prompt`, `#hero-gen` | `llama.prompt_tokens_per_sec` / `generation_tokens_per_sec` | `fmtLiveTpsHero`: `null`→`—`, `0`→`0.0` |
| Prompt Speed / Generation Speed | `#m-prompt`, `#m-gen` | to samo | `fmtLiveTps` → `"0.0 t/s"` |
| Context (KV Cache) | `#m-ctx` | `kv_cache_tokens / kv_cache_max` | `"120 / 8192 (1.5%)"` albo `—` |
| Slots | `#m-slots` | `slots_idle` / `slots_processing` | `"1 idle / 0 busy"` |
| Server Status | `#m-status` | `llama.status` | `"ok"` |
| Hero Requests | `#hero-reqs` | `requests_processing` | liczba albo `—` |

Karta modelu `#spec-ctx` to **preset/props context_size**, nie live KV.

### Lifetime

| Kafel | DOM | Pole |
| --- | --- | --- |
| Prompt processed | `#u-prompt` | `usage.prompt_tokens` |
| Generated | `#u-gen` | `usage.predicted_tokens` |
| **Cache hits** | `#u-cache`, `#u-cache-ratio` | `usage.cached_tokens`, `cache_hit_ratio` |
| Saved vs Luna / Qwen | `#u-luna`, `#u-qwen` | USD z lifetime tokenów |
| GPU Energy Cost | `#e-*` | `energy.lifetime.*` — **osobny** subsystem |

`applyUsage` / `applyWsPayload` nie mieszają lifetime speed z live t/s.

---

## 7. Cztery rzeczy do weryfikacji (to ma sprawdzić prompt)

### A. Cache hits = 0 mimo silnego LCP — **otwarty bug, największy priorytet**

Dlaczego UI kłamie:

1. `/metrics` nie eksportuje cache hits.
2. `/slots` ma `n_prompt_tokens_cache` (live, per-request) — **parser go nie czyta**.
3. `parse_cache_n` matchuje wyłącznie `"cache_n":` (JSON). Test **świadomie** odrzuca log llama.cpp:

```text
slot update_slots: ... cache_n = 12345     → None
{"timings":{"cache_n":256}}               → Some(256)
```

4. Chat proxy (`/api/chat`) złapie `timings.cache_n` **tylko** gdy klient idzie przez monitor. Cursor / zewnętrzny klient → 0, chyba że log ma JSON.
5. OpenAI `usage.prompt_tokens_details.cached_tokens` — ignorowane.
6. Dedup 2 s + `n == 0` skip — OK, nie tłumaczy stałego zera.

Wniosek: lifetime **Cache hits** mierzy „ile razy złapaliśmy JSON cache_n w logu/SSE”, nie „ile tokenów LCP faktycznie reused”.

Kierunki naprawy (do decyzji w promptcie, nie robić wszystkich naraz bez specyfikacji):

- Akumulować `n_prompt_tokens_cache` z `/slots` przy **zakończeniu** requestu (`is_processing` true→false), nie w każdym pollu (inaczej zsumujesz ten sam cache 10× na sekundę).
- Parsować tekstowe `cache_n = N` z logów.
- Czytać `timings.cache_n` / `cached_tokens` z dowolnego źródła requestu.
- Osobny kafel **live** cache (bieżący request) vs lifetime suma.

Acceptance (szkic):

- Request z LCP (`cache_n` albo `n_prompt_tokens_cache` > 0) zwiększa `usage.cached_tokens` **raz**.
- Idle po requestcie nie dolicza dalej.
- Log `cache_n = 100` (bez JSON) albo `/slots.n_prompt_tokens_cache` — pokrycie testem fixture.
- Chat poza `/api/chat` nadal liczy (log albo slots).
- `cargo test` + ręcznie: drugi identyczny prompt, ratio > 0.

### B. Prompt speed — **naprawione w backendzie; zweryfikować vs żywy serwer**

Kod: `apply_metrics` kopiuje gauge; `0` ≠ lifetime average.

Ryzyko residualne:

- llama.cpp opisuje gauge jako „Average prompt throughput”, implementacja to `processed / time` **bieżącego** `n_prompt_tokens_processed`. Jeśli upstream **nie zeruje** po end-of-request, UI pokaże ostatni request, nie `0`. Wtedy to bug llama.cpp albo trzeba brać t/s tylko gdy `requests_processing > 0` / `is_processing`.
- `None` (fail `/metrics`) renderuje `—`, nie poprzednią wartość — OK.

Acceptance:

- Idle, `/metrics` success, gauge 0 → kafel `0.0 t/s`, **nie** `tokens_total/seconds_total`.
- W trakcie prompt eval → wartość zbliżona do `llamacpp:prompt_tokens_seconds`.
- `/metrics` 5xx → `—`.
- Istniejące testy `live_prompt_speed_zero_*` muszą przejść; nie przywracać dzielenia totals.

### C. Generation speed — analogicznie naprawione

Acceptance: po zakończeniu requestu, gdy llama.cpp zwraca `predicted_tokens_seconds 0` → UI `0.0 t/s`. Badge `#badge-server` dodaje `t/s` tylko gdy `generation_tokens_per_sec > 0`.

Jeśli żywy serwer **trzyma** ostatnie t/s: w promptcie dodać regułę „gating”: pokaż generation t/s tylko gdy `requests_processing > 0` lub slot `is_processing`.

### D. Context / KV — used vs capacity vs high-water **częściowo naprawione**

Jest:

- used = current slot occupancy z `/slots`
- max = `sum(n_ctx)` = capacity
- high-water `n_tokens_max` **nie** wchodzi do kafelka

Braki:

- UI nie pokazuje high-water wcale (może być OK — ale user chce **rozróżnienia** trzech pojęć; dziś kafel mówi tylko `used / capacity %`).
- Po requestcie occupancy **nie spada do 0**, jeśli `task_prev` zostawia `n_prompt_tokens` — to jest live KV pod LCP, nie bug n_tokens_max. Etykieta „Context (KV Cache)” tego nie wyjaśnia.
- Brak fixture prawdziwego `/slots` z `n_prompt_tokens_cache`.
- `/slots` wyłączone (`--no-slots` w extra args) → cały kafel `—`, nawet gdy `n_tokens_max` istnieje (świadomie: high-water ≠ used).

Acceptance:

- Nowy session po pełnym kontekście: used spada (test już jest).
- Idle bez taska: `0 / n_ctx`.
- Parallel: suma used i suma `n_ctx`.
- Nigdy nie używać `n_tokens_max` jako licznika used.
- (Opcjonalnie w promptcie) trzy jawne wartości: live used, capacity, high-water.

---

## 8. Przykładowe surowe odpowiedzi

### 8.1 Fixture `/metrics` — `tests/fixtures/prometheus_metrics.txt`

```text
# HELP llamacpp:prompt_tokens_seconds Prompt tokens per second (gauge)
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 1234.5
# HELP llamacpp:predicted_tokens_seconds Predicted tokens per second (gauge)
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 56.7
# HELP llamacpp:prompt_tokens_total Total prompt tokens processed
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 10000
# HELP llamacpp:prompt_seconds_total Total prompt processing seconds
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 8.1
# HELP llamacpp:tokens_predicted_total Total predicted tokens
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 5000
# HELP llamacpp:tokens_predicted_seconds_total Total predicted seconds
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 88.2
# HELP llamacpp:n_tokens_max High watermark of the context size observed
# TYPE llamacpp:n_tokens_max counter
llamacpp:n_tokens_max 131072
# HELP llamacpp:requests_processing Current requests processing
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
```

Idle (do testów / promptu) — tego **nie ma** w repo, trzeba dodać:

```text
llamacpp:prompt_tokens_seconds 0
llamacpp:predicted_tokens_seconds 0
llamacpp:prompt_tokens_total 10000
llamacpp:prompt_seconds_total 8.1
llamacpp:tokens_predicted_total 5000
llamacpp:tokens_predicted_seconds_total 88.2
llamacpp:n_tokens_max 8192
llamacpp:requests_processing 0
```

Live t/s musi być `0`, nie `10000/8.1 ≈ 1234.6` ani `5000/88.2 ≈ 56.7`.

### 8.2 `/slots` — przykłady z kodu llama.cpp (brak fixture w repo)

Busy, LCP mocno działa:

```json
[{
  "id": 0,
  "n_ctx": 131072,
  "speculative": false,
  "is_processing": true,
  "id_task": 42,
  "n_prompt_tokens": 8000,
  "n_prompt_tokens_processed": 12,
  "n_prompt_tokens_cache": 7988,
  "next_token": { "has_next_token": true, "n_decoded": 40, "n_remain": -1, "has_new_line": false }
}]
```

Oczekiwane **dziś**: used=8000, max=131072, cache hits lifetime **bez zmian**.
Oczekiwane **po naprawie cache**: +7988 do lifetime (raz, na transition busy→idle), nie 8000.

Idle bez taska:

```json
[{ "id": 0, "n_ctx": 65536, "speculative": false, "is_processing": false }]
```

→ used=0, max=65536.

Starszy serwer z `n_past`:

```json
[{ "id": 0, "n_ctx": 8192, "n_past": 1024, "n_decoded": 50, "is_processing": true }]
```

→ used=1024 (nie 1074).

### 8.3 `/health`

```json
{"status": "ok"}
```

---

## 9. Testy, których brakuje (wrzucić do promptu)

| Luka | Po co |
| --- | --- |
| Fixture prawdziwego `/slots` (busy + idle + LCP) | Parser occupancy + cache |
| Poller: `/metrics` fail nie woła `apply_prometheus(0,0)` | Już jest test usage, brak integracji pollera |
| `parse_cache_n` dla `cache_n = 12` (log tekstowy) | Główna przyczyna zera |
| Transition `/slots` processing→idle dolicza `n_prompt_tokens_cache` raz | Jeśli taka będzie specyfikacja |
| Frontend nie da się unit-testować w CI — tylko kontrakt JSON (`null` vs `0`) | Test `live_zero_serializes_as_zero_not_null` |

CI: `cargo fmt -- --check && cargo clippy -- -D warnings && cargo test && cargo build --release`.

---

## 10. Szkielet promptu dla Cursora (do wypełnienia po surowych dumpach)

Nie implementować tu poprawek — to materiał na prompt:

1. **Nie ruszaj** fallbacku lifetime t/s ani `n_tokens_max` jako KV used — to zrobione (`b0a524a`, `4b68c8b`). Testy `live_*_zero_is_not_lifetime_average` i `slots_kv_usage_drops_after_new_session_*` muszą zostać zielone.
2. **Cache hits:** dziś 0, bo źródło to wyłącznie JSON `"cache_n"`, a LCP siedzi w `/slots.n_prompt_tokens_cache` i w tekstowych logach. Zdefiniuj dokładnie: live vs lifetime, kiedy inkrement, jak uniknąć podwójnego zliczania (poll 1 Hz vs log vs SSE).
3. **Prompt/Generation speed:** skopiuj gauge; `0` jest poprawnym idle. Jeśli surowy dump pokaże niezerowy gauge przy `requests_processing 0`, dodaj gating od busy — nie dziel totals.
4. **Context/KV:** used = occupancy slotu, max = capacity `n_ctx`, high-water osobno albo wcale. Nie mylić z `prompt_n+cache_n+predicted_n` (per-request) ani z lifetime tokens.
5. Testy fixture-driven jak `gpu/` i `metrics.rs`. Dodać fixture `/slots`.
6. Frontend: `null` → `—`, `0` → `0.0 t/s`. Zmiana `static/*` wymaga rebuild Rust.

Potrzebne od użytkownika przed promptem (jeśli możliwe):

```bash
curl -s localhost:$PORT/metrics
curl -s localhost:$PORT/slots
curl -s localhost:$PORT/health
# jedna linia logu z LCP + ostatni timings z /v1/chat/completions
```

Bez tego Cache hits i ewentualny sticky-gauge t/s to hipotezy oparte o kod llama.cpp, nie o ten konkretny binarek.
