use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// A live generation print older than this is shown as last, not live.
pub const LIVE_STALE: Duration = Duration::from_secs(8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InferencePhase {
    #[default]
    Idle,
    Prefill,
    Generation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeedKind {
    Live,
    Last,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotPhase {
    Idle,
    Prefill,
    Generation,
}

#[derive(Debug, Clone)]
pub struct SlotLiveState {
    pub task_id: Option<u64>,
    pub phase: SlotPhase,
    pub prompt_speed: Option<f64>,
    pub generation_speed: Option<f64>,
    pub generation_speed_avg: Option<f64>,
    pub last_update: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiveLogEvent {
    Launch {
        slot_id: u32,
        task_id: u64,
    },
    Prefill {
        slot_id: u32,
        task_id: u64,
        prompt_speed: f64,
    },
    Generation {
        slot_id: u32,
        task_id: u64,
        tg: Option<f64>,
        tg_3s: f64,
    },
    PromptEvalDone {
        slot_id: u32,
        task_id: u64,
        tokens_per_sec: f64,
    },
    EvalDone {
        slot_id: u32,
        task_id: u64,
        tokens_per_sec: f64,
    },
    Release {
        slot_id: u32,
        task_id: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiveDashboardView {
    pub phase: InferencePhase,
    pub prompt_tokens_per_sec: Option<f64>,
    pub generation_tokens_per_sec: Option<f64>,
    pub generation_tokens_per_sec_avg: Option<f64>,
    pub prompt_speed_kind: Option<SpeedKind>,
    pub generation_speed_kind: Option<SpeedKind>,
}

impl Default for LiveDashboardView {
    fn default() -> Self {
        Self {
            phase: InferencePhase::Idle,
            prompt_tokens_per_sec: None,
            generation_tokens_per_sec: None,
            generation_tokens_per_sec_avg: None,
            prompt_speed_kind: None,
            generation_speed_kind: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LiveSlotTracker {
    slots: BTreeMap<u32, SlotLiveState>,
}

impl LiveSlotTracker {
    pub fn apply_line(&mut self, line: &str, now: Instant) -> bool {
        let Some(event) = parse_live_log_event(line) else {
            return false;
        };
        self.apply_event(event, now);
        true
    }

    pub fn apply_event(&mut self, event: LiveLogEvent, now: Instant) {
        match event {
            LiveLogEvent::Launch { slot_id, task_id } => {
                self.begin_task(slot_id, task_id, now, SlotPhase::Idle);
            }
            LiveLogEvent::Prefill {
                slot_id,
                task_id,
                prompt_speed,
            } => {
                let slot = self.begin_task(slot_id, task_id, now, SlotPhase::Prefill);
                slot.phase = SlotPhase::Prefill;
                slot.prompt_speed = Some(prompt_speed);
                slot.last_update = now;
            }
            LiveLogEvent::Generation {
                slot_id,
                task_id,
                tg,
                tg_3s,
            } => {
                let slot = self.begin_task(slot_id, task_id, now, SlotPhase::Generation);
                slot.phase = SlotPhase::Generation;
                slot.generation_speed = Some(tg_3s);
                if let Some(avg) = tg {
                    slot.generation_speed_avg = Some(avg);
                }
                slot.last_update = now;
            }
            LiveLogEvent::PromptEvalDone {
                slot_id,
                task_id,
                tokens_per_sec,
            } => {
                let slot = self.begin_task(slot_id, task_id, now, SlotPhase::Idle);
                slot.prompt_speed = Some(tokens_per_sec);
                slot.last_update = now;
            }
            LiveLogEvent::EvalDone {
                slot_id,
                task_id,
                tokens_per_sec,
            } => {
                let slot = self.begin_task(slot_id, task_id, now, SlotPhase::Idle);
                if slot.phase != SlotPhase::Generation || slot.generation_speed.is_none() {
                    slot.generation_speed = Some(tokens_per_sec);
                }
                slot.last_update = now;
            }
            LiveLogEvent::Release { slot_id, task_id } => {
                let slot = self.begin_task(slot_id, task_id, now, SlotPhase::Idle);
                slot.phase = SlotPhase::Idle;
                slot.last_update = now;
            }
        }
    }

    fn begin_task(
        &mut self,
        slot_id: u32,
        task_id: u64,
        now: Instant,
        if_new_phase: SlotPhase,
    ) -> &mut SlotLiveState {
        let slot = self.slots.entry(slot_id).or_insert_with(|| SlotLiveState {
            task_id: None,
            phase: SlotPhase::Idle,
            prompt_speed: None,
            generation_speed: None,
            generation_speed_avg: None,
            last_update: now,
        });
        if slot.task_id != Some(task_id) {
            slot.task_id = Some(task_id);
            slot.phase = if_new_phase;
            // Previous speeds remain as last until this task overwrites them.
        }
        slot
    }

    #[cfg(test)]
    fn slot(&self, slot_id: u32) -> Option<&SlotLiveState> {
        self.slots.get(&slot_id)
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn dashboard(&self, now: Instant) -> LiveDashboardView {
        if self.slots.is_empty() {
            return LiveDashboardView::default();
        }

        let mut prompt_live = 0.0;
        let mut prompt_live_n = 0u32;
        let mut gen_live = 0.0;
        let mut gen_live_n = 0u32;
        let mut gen_avg_live = 0.0;
        let mut gen_avg_n = 0u32;
        let mut last_prompt: Option<(Instant, f64)> = None;
        let mut last_gen: Option<(Instant, f64)> = None;
        let mut last_gen_avg: Option<(Instant, f64)> = None;
        let mut any_prefill = false;
        let mut any_generation = false;

        for slot in self.slots.values() {
            let live = slot_is_live(slot, now);
            if live && slot.phase == SlotPhase::Prefill {
                any_prefill = true;
                if let Some(s) = slot.prompt_speed {
                    prompt_live += s;
                    prompt_live_n += 1;
                }
            }
            if live && slot.phase == SlotPhase::Generation {
                any_generation = true;
                if let Some(s) = slot.generation_speed {
                    gen_live += s;
                    gen_live_n += 1;
                }
                if let Some(s) = slot.generation_speed_avg {
                    gen_avg_live += s;
                    gen_avg_n += 1;
                }
            }
            if let Some(s) = slot.prompt_speed
                && last_prompt.is_none_or(|(t, _)| slot.last_update >= t)
            {
                last_prompt = Some((slot.last_update, s));
            }
            if let Some(s) = slot.generation_speed
                && last_gen.is_none_or(|(t, _)| slot.last_update >= t)
            {
                last_gen = Some((slot.last_update, s));
            }
            if let Some(s) = slot.generation_speed_avg
                && last_gen_avg.is_none_or(|(t, _)| slot.last_update >= t)
            {
                last_gen_avg = Some((slot.last_update, s));
            }
        }

        let phase = if any_generation {
            InferencePhase::Generation
        } else if any_prefill {
            InferencePhase::Prefill
        } else {
            InferencePhase::Idle
        };

        let (prompt, prompt_kind) = if prompt_live_n > 0 {
            (Some(prompt_live), Some(SpeedKind::Live))
        } else {
            (
                last_prompt.map(|(_, s)| s),
                last_prompt.map(|_| SpeedKind::Last),
            )
        };
        let (generation, generation_kind) = if gen_live_n > 0 {
            (Some(gen_live), Some(SpeedKind::Live))
        } else {
            (last_gen.map(|(_, s)| s), last_gen.map(|_| SpeedKind::Last))
        };
        let generation_avg = if gen_avg_n > 0 {
            Some(gen_avg_live)
        } else {
            last_gen_avg.map(|(_, s)| s)
        };

        LiveDashboardView {
            phase,
            prompt_tokens_per_sec: prompt,
            generation_tokens_per_sec: generation,
            generation_tokens_per_sec_avg: generation_avg,
            prompt_speed_kind: prompt_kind,
            generation_speed_kind: generation_kind,
        }
    }
}

fn slot_is_live(slot: &SlotLiveState, now: Instant) -> bool {
    slot.phase != SlotPhase::Idle && now.saturating_duration_since(slot.last_update) <= LIVE_STALE
}

pub fn parse_live_log_event(line: &str) -> Option<LiveLogEvent> {
    let (slot_id, task_id) = parse_id_task(line)?;

    if line.contains("stop processing") {
        return Some(LiveLogEvent::Release { slot_id, task_id });
    }
    if (line.contains("launch_slot") || line.contains("processing task"))
        && !line.contains("prompt processing")
        && !line.contains("tg_3s")
        && !line.contains("eval time")
    {
        return Some(LiveLogEvent::Launch { slot_id, task_id });
    }

    if let Some(tg_3s) = find_tagged_f64(line, "tg_3s") {
        let tg = find_tagged_f64_exact_tg(line);
        return Some(LiveLogEvent::Generation {
            slot_id,
            task_id,
            tg,
            tg_3s,
        });
    }

    if line.contains("prompt processing")
        && let Some(prompt_speed) = speed_before_tokens_per_second(line)
    {
        return Some(LiveLogEvent::Prefill {
            slot_id,
            task_id,
            prompt_speed,
        });
    }

    if line.contains("prompt eval time")
        && let Some(tokens_per_sec) = speed_before_tokens_per_second(line)
    {
        return Some(LiveLogEvent::PromptEvalDone {
            slot_id,
            task_id,
            tokens_per_sec,
        });
    } else if line.contains("eval time")
        && let Some(tokens_per_sec) = speed_before_tokens_per_second(line)
    {
        return Some(LiveLogEvent::EvalDone {
            slot_id,
            task_id,
            tokens_per_sec,
        });
    }

    None
}

fn parse_id_task(line: &str) -> Option<(u32, u64)> {
    let id = scan_key_u64(line, "id")?;
    let task = scan_key_u64(line, "task")?;
    u32::try_from(id).ok().map(|id| (id, task))
}

fn scan_key_u64(line: &str, key: &str) -> Option<u64> {
    let mut search_from = 0;
    while let Some(rel) = line[search_from..].find(key) {
        let start = search_from + rel;
        let before_ok = start == 0 || !line.as_bytes()[start - 1].is_ascii_alphanumeric();
        let after = start + key.len();
        let after_ok = line
            .as_bytes()
            .get(after)
            .is_none_or(|b| !b.is_ascii_alphanumeric() && *b != b'_');
        if before_ok && after_ok {
            let rest = line[after..].trim_start();
            let rest = rest.strip_prefix(':').unwrap_or(rest).trim_start();
            let rest = rest.strip_prefix('|').unwrap_or(rest).trim_start();
            if let Some(n) = parse_leading_u64(rest) {
                return Some(n);
            }
        }
        search_from = start + 1;
    }
    None
}

fn parse_leading_u64(s: &str) -> Option<u64> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    s[..end].parse().ok()
}

fn parse_leading_f64(s: &str) -> Option<f64> {
    let end = s
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
        })
        .unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    s[..end]
        .parse()
        .ok()
        .filter(|v: &f64| v.is_finite() && *v >= 0.0)
}

fn find_tagged_f64(hay: &str, tag: &str) -> Option<f64> {
    let i = hay.find(tag)?;
    let rest = hay[i + tag.len()..].trim_start();
    let rest = rest.strip_prefix('=').unwrap_or(rest).trim_start();
    parse_leading_f64(rest)
}

/// `tg = 28.66` without matching `tg_3s`.
fn find_tagged_f64_exact_tg(hay: &str) -> Option<f64> {
    let mut search_from = 0;
    while let Some(rel) = hay[search_from..].find("tg") {
        let start = search_from + rel;
        let after = start + 2;
        if hay[after..].starts_with("_3s") {
            search_from = after;
            continue;
        }
        let before_ok = start == 0 || !hay.as_bytes()[start - 1].is_ascii_alphanumeric();
        if !before_ok {
            search_from = after;
            continue;
        }
        let rest = hay[after..].trim_start();
        let rest = rest.strip_prefix('=').unwrap_or(rest).trim_start();
        if let Some(v) = parse_leading_f64(rest) {
            return Some(v);
        }
        search_from = after;
    }
    None
}

fn speed_before_tokens_per_second(line: &str) -> Option<f64> {
    let i = line.find("tokens per second")?;
    let before = line[..i].trim_end();
    before
        .rsplit(|c: char| c.is_whitespace() || c == '/')
        .find(|tok| !tok.is_empty())
        .and_then(parse_leading_f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> Instant {
        Instant::now()
    }

    #[test]
    fn parse_generation_tg_3s_preferred() {
        let event = parse_live_log_event(
            "slot print_timing: id 1 | task 916 | n_gen = 276, tg = 28.66 t/s, tg_3s = 30.85 t/s",
        )
        .unwrap();
        match event {
            LiveLogEvent::Generation {
                slot_id,
                task_id,
                tg,
                tg_3s,
            } => {
                assert_eq!(slot_id, 1);
                assert_eq!(task_id, 916);
                assert_eq!(tg, Some(28.66));
                assert!((tg_3s - 30.85).abs() < 1e-9);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn generation_line_switches_prefill_to_generation() {
        let now = t0();
        let mut t = LiveSlotTracker::default();
        t.apply_line(
            "slot print_timing: id 1 | task 916 | prompt processing, n_tokens = 512, progress = 0.50, t = 0.85 s / 602.15 tokens per second",
            now,
        );
        assert_eq!(t.slot(1).unwrap().phase, SlotPhase::Prefill);
        t.apply_line(
            "slot print_timing: id 1 | task 916 | n_gen = 276, tg = 28.66 t/s, tg_3s = 30.85 t/s",
            now + Duration::from_secs(3),
        );
        let slot = t.slot(1).unwrap();
        assert_eq!(slot.phase, SlotPhase::Generation);
        assert!((slot.generation_speed.unwrap() - 30.85).abs() < 1e-9);
        assert!((slot.prompt_speed.unwrap() - 602.15).abs() < 1e-9);
    }

    #[test]
    fn later_tg_3s_updates_live_speed() {
        let now = t0();
        let mut t = LiveSlotTracker::default();
        t.apply_line(
            "slot print_timing: id 1 | task 916 | n_gen = 100, tg = 27.73 t/s, tg_3s = 28.01 t/s",
            now,
        );
        t.apply_line(
            "slot print_timing: id 1 | task 916 | n_gen = 276, tg = 28.66 t/s, tg_3s = 30.85 t/s",
            now + Duration::from_secs(3),
        );
        let slot = t.slot(1).unwrap();
        assert!((slot.generation_speed.unwrap() - 30.85).abs() < 1e-9);
        assert!((slot.generation_speed_avg.unwrap() - 28.66).abs() < 1e-9);
        let view = t.dashboard(now + Duration::from_secs(3));
        assert_eq!(view.phase, InferencePhase::Generation);
        assert_eq!(view.generation_speed_kind, Some(SpeedKind::Live));
        assert!((view.generation_tokens_per_sec.unwrap() - 30.85).abs() < 1e-9);
    }

    #[test]
    fn release_goes_idle_and_keeps_last_speed() {
        let now = t0();
        let mut t = LiveSlotTracker::default();
        t.apply_line(
            "slot print_timing: id 1 | task 916 | n_gen = 988, tg = 29.22 t/s, tg_3s = 31.01 t/s",
            now,
        );
        t.apply_line(
            "slot print_timing: id 1 | task 916 | stop processing: n_tokens = 2048, truncated = 0",
            now + Duration::from_secs(1),
        );
        let slot = t.slot(1).unwrap();
        assert_eq!(slot.phase, SlotPhase::Idle);
        assert!((slot.generation_speed.unwrap() - 31.01).abs() < 1e-9);
        let view = t.dashboard(now + Duration::from_secs(1));
        assert_eq!(view.phase, InferencePhase::Idle);
        assert_eq!(view.generation_speed_kind, Some(SpeedKind::Last));
        assert!((view.generation_tokens_per_sec.unwrap() - 31.01).abs() < 1e-9);
    }

    #[test]
    fn new_task_does_not_treat_previous_speed_as_live() {
        let now = t0();
        let mut t = LiveSlotTracker::default();
        t.apply_line(
            "slot print_timing: id 1 | task 916 | n_gen = 276, tg = 28.66 t/s, tg_3s = 30.85 t/s",
            now,
        );
        t.apply_line(
            "slot launch_slot_: id 1 | task 1001 | processing task, is_child = 0",
            now + Duration::from_secs(1),
        );
        let slot = t.slot(1).unwrap();
        assert_eq!(slot.task_id, Some(1001));
        assert_eq!(slot.phase, SlotPhase::Idle);
        assert!((slot.generation_speed.unwrap() - 30.85).abs() < 1e-9);
        let view = t.dashboard(now + Duration::from_secs(1));
        assert_eq!(view.phase, InferencePhase::Idle);
        assert_eq!(view.generation_speed_kind, Some(SpeedKind::Last));
        assert!((view.generation_tokens_per_sec.unwrap() - 30.85).abs() < 1e-9);
    }

    #[test]
    fn two_slots_are_independent() {
        let now = t0();
        let mut t = LiveSlotTracker::default();
        t.apply_line(
            "slot print_timing: id 0 | task 10 | prompt processing, n_tokens = 100, progress = 0.20, t = 0.20 s / 500.00 tokens per second",
            now,
        );
        t.apply_line(
            "slot print_timing: id 1 | task 11 | n_gen = 276, tg = 28.66 t/s, tg_3s = 30.85 t/s",
            now,
        );
        assert_eq!(t.slot(0).unwrap().phase, SlotPhase::Prefill);
        assert_eq!(t.slot(1).unwrap().phase, SlotPhase::Generation);
        let view = t.dashboard(now);
        assert_eq!(view.phase, InferencePhase::Generation);
        assert_eq!(view.prompt_speed_kind, Some(SpeedKind::Live));
        assert_eq!(view.generation_speed_kind, Some(SpeedKind::Live));
        assert!((view.prompt_tokens_per_sec.unwrap() - 500.0).abs() < 1e-9);
        assert!((view.generation_tokens_per_sec.unwrap() - 30.85).abs() < 1e-9);
    }

    #[test]
    fn eval_time_sets_final_last_generation_speed() {
        let now = t0();
        let mut t = LiveSlotTracker::default();
        t.apply_line(
            "slot print_timing: id 1 | task 916 |          eval time =   3378.12 ms /   988 tokens (    3.42 ms per token,    29.24 tokens per second)",
            now,
        );
        let slot = t.slot(1).unwrap();
        assert!((slot.generation_speed.unwrap() - 29.24).abs() < 1e-9);
        t.apply_line(
            "slot print_timing: id 1 | task 916 | stop processing: n_tokens = 2048, truncated = 0",
            now + Duration::from_millis(10),
        );
        let view = t.dashboard(now + Duration::from_millis(10));
        assert_eq!(view.generation_speed_kind, Some(SpeedKind::Last));
        assert!((view.generation_tokens_per_sec.unwrap() - 29.24).abs() < 1e-9);
    }

    #[test]
    fn eval_time_does_not_override_active_tg_3s() {
        let now = t0();
        let mut t = LiveSlotTracker::default();
        t.apply_line(
            "slot print_timing: id 1 | task 916 | n_gen = 988, tg = 29.22 t/s, tg_3s = 31.01 t/s",
            now,
        );
        t.apply_line(
            "slot print_timing: id 1 | task 916 |          eval time =   3378.12 ms /   988 tokens (    3.42 ms per token,    29.24 tokens per second)",
            now + Duration::from_millis(5),
        );
        let slot = t.slot(1).unwrap();
        assert_eq!(slot.phase, SlotPhase::Generation);
        assert!((slot.generation_speed.unwrap() - 31.01).abs() < 1e-9);
    }

    #[test]
    fn fixture_lines_parse() {
        let body = include_str!("../../tests/fixtures/llama_server_slot_log.txt");
        let events: Vec<_> = body.lines().filter_map(parse_live_log_event).collect();
        assert!(events.len() >= 10);
        assert!(events.iter().any(|e| matches!(
            e,
            LiveLogEvent::Generation { tg_3s, .. } if (*tg_3s - 30.85).abs() < 1e-9
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            LiveLogEvent::Prefill { prompt_speed, .. } if (*prompt_speed - 602.15).abs() < 1e-9
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            LiveLogEvent::Prefill { prompt_speed, .. } if (*prompt_speed - 151.71).abs() < 1e-9
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            LiveLogEvent::Generation { tg_3s, .. } if (*tg_3s - 11.44).abs() < 1e-9
        )));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, LiveLogEvent::Release { task_id: 916, .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, LiveLogEvent::Launch { task_id: 1001, .. }))
        );
        assert!(events.iter().any(|e| matches!(
            e,
            LiveLogEvent::EvalDone { tokens_per_sec, .. } if (*tokens_per_sec - 29.24).abs() < 1e-9
        )));
    }

    #[test]
    fn stale_tg_3s_is_last_not_live() {
        let now = t0();
        let mut t = LiveSlotTracker::default();
        t.apply_line(
            "slot print_timing: id 1 | task 916 | n_gen = 276, tg = 28.66 t/s, tg_3s = 30.85 t/s",
            now,
        );
        let view = t.dashboard(now + LIVE_STALE + Duration::from_millis(1));
        assert_eq!(view.phase, InferencePhase::Idle);
        assert_eq!(view.generation_speed_kind, Some(SpeedKind::Last));
        assert!((view.generation_tokens_per_sec.unwrap() - 30.85).abs() < 1e-9);
    }

    #[test]
    fn parse_n_decoded_tg_3s_with_padded_ids() {
        let event = parse_live_log_event(
            "3.11.14.454 I slot print_timing: id  0 | task 0 | n_decoded =    100, tg =  11.44 t/s, tg_3s =  11.44 t/s",
        )
        .unwrap();
        match event {
            LiveLogEvent::Generation {
                slot_id,
                task_id,
                tg,
                tg_3s,
            } => {
                assert_eq!(slot_id, 0);
                assert_eq!(task_id, 0);
                assert_eq!(tg, Some(11.44));
                assert!((tg_3s - 11.44).abs() < 1e-9);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn two_generating_slots_sum_live_throughput() {
        let now = t0();
        let mut t = LiveSlotTracker::default();
        t.apply_line(
            "slot print_timing: id 0 | task 1 | n_gen = 100, tg = 20.00 t/s, tg_3s = 21.00 t/s",
            now,
        );
        t.apply_line(
            "slot print_timing: id 1 | task 2 | n_gen = 100, tg = 30.00 t/s, tg_3s = 31.00 t/s",
            now,
        );
        let view = t.dashboard(now);
        assert_eq!(view.generation_speed_kind, Some(SpeedKind::Live));
        assert!((view.generation_tokens_per_sec.unwrap() - 52.0).abs() < 1e-9);
    }
}
