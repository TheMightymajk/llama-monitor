use crate::llama::server::ServerConfig;
use crate::models::parse_gguf_filename;

/// Where a field value came from (priority: props > models > process > preset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldSource {
    #[default]
    None,
    Props,
    Models,
    ProcessConfig,
    Preset,
}

impl FieldSource {
    fn rank(self) -> u8 {
        match self {
            Self::Props => 4,
            Self::Models => 3,
            Self::ProcessConfig => 2,
            Self::Preset => 1,
            Self::None => 0,
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SourcedString {
    pub value: Option<String>,
    pub source: FieldSource,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SourcedU64 {
    pub value: Option<u64>,
    pub source: FieldSource,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SourcedU32 {
    pub value: Option<u32>,
    pub source: FieldSource,
}

/// Model identity detected from the live llama-server (not the selected preset).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RunningModelInfo {
    /// True when at least one live server source (/props or /v1/models) contributed.
    pub detected: bool,
    /// Highest-priority live source that contributed any field.
    pub primary_source: FieldSource,
    pub name: SourcedString,
    pub model_id: SourcedString,
    pub gguf_file: SourcedString,
    pub model_path: SourcedString,
    pub context_size: SourcedU64,
    pub native_context: SourcedU64,
    pub total_slots: SourcedU32,
    pub quant: SourcedString,
}

impl RunningModelInfo {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Display name: prefer human name, then id, then gguf stem.
    #[allow(dead_code)]
    pub fn display_name(&self) -> Option<&str> {
        self.name
            .value
            .as_deref()
            .filter(|s| !s.is_empty())
            .or_else(|| self.model_id.value.as_deref().filter(|s| !s.is_empty()))
            .or_else(|| {
                self.gguf_file
                    .value
                    .as_deref()
                    .map(|f| f.strip_suffix(".gguf").unwrap_or(f))
                    .filter(|s| !s.is_empty())
            })
    }
}

/// Partial scrape from one endpoint before merge.
#[derive(Debug, Clone, Default)]
pub struct ModelDiscoveryPartial {
    pub model_path: Option<String>,
    pub model_alias: Option<String>,
    pub model_id: Option<String>,
    pub context_size: Option<u64>,
    pub native_context: Option<u64>,
    pub total_slots: Option<u32>,
}

/// Parse GET /props JSON (tolerant of unknown fields and legacy shapes).
pub fn parse_props_json(body: &str) -> Result<ModelDiscoveryPartial, String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("invalid /props JSON: {e}"))?;
    if !v.is_object() {
        return Err("invalid /props JSON: expected object".into());
    }

    let mut partial = ModelDiscoveryPartial::default();

    if let Some(path) = v.get("model_path").and_then(|x| x.as_str()) {
        let path = path.trim();
        if !path.is_empty() {
            partial.model_path = Some(path.to_string());
        }
    }
    if let Some(alias) = v
        .get("model_alias")
        .and_then(|x| x.as_str())
        .or_else(|| v.get("alias").and_then(|x| x.as_str()))
    {
        let alias = alias.trim();
        if !alias.is_empty() {
            partial.model_alias = Some(alias.to_string());
        }
    }
    if let Some(slots) = v.get("total_slots").and_then(as_u64) {
        partial.total_slots = Some(slots as u32);
    }

    if let Some(dgs) = v.get("default_generation_settings") {
        if partial.context_size.is_none()
            && let Some(n) = dgs.get("n_ctx").and_then(as_u64)
        {
            partial.context_size = Some(n);
        }
        // Legacy: model path lived under default_generation_settings.model
        if partial.model_path.is_none()
            && let Some(m) = dgs.get("model").and_then(|x| x.as_str())
        {
            let m = m.trim();
            if !m.is_empty() {
                partial.model_path = Some(m.to_string());
            }
        }
    }

    Ok(partial)
}

/// Parse GET /v1/models JSON (OpenAI-compatible list).
pub fn parse_v1_models_json(body: &str) -> Result<ModelDiscoveryPartial, String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("invalid /v1/models JSON: {e}"))?;
    if !v.is_object() {
        return Err("invalid /v1/models JSON: expected object".into());
    }

    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .cloned()
        .unwrap_or_default();

    if data.is_empty() {
        return Ok(ModelDiscoveryPartial::default());
    }

    let first = &data[0];
    let mut partial = ModelDiscoveryPartial::default();

    if let Some(id) = first.get("id").and_then(|x| x.as_str()) {
        let id = id.trim();
        if !id.is_empty() {
            partial.model_id = Some(id.to_string());
            // When id looks like a filesystem path / .gguf, treat as path too.
            if id.contains('/') || id.ends_with(".gguf") {
                partial.model_path = Some(id.to_string());
            }
        }
    }

    if let Some(meta) = first.get("meta") {
        if !meta.is_null()
            && let Some(n) = meta.get("n_ctx_train").and_then(as_u64)
        {
            partial.native_context = Some(n);
        }
        if !meta.is_null()
            && partial.context_size.is_none()
            && let Some(n) = meta.get("n_ctx").and_then(as_u64)
        {
            partial.context_size = Some(n);
        }
    }

    Ok(partial)
}

fn as_u64(v: &serde_json::Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_i64().and_then(|i| u64::try_from(i).ok()))
        .or_else(|| {
            v.as_f64()
                .and_then(|f| if f >= 0.0 { Some(f as u64) } else { None })
        })
}

fn set_string(field: &mut SourcedString, value: Option<String>, source: FieldSource) {
    let Some(value) = value.filter(|s| !s.is_empty()) else {
        return;
    };
    if field.value.is_some() && field.source.rank() >= source.rank() {
        return;
    }
    field.value = Some(value);
    field.source = source;
}

fn set_u64(field: &mut SourcedU64, value: Option<u64>, source: FieldSource) {
    let Some(value) = value else {
        return;
    };
    if field.value.is_some() && field.source.rank() >= source.rank() {
        return;
    }
    field.value = Some(value);
    field.source = source;
}

fn set_u32(field: &mut SourcedU32, value: Option<u32>, source: FieldSource) {
    let Some(value) = value else {
        return;
    };
    if field.value.is_some() && field.source.rank() >= source.rank() {
        return;
    }
    field.value = Some(value);
    field.source = source;
}

fn apply_partial(
    info: &mut RunningModelInfo,
    partial: &ModelDiscoveryPartial,
    source: FieldSource,
) {
    set_string(&mut info.model_path, partial.model_path.clone(), source);
    set_string(&mut info.model_id, partial.model_id.clone(), source);
    set_string(&mut info.name, partial.model_alias.clone(), source);
    set_u64(&mut info.context_size, partial.context_size, source);
    set_u64(&mut info.native_context, partial.native_context, source);
    set_u32(&mut info.total_slots, partial.total_slots, source);

    if source.rank() > info.primary_source.rank()
        && (partial.model_path.is_some()
            || partial.model_alias.is_some()
            || partial.model_id.is_some()
            || partial.context_size.is_some()
            || partial.native_context.is_some()
            || partial.total_slots.is_some())
    {
        info.primary_source = source;
    }
}

fn derive_from_path(info: &mut RunningModelInfo) {
    let Some(path) = info.model_path.value.clone() else {
        return;
    };
    let filename = path.rsplit(['/', '\\']).next().unwrap_or(&path).to_string();
    if filename.ends_with(".gguf") {
        set_string(
            &mut info.gguf_file,
            Some(filename.clone()),
            info.model_path.source,
        );
        let (model_name, quant) = parse_gguf_filename(&filename);
        if info.name.value.is_none() {
            set_string(&mut info.name, model_name, info.model_path.source);
        }
        set_string(&mut info.quant, quant, info.model_path.source);
    } else if info.gguf_file.value.is_none() && !filename.is_empty() {
        set_string(&mut info.gguf_file, Some(filename), info.model_path.source);
    }
}

fn derive_from_id(info: &mut RunningModelInfo) {
    let Some(id) = info.model_id.value.clone() else {
        return;
    };
    if id.ends_with(".gguf") || id.contains('/') {
        return; // handled as path
    }
    if info.name.value.is_none() {
        set_string(&mut info.name, Some(id.clone()), info.model_id.source);
    }
    let (model_name, quant) = parse_gguf_filename(&format!("{id}.gguf"));
    if info.name.value.is_none() {
        set_string(&mut info.name, model_name, info.model_id.source);
    }
    if info.quant.value.is_none() {
        set_string(&mut info.quant, quant, info.model_id.source);
    }
}

/// Merge discovery sources with fixed priority: props > models > process > preset.
pub fn merge_running_model(
    props: Option<&ModelDiscoveryPartial>,
    models: Option<&ModelDiscoveryPartial>,
    process: Option<&ServerConfig>,
    preset_name: Option<&str>,
    preset_model_path: Option<&str>,
    preset_context: Option<u64>,
) -> RunningModelInfo {
    let mut info = RunningModelInfo::default();

    if let Some(p) = props {
        apply_partial(&mut info, p, FieldSource::Props);
    }
    if let Some(m) = models {
        apply_partial(&mut info, m, FieldSource::Models);
    }

    let live = info.primary_source == FieldSource::Props
        || info.primary_source == FieldSource::Models
        || props.is_some_and(|p| {
            p.model_path.is_some()
                || p.model_alias.is_some()
                || p.context_size.is_some()
                || p.total_slots.is_some()
        })
        || models.is_some_and(|m| {
            m.model_id.is_some() || m.model_path.is_some() || m.native_context.is_some()
        });

    if let Some(cfg) = process {
        if !cfg.model_path.is_empty() {
            set_string(
                &mut info.model_path,
                Some(cfg.model_path.clone()),
                FieldSource::ProcessConfig,
            );
        }
        if cfg.context_size > 0 {
            set_u64(
                &mut info.context_size,
                Some(cfg.context_size),
                FieldSource::ProcessConfig,
            );
        }
        if cfg.parallel_slots > 0 {
            set_u32(
                &mut info.total_slots,
                Some(cfg.parallel_slots),
                FieldSource::ProcessConfig,
            );
        }
    }

    if let Some(path) = preset_model_path.filter(|s| !s.is_empty()) {
        set_string(
            &mut info.model_path,
            Some(path.to_string()),
            FieldSource::Preset,
        );
    }
    if let Some(name) = preset_name.filter(|s| !s.is_empty()) {
        set_string(&mut info.name, Some(name.to_string()), FieldSource::Preset);
    }
    if let Some(ctx) = preset_context.filter(|c| *c > 0) {
        set_u64(&mut info.context_size, Some(ctx), FieldSource::Preset);
    }

    derive_from_path(&mut info);
    derive_from_id(&mut info);

    // `detected` means a live llama-server endpoint contributed (not just preset/process).
    info.detected = live
        || matches!(
            info.primary_source,
            FieldSource::Props | FieldSource::Models
        );

    // If only process/preset filled fields, primary_source may still be None — fix it.
    if info.primary_source == FieldSource::None {
        for src in [
            info.model_path.source,
            info.name.source,
            info.model_id.source,
            info.context_size.source,
        ] {
            if src.rank() > info.primary_source.rank() {
                info.primary_source = src;
            }
        }
    }

    // Live detection: any props/models field source present
    if !info.detected {
        info.detected = [
            info.model_path.source,
            info.name.source,
            info.model_id.source,
            info.gguf_file.source,
            info.context_size.source,
            info.native_context.source,
            info.total_slots.source,
        ]
        .iter()
        .any(|s| matches!(s, FieldSource::Props | FieldSource::Models));
    }

    info
}

/// Sticky health tracker: clear running model only after N consecutive failures.
#[derive(Debug, Default)]
pub struct HealthStickiness {
    consecutive_failures: u32,
    threshold: u32,
}

impl HealthStickiness {
    pub fn new(threshold: u32) -> Self {
        Self {
            consecutive_failures: 0,
            threshold: threshold.max(1),
        }
    }

    /// Returns true when the caller should clear RunningModelInfo.
    pub fn on_health_result(&mut self, ok: bool) -> bool {
        if ok {
            self.consecutive_failures = 0;
            false
        } else {
            self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            self.consecutive_failures >= self.threshold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_props_modern() {
        let body = include_str!("../../tests/fixtures/llama_props.json");
        let p = parse_props_json(body).unwrap();
        assert_eq!(
            p.model_path.as_deref(),
            Some("/models/Qwen3.8-27B-UD-Q4_K_M.gguf")
        );
        assert_eq!(p.model_alias.as_deref(), Some("Qwen3.8-27B-UD-Q4_K_M"));
        assert_eq!(p.context_size, Some(131072));
        assert_eq!(p.total_slots, Some(1));
    }

    #[test]
    fn parse_props_legacy_model_field() {
        let body = include_str!("../../tests/fixtures/llama_props_legacy.json");
        let p = parse_props_json(body).unwrap();
        assert_eq!(
            p.model_path.as_deref(),
            Some("/opt/models/legacy-model.Q5_K_M.gguf")
        );
        assert_eq!(p.context_size, Some(8192));
        assert_eq!(p.total_slots, Some(2));
    }

    #[test]
    fn parse_props_missing_optional_fields() {
        let p = parse_props_json(r#"{"total_slots":1}"#).unwrap();
        assert!(p.model_path.is_none());
        assert!(p.model_alias.is_none());
        assert!(p.context_size.is_none());
        assert_eq!(p.total_slots, Some(1));
    }

    #[test]
    fn parse_props_unknown_fields_ok() {
        let body = include_str!("../../tests/fixtures/llama_props.json");
        assert!(parse_props_json(body).is_ok());
    }

    #[test]
    fn parse_props_invalid_json() {
        assert!(parse_props_json("not-json").is_err());
        assert!(parse_props_json("[1,2,3]").is_err());
    }

    #[test]
    fn parse_v1_models_ok() {
        let body = include_str!("../../tests/fixtures/llama_v1_models.json");
        let p = parse_v1_models_json(body).unwrap();
        assert_eq!(p.model_id.as_deref(), Some("Qwen3.8-27B-UD-Q4_K_M"));
        assert_eq!(p.native_context, Some(262144));
        assert!(p.model_path.is_none());
    }

    #[test]
    fn parse_v1_models_path_as_id() {
        let body = include_str!("../../tests/fixtures/llama_v1_models_path_id.json");
        let p = parse_v1_models_json(body).unwrap();
        assert!(p.model_path.as_ref().unwrap().ends_with(".gguf"));
        assert!(p.native_context.is_none()); // meta null
    }

    #[test]
    fn parse_v1_models_empty_list() {
        let p = parse_v1_models_json(r#"{"object":"list","data":[]}"#).unwrap();
        assert!(p.model_id.is_none());
        assert!(p.model_path.is_none());
    }

    #[test]
    fn parse_v1_models_invalid_json() {
        assert!(parse_v1_models_json("{").is_err());
    }

    #[test]
    fn merge_props_over_models() {
        let props = ModelDiscoveryPartial {
            model_path: Some("/from/props.gguf".into()),
            model_alias: Some("FromProps".into()),
            context_size: Some(4096),
            ..Default::default()
        };
        let models = ModelDiscoveryPartial {
            model_id: Some("FromModels".into()),
            model_path: Some("/from/models.gguf".into()),
            native_context: Some(8192),
            context_size: Some(2048),
            ..Default::default()
        };
        let info = merge_running_model(Some(&props), Some(&models), None, None, None, None);
        assert!(info.detected);
        assert_eq!(info.primary_source, FieldSource::Props);
        assert_eq!(info.model_path.value.as_deref(), Some("/from/props.gguf"));
        assert_eq!(info.name.value.as_deref(), Some("FromProps"));
        assert_eq!(info.context_size.value, Some(4096));
        assert_eq!(info.native_context.value, Some(8192));
        assert_eq!(info.model_id.value.as_deref(), Some("FromModels"));
        assert_eq!(info.gguf_file.value.as_deref(), Some("props.gguf"));
    }

    #[test]
    fn merge_fallback_props_to_models() {
        let models = ModelDiscoveryPartial {
            model_id: Some("OnlyModels".into()),
            native_context: Some(65536),
            ..Default::default()
        };
        let info = merge_running_model(None, Some(&models), None, None, None, None);
        assert!(info.detected);
        assert_eq!(info.primary_source, FieldSource::Models);
        assert_eq!(info.name.value.as_deref(), Some("OnlyModels"));
        assert_eq!(info.native_context.value, Some(65536));
    }

    #[test]
    fn merge_process_then_preset_fallback() {
        let cfg = ServerConfig {
            model_path: "/proc/model.Q4_0.gguf".into(),
            context_size: 16384,
            parallel_slots: 2,
            ..empty_server_config()
        };
        let info = merge_running_model(
            None,
            None,
            Some(&cfg),
            Some("Preset Name"),
            Some("/preset/other.gguf"),
            Some(8192),
        );
        assert!(!info.detected);
        assert_eq!(
            info.model_path.value.as_deref(),
            Some("/proc/model.Q4_0.gguf")
        );
        assert_eq!(info.model_path.source, FieldSource::ProcessConfig);
        assert_eq!(info.context_size.value, Some(16384));
        // name: process path derives name; process path rank > preset for path-derived name
        assert!(info.name.value.is_some());
        assert_eq!(info.gguf_file.value.as_deref(), Some("model.Q4_0.gguf"));
    }

    #[test]
    fn health_stickiness_single_timeout_keeps_state() {
        let mut h = HealthStickiness::new(3);
        assert!(!h.on_health_result(false));
        assert!(!h.on_health_result(false));
        assert!(h.on_health_result(false));
    }

    #[test]
    fn health_stickiness_recovery_resets() {
        let mut h = HealthStickiness::new(3);
        assert!(!h.on_health_result(false));
        assert!(!h.on_health_result(true));
        assert!(!h.on_health_result(false));
        assert!(!h.on_health_result(false));
        assert!(h.on_health_result(false));
    }

    #[test]
    fn full_merge_from_fixtures() {
        let props =
            parse_props_json(include_str!("../../tests/fixtures/llama_props.json")).unwrap();
        let models =
            parse_v1_models_json(include_str!("../../tests/fixtures/llama_v1_models.json"))
                .unwrap();
        let info = merge_running_model(
            Some(&props),
            Some(&models),
            None,
            Some("Preset"),
            None,
            None,
        );
        assert!(info.detected);
        assert_eq!(info.display_name(), Some("Qwen3.8-27B-UD-Q4_K_M"));
        assert_eq!(
            info.gguf_file.value.as_deref(),
            Some("Qwen3.8-27B-UD-Q4_K_M.gguf")
        );
        assert_eq!(info.context_size.value, Some(131072));
        assert_eq!(info.native_context.value, Some(262144));
        assert_eq!(
            info.model_id.value.as_deref(),
            Some("Qwen3.8-27B-UD-Q4_K_M")
        );
    }

    fn empty_server_config() -> ServerConfig {
        ServerConfig {
            model_path: String::new(),
            context_size: 0,
            ctk: String::new(),
            ctv: String::new(),
            tensor_split: String::new(),
            batch_size: 0,
            ubatch_size: 0,
            no_mmap: false,
            port: 8080,
            ngram_spec: false,
            parallel_slots: 0,
            gpu_layers: None,
            mlock: false,
            flash_attn: String::new(),
            split_mode: String::new(),
            main_gpu: None,
            threads: None,
            threads_batch: None,
            rope_scaling: String::new(),
            rope_freq_base: None,
            rope_freq_scale: None,
            draft_model: String::new(),
            draft_min: None,
            draft_max: None,
            spec_ngram_size: None,
            seed: None,
            system_prompt_file: String::new(),
            extra_args: String::new(),
        }
    }
}
