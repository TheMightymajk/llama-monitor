pub const INDEX_HTML: &str = include_str!("../../static/index.html");
pub const STYLE_CSS: &str = include_str!("../../static/style.css");
pub const APP_JS: &str = include_str!("../../static/app.js");
pub const MANIFEST_JSON: &str = include_str!("../../static/manifest.json");
pub const SW_JS: &str = include_str!("../../static/sw.js");
pub const ICON_SVG: &str = include_str!("../../static/icon.svg");
pub const MARKED_JS: &str = include_str!("../../static/vendor/marked.min.js");

#[cfg(test)]
mod fmt_tokens_js {
    fn fmt_tokens_source() -> &'static str {
        let js = super::APP_JS;
        let start = js
            .find("function fmtTokens(")
            .expect("static/app.js must define function fmtTokens");
        let after = &js[start..];
        let end = after[1..]
            .find("\nfunction ")
            .expect("fmtTokens must be followed by another function");
        &after[..=end]
    }

    fn assert_fmt_tokens_cases(cases: &[(&str, &str)]) {
        let mut script = String::from(fmt_tokens_source());
        script.push_str("\nconst cases = [\n");
        for (n, expected) in cases {
            script.push_str(&format!("  [{n}, {expected:?}],\n"));
        }
        script.push_str("];\n");
        script.push_str(
            r#"
let failed = 0;
for (const [n, expected] of cases) {
  const got = fmtTokens(n);
  if (got !== expected) {
    console.error(`fmtTokens(${n}): got ${JSON.stringify(got)}, expected ${JSON.stringify(expected)}`);
    failed++;
  }
}
if (failed) process.exit(1);
"#,
        );
        let output = std::process::Command::new("node")
            .arg("-e")
            .arg(&script)
            .output()
            .expect("node must be available to test fmtTokens");
        assert!(
            output.status.success(),
            "fmtTokens JS tests failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn fmt_tokens_compact_units_k_m_b_t() {
        assert_fmt_tokens_cases(&[
            ("1000", "1.0k"),
            ("1000000", "1.0M"),
            ("1000000000", "1.0B"),
            ("1000000000000", "1.0T"),
            ("753500", "753.5k"),
            ("4660000", "4.66M"),
            ("34200000", "34.2M"),
            ("1240000000", "1.24B"),
            ("2180000000000", "2.18T"),
        ]);
    }

    #[test]
    fn fmt_tokens_values_above_u32_do_not_wrap() {
        assert_fmt_tokens_cases(&[
            ("4294967295", "4.29B"),
            ("4294967296", "4.29B"),
            ("5000000000", "5.0B"),
            ("12000000000", "12.0B"),
            ("3000000000", "3.0B"),
            ("17000000000", "17.0B"),
        ]);
    }

    #[test]
    fn fmt_tokens_does_not_use_32bit_bitwise_coercion() {
        let src = fmt_tokens_source();
        assert!(
            src.contains("Number.MAX_SAFE_INTEGER"),
            "fmtTokens must document JS Number.MAX_SAFE_INTEGER"
        );
        assert!(!src.contains("| 0"), "fmtTokens must not use | 0");
        assert!(!src.contains(">>> 0"), "fmtTokens must not use >>> 0");
        assert!(
            !src.contains("<<") && !src.contains(">>"),
            "fmtTokens must not use bitwise shifts"
        );
    }

    #[test]
    fn peak_and_mtp_ui_bind_to_usage_lifetime() {
        let js = super::APP_JS;
        assert!(js.contains("usage.peak_context_tokens"));
        assert!(js.contains("usage.mtp_acceptance_ratio"));
        assert!(js.contains("lifetime high-water"));
        assert!(js.contains("lifetime speculative decode"));
        assert!(js.contains("drafted · lifetime"));
        assert!(!js.contains("session high-water"));
        assert!(!js.contains("l.n_tokens_max"));
        assert!(!js.contains("l.spec_acceptance_ratio"));
        assert!(!js.contains("l.spec_accepted_tokens"));
        assert!(!js.contains("l.spec_draft_tokens"));
        assert!(super::INDEX_HTML.contains("lifetime high-water"));
        assert!(super::INDEX_HTML.contains("lifetime speculative decode"));
        assert!(!super::INDEX_HTML.contains("session high-water"));
    }

    #[test]
    fn live_speed_ui_binds_inference_phase() {
        let js = super::APP_JS;
        assert!(js.contains("l.inference_phase"));
        assert!(js.contains("phase === 'prefill'"));
        assert!(js.contains("phase === 'generating'"));
        assert!(super::INDEX_HTML.contains("id=\"m-prompt-sub\""));
        assert!(super::INDEX_HTML.contains("id=\"m-gen-sub\""));
    }
}
