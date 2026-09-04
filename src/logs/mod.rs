//! Log buffer and external file follow (tail -F semantics).

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::state::AppState;

/// Hard cap on retained log lines in memory.
pub const MAX_LOG_LINES: usize = 2000;
/// On first open / rotation, seed at most this many trailing lines.
pub const INITIAL_TAIL_LINES: usize = 500;
/// Reject / truncate individual lines above this size.
pub const MAX_LINE_BYTES: usize = 64 * 1024;
const POLL_READ_CHUNK: usize = 64 * 1024;
const EXTERNAL_LOG_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// Background task: follow configured external log file (or idle when unset).
pub async fn external_log_poller(state: AppState) {
    let mut tailer: Option<ExternalLogTailer> = None;
    let mut last_path: Option<PathBuf> = None;

    loop {
        let path = state.external_log_path.lock().unwrap().clone();

        if path != last_path {
            // Path change: drop previous view so sources never mix.
            state.log_buffer.lock().unwrap().clear();
            tailer = path.as_ref().map(|p| ExternalLogTailer::new(p.clone()));
            last_path = path.clone();
            if path.is_none() {
                let mut src = state.log_source.lock().unwrap();
                if src.kind == LogSourceKind::ExternalFile {
                    *src = LogSourceInfo::default();
                }
            }
        }

        if let Some(ref mut t) = tailer {
            let new_lines = {
                let mut buf = state.log_buffer.lock().unwrap();
                t.poll(&mut buf)
            };
            for line in &new_lines {
                if let Some(n) = crate::usage::parse_cache_n(line) {
                    let mut usage = state.usage.lock().unwrap();
                    if usage.add_cached(n) {
                        let _ = usage.maybe_save(&state.usage_path, false);
                    }
                }
            }
            let line_count = state.log_buffer.lock().unwrap().len();
            let mut src = state.log_source.lock().unwrap();
            src.kind = LogSourceKind::ExternalFile;
            src.status = t.status();
            src.file_name = t.file_name();
            src.error = t.error().map(|s| s.to_string());
            src.line_count = line_count;
        }

        tokio::time::sleep(EXTERNAL_LOG_POLL_INTERVAL).await;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogSourceKind {
    ManagedProcess,
    ExternalFile,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogSourceStatus {
    Connected,
    WaitingForFile,
    Error,
    Idle,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LogSourceInfo {
    pub kind: LogSourceKind,
    pub status: LogSourceStatus,
    pub file_name: Option<String>,
    pub error: Option<String>,
    pub line_count: usize,
}

impl Default for LogSourceInfo {
    fn default() -> Self {
        Self {
            kind: LogSourceKind::None,
            status: LogSourceStatus::Idle,
            file_name: None,
            error: None,
            line_count: 0,
        }
    }
}

/// Bounded ring of log lines.
#[derive(Debug, Default)]
pub struct LogBuffer {
    lines: VecDeque<String>,
    max_lines: usize,
}

impl LogBuffer {
    pub fn new(max_lines: usize) -> Self {
        Self {
            lines: VecDeque::with_capacity(max_lines.min(1024)),
            max_lines: max_lines.max(1),
        }
    }

    pub fn clear(&mut self) {
        self.lines.clear();
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn push_line(&mut self, mut line: String) {
        if line.len() > MAX_LINE_BYTES {
            line.truncate(MAX_LINE_BYTES);
            line.push_str("…[truncated]");
        }
        if self.lines.len() >= self.max_lines {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    pub fn snapshot(&self) -> Vec<String> {
        self.lines.iter().cloned().collect()
    }
}

/// Expand a leading `~/` to the user home directory.
pub fn expand_tilde(path: &str) -> PathBuf {
    let path = path.trim();
    if path.is_empty() {
        return PathBuf::new();
    }
    if path == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        let mut home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        home.push(rest);
        return home;
    }
    PathBuf::from(path)
}

fn lossy_line_from_bytes(bytes: &[u8]) -> String {
    let mut s = String::from_utf8_lossy(bytes).into_owned();
    if s.ends_with('\r') {
        s.pop();
    }
    s
}

/// Split buffer on `\n`, keeping an incomplete trailing fragment.
pub fn split_lines(buf: &mut Vec<u8>) -> Vec<String> {
    let mut lines = Vec::new();
    loop {
        let Some(pos) = buf.iter().position(|&b| b == b'\n') else {
            break;
        };
        let mut line_bytes = buf.drain(..=pos).collect::<Vec<u8>>();
        line_bytes.pop(); // drop \n
        if line_bytes.len() > MAX_LINE_BYTES {
            line_bytes.truncate(MAX_LINE_BYTES);
            let mut s = lossy_line_from_bytes(&line_bytes);
            s.push_str("…[truncated]");
            lines.push(s);
        } else {
            lines.push(lossy_line_from_bytes(&line_bytes));
        }
    }
    lines
}

#[derive(Debug)]
struct FileIdentity {
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    len: u64,
}

#[cfg(unix)]
fn file_identity(meta: &std::fs::Metadata) -> FileIdentity {
    use std::os::unix::fs::MetadataExt;
    FileIdentity {
        dev: meta.dev(),
        ino: meta.ino(),
        len: meta.len(),
    }
}

#[cfg(not(unix))]
fn file_identity(meta: &std::fs::Metadata) -> FileIdentity {
    FileIdentity { len: meta.len() }
}

#[cfg(unix)]
fn same_file(a: &FileIdentity, b: &FileIdentity) -> bool {
    a.dev == b.dev && a.ino == b.ino
}

#[cfg(not(unix))]
fn same_file(_a: &FileIdentity, _b: &FileIdentity) -> bool {
    true
}

/// Follow a log file like `tail -F`.
pub struct ExternalLogTailer {
    path: PathBuf,
    file: Option<File>,
    identity: Option<FileIdentity>,
    offset: u64,
    incomplete: Vec<u8>,
    discard_until_newline: bool,
    status: LogSourceStatus,
    error: Option<String>,
    seeded: bool,
}

impl ExternalLogTailer {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            file: None,
            identity: None,
            offset: 0,
            incomplete: Vec::new(),
            discard_until_newline: false,
            status: LogSourceStatus::WaitingForFile,
            error: None,
            seeded: false,
        }
    }

    pub fn status(&self) -> LogSourceStatus {
        self.status
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn file_name(&self) -> Option<String> {
        self.path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
    }

    /// Poll once. Returns newly completed lines added to `buffer`.
    pub fn poll(&mut self, buffer: &mut LogBuffer) -> Vec<String> {
        match self.poll_inner(buffer) {
            Ok(lines) => lines,
            Err(e) => {
                self.status = LogSourceStatus::Error;
                self.error = Some(e);
                self.file = None;
                self.identity = None;
                Vec::new()
            }
        }
    }

    fn poll_inner(&mut self, buffer: &mut LogBuffer) -> Result<Vec<String>, String> {
        let meta = match std::fs::metadata(&self.path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.status = LogSourceStatus::WaitingForFile;
                self.error = None;
                self.file = None;
                self.identity = None;
                self.offset = 0;
                self.incomplete.clear();
                self.discard_until_newline = false;
                self.seeded = false;
                return Ok(Vec::new());
            }
            Err(e) => return Err(format!("stat {}: {e}", self.path.display())),
        };

        if !meta.is_file() {
            return Err(format!("{} is not a regular file", self.path.display()));
        }

        let id = file_identity(&meta);
        let mut new_lines = Vec::new();

        let need_reopen = match &self.identity {
            None => true,
            Some(prev) => !same_file(prev, &id) || self.file.is_none(),
        };

        if let Some(prev) = &self.identity
            && same_file(prev, &id)
            && id.len < self.offset
        {
            self.offset = 0;
            self.incomplete.clear();
            self.discard_until_newline = false;
            if let Some(f) = self.file.as_mut() {
                f.seek(SeekFrom::Start(0))
                    .map_err(|e| format!("seek after truncate: {e}"))?;
            }
        }

        if need_reopen {
            new_lines.extend(self.open_file(&id, buffer)?);
        }

        new_lines.extend(self.read_new_bytes(buffer)?);
        self.identity = Some(FileIdentity {
            #[cfg(unix)]
            dev: id.dev,
            #[cfg(unix)]
            ino: id.ino,
            len: std::fs::metadata(&self.path)
                .map(|m| m.len())
                .unwrap_or(id.len),
        });
        self.status = LogSourceStatus::Connected;
        self.error = None;
        Ok(new_lines)
    }

    fn open_file(
        &mut self,
        id: &FileIdentity,
        buffer: &mut LogBuffer,
    ) -> Result<Vec<String>, String> {
        let file = OpenOptions::new()
            .read(true)
            .open(&self.path)
            .map_err(|e| format!("open {}: {e}", self.path.display()))?;

        self.file.replace(file);
        self.incomplete.clear();
        self.discard_until_newline = false;
        let mut new_lines = Vec::new();

        if !self.seeded {
            let lines = read_last_lines(&self.path, INITIAL_TAIL_LINES)
                .map_err(|e| format!("seed {}: {e}", self.path.display()))?;
            for line in &lines {
                buffer.push_line(line.clone());
            }
            new_lines = lines;
            let len = id.len;
            if let Some(f) = self.file.as_mut() {
                f.seek(SeekFrom::Start(len))
                    .map_err(|e| format!("seek eof: {e}"))?;
            }
            self.offset = len;
            self.seeded = true;
        } else {
            self.offset = 0;
            if let Some(f) = self.file.as_mut() {
                f.seek(SeekFrom::Start(0))
                    .map_err(|e| format!("seek start: {e}"))?;
            }
        }
        self.identity = Some(FileIdentity {
            #[cfg(unix)]
            dev: id.dev,
            #[cfg(unix)]
            ino: id.ino,
            len: id.len,
        });
        Ok(new_lines)
    }

    fn read_new_bytes(&mut self, buffer: &mut LogBuffer) -> Result<Vec<String>, String> {
        let Some(file) = self.file.as_mut() else {
            return Ok(Vec::new());
        };
        file.seek(SeekFrom::Start(self.offset))
            .map_err(|e| format!("seek: {e}"))?;

        let mut data = Vec::new();
        let mut chunk = vec![0u8; POLL_READ_CHUNK];
        loop {
            let n = file.read(&mut chunk).map_err(|e| format!("read: {e}"))?;
            if n == 0 {
                break;
            }
            self.offset += n as u64;
            data.extend_from_slice(&chunk[..n]);
        }
        Ok(self.ingest_bytes(&data, buffer))
    }

    fn ingest_bytes(&mut self, bytes: &[u8], buffer: &mut LogBuffer) -> Vec<String> {
        let mut input = bytes;
        if self.discard_until_newline {
            if let Some(pos) = input.iter().position(|&b| b == b'\n') {
                input = &input[pos + 1..];
                self.discard_until_newline = false;
            } else {
                return Vec::new();
            }
        }

        self.incomplete.extend_from_slice(input);

        if self.incomplete.len() > MAX_LINE_BYTES && !self.incomplete.contains(&b'\n') {
            let truncated: Vec<u8> = self.incomplete.drain(..MAX_LINE_BYTES).collect();
            let mut s = lossy_line_from_bytes(&truncated);
            s.push_str("…[truncated]");
            buffer.push_line(s.clone());
            self.incomplete.clear();
            self.discard_until_newline = true;
            return vec![s];
        }

        let lines = split_lines(&mut self.incomplete);
        for line in &lines {
            buffer.push_line(line.clone());
        }
        lines
    }
}

/// Read the last `max_lines` complete lines from `path` (for initial seed).
pub fn read_last_lines(path: &Path, max_lines: usize) -> Result<Vec<String>, String> {
    let mut file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    let len = file.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    if len == 0 {
        return Ok(Vec::new());
    }

    let mut pos = len;
    let mut acc: Vec<u8> = Vec::new();
    let mut newline_count = 0usize;
    let chunk_size = 8192u64;

    while pos > 0 && newline_count <= max_lines {
        let read_size = chunk_size.min(pos);
        pos -= read_size;
        file.seek(SeekFrom::Start(pos)).map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; read_size as usize];
        file.read_exact(&mut buf).map_err(|e| e.to_string())?;
        newline_count += buf.iter().filter(|&&b| b == b'\n').count();
        acc.splice(0..0, buf);
        if acc.len() > MAX_LINE_BYTES.saturating_mul(max_lines.saturating_add(1)) {
            break;
        }
    }

    if pos > 0
        && let Some(i) = acc.iter().position(|&b| b == b'\n')
    {
        acc.drain(..=i);
    }

    let mut text = acc;
    if let Some(last_nl) = text.iter().rposition(|&b| b == b'\n') {
        text.truncate(last_nl + 1);
    } else if pos != 0 {
        text.clear();
    }

    let mut lines = Vec::new();
    let mut start = 0;
    for (i, &b) in text.iter().enumerate() {
        if b == b'\n' {
            let mut slice = &text[start..i];
            if slice.last() == Some(&b'\r') {
                slice = &slice[..slice.len() - 1];
            }
            let line = if slice.len() > MAX_LINE_BYTES {
                let mut s = lossy_line_from_bytes(&slice[..MAX_LINE_BYTES]);
                s.push_str("…[truncated]");
                s
            } else {
                lossy_line_from_bytes(slice)
            };
            lines.push(line);
            start = i + 1;
        }
    }
    if start < text.len() && pos == 0 {
        let slice = &text[start..];
        let line = if slice.len() > MAX_LINE_BYTES {
            let mut s = lossy_line_from_bytes(&slice[..MAX_LINE_BYTES]);
            s.push_str("…[truncated]");
            s
        } else {
            lossy_line_from_bytes(slice)
        };
        lines.push(line);
    }

    if lines.len() > max_lines {
        lines = lines.split_off(lines.len() - max_lines);
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("llama-monitor-logs-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn expand_tilde_home() {
        let p = expand_tilde("~/foo/bar.log");
        assert!(p.ends_with("foo/bar.log"));
        assert!(!p.to_string_lossy().starts_with('~'));
    }

    #[test]
    fn split_lines_crlf_and_partial() {
        let mut buf = b"a\r\nb\r\nc".to_vec();
        let lines = split_lines(&mut buf);
        assert_eq!(lines, vec!["a", "b"]);
        assert_eq!(buf, b"c");
    }

    #[test]
    fn lossy_invalid_utf8() {
        let mut buf = vec![b'h', b'i', 0xff, b'\n'];
        let lines = split_lines(&mut buf);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("hi"));
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn buffer_max_lines() {
        let mut b = LogBuffer::new(3);
        b.push_line("1".into());
        b.push_line("2".into());
        b.push_line("3".into());
        b.push_line("4".into());
        assert_eq!(b.snapshot(), vec!["2", "3", "4"]);
    }

    #[test]
    fn buffer_truncates_long_line() {
        let mut b = LogBuffer::new(10);
        let long = "x".repeat(MAX_LINE_BYTES + 100);
        b.push_line(long);
        let s = &b.snapshot()[0];
        assert!(s.len() < MAX_LINE_BYTES + 20);
        assert!(s.ends_with("…[truncated]"));
    }

    #[test]
    fn read_last_lines_large_file() {
        let dir = tmp_dir();
        let path = dir.join("big.log");
        {
            let mut f = File::create(&path).unwrap();
            for i in 0..2000 {
                writeln!(f, "line-{i}").unwrap();
            }
        }
        let lines = read_last_lines(&path, 500).unwrap();
        assert_eq!(lines.len(), 500);
        assert_eq!(lines.first().unwrap(), "line-1500");
        assert_eq!(lines.last().unwrap(), "line-1999");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_file_seed() {
        let dir = tmp_dir();
        let path = dir.join("empty.log");
        File::create(&path).unwrap();
        let lines = read_last_lines(&path, 500).unwrap();
        assert!(lines.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tailer_append_full_line() {
        let dir = tmp_dir();
        let path = dir.join("app.log");
        std::fs::write(&path, "one\n").unwrap();

        let mut buf = LogBuffer::new(2000);
        let mut tail = ExternalLogTailer::new(path.clone());
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.snapshot(), vec!["one"]);
        assert_eq!(tail.status(), LogSourceStatus::Connected);

        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            writeln!(f, "two").unwrap();
        }
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.snapshot(), vec!["one", "two"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tailer_partial_line_then_complete() {
        let dir = tmp_dir();
        let path = dir.join("partial.log");
        File::create(&path).unwrap();

        let mut buf = LogBuffer::new(2000);
        let mut tail = ExternalLogTailer::new(path.clone());
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.len(), 0);

        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            write!(f, "hel").unwrap();
        }
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.len(), 0);

        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            write!(f, "lo").unwrap();
            writeln!(f).unwrap();
        }
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.snapshot(), vec!["hello"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tailer_waiting_then_create() {
        let dir = tmp_dir();
        let path = dir.join("late.log");
        let mut buf = LogBuffer::new(2000);
        let mut tail = ExternalLogTailer::new(path.clone());
        let _ = tail.poll(&mut buf);
        assert_eq!(tail.status(), LogSourceStatus::WaitingForFile);

        std::fs::write(&path, "appeared\n").unwrap();
        let _ = tail.poll(&mut buf);
        assert_eq!(tail.status(), LogSourceStatus::Connected);
        assert_eq!(buf.snapshot(), vec!["appeared"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tailer_truncation() {
        let dir = tmp_dir();
        let path = dir.join("trunc.log");
        std::fs::write(&path, "a\nb\nc\n").unwrap();

        let mut buf = LogBuffer::new(2000);
        let mut tail = ExternalLogTailer::new(path.clone());
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.len(), 3);

        std::fs::write(&path, "").unwrap();
        let _ = tail.poll(&mut buf);

        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            writeln!(f, "fresh").unwrap();
        }
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.snapshot().last().map(String::as_str), Some("fresh"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tailer_replace_file_no_dup_seed() {
        let dir = tmp_dir();
        let path = dir.join("rot.log");
        std::fs::write(&path, "old1\nold2\n").unwrap();

        let mut buf = LogBuffer::new(2000);
        let mut tail = ExternalLogTailer::new(path.clone());
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.snapshot(), vec!["old1", "old2"]);

        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, "new1\n").unwrap();
        let _ = tail.poll(&mut buf);
        let snap = buf.snapshot();
        assert_eq!(snap.iter().filter(|l| *l == "new1").count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tailer_no_permission() {
        let dir = tmp_dir();
        let path = dir.join("noperm.log");
        std::fs::write(&path, "secret\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&path).unwrap().permissions();
            perms.set_mode(0o000);
            std::fs::set_permissions(&path, perms).unwrap();

            let mut buf = LogBuffer::new(2000);
            let mut tail = ExternalLogTailer::new(path.clone());
            let _ = tail.poll(&mut buf);
            assert_eq!(tail.status(), LogSourceStatus::Error);

            let mut perms = std::fs::metadata(&path).unwrap().permissions();
            perms.set_mode(0o644);
            std::fs::set_permissions(&path, perms).unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tailer_crlf() {
        let dir = tmp_dir();
        let path = dir.join("crlf.log");
        std::fs::write(&path, b"a\r\nb\r\n").unwrap();
        let mut buf = LogBuffer::new(2000);
        let mut tail = ExternalLogTailer::new(path);
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.snapshot(), vec!["a", "b"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn managed_process_buffer_still_works() {
        let mut buf = LogBuffer::new(2000);
        buf.push_line("[monitor] started".into());
        buf.push_line("slot update".into());
        assert_eq!(buf.len(), 2);
        buf.clear();
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn tailer_invalid_utf8_line() {
        let dir = tmp_dir();
        let path = dir.join("utf8.log");
        std::fs::write(&path, [b'o', b'k', 0xff, b'\n']).unwrap();
        let mut buf = LogBuffer::new(2000);
        let mut tail = ExternalLogTailer::new(path);
        let _ = tail.poll(&mut buf);
        assert_eq!(buf.len(), 1);
        assert!(buf.snapshot()[0].starts_with("ok"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
