'use strict';

function $(id) {
    return document.getElementById(id);
}

function text(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
}

function switchTab(name) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    $('page-' + name).classList.add('active');
    $('tab-' + name).classList.add('active');
}

let presets = [];
let serverRunning = false;
let serverStartedAt = null;
let wsConnected = false;
let settingsSaveTimer = null;
let lastRunningModel = null;
let logAutoScroll = true;
let logUserPinnedBottom = true;
let logSelecting = false;
let lastLogView = { logs: [], log_source: {} };
let lastRenderedLogs = [];
let lastEnergySnapshot = null;

function logSourceLabel(kind) {
    if (kind === 'external_file') return 'External file';
    if (kind === 'managed_process') return 'Managed process';
    return 'No source';
}

function logStatusLabel(status) {
    if (status === 'connected') return 'Connected';
    if (status === 'waiting_for_file') return 'Waiting for file';
    if (status === 'error') return 'Error';
    return 'Idle';
}

function logPanelHasLiveSelection() {
    if (logSelecting) return true;
    const el = $('log-panel');
    const sel = window.getSelection();
    if (!el || !sel || sel.rangeCount === 0 || sel.isCollapsed) return false;
    return el.contains(sel.anchorNode) || el.contains(sel.focusNode);
}

function logsArePrefix(prev, next) {
    if (prev.length > next.length) return false;
    for (let i = 0; i < prev.length; i++) {
        if (prev[i] !== next[i]) return false;
    }
    return true;
}

function applyLogPanel(d) {
    if (d) lastLogView = d;
    const src = lastLogView.log_source || {};
    text('log-source-kind', logSourceLabel(src.kind));
    text('log-source-file', src.file_name || '—');
    const st = logStatusLabel(src.status);
    text('log-source-status', src.error ? st + ': ' + src.error : st);

    const all = lastLogView.logs || [];
    const filter = ($('log-filter').value || '').trim().toLowerCase();
    const visible = filter
        ? all.filter(l => String(l).toLowerCase().includes(filter))
        : all;
    text('log-line-count', visible.length + ' / ' + all.length);

    const el = $('log-panel');
    const selecting = logPanelHasLiveSelection();
    const unchanged = visible.length === lastRenderedLogs.length
        && logsArePrefix(lastRenderedLogs, visible);
    const canAppend = lastRenderedLogs.length > 0
        && visible.length > lastRenderedLogs.length
        && logsArePrefix(lastRenderedLogs, visible);
    const textNode = el.firstChild && el.firstChild.nodeType === Node.TEXT_NODE
        ? el.firstChild
        : null;

    if (unchanged) {
        // already on screen
    } else if (canAppend && textNode) {
        // appendData keeps existing ranges; replacing textContent would not
        textNode.appendData('\n' + visible.slice(lastRenderedLogs.length).join('\n'));
        lastRenderedLogs = visible.slice();
    } else if (!selecting) {
        el.textContent = visible.join('\n');
        lastRenderedLogs = visible.slice();
    }
    if (logAutoScroll && logUserPinnedBottom && !selecting) {
        el.scrollTop = el.scrollHeight;
    }
    text('badge-logs', all.length > 0 ? ' ' + all.length : '');
}

function collectSettings() {
    return {
        preset_id: $('preset-select').value,
        port: parseInt($('port').value) || 8080,
        llama_server_path: $('set-server-path').value,
        llama_server_cwd: $('set-server-cwd').value,
        models_dir: '',
        external_log_file: $('set-external-log').value,
    };
}

function saveSettings() {
    clearTimeout(settingsSaveTimer);
    settingsSaveTimer = setTimeout(() => {
        fetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(collectSettings()),
        }).catch(() => {});
    }, 400);
}

function applySettings(s) {
    if (!s) return;
    if (s.port) $('port').value = s.port;
    if (s.llama_server_path !== undefined) $('set-server-path').value = s.llama_server_path;
    if (s.llama_server_cwd !== undefined) $('set-server-cwd').value = s.llama_server_cwd;
    if (s.external_log_file !== undefined) $('set-external-log').value = s.external_log_file;
}

$('controls').addEventListener('input', saveSettings);
$('controls').addEventListener('change', saveSettings);

async function loadPresets(selectId) {
    const [presetsResp, settingsResp] = await Promise.all([
        fetch('/api/presets'),
        selectId === undefined ? fetch('/api/settings') : Promise.resolve(null),
    ]);
    presets = await presetsResp.json();
    const saved = settingsResp ? await settingsResp.json() : null;

    const sel = $('preset-select');
    sel.replaceChildren();
    presets.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.name;
        sel.appendChild(opt);
    });

    const targetId = selectId ?? (saved?.preset_id || null);
    if (targetId && presets.find(p => p.id === targetId)) {
        sel.value = targetId;
    } else if (presets.length > 0) {
        sel.value = presets[0].id;
    }

    if (selectId === undefined && saved) applySettings(saved);
    saveSettings();
    refreshModelCard();
}

loadPresets();
loadGpuEnv();

async function loadGpuEnv() {
    try {
        const resp = await fetch('/api/gpu-env');
        const data = await resp.json();
        const env = data.env;
        const archs = data.architectures;
        const detected = data.detected;

        const sel = $('gpu-env-arch');
        sel.replaceChildren();
        archs.forEach(a => {
            const opt = document.createElement('option');
            opt.value = a.id;
            let label = a.name;
            if (detected && detected.arch === a.id) label += ' (detected)';
            opt.textContent = label;
            sel.appendChild(opt);
        });
        sel.value = env.arch;

        $('gpu-env-devices').value = env.devices;
        $('gpu-env-rocm-path').value = env.rocm_path || '/opt/rocm';

        const infoEl = $('gpu-detected-info');
        const summaryInfo = $('gpu-env-info');
        if (detected) {
            infoEl.textContent = 'Detected: ' + detected.count + 'x ' + detected.arch + ' (' + detected.names.join(', ') + ')';
            summaryInfo.textContent = '\u2014 ' + detected.count + 'x ' + detected.arch;
        } else {
            infoEl.textContent = 'No GPU detected via rocminfo/nvidia-smi';
            summaryInfo.textContent = '';
        }
    } catch (err) {
        console.error('Failed to load GPU env:', err);
    }
}

function openConfigModal() {
    if (lastEnergySnapshot) {
        const price = lastEnergySnapshot.price_per_kwh;
        const threshold = lastEnergySnapshot.inference_util_threshold;
        if (Number.isFinite(price)) $('set-energy-price').value = price;
        if (Number.isFinite(threshold)) $('set-energy-util').value = threshold;
    }
    $('config-modal').classList.add('open');
}
function closeConfigModal() { $('config-modal').classList.remove('open'); }

function saveConfig() {
    const pricePerKwh = parseFloat($('set-energy-price').value);
    const inferenceUtilThreshold = parseFloat($('set-energy-util').value);
    if (!Number.isFinite(pricePerKwh) || pricePerKwh < 0 ||
        !Number.isFinite(inferenceUtilThreshold) ||
        inferenceUtilThreshold < 0 || inferenceUtilThreshold > 100) {
        showToast('Energy settings are invalid', 'error');
        return;
    }

    clearTimeout(settingsSaveTimer);
    fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(collectSettings()),
    }).catch(() => {});

    const env = {
        arch: $('gpu-env-arch').value,
        devices: $('gpu-env-devices').value.trim(),
        rocm_path: $('gpu-env-rocm-path').value.trim() || '/opt/rocm',
        extra_env: [],
    };
    fetch('/api/gpu-env', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(env),
    }).catch(() => {});

    fetch('/api/energy/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            price_per_kwh: pricePerKwh,
            inference_util_threshold: inferenceUtilThreshold,
        }),
    }).catch(() => {});

    closeConfigModal();
    showToast('Configuration saved', 'success');
}

let fbTargetId = '';
let fbFilter = '';
let fbCurrentPath = '';

function openFileBrowser(targetId, filter) {
    fbTargetId = targetId;
    fbFilter = filter === 'dir' ? '' : (filter || '');
    const modal = $('file-browser-modal');
    const current = $(targetId).value;
    let startPath = '';
    if (current) {
        const parts = current.split('/');
        parts.pop();
        startPath = parts.join('/') || '/';
    }
    $('btn-fb-select').style.display = filter === 'dir' ? '' : 'none';
    modal.classList.add('open');
    fileBrowserGo(startPath);
}

function closeFileBrowser() {
    $('file-browser-modal').classList.remove('open');
}

function setFbEmpty(message) {
    const entriesEl = $('fb-entries');
    entriesEl.replaceChildren();
    const empty = document.createElement('div');
    empty.className = 'fb-empty';
    empty.textContent = message;
    entriesEl.appendChild(empty);
}

async function fileBrowserGo(path) {
    const entriesEl = $('fb-entries');
    setFbEmpty('Loading...');
    const params = new URLSearchParams();
    if (path) params.set('path', path);
    if (fbFilter) params.set('filter', fbFilter);
    try {
        const resp = await fetch('/api/browse?' + params);
        const data = await resp.json();
        if (data.error) {
            setFbEmpty(data.error);
            return;
        }
        fbCurrentPath = data.path;
        $('fb-path-input').value = data.path;
        if (!data.entries.length) {
            setFbEmpty('Empty directory');
            return;
        }
        entriesEl.replaceChildren();
        data.entries.forEach(e => {
            const row = document.createElement('div');
            row.className = e.is_dir ? 'fb-entry fb-entry-dir' : 'fb-entry fb-entry-file fb-match';
            const icon = document.createElement('span');
            icon.className = 'fb-entry-icon';
            icon.textContent = e.is_dir ? '\u{1F4C1}' : '\u{1F4C4}';
            const name = document.createElement('span');
            name.className = 'fb-entry-name';
            name.textContent = e.name;
            row.appendChild(icon);
            row.appendChild(name);
            if (!e.is_dir) {
                const size = document.createElement('span');
                size.className = 'fb-entry-size';
                size.textContent = e.size_display;
                row.appendChild(size);
            }
            row.addEventListener('click', () => {
                if (e.is_dir) fileBrowserGo(e.path);
                else fileBrowserSelect(e.path);
            });
            entriesEl.appendChild(row);
        });
    } catch (err) {
        setFbEmpty('Error: ' + err.message);
    }
}

function fileBrowserUp() {
    if (fbCurrentPath && fbCurrentPath !== '/') {
        const parts = fbCurrentPath.split('/');
        parts.pop();
        fileBrowserGo(parts.join('/') || '/');
    }
}

function fileBrowserSelect(path) {
    $(fbTargetId).value = path || fbCurrentPath;
    $(fbTargetId).dispatchEvent(new Event('input', { bubbles: true }));
    closeFileBrowser();
}

function showToast(message, type = 'error') {
    const container = $('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.textContent = message;
    container.appendChild(toast);
    requestAnimationFrame(() => { toast.classList.add('show'); });
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

let confirmResolver = null;

function closeConfirmModal(result) {
    $('confirm-modal').classList.remove('open');
    if (confirmResolver) {
        const resolve = confirmResolver;
        confirmResolver = null;
        resolve(result);
    }
}

function confirmAction(title, message, confirmLabel, danger, details = []) {
    return new Promise(resolve => {
        confirmResolver = resolve;
        text('confirm-title', title);
        text('confirm-message', message);
        const detailList = $('confirm-details');
        detailList.replaceChildren();
        details.forEach(detail => {
            const item = document.createElement('li');
            item.textContent = detail;
            detailList.appendChild(item);
        });
        detailList.hidden = details.length === 0;
        const ok = $('confirm-ok');
        ok.textContent = confirmLabel;
        ok.className = danger ? 'btn btn-danger' : 'btn btn-start';
        $('confirm-modal').classList.add('open');
    });
}

function setVal(id, v) { $(id).value = v ?? ''; }
function setChk(id, v) { $(id).checked = !!v; }
function setOpt(id, v) { $(id).value = v || ''; }
function numOrEmpty(id, v) { $(id).value = v != null ? v : ''; }

function clearFieldErrors() {
    document.querySelectorAll('#preset-form .field-error').forEach(el => el.classList.remove('field-error'));
}

function openPresetModal(mode) {
    const modal = $('preset-modal');
    const title = $('modal-title');
    const form = $('preset-form');
    form.reset();
    clearFieldErrors();

    if (mode === 'edit') {
        const id = $('preset-select').value;
        const p = presets.find(pr => pr.id === id);
        if (!p) { showToast('No preset selected', 'warn'); return; }
        title.textContent = 'Edit Preset';
        setVal('modal-preset-id', p.id);
        setVal('modal-name', p.name);
        setVal('modal-model-path', p.model_path);
        numOrEmpty('modal-gpu-layers', p.gpu_layers);
        setChk('modal-no-mmap', p.no_mmap);
        setChk('modal-mlock', p.mlock);
        setVal('modal-context-size', p.context_size || 128000);
        setVal('modal-ctk', p.ctk || 'q8_0');
        setVal('modal-ctv', p.ctv || 'f16');
        setOpt('modal-flash-attn', p.flash_attn);
        setVal('modal-batch-size', p.batch_size || 2048);
        setVal('modal-ubatch-size', p.ubatch_size || p.batch_size || 2048);
        setVal('modal-parallel-slots', p.parallel_slots || 1);
        setVal('modal-tensor-split', p.tensor_split);
        setOpt('modal-split-mode', p.split_mode);
        numOrEmpty('modal-main-gpu', p.main_gpu);
        numOrEmpty('modal-threads', p.threads);
        numOrEmpty('modal-threads-batch', p.threads_batch);
        setOpt('modal-rope-scaling', p.rope_scaling);
        numOrEmpty('modal-rope-freq-base', p.rope_freq_base);
        numOrEmpty('modal-rope-freq-scale', p.rope_freq_scale);
        setChk('modal-ngram-spec', p.ngram_spec);
        numOrEmpty('modal-spec-ngram-size', p.spec_ngram_size);
        numOrEmpty('modal-draft-min', p.draft_min);
        numOrEmpty('modal-draft-max', p.draft_max);
        setVal('modal-draft-model', p.draft_model);
        numOrEmpty('modal-seed', p.seed);
        setVal('modal-system-prompt-file', p.system_prompt_file);
        setVal('modal-extra-args', p.extra_args);
    } else {
        title.textContent = 'New Preset';
        setVal('modal-preset-id', '');
        setVal('modal-context-size', 128000);
        setVal('modal-ctk', 'q8_0');
        setVal('modal-ctv', 'f16');
        setVal('modal-batch-size', 2048);
        setVal('modal-ubatch-size', 2048);
        setVal('modal-parallel-slots', 1);
    }

    modal.classList.add('open');
    const body = modal.querySelector('.modal-body');
    if (body) body.scrollTop = 0;
}

function closePresetModal() {
    $('preset-modal').classList.remove('open');
}

function intOrNull(id) { const v = $(id).value; return v !== '' ? parseInt(v) : null; }
function floatOrNull(id) { const v = $(id).value; return v !== '' ? parseFloat(v) : null; }
function strVal(id) { return $(id).value.trim(); }

async function savePreset(event) {
    event.preventDefault();
    clearFieldErrors();

    const id = $('modal-preset-id').value;
    const preset = {
        name: strVal('modal-name'),
        model_path: strVal('modal-model-path'),
        gpu_layers: intOrNull('modal-gpu-layers'),
        no_mmap: $('modal-no-mmap').checked,
        mlock: $('modal-mlock').checked,
        context_size: parseInt($('modal-context-size').value) || 128000,
        ctk: strVal('modal-ctk') || 'q8_0',
        ctv: strVal('modal-ctv') || 'f16',
        flash_attn: strVal('modal-flash-attn'),
        batch_size: parseInt($('modal-batch-size').value) || 2048,
        ubatch_size: parseInt($('modal-ubatch-size').value) || 2048,
        parallel_slots: parseInt($('modal-parallel-slots').value) || 1,
        tensor_split: strVal('modal-tensor-split'),
        split_mode: strVal('modal-split-mode'),
        main_gpu: intOrNull('modal-main-gpu'),
        threads: intOrNull('modal-threads'),
        threads_batch: intOrNull('modal-threads-batch'),
        rope_scaling: strVal('modal-rope-scaling'),
        rope_freq_base: floatOrNull('modal-rope-freq-base'),
        rope_freq_scale: floatOrNull('modal-rope-freq-scale'),
        ngram_spec: $('modal-ngram-spec').checked,
        spec_ngram_size: intOrNull('modal-spec-ngram-size'),
        draft_min: intOrNull('modal-draft-min'),
        draft_max: intOrNull('modal-draft-max'),
        draft_model: strVal('modal-draft-model'),
        seed: intOrNull('modal-seed'),
        system_prompt_file: strVal('modal-system-prompt-file'),
        extra_args: strVal('modal-extra-args'),
    };

    let valid = true;
    if (!preset.name) {
        $('modal-name').classList.add('field-error');
        valid = false;
    }
    if (!preset.model_path) {
        $('modal-model-path').classList.add('field-error');
        valid = false;
    }
    if (!valid) {
        showToast('Please fill in all required fields', 'error');
        return;
    }

    const saveBtn = $('btn-modal-save');
    saveBtn.classList.add('saving');
    saveBtn.textContent = 'Saving...';

    try {
        let resp;
        let savedId;
        if (id) {
            resp = await fetch('/api/presets/' + encodeURIComponent(id), {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(preset),
            });
            if (!resp.ok) {
                const err = await resp.text().catch(() => 'Unknown error');
                showToast('Save failed: ' + err, 'error');
                return;
            }
            savedId = id;
        } else {
            resp = await fetch('/api/presets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(preset),
            });
            if (!resp.ok) {
                const err = await resp.text().catch(() => 'Unknown error');
                showToast('Save failed: ' + err, 'error');
                return;
            }
            const data = await resp.json();
            savedId = data.id || null;
        }
        closePresetModal();
        await loadPresets(savedId);
        showToast('Preset saved', 'success');
    } catch (err) {
        showToast('Save failed: ' + err.message, 'error');
    } finally {
        saveBtn.classList.remove('saving');
        saveBtn.textContent = 'Save';
    }
}

async function copyPreset() {
    const id = $('preset-select').value;
    const p = presets.find(pr => pr.id === id);
    if (!p) { showToast('No preset selected', 'warn'); return; }

    const copy = Object.assign({}, p);
    delete copy.id;
    copy.name = p.name + ' (copy)';

    try {
        const resp = await fetch('/api/presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(copy),
        });
        if (!resp.ok) {
            const err = await resp.text().catch(() => 'Unknown error');
            showToast('Copy failed: ' + err, 'error');
            return;
        }
        const data = await resp.json();
        await loadPresets(data.preset?.id || null);
        showToast('Preset copied', 'success');
    } catch (err) {
        showToast('Copy failed: ' + err.message, 'error');
    }
}

async function deletePreset() {
    const id = $('preset-select').value;
    const p = presets.find(pr => pr.id === id);
    if (!p) { showToast('No preset selected', 'warn'); return; }
    const ok = await confirmAction('Delete preset', 'Delete preset "' + p.name + '"? This cannot be undone.', 'Delete', true);
    if (!ok) return;

    try {
        const resp = await fetch('/api/presets/' + encodeURIComponent(id), { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.text().catch(() => 'Unknown error');
            showToast('Delete failed: ' + err, 'error');
            return;
        }
        await loadPresets();
        showToast('Preset deleted', 'success');
    } catch (err) {
        showToast('Delete failed: ' + err.message, 'error');
    }
}

async function resetPresets() {
    const ok = await confirmAction(
        'Reset presets',
        'Reset all presets to built-in defaults? Custom presets will be removed.',
        'Reset',
        true
    );
    if (!ok) return;
    try {
        const resp = await fetch('/api/presets/reset', { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.text().catch(() => 'Unknown error');
            showToast('Reset failed: ' + err, 'error');
            return;
        }
        await loadPresets();
        showToast('Presets reset to defaults', 'success');
    } catch (err) {
        showToast('Reset failed: ' + err.message, 'error');
    }
}

['modal-name', 'modal-model-path'].forEach(id => {
    $(id).addEventListener('input', function() {
        this.classList.remove('field-error');
    });
});

function selectedPreset() {
    const id = $('preset-select').value;
    return presets.find(pr => pr.id === id) || {};
}

function getConfig() {
    const p = selectedPreset();
    return {
        model_path: p.model_path || '',
        context_size: p.context_size || 128000,
        ctk: p.ctk || 'q8_0',
        ctv: p.ctv || 'f16',
        tensor_split: p.tensor_split || '',
        batch_size: p.batch_size || 2048,
        ubatch_size: p.ubatch_size || p.batch_size || 2048,
        no_mmap: !!p.no_mmap,
        port: parseInt($('port').value) || 8080,
        ngram_spec: !!p.ngram_spec,
        parallel_slots: p.parallel_slots || 1,
        gpu_layers: p.gpu_layers ?? null,
        mlock: !!p.mlock,
        flash_attn: p.flash_attn || '',
        split_mode: p.split_mode || '',
        main_gpu: p.main_gpu ?? null,
        threads: p.threads ?? null,
        threads_batch: p.threads_batch ?? null,
        rope_scaling: p.rope_scaling || '',
        rope_freq_base: p.rope_freq_base ?? null,
        rope_freq_scale: p.rope_freq_scale ?? null,
        draft_model: p.draft_model || '',
        draft_min: p.draft_min ?? null,
        draft_max: p.draft_max ?? null,
        spec_ngram_size: p.spec_ngram_size ?? null,
        seed: p.seed ?? null,
        system_prompt_file: p.system_prompt_file || '',
        extra_args: p.extra_args || '',
    };
}

function ggufMeta(path) {
    const filename = (path || '').split('/').pop() || '';
    const stem = filename.replace(/\.gguf$/i, '').replace(/-\d{5}-of-\d{5}$/, '');
    const match = stem.match(/-(UD-Q.+|[QI]Q.+|Q[\dA-Z_]+|F16|F32|BF16)$/i);
    return {
        filename: filename || '—',
        name: match ? stem.slice(0, match.index) : (stem || '—'),
        quant: match ? match[1] : '—',
    };
}

function dash(v) {
    if (v === null || v === undefined || v === '') return '—';
    return String(v);
}

function sourcedVal(field) {
    if (!field || field.value == null || field.value === '') return null;
    return field.value;
}

function refreshModelCard() {
    const p = selectedPreset();
    const presetLabel = p.name || ggufMeta(p.model_path).name || '—';
    text('spec-preset', presetLabel);

    const rm = lastRunningModel;
    if (rm && rm.detected) {
        const name = sourcedVal(rm.name) || sourcedVal(rm.model_id) || sourcedVal(rm.gguf_file) || '—';
        text('hero-model', name);
        text('hero-model-sub', 'detected from llama-server');
        const src = rm.primary_source || 'props';
        text('spec-source', src === 'props' || src === 'models' ? 'Detected from llama-server' : String(src));
        text('spec-name', dash(sourcedVal(rm.name) || sourcedVal(rm.model_id)));
        text('spec-gguf', dash(sourcedVal(rm.gguf_file)));
        text('spec-model-id', dash(sourcedVal(rm.model_id)));
        text('spec-quant', dash(sourcedVal(rm.quant)));
        text('spec-ctx', rm.context_size && rm.context_size.value != null
            ? Number(rm.context_size.value).toLocaleString() : '—');
        text('spec-ctx-native', rm.native_context && rm.native_context.value != null
            ? Number(rm.native_context.value).toLocaleString() : '—');
        text('spec-slots', rm.total_slots && rm.total_slots.value != null
            ? String(rm.total_slots.value) : '—');
        text('spec-model-path', dash(sourcedVal(rm.model_path)));
        return;
    }

    // No live detection: show selected preset as reference only (not as "running")
    text('hero-model', presetLabel);
    text('hero-model-sub', 'selected preset');
    text('spec-source', 'selected preset');
    text('spec-name', '—');
    text('spec-gguf', '—');
    text('spec-model-id', '—');
    text('spec-quant', '—');
    text('spec-ctx', '—');
    text('spec-ctx-native', '—');
    text('spec-slots', '—');
    text('spec-model-path', '—');
}

async function doStart() {
    const config = getConfig();
    const p = selectedPreset();
    if (!config.model_path) {
        showToast('No model path set. Edit the preset to select a model.', 'error');
        return;
    }
    const ok = await confirmAction(
        'Start llama-server',
        'Start "' + (p.name || 'selected preset') + '" on port ' + config.port + '?',
        'Start',
        false
    );
    if (!ok) return;
    $('btn-start').disabled = true;
    const resp = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
    });
    const data = await resp.json();
    if (!data.ok) {
        showToast('Start failed: ' + (data.error || 'unknown'), 'error');
        $('btn-start').disabled = false;
    }
}

async function doStop() {
    const ok = await confirmAction(
        'Stop llama-server',
        'Stop the running llama-server? Active requests will be interrupted.',
        'Stop',
        true
    );
    if (!ok) return;
    $('btn-stop').disabled = true;
    await fetch('/api/stop', { method: 'POST' });
}

function formatUptime(startedAt) {
    if (!startedAt) return '—';
    const sec = Math.max(0, Math.floor(Date.now() / 1000 - startedAt));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}

function setWsStatus(connected) {
    wsConnected = connected;
    const dot = $('ws-dot');
    dot.className = 'status-dot ' + (connected ? 'running' : 'error');
    text('ws-text', connected ? 'WebSocket' : 'WebSocket down');
}

function gpuSummary(gpu) {
    if (gpu == null) {
        return { util: null, vramUsed: 0, vramTotal: 0, temp: null, unavailable: true };
    }
    const cards = Object.values(gpu);
    if (!cards.length) {
        return { util: null, vramUsed: 0, vramTotal: 0, temp: null, unavailable: false };
    }
    let vramUsed = 0, vramTotal = 0, temp = 0, load = 0;
    for (const c of cards) {
        vramUsed += c.vram_used || 0;
        vramTotal += c.vram_total || 0;
        temp = Math.max(temp, c.temp || 0);
        load = Math.max(load, c.load || 0);
    }
    return { util: load, vramUsed, vramTotal, temp, unavailable: false };
}

function fmtMib(mib) {
    if (!mib) return '0.0';
    return (mib / 1024).toFixed(1);
}

function fmtTokens(n) {
    if (n == null || Number.isNaN(n)) return '—';
    const v = Number(n);
    if (v >= 1_000_000) return (v / 1_000_000).toFixed(2) + 'M';
    if (v >= 10_000) return (v / 1_000).toFixed(1) + 'k';
    return v.toLocaleString();
}

function fmtUsd(n) {
    if (n == null || Number.isNaN(n)) return '—';
    const v = Number(n);
    if (v === 0) return '$0.00';
    if (v > 0 && v < 0.01) return '$' + v.toFixed(4);
    return '$' + v.toFixed(2);
}

function formatEnergyKwh(kwh) {
    if (!Number.isFinite(kwh)) return '—';
    if (kwh > 0 && kwh < 0.01) return (kwh * 1000).toFixed(1) + ' Wh';
    return kwh.toFixed(3) + ' kWh';
}

function formatPln(value) {
    if (!Number.isFinite(value)) return '—';
    if (value > 0 && value < 0.01) return '<0.01 PLN';
    return value.toFixed(2) + ' PLN';
}

function applyEnergyTotals(prefix, totals, hasMeasurements) {
    const values = totals || {};
    text(prefix + '-cost', hasMeasurements ? formatPln(values.inference_cost_pln) : '—');
    text(prefix + '-energy', hasMeasurements ? formatEnergyKwh(values.inference_energy_kwh) : '—');
}

function applyEnergy(e) {
    lastEnergySnapshot = e || null;
    const lifetime = e?.lifetime || {};
    const hasMeasurements = !!e?.first_measurement_at ||
        (Number.isFinite(lifetime.energy_kwh) && lifetime.energy_kwh !== 0) ||
        (Number.isFinite(lifetime.inference_energy_kwh) && lifetime.inference_energy_kwh !== 0);
    const tariff = Number.isFinite(e?.price_per_kwh)
        ? e.price_per_kwh.toFixed(2) + ' PLN/kWh'
        : '— PLN/kWh';

    text('e-inference-cost', hasMeasurements ? formatPln(lifetime.inference_cost_pln) : '—');
    text('e-inference-energy', hasMeasurements ? formatEnergyKwh(lifetime.inference_energy_kwh) : '—');
    text('e-tariff', tariff);
    text('e-detail-tariff', tariff);

    applyEnergyTotals('e-sess', e?.session, hasMeasurements);
    applyEnergyTotals('e-today', e?.today, hasMeasurements);
    applyEnergyTotals('e-7d', e?.last_7_days, hasMeasurements);
    applyEnergyTotals('e-life-inf', lifetime, hasMeasurements);
    text('e-life-total-cost', hasMeasurements ? formatPln(lifetime.cost_pln) : '—');
    text('e-life-total-energy', hasMeasurements ? formatEnergyKwh(lifetime.energy_kwh) : '—');
    text('e-first', e?.first_measurement_at || '—');
    text('e-tz', e?.timezone_label || '—');

    const warning = e?.save_warning || '';
    text('e-warn', warning);
    $('e-warn').hidden = !warning;
}

function applyUsage(u) {
    if (!u) {
        text('u-prompt', '—');
        text('u-gen', '—');
        text('u-cache', '—');
        text('u-cache-ratio', '—');
        text('u-total', '—');
        text('u-luna', '—');
        text('u-qwen', '—');
        return;
    }
    text('u-prompt', fmtTokens(u.prompt_tokens));
    text('u-gen', fmtTokens(u.predicted_tokens));
    text('u-cache', fmtTokens(u.cached_tokens));
    const total = u.total_prompt_tokens != null
        ? u.total_prompt_tokens
        : (Number(u.prompt_tokens) || 0) + (Number(u.cached_tokens) || 0);
    text('u-total', fmtTokens(total));
    const reuse = u.cache_reuse_ratio != null ? u.cache_reuse_ratio : u.cache_hit_ratio;
    const ratio = reuse != null ? (reuse * 100).toFixed(1) + '% reuse' : '—';
    text('u-cache-ratio', ratio);
    text('u-luna', fmtUsd(u.saved_luna_usd));
    text('u-qwen', fmtUsd(u.saved_qwen_usd));
    if (u.rates) {
        const tag = $('usage-rates-tag');
        const lunaCached = u.rates.luna_cached_input_per_m != null
            ? u.rates.luna_cached_input_per_m
            : u.rates.luna_input_per_m;
        const qwenCached = u.rates.qwen_cached_input_per_m != null
            ? u.rates.qwen_cached_input_per_m
            : u.rates.qwen_input_per_m;
        tag.title =
            'GPT-5.6 Luna $' + u.rates.luna_input_per_m + '/cached $' + lunaCached + '/$' + u.rates.luna_output_per_m +
            ' · Qwen3.8-27B $' + u.rates.qwen_input_per_m + '/cached $' + qwenCached + '/$' + u.rates.qwen_output_per_m +
            ' per 1M in/cached/out · Saved = estimated API cost (GPU energy not subtracted) · ' + (u.rates.label || '');
    }
}

async function resetUsage() {
    const ok = await confirmAction(
        'Reset lifetime stats',
        'Clear all lifetime token counters and savings? This cannot be undone.',
        'Reset',
        true
    );
    if (!ok) return;
    const resp = await fetch('/api/usage/reset', { method: 'POST' });
    const data = await resp.json();
    if (!data.ok) {
        showToast('Reset failed', 'error');
        return;
    }
    applyUsage(data.usage);
    showToast('Lifetime stats reset', 'success');
}

async function resetEnergy() {
    const ok = await confirmAction(
        'Reset lifetime energy',
        'Reset lifetime GPU energy statistics? This permanently removes all stored GPU energy and cost history. Token statistics will not be affected. This action cannot be undone.',
        'Reset',
        true,
        [
            'Lifetime GPU energy',
            'Lifetime inference energy',
            'Lifetime GPU costs',
            'Daily energy history',
        ]
    );
    if (!ok) return;
    try {
        const resp = await fetch('/api/energy/reset-lifetime', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirm: true }),
        });
        const data = await resp.json();
        if (!resp.ok || !data.ok) {
            showToast('Energy reset failed: ' + (data.error || 'unknown error'), 'error');
            return;
        }
        showToast('Lifetime energy reset', 'success');
    } catch (err) {
        showToast('Energy reset failed: ' + err.message, 'error');
    }
}

function fmtLiveTps(v) {
    if (v == null || Number.isNaN(Number(v))) return '—';
    return Number(v).toFixed(1) + ' t/s';
}

function fmtLiveTpsHero(v) {
    if (v == null || Number.isNaN(Number(v))) return '—';
    return Number(v).toFixed(1);
}

function applyWsPayload(d) {
    serverRunning = d.server_running;
    serverStartedAt = d.server_started_at || null;
    const dot = $('status-dot');
    dot.className = 'status-dot ' + (serverRunning ? 'running' : 'stopped');
    text('status-text', serverRunning ? 'llama-server' : 'Stopped');
    $('btn-start').disabled = serverRunning;
    $('btn-stop').disabled = !serverRunning;
    text('uptime-text', formatUptime(serverStartedAt));

    const l = d.llama || {};
    text('m-prompt', fmtLiveTps(l.prompt_tokens_per_sec));
    text('m-gen', fmtLiveTps(l.generation_tokens_per_sec));
    text('hero-prompt', fmtLiveTpsHero(l.prompt_tokens_per_sec));
    text('hero-gen', fmtLiveTpsHero(l.generation_tokens_per_sec));
    if (l.kv_cache_tokens != null && l.kv_cache_max != null) {
        const pct = l.kv_cache_max > 0
            ? ((l.kv_cache_tokens / l.kv_cache_max) * 100).toFixed(1)
            : '0.0';
        text('m-ctx', l.kv_cache_tokens + ' / ' + l.kv_cache_max + ' (' + pct + '%)');
    } else {
        text('m-ctx', '—');
    }
    if (l.n_tokens_max != null) {
        text('m-peak', fmtTokens(l.n_tokens_max));
        const slotCount = (l.slots_idle != null && l.slots_processing != null)
            ? l.slots_idle + l.slots_processing
            : null;
        if (l.kv_cache_max != null && slotCount === 1 && l.kv_cache_max > 0) {
            const peakPct = ((l.n_tokens_max / l.kv_cache_max) * 100).toFixed(1);
            text('m-peak-sub', fmtTokens(l.kv_cache_max) + ' cap · ' + peakPct + '%');
        } else {
            text('m-peak-sub', 'session high-water');
        }
    } else {
        text('m-peak', '—');
        text('m-peak-sub', 'session high-water');
    }
    if (l.spec_acceptance_ratio != null && !Number.isNaN(Number(l.spec_acceptance_ratio))) {
        text('m-mtp', (Number(l.spec_acceptance_ratio) * 100).toFixed(1) + '%');
        const accepted = l.spec_accepted_tokens != null ? fmtTokens(l.spec_accepted_tokens) : '—';
        const drafted = l.spec_draft_tokens != null ? fmtTokens(l.spec_draft_tokens) : '—';
        text('m-mtp-sub', accepted + ' accepted / ' + drafted + ' drafted');
    } else {
        text('m-mtp', '—');
        text('m-mtp-sub', 'speculative decode');
    }
    if (l.slots_idle != null && l.slots_processing != null) {
        text('m-slots', l.slots_idle + ' idle / ' + l.slots_processing + ' busy');
    } else {
        text('m-slots', '—');
    }
    text('hero-reqs', l.requests_processing != null ? String(l.requests_processing) : '—');

    applyUsage(d.usage);
    applyEnergy(d.energy);
    lastRunningModel = d.running_model || null;
    refreshModelCard();

    const statusEl = $('m-status');
    statusEl.textContent = l.status || '—';
    statusEl.className = 'metric-value ' + (l.status === 'ok' ? 'status-ok' : l.status === 'no slot available' ? 'status-busy' : (l.status ? 'status-err' : ''));

    const summary = gpuSummary(d.gpu);
    const gpuEmpty = $('gpu-empty');
    if (summary.unavailable) {
        text('hero-gpu', '—');
        text('hero-vram', '—');
        text('hero-vram-sub', 'GPU memory');
        text('hero-temp', '—');
        gpuEmpty.style.display = '';
        gpuEmpty.textContent = 'GPU metrics unavailable (rocm-smi / nvidia-smi failed).';
    } else if (summary.util == null) {
        text('hero-gpu', '—');
        text('hero-vram', '—');
        text('hero-vram-sub', 'GPU memory');
        text('hero-temp', '—');
        gpuEmpty.style.display = '';
        gpuEmpty.textContent = 'No GPU metrics on this machine. AMD R9700 / ROCm data appears when amd-smi or rocm-smi is available.';
    } else {
        text('hero-gpu', summary.util + '%');
        const vpct = summary.vramTotal > 0 ? Math.round((summary.vramUsed / summary.vramTotal) * 100) : 0;
        text('hero-vram', fmtMib(summary.vramUsed) + ' GB');
        text('hero-vram-sub', fmtMib(summary.vramTotal) + ' GB total · ' + vpct + '%');
        text('hero-temp', Math.round(summary.temp) + '°C');
        gpuEmpty.style.display = 'none';
    }

    const tbody = $('gpu-rows');
    tbody.replaceChildren();
    Object.entries(d.gpu || {}).forEach(([card, m]) => {
        const pwr = m.power_consumption;
        const capped = pwr != null && pwr >= m.power_limit && m.power_limit > 0;
        const vpct = m.vram_total > 0 ? Math.round((m.vram_used / m.vram_total) * 100) : 0;
        let powerText = '—';
        let powerTitle = 'Power reading unavailable';
        if (pwr != null) {
            powerText = capped
                ? pwr.toFixed(1) + 'W!'
                : pwr.toFixed(1) + 'W / ' + m.power_limit + 'W';
            powerTitle = m.power_kind === 'average'
                ? 'Average package power (not instantaneous socket power)'
                : 'Current GPU power';
        }
        const tr = document.createElement('tr');
        const cells = [
            ['card value', card, null],
            ['value temp', Math.round(m.temp) + 'C', null],
            ['value load', m.load + '%', null],
            ['value vram', vpct + '%', null],
            [capped ? 'value capped' : 'value power', powerText, powerTitle],
            ['value sclk', m.sclk_mhz + 'MHz', null],
            ['value mclk', m.mclk_mhz + 'MHz', null],
        ];
        cells.forEach(([cls, value, title]) => {
            const td = document.createElement('td');
            td.className = cls;
            td.textContent = value;
            if (title) td.title = title;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    applyLogPanel(d);

    const badgeParts = [];
    if (serverRunning) badgeParts.push('Running');
    if (l.generation_tokens_per_sec != null && l.generation_tokens_per_sec > 0) {
        badgeParts.push(l.generation_tokens_per_sec.toFixed(1) + 't/s');
    }
    const gpuEntries = Object.entries(d.gpu || {});
    if (gpuEntries.length > 0) badgeParts.push(Math.max(...gpuEntries.map(([, m]) => m.temp)).toFixed(0) + 'C');
    text('badge-server', badgeParts.length ? ' ' + badgeParts.join(' · ') : ' Stopped');
    text('badge-chat', chatHistory.length > 0 ? ' ' + chatHistory.length + ' msg' : '');
    refreshModelCard();
}

function connectWs() {
    const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    const ws = new WebSocket(proto + location.host + '/ws');
    ws.onopen = () => setWsStatus(true);
    ws.onmessage = e => {
        try {
            applyWsPayload(JSON.parse(e.data));
        } catch (err) {
            console.error('WS payload error', err);
        }
    };
    ws.onerror = () => {};
    ws.onclose = () => {
        setWsStatus(false);
        text('status-text', serverRunning ? 'llama-server' : 'Disconnected');
        setTimeout(connectWs, 1500);
    };
}

connectWs();
setInterval(() => {
    if (serverStartedAt) text('uptime-text', formatUptime(serverStartedAt));
}, 1000);

if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
}

const MD_ALLOWED = new Set(['P', 'BR', 'STRONG', 'B', 'EM', 'I', 'CODE', 'PRE', 'UL', 'OL', 'LI', 'BLOCKQUOTE', 'A', 'H1', 'H2', 'H3', 'H4', 'SPAN', 'TABLE', 'THEAD', 'TBODY', 'TR', 'TH', 'TD', 'HR', 'DEL']);

function sanitizeNode(node) {
    const children = Array.from(node.childNodes);
    for (const child of children) {
        if (child.nodeType === Node.ELEMENT_NODE) {
            if (!MD_ALLOWED.has(child.tagName)) {
                const textNode = document.createTextNode(child.textContent);
                node.replaceChild(textNode, child);
                continue;
            }
            [...child.attributes].forEach(attr => {
                const name = attr.name.toLowerCase();
                const ok = child.tagName === 'A' && name === 'href' && /^(https?:|mailto:)/i.test(attr.value);
                if (!ok) child.removeAttribute(attr.name);
            });
            if (child.tagName === 'A') child.setAttribute('rel', 'noopener noreferrer');
            sanitizeNode(child);
        }
    }
}

function renderMd(src) {
    let html = src.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/\n/g, '<br>');
    if (typeof marked !== 'undefined') {
        try { html = marked.parse(src); } catch (_) {}
    }
    const wrap = document.createElement('div');
    wrap.innerHTML = html;
    sanitizeNode(wrap);
    return wrap;
}

let chatHistory = [];
let chatBusy = false;

function clearChat() {
    chatHistory = [];
    $('chat-messages').replaceChildren();
}

function chatScroll() {
    const c = $('chat-messages');
    c.scrollTop = c.scrollHeight;
}

function appendMsg(role, txt) {
    const el = document.createElement('div');
    el.className = 'msg msg-' + role;
    el.textContent = txt;
    $('chat-messages').appendChild(el);
    chatScroll();
    return el;
}

async function sendChat() {
    if (chatBusy) return;
    const input = $('chat-input');
    const value = input.value.trim();
    if (!value) return;
    input.value = '';

    chatHistory.push({ role: 'user', content: value });
    appendMsg('user', value);

    const chatPort = $('port').value || '8080';
    const url = '/api/chat?port=' + encodeURIComponent(chatPort);

    chatBusy = true;
    $('btn-send').disabled = true;

    let thinkEl = null;
    let thinkContent = '';
    const msgEl = appendMsg('assistant', '');
    let msgContent = '';

    try {
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: chatHistory,
                stream: true,
                temperature: 1.0,
                top_p: 0.95,
                top_k: 40,
                min_p: 0.01,
                repeat_penalty: 1.0,
            }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
            const { done, value: chunk } = await reader.read();
            if (done) break;
            buf += decoder.decode(chunk, { stream: true });

            const lines = buf.split('\n');
            buf = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6).trim();
                if (payload === '[DONE]') continue;
                try {
                    const obj = JSON.parse(payload);
                    const delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
                    if (!delta) continue;

                    const rc = delta.reasoning_content || '';
                    if (rc) {
                        thinkContent += rc;
                        if (!thinkEl) {
                            thinkEl = document.createElement('details');
                            thinkEl.className = 'msg msg-thinking';
                            const summary = document.createElement('summary');
                            summary.textContent = 'thinking...';
                            const span = document.createElement('span');
                            thinkEl.appendChild(summary);
                            thinkEl.appendChild(span);
                            $('chat-messages').insertBefore(thinkEl, msgEl);
                        }
                        thinkEl.querySelector('span').textContent = thinkContent;
                    }

                    const c = delta.content || '';
                    if (c) {
                        msgContent += c;
                        const rendered = renderMd(msgContent);
                        msgEl.replaceChildren(...rendered.childNodes);
                    }
                } catch (_) {}
            }
            chatScroll();
        }
    } catch (err) {
        msgEl.textContent = '[error] ' + err.message;
        msgEl.style.color = 'var(--error)';
    }

    if (msgContent) {
        chatHistory.push({ role: 'assistant', content: msgContent });
    }
    chatBusy = false;
    $('btn-send').disabled = false;
}

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});
$('btn-start').addEventListener('click', doStart);
$('btn-stop').addEventListener('click', doStop);
$('btn-config').addEventListener('click', openConfigModal);
$('btn-config-close').addEventListener('click', closeConfigModal);
$('btn-config-cancel').addEventListener('click', closeConfigModal);
$('btn-config-save').addEventListener('click', saveConfig);
$('btn-preset-new').addEventListener('click', () => openPresetModal('new'));
$('btn-preset-edit').addEventListener('click', () => openPresetModal('edit'));
$('btn-preset-copy').addEventListener('click', copyPreset);
$('btn-preset-delete').addEventListener('click', deletePreset);
$('btn-preset-reset').addEventListener('click', resetPresets);
$('btn-usage-reset').addEventListener('click', resetUsage);
$('btn-energy-reset').addEventListener('click', resetEnergy);
$('btn-preset-close').addEventListener('click', closePresetModal);
$('btn-preset-cancel').addEventListener('click', closePresetModal);
$('preset-form').addEventListener('submit', savePreset);
$('preset-select').addEventListener('change', () => { saveSettings(); refreshModelCard(); });
$('btn-chat-clear').addEventListener('click', clearChat);
$('btn-send').addEventListener('click', sendChat);
$('browse-server-path').addEventListener('click', () => openFileBrowser('set-server-path', 'executable'));
$('browse-server-cwd').addEventListener('click', () => openFileBrowser('set-server-cwd', 'dir'));
$('log-panel').addEventListener('mousedown', e => {
    if (e.button === 0) logSelecting = true;
});
window.addEventListener('mouseup', () => {
    logSelecting = false;
});
window.addEventListener('blur', () => {
    logSelecting = false;
});
$('log-panel').addEventListener('scroll', () => {
    const el = $('log-panel');
    logUserPinnedBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
});
$('log-autoscroll').addEventListener('change', () => {
    logAutoScroll = $('log-autoscroll').checked;
    if (logAutoScroll) {
        logUserPinnedBottom = true;
        applyLogPanel(null);
    }
});
$('log-filter').addEventListener('input', () => applyLogPanel(null));
$('btn-log-bottom').addEventListener('click', () => {
    logUserPinnedBottom = true;
    const el = $('log-panel');
    el.scrollTop = el.scrollHeight;
});
$('btn-log-clear').addEventListener('click', () => {
    fetch('/api/logs/clear', { method: 'POST' })
        .then(() => {
            lastLogView = Object.assign({}, lastLogView, { logs: [] });
            applyLogPanel(null);
        })
        .catch(() => {});
});
$('browse-model-path').addEventListener('click', () => openFileBrowser('modal-model-path', 'gguf'));
$('btn-fb-close').addEventListener('click', closeFileBrowser);
$('btn-fb-cancel').addEventListener('click', closeFileBrowser);
$('btn-fb-select').addEventListener('click', () => fileBrowserSelect());
$('btn-fb-up').addEventListener('click', fileBrowserUp);
$('fb-path-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') fileBrowserGo(e.target.value);
});
$('confirm-ok').addEventListener('click', () => closeConfirmModal(true));
$('confirm-cancel').addEventListener('click', () => closeConfirmModal(false));
$('confirm-close').addEventListener('click', () => closeConfirmModal(false));
$('chat-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
$('config-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeConfigModal();
});
$('file-browser-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeFileBrowser();
});
$('preset-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closePresetModal();
});
$('confirm-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeConfirmModal(false);
});
document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if ($('file-browser-modal').classList.contains('open')) {
        closeFileBrowser();
        e.stopImmediatePropagation();
        return;
    }
    if ($('confirm-modal').classList.contains('open')) {
        closeConfirmModal(false);
        return;
    }
    if ($('config-modal').classList.contains('open')) closeConfigModal();
    else if ($('preset-modal').classList.contains('open')) closePresetModal();
}, true);

if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
}
