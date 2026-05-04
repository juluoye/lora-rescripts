import { $, escapeHtml } from '../utils/dom.js';

export function createSavedConfigsController({
  api,
  state,
  trainingTypes,
  createDefaultConfig,
  mergeConfigPatch,
  resetTransientState,
  saveDraft,
  renderView,
  renderNavigator,
  showToast,
  closeBuiltinPicker,
}) {
  function getModalParts() {
    return {
      modal: $('#builtin-picker-modal'),
      title: $('#builtin-picker-title'),
      pathEl: $('#builtin-picker-path'),
      list: $('#builtin-picker-list'),
      footer: document.querySelector('.builtin-picker-footer'),
    };
  }

  function saveCurrentParams() {
    const defaultName = state.config.output_name || state.config.pretrained_model_name_or_path?.split(/[/\\]/).pop()?.replace(/\.[^.]+$/, '') || '';
    const { modal, title, pathEl, list } = getModalParts();
    if (!modal || !title || !pathEl || !list) return;

    title.textContent = '保存当前参数';
    pathEl.textContent = '请输入保存名称，保存后会直接写入本地文件。';
    list.innerHTML = `
      <div class="save-params-form">
        <input type="text" id="save-params-name" class="text-input" value="${escapeHtml(defaultName)}" placeholder="输入参数名称">
        <button class="btn btn-primary btn-sm" type="button" id="save-params-confirm">保存</button>
      </div>
    `;
    modal.classList.add('open');

    const nameInput = $('#save-params-name');
    const confirmBtn = $('#save-params-confirm');
    const submit = async () => {
      const name = nameInput?.value?.trim();
      if (!name) {
        if (pathEl) pathEl.textContent = '请输入保存名称。';
        nameInput?.focus();
        return;
      }
      try {
        // 保存原始 UI 状态（而非 buildRunConfig 转换后的后端 payload），
        // 这样 LyCORIS 算法、日志前缀等 UI 专属字段不会丢失。
        const payload = {};
        for (const [key, value] of Object.entries(state.config)) {
          if (value !== '' && value != null) payload[key] = value;
        }
        payload.__training_type__ = state.activeTrainingType;
        await api.saveConfig(name, payload);
        saveDraft();
        state.hasLocalDraft = true;
        modal.classList.remove('open');
        showToast('参数已保存：' + name);
        if (state.activeModule === 'config') {
          renderView('config');
        } else {
          renderNavigator();
        }
      } catch (error) {
        if (pathEl) pathEl.textContent = error.message || '保存失败。';
        if (nameInput) {
          nameInput.style.borderColor = 'var(--danger, #d9534f)';
          nameInput.focus();
          nameInput.select();
        }
      }
    };

    confirmBtn?.addEventListener('click', submit, { once: true });
    nameInput?.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submit();
      }
    }, { once: true });
    nameInput?.focus();
    nameInput?.select();
  }

  async function loadSavedParams() {
    const { modal, title, pathEl, list, footer } = getModalParts();
    if (!modal || !title || !pathEl || !list) return;

    title.textContent = '读取已保存参数';
    pathEl.textContent = '选择一个已保存的参数，点击后立即载入。';
    list.innerHTML = '<div class="builtin-picker-empty"><span>加载中...</span></div>';
    if (footer) footer.innerHTML = `<button class="btn btn-outline btn-sm" type="button" id="builtin-picker-cancel" onclick="closeBuiltinPicker()">取消</button>`;
    modal.classList.add('open');

    try {
      const response = await api.listSavedConfigs();
      const configs = response?.data?.configs || [];
      if (!configs.length) {
        list.innerHTML = '<div class="builtin-picker-empty"><span>未检测到内容</span></div>';
        return;
      }
      list.innerHTML = configs.map((configItem) => `
        <div class="builtin-picker-item" type="button">
          <span class="builtin-picker-name">${escapeHtml(configItem.name)}</span>
          <span class="builtin-picker-time">${new Date(configItem.time).toLocaleString('zh-CN')}</span>
          <button class="btn btn-outline btn-sm btn-picker-action" type="button" onclick="previewSavedConfig('${escapeHtml(configItem.name)}')">预览</button>
          <button class="btn btn-outline btn-sm btn-picker-action" type="button" onclick="loadNamedConfig('${escapeHtml(configItem.name)}')">载入</button>
          <button class="btn btn-outline btn-sm btn-picker-action" type="button" onclick="event.stopPropagation(); renameSavedConfig('${escapeHtml(configItem.name)}')">重命名</button>
          <button class="builtin-picker-delete-btn" type="button" title="删除" onclick="event.stopPropagation(); deleteSavedConfig('${escapeHtml(configItem.name)}')">✕</button>
        </div>
      `).join('');
    } catch (error) {
      pathEl.textContent = error.message || '读取列表失败。';
      list.innerHTML = '<div class="builtin-picker-empty"><span>未检测到内容</span></div>';
    }
  }

  async function loadNamedConfig(name) {
    const pathEl = $('#builtin-picker-path');
    try {
      const response = await api.loadSavedConfig(name);
      const data = response?.data;
      if (!data) {
        throw new Error('参数内容为空。');
      }

      const savedType = data.__training_type__ || data.model_train_type || '';
      delete data.__training_type__;

      normalizeLegacySavedConfig(data);

      if (savedType && savedType !== state.activeTrainingType) {
        const typeExists = trainingTypes.some((trainingType) => trainingType.id === savedType);
        if (typeExists) {
          state.activeTrainingType = savedType;
          localStorage.setItem('sd-rescripts:training-type', savedType);
          state.config = createDefaultConfig(savedType);
        }
      }
      mergeConfigPatch(data);
      state.hasLocalDraft = true;
      resetTransientState();
      saveDraft();
      closeBuiltinPicker();
      showToast(`已载入参数：${name}${savedType ? ` (${savedType})` : ''}`);
      if (state.activeModule === 'config') {
        renderView('config');
      } else {
        renderNavigator();
      }
    } catch (error) {
      if (pathEl) {
        pathEl.textContent = error.message || '读取参数失败。';
      }
    }
  }

  async function deleteSavedConfig(name) {
    try {
      await api.deleteSavedConfig(name);
      showToast('已删除：' + name);
      loadSavedParams();
    } catch (error) {
      showToast(error.message || '删除失败');
    }
  }

  async function renameSavedConfig(oldName) {
    const { title, pathEl, list, footer } = getModalParts();
    if (!title || !pathEl || !list) return;

    title.textContent = '重命名参数';
    pathEl.textContent = `当前名称：${oldName}`;
    list.innerHTML = `
      <div style="padding: 16px;">
        <input type="text" id="rename-config-input" value="${escapeHtml(oldName)}"
          style="width:100%;padding:8px 12px;border:1px solid var(--border);border-radius:6px;font-size:0.88rem;background:var(--bg-main);color:var(--text-main);"
          placeholder="输入新名称" />
      </div>
    `;
    if (footer) footer.innerHTML = `
      <button class="btn btn-outline btn-sm" type="button" onclick="loadSavedParams()">← 返回列表</button>
      <button class="btn btn-primary btn-sm" type="button" id="rename-config-confirm">确认重命名</button>
    `;

    const input = $('#rename-config-input');
    const confirmBtn = $('#rename-config-confirm');

    const doRename = async () => {
      const newName = input?.value?.trim();
      if (!newName) {
        pathEl.textContent = '请输入新名称。';
        input?.focus();
        return;
      }
      if (newName === oldName) {
        loadSavedParams();
        return;
      }
      try {
        await api.renameSavedConfig(oldName, newName);
        showToast('已重命名：' + oldName + ' → ' + newName);
        loadSavedParams();
      } catch (error) {
        pathEl.textContent = error.message || '重命名失败。';
        if (input) {
          input.style.borderColor = 'var(--danger, #d9534f)';
          input.focus();
          input.select();
        }
      }
    };

    confirmBtn?.addEventListener('click', doRename, { once: true });
    input?.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        doRename();
      }
    }, { once: true });
    input?.focus();
    input?.select();
  }

  async function previewSavedConfig(name) {
    const { title, pathEl, list, footer } = getModalParts();
    if (!title || !pathEl || !list) return;

    title.textContent = `参数预览：${name}`;
    pathEl.textContent = '加载中...';
    list.innerHTML = '<div class="builtin-picker-empty"><span>加载中...</span></div>';
    if (footer) footer.innerHTML = `<button class="btn btn-outline btn-sm" type="button" onclick="loadSavedParams()">← 返回列表</button><button class="btn btn-outline btn-sm" type="button" id="builtin-picker-cancel" onclick="closeBuiltinPicker()">取消</button>`;

    try {
      const response = await api.loadSavedConfig(name);
      const data = response?.data;
      if (!data) throw new Error('参数内容为空。');
      const entries = Object.entries(data);
      pathEl.textContent = `共 ${entries.length} 个参数`;
      list.innerHTML = `
        <div class="params-preview-list">
          ${entries.map(([key, value]) => {
            const display = typeof value === 'object' ? JSON.stringify(value) : String(value ?? '');
            return `<div class="params-preview-row"><span class="params-key">${escapeHtml(key)}</span><span class="params-val">${escapeHtml(display)}</span></div>`;
          }).join('')}
        </div>
      `;
    } catch (error) {
      pathEl.textContent = error.message || '预览失败。';
    }
  }

  function bindGlobals(targetWindow) {
    targetWindow.saveCurrentParams = saveCurrentParams;
    targetWindow.loadSavedParams = loadSavedParams;
    targetWindow.loadNamedConfig = loadNamedConfig;
    targetWindow.deleteSavedConfig = deleteSavedConfig;
    targetWindow.renameSavedConfig = renameSavedConfig;
    targetWindow.previewSavedConfig = previewSavedConfig;
  }

  return {
    saveCurrentParams,
    loadSavedParams,
    loadNamedConfig,
    deleteSavedConfig,
    renameSavedConfig,
    previewSavedConfig,
    bindGlobals,
  };
}

function normalizeLegacySavedConfig(data) {
  if (Array.isArray(data.network_args) && !data.lycoris_algo) {
    const argMap = {};
    for (const item of data.network_args) {
      const eq = String(item).indexOf('=');
      if (eq > 0) argMap[String(item).slice(0, eq).trim()] = String(item).slice(eq + 1).trim();
    }
    if (argMap.algo) {
      data.lycoris_algo = argMap.algo;
      if (!data.network_module) data.network_module = 'lycoris.kohya';
    }
    if (argMap.conv_dim != null) { const n = Number(argMap.conv_dim); data.conv_dim = Number.isNaN(n) ? '' : n; }
    if (argMap.conv_alpha != null) { const n = Number(argMap.conv_alpha); data.conv_alpha = Number.isNaN(n) ? '' : n; }
    if (argMap.dropout != null) { const n = Number(argMap.dropout); data.dropout = Number.isNaN(n) ? '' : n; }
    if (argMap.train_norm != null) data.train_norm = argMap.train_norm === 'True';
    if (argMap.factor != null) { const n = Number(argMap.factor); data.lokr_factor = Number.isNaN(n) ? '' : n; }
    if (argMap.dora_wd != null) data.dora_wd = argMap.dora_wd === 'True';
    if (argMap.scale_weight_norms != null) { const n = Number(argMap.scale_weight_norms); data.scale_weight_norms = Number.isNaN(n) ? '' : n; }

    const structured = new Set(['algo', 'conv_dim', 'conv_alpha', 'dropout', 'train_norm', 'factor', 'dora_wd', 'scale_weight_norms']);
    const remaining = data.network_args.filter((arg) => {
      const key = String(arg).split('=')[0].trim();
      return !structured.has(key);
    });
    if (remaining.length > 0) data.network_args_custom = remaining.join('\n');
    delete data.network_args;
  }

  if (Array.isArray(data.optimizer_args) && !data.optimizer_args_custom) {
    const prodigyRestore = {};
    const remainingArgs = [];
    for (const arg of data.optimizer_args) {
      const eqIdx = String(arg).indexOf('=');
      const key = eqIdx > 0 ? String(arg).slice(0, eqIdx).trim() : '';
      const value = eqIdx > 0 ? String(arg).slice(eqIdx + 1).trim() : '';
      if (key === 'd_coef') { prodigyRestore.prodigy_d_coef = value; }
      else if (key === 'd0') { prodigyRestore.prodigy_d0 = value; }
      else { remainingArgs.push(String(arg)); }
    }
    if (prodigyRestore.prodigy_d_coef != null) data.prodigy_d_coef = prodigyRestore.prodigy_d_coef;
    if (prodigyRestore.prodigy_d0 != null) data.prodigy_d0 = prodigyRestore.prodigy_d0;
    if (remainingArgs.length > 0) data.optimizer_args_custom = remainingArgs.join('\n');
    delete data.optimizer_args;
  }

  if (Array.isArray(data.lr_scheduler_args)) {
    data.lr_scheduler_args = data.lr_scheduler_args.join('\n');
  }

  if (Array.isArray(data.base_weights)) {
    data.base_weights = data.base_weights.join('\n');
    if (!data.enable_base_weight) data.enable_base_weight = true;
  }
}
