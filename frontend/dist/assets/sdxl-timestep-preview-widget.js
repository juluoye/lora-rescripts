const STYLE_ID = "sdxl-timestep-preview-widget-style";
const ROOT_ID = "sdxl-timestep-preview-widget-root";
const POLL_INTERVAL_MS = 2000;
const COLLAPSE_STORAGE_KEY_PREFIX = "timestep-preview-widget-collapsed:";

let cleanupCurrentWidget = null;

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
#${ROOT_ID} {
  margin: 0 20px 16px 20px;
  border: 1px solid #d9e3f0;
  border-radius: 12px;
  background: linear-gradient(180deg, #ffffff 0%, #f6fbff 100%);
  box-shadow: 0 10px 28px rgba(64, 158, 255, 0.08);
}
#${ROOT_ID} .sdxl-timestep-preview-card {
  padding: 16px;
}
#${ROOT_ID} .sdxl-timestep-preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 6px;
}
#${ROOT_ID} .sdxl-timestep-preview-title {
  font-size: 16px;
  font-weight: 600;
  color: #213547;
}
#${ROOT_ID} .sdxl-timestep-preview-toggle {
  border: 1px solid #c8d7eb;
  background: #ffffff;
  color: #409eff;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 12px;
  line-height: 1;
  cursor: pointer;
}
#${ROOT_ID}[data-collapsed="true"] .sdxl-timestep-preview-body {
  display: none;
}
#${ROOT_ID} .sdxl-timestep-preview-desc {
  font-size: 13px;
  color: #5f6b7a;
  margin-bottom: 12px;
  line-height: 1.6;
}
#${ROOT_ID} .sdxl-timestep-preview-summary {
  font-size: 13px;
  color: #213547;
  white-space: pre-wrap;
  line-height: 1.6;
  margin-bottom: 12px;
}
#${ROOT_ID} .sdxl-timestep-preview-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
#${ROOT_ID} .sdxl-timestep-preview-actions button {
  border: 1px solid #409eff;
  background: #409eff;
  color: #ffffff;
  border-radius: 8px;
  padding: 8px 14px;
  font-size: 13px;
  cursor: pointer;
}
#${ROOT_ID} .sdxl-timestep-preview-actions button.secondary {
  background: #ffffff;
  color: #409eff;
}
#${ROOT_ID} .sdxl-timestep-preview-actions button[disabled] {
  opacity: 0.55;
  cursor: not-allowed;
}
#${ROOT_ID} .sdxl-timestep-preview-image-wrap {
  border: 1px solid #d9e3f0;
  background: #f8fbff;
  border-radius: 10px;
  padding: 10px;
}
#${ROOT_ID} .sdxl-timestep-preview-image {
  width: 100%;
  display: block;
  border-radius: 6px;
}
#${ROOT_ID} .sdxl-timestep-preview-empty {
  color: #7c8796;
  font-size: 13px;
  line-height: 1.6;
}
`;
  document.head.appendChild(style);
}

function apiGet(url) {
  return fetch(url, { cache: "no-store" }).then((res) => res.json());
}

function apiPost(url, body = {}) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((res) => res.json());
}

function findMountTarget() {
  const rightContainer = document.querySelector(".example-container .right-container");
  if (!rightContainer) {
    return null;
  }

  const paramsSection = rightContainer.querySelector(".params-section");
  if (!paramsSection) {
    return rightContainer;
  }

  return paramsSection.closest("section") || paramsSection;
}

function createPreviewConfig(previewMode) {
  if (previewMode === "anima") {
    return {
      previewMode,
      title: "Anima Timestep 采样 / 权重预览",
      description: "读取当前右侧参数预览中的 Anima timestep 采样与 weighting_scheme 配置，一键打开交互式曲线窗口，并显示最近一次保存的快照。",
      emptyText: "还没有可用快照。先点击“打开预览工具”即可生成。",
      defaults: {
        mode: "shift",
        scale: 1.0,
        shift: 3.0,
        weight_mode: "uniform",
        weight_scale: 1.0,
        weight_shift: 1.0,
        logit_mean: 0.0,
        logit_std: 1.0,
        mode_scale: 1.29,
        min_timestep: 0,
        max_timestep: 1000,
        width: 1024,
        height: 1024,
      },
      summaryLines(config) {
        return [
          `当前采样模式: ${config.mode}`,
          `采样 sigmoid_scale: ${config.scale}`,
          `采样 discrete_flow_shift: ${config.shift}`,
          `当前 weighting_scheme: ${config.weight_mode}`,
          `logit_mean: ${config.logit_mean}`,
          `logit_std: ${config.logit_std}`,
          `mode_scale: ${config.mode_scale}`,
          `resolution: ${config.width} x ${config.height}`,
          `min_timestep: ${config.min_timestep}`,
          `max_timestep: ${config.max_timestep}`,
        ];
      },
      extract(text) {
        const defaults = this.defaults;
        const readString = (key, fallback) => {
          const match = text.match(new RegExp(`${key}\\s*=\\s*["']([^"'\\n]+)["']`, "i"));
          return match ? match[1] : fallback;
        };
        const readNumber = (key, fallback) => {
          const match = text.match(new RegExp(`${key}\\s*=\\s*([-+]?\\d+(?:\\.\\d+)?)`, "i"));
          return match ? Number(match[1]) : fallback;
        };
        const readResolution = (fallbackWidth, fallbackHeight) => {
          const match = text.match(/resolution\s*=\s*["']\s*(\d+)\s*,\s*(\d+)\s*["']/i);
          if (!match) {
            return { width: fallbackWidth, height: fallbackHeight };
          }
          return {
            width: Number(match[1]),
            height: Number(match[2]),
          };
        };

        const mode = readString("timestep_sampling", defaults.mode);
        const weightMode = readString("weighting_scheme", defaults.weight_mode);
        const resolution = readResolution(defaults.width, defaults.height);
        return {
          preview_mode: "anima",
          mode: ["sigma", "uniform", "sigmoid", "shift", "flux_shift"].includes(mode) ? mode : defaults.mode,
          scale: readNumber("sigmoid_scale", defaults.scale),
          shift: readNumber("discrete_flow_shift", defaults.shift),
          weight_mode: ["sigma_sqrt", "logit_normal", "mode", "cosmap", "none", "uniform"].includes(weightMode)
            ? weightMode
            : defaults.weight_mode,
          weight_scale: readNumber("mode_scale", defaults.weight_scale),
          weight_shift: defaults.weight_shift,
          logit_mean: readNumber("logit_mean", defaults.logit_mean),
          logit_std: readNumber("logit_std", defaults.logit_std),
          mode_scale: readNumber("mode_scale", defaults.mode_scale),
          min_timestep: readNumber("min_timestep", defaults.min_timestep),
          max_timestep: readNumber("max_timestep", defaults.max_timestep),
          width: resolution.width,
          height: resolution.height,
        };
      },
    };
  }

  return {
    previewMode,
    title: "SDXL Timestep 采样 / Loss 加权预览",
    description: "读取当前右侧参数预览中的 timestep 采样与 loss 加权配置，一键打开交互式曲线窗口，并显示最近一次保存的快照。",
    emptyText: "还没有可用快照。先点击“打开预览工具”即可生成。",
    defaults: {
      mode: "uniform",
      scale: 1.0,
      shift: 1.0,
      weight_mode: "none",
      weight_scale: 1.0,
      weight_shift: 1.0,
      logit_mean: 0.0,
      logit_std: 1.0,
      mode_scale: 1.29,
      min_timestep: 0,
      max_timestep: 1000,
      width: 1024,
      height: 1024,
    },
    summaryLines(config) {
      return [
        `当前采样模式: ${config.mode}`,
        `采样 sigmoid_scale: ${config.scale}`,
        `采样 shift: ${config.shift}`,
        `当前 loss 加权: ${config.weight_mode}`,
        `加权 sigmoid_scale: ${config.weight_scale}`,
        `加权 shift: ${config.weight_shift}`,
        `min_timestep: ${config.min_timestep}`,
        `max_timestep: ${config.max_timestep}`,
      ];
    },
    extract(text) {
      const defaults = this.defaults;
      const readString = (key, fallback) => {
        const match = text.match(new RegExp(`${key}\\s*=\\s*["']([^"'\\n]+)["']`, "i"));
        return match ? match[1] : fallback;
      };
      const readNumber = (key, fallback) => {
        const match = text.match(new RegExp(`${key}\\s*=\\s*([-+]?\\d+(?:\\.\\d+)?)`, "i"));
        return match ? Number(match[1]) : fallback;
      };

      const mode = readString("timestep_sampling", defaults.mode);
      const weightMode = readString("timestep_loss_weighting", defaults.weight_mode);
      return {
        preview_mode: "sdxl",
        mode: ["uniform", "sigmoid", "shift"].includes(mode) ? mode : defaults.mode,
        scale: readNumber("timestep_sigmoid_scale", defaults.scale),
        shift: readNumber("timestep_shift", defaults.shift),
        weight_mode: ["none", "linear", "cosine", "sigmoid", "shift"].includes(weightMode)
          ? weightMode
          : defaults.weight_mode,
        weight_scale: readNumber("timestep_loss_weight_sigmoid_scale", defaults.weight_scale),
        weight_shift: readNumber("timestep_loss_weight_shift", defaults.weight_shift),
        logit_mean: defaults.logit_mean,
        logit_std: defaults.logit_std,
        mode_scale: defaults.mode_scale,
        min_timestep: readNumber("min_timestep", defaults.min_timestep),
        max_timestep: readNumber("max_timestep", defaults.max_timestep),
        width: defaults.width,
        height: defaults.height,
      };
    },
  };
}

function formatConfigSummary(routeConfig, config) {
  return routeConfig.summaryLines(config).join("\n");
}

function createWidgetShell(routeConfig) {
  const root = document.createElement("section");
  root.id = ROOT_ID;
  root.innerHTML = `
    <div class="sdxl-timestep-preview-card">
      <div class="sdxl-timestep-preview-header">
        <div class="sdxl-timestep-preview-title">${routeConfig.title}</div>
        <button type="button" class="sdxl-timestep-preview-toggle" data-role="toggle">折叠</button>
      </div>
      <div class="sdxl-timestep-preview-body">
        <div class="sdxl-timestep-preview-desc">${routeConfig.description}</div>
        <div class="sdxl-timestep-preview-summary">正在读取当前配置...</div>
        <div class="sdxl-timestep-preview-actions">
          <button type="button" data-role="open">打开预览工具</button>
          <button type="button" class="secondary" data-role="refresh">刷新快照</button>
        </div>
        <div class="sdxl-timestep-preview-image-wrap">
          <div class="sdxl-timestep-preview-empty" data-role="empty">${routeConfig.emptyText}</div>
          <img class="sdxl-timestep-preview-image" data-role="image" alt="timestep preview" style="display:none;" />
        </div>
      </div>
    </div>
  `;
  return root;
}

function extractCurrentConfigFromPreview(routeConfig) {
  const previewNode = document.querySelector(".params-section .el-scrollbar__view");
  const text = String(previewNode?.textContent || "");
  if (!text.trim()) {
    return {
      preview_mode: routeConfig.previewMode,
      ...routeConfig.defaults,
    };
  }
  return routeConfig.extract(text);
}

function mountTimestepPreviewWidget(previewMode) {
  if (cleanupCurrentWidget) {
    cleanupCurrentWidget();
    cleanupCurrentWidget = null;
  }

  ensureStyle();

  const routeConfig = createPreviewConfig(previewMode);
  const storageKey = `${COLLAPSE_STORAGE_KEY_PREFIX}${routeConfig.previewMode}`;

  let disposed = false;
  let pollTimer = null;
  let attachTimer = null;
  let root = null;
  let collapsed = false;

  const loadCollapsedState = () => {
    try {
      return window.localStorage.getItem(storageKey) === "true";
    } catch (_error) {
      return false;
    }
  };

  const saveCollapsedState = (value) => {
    try {
      window.localStorage.setItem(storageKey, value ? "true" : "false");
    } catch (_error) {
      // ignore storage write errors
    }
  };

  const applyCollapsedState = () => {
    if (!root) {
      return;
    }
    root.dataset.collapsed = collapsed ? "true" : "false";
    const toggle = root.querySelector("[data-role='toggle']");
    if (toggle) {
      toggle.textContent = collapsed ? "展开" : "折叠";
    }
  };

  const stopPolling = () => {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  };

  const stopAttachTimer = () => {
    if (attachTimer) {
      clearInterval(attachTimer);
      attachTimer = null;
    }
  };

  const renderCurrentConfig = () => {
    if (!root) {
      return null;
    }
    const config = extractCurrentConfigFromPreview(routeConfig);
    const summary = root.querySelector(".sdxl-timestep-preview-summary");
    if (summary) {
      summary.textContent = formatConfigSummary(routeConfig, config);
    }
    return config;
  };

  const refreshPreviewImage = async () => {
    if (disposed || !root) {
      return;
    }

    renderCurrentConfig();

    try {
      const response = await apiGet(`/api/timestep_preview/status?preview_mode=${encodeURIComponent(routeConfig.previewMode)}`);
      if (response.status !== "success") {
        throw new Error(response.message || "读取预览状态失败。");
      }

      const payload = response.data || {};
      const img = root.querySelector("[data-role='image']");
      const empty = root.querySelector("[data-role='empty']");

      if (!img || !empty) {
        return;
      }

      if (payload.exists && payload.image_url) {
        img.src = `${payload.image_url}?t=${payload.mtime || Date.now()}`;
        img.style.display = "block";
        empty.style.display = "none";
      } else {
        img.style.display = "none";
        empty.style.display = "block";
        const launchStatus = payload.launch_status || {};
        empty.textContent = launchStatus.detail || routeConfig.emptyText;
      }
    } catch (error) {
      const empty = root.querySelector("[data-role='empty']");
      const img = root.querySelector("[data-role='image']");
      if (!empty || !img) {
        return;
      }
      img.style.display = "none";
      empty.style.display = "block";
      empty.textContent = String(error?.message || error);
    } finally {
      stopPolling();
      if (!disposed) {
        pollTimer = window.setTimeout(refreshPreviewImage, POLL_INTERVAL_MS);
      }
    }
  };

  const handleOpenPreview = async () => {
    const config = renderCurrentConfig() || extractCurrentConfigFromPreview(routeConfig);
    const openButton = root.querySelector("[data-role='open']");
    if (!openButton) {
      return;
    }
    openButton.disabled = true;

    try {
      const response = await apiPost("/api/timestep_preview/open", config);
      if (response.status !== "success") {
        throw new Error(response.message || "打开预览工具失败。");
      }
      await refreshPreviewImage();
    } catch (error) {
      window.alert(String(error?.message || error));
    } finally {
      openButton.disabled = false;
    }
  };

  const attachWidget = () => {
    if (disposed || root) {
      return;
    }

    const mountTarget = findMountTarget();
    if (!mountTarget || !mountTarget.parentNode) {
      return;
    }

    root = createWidgetShell(routeConfig);
    mountTarget.parentNode.insertBefore(root, mountTarget);
    collapsed = loadCollapsedState();

    root.querySelector("[data-role='open']").onclick = handleOpenPreview;
    root.querySelector("[data-role='refresh']").onclick = refreshPreviewImage;
    root.querySelector("[data-role='toggle']").onclick = () => {
      collapsed = !collapsed;
      saveCollapsedState(collapsed);
      applyCollapsedState();
    };

    applyCollapsedState();
    renderCurrentConfig();
    refreshPreviewImage();
  };

  attachTimer = window.setInterval(() => {
    if (disposed || root) {
      return;
    }
    attachWidget();
  }, 300);

  attachWidget();

  cleanupCurrentWidget = () => {
    disposed = true;
    stopPolling();
    stopAttachTimer();
    if (root) {
      root.remove();
      root = null;
    }
  };

  return cleanupCurrentWidget;
}

export function mountSdxlTimestepPreviewWidget() {
  return mountTimestepPreviewWidget("sdxl");
}

export function mountAnimaTimestepPreviewWidget() {
  return mountTimestepPreviewWidget("anima");
}
