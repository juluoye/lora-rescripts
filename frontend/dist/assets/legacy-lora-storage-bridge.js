(function () {
  const PAGE_KEYS = {
    "master.html": "master",
    "flux.html": "flux",
    "sd3.html": "sd3",
    "sdxl.html": "sdxl",
    "newbie.html": "newbie",
    "basic.html": "basic",
    "anima.html": "anima",
    "anima-finetune.html": "anima-finetune"
  };

  const pathParts = window.location.pathname.split("/");
  const fileName = pathParts[pathParts.length - 1] || "";
  const pageId = PAGE_KEYS[fileName];
  if (!pageId) {
    return;
  }

  const savedKey = "configs-lora-" + pageId;
  const autosaveKey = savedKey + "-autosave";
  const sourceMap = new Map();
  const safeNameMap = new Map();
  let syncScheduled = false;
  let initialized = false;

  function toSafeName(value) {
    return String(value || "").replace(/[<>:"/\\|?*]+/g, "_").trim();
  }

  function toEntry(name, time) {
    return {
      name: name,
      time: typeof time === "number" ? new Date(time).toLocaleString("zh-CN", { hour12: false }) : "",
      value: {}
    };
  }

  function readSavedList() {
    try {
      const payload = JSON.parse(localStorage.getItem(savedKey) || "[]");
      return Array.isArray(payload) ? payload : [];
    } catch (error) {
      return [];
    }
  }

  function writeSavedList(items) {
    localStorage.setItem(savedKey, JSON.stringify(items));
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok || payload.status !== "success") {
      throw new Error(payload.message || ("请求失败: " + response.status));
    }
    return payload.data || {};
  }

  async function hydrateFromLocalToml() {
    try {
      const data = await fetchJson("/api/legacy_lora_configs/list?page=" + encodeURIComponent(pageId));
      const localConfigs = Array.isArray(data.configs) ? data.configs : [];
      if (!localConfigs.length) {
        return;
      }

      const currentItems = readSavedList();
      const currentNames = new Set(
        currentItems
          .map(function (item) { return item && typeof item.time === "string" ? item.time : ""; })
          .filter(Boolean)
      );
      const nextItems = currentItems.slice();

      for (const item of localConfigs) {
        sourceMap.set(item.name, item.source || "legacy-local");
        safeNameMap.set(item.name, item.name);
        if (!currentNames.has(item.name)) {
          nextItems.push(toEntry(item.name, item.time));
        }
      }

      nextItems.sort(function (a, b) {
        return String(b.time || "").localeCompare(String(a.time || ""));
      });
      writeSavedList(nextItems);
    } catch (error) {
      console.warn("legacy storage bridge hydrate failed", error);
    }
  }

  async function loadIntoSavedEntries(items) {
    const pending = [];
    for (const item of items) {
      if (!item || item.value && Object.keys(item.value).length) {
        continue;
      }
      const configName = item.time;
      if (!configName) {
        continue;
      }
      const requestName = safeNameMap.get(configName) || toSafeName(configName);
      pending.push(
        fetchJson(
          "/api/legacy_lora_configs/load?page=" + encodeURIComponent(pageId)
          + "&name=" + encodeURIComponent(requestName)
          + "&source=" + encodeURIComponent(sourceMap.get(configName) || "legacy-local")
        ).then(function (data) {
          item.value = data.config || {};
        }).catch(function () {
          item.value = item.value || {};
        })
      );
    }
    await Promise.all(pending);
    writeSavedList(items);
  }

  function buildSavePayload(entry) {
    if (!entry || !entry.time || !entry.value || typeof entry.value !== "object") {
      return null;
    }
    return {
      page: pageId,
      name: entry.time,
      config: entry.value
    };
  }

  async function flushSavedList() {
    syncScheduled = false;
    if (!initialized) {
      return;
    }

    const items = readSavedList();
    const requests = [];
    for (const entry of items) {
      const payload = buildSavePayload(entry);
      if (!payload) {
        continue;
      }
      requests.push(
        fetchJson("/api/legacy_lora_configs/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }).catch(function (error) {
          console.warn("legacy storage bridge save failed", error);
        })
      );
    }

    const autosaveRaw = localStorage.getItem(autosaveKey);
    if (autosaveRaw) {
      try {
        const autosaveValue = JSON.parse(autosaveRaw);
        if (autosaveValue && typeof autosaveValue === "object") {
          requests.push(
            fetchJson("/api/legacy_lora_configs/save", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                page: pageId,
                name: "latest",
                config: autosaveValue
              })
            }).catch(function (error) {
              console.warn("legacy storage bridge autosave sync failed", error);
            })
          );
        }
      } catch (error) {
        console.warn("legacy storage bridge autosave parse failed", error);
      }
    }

    await Promise.all(requests);
  }

  function scheduleSync() {
    if (syncScheduled) {
      return;
    }
    syncScheduled = true;
    window.setTimeout(flushSavedList, 300);
  }

  const originalSetItem = localStorage.setItem.bind(localStorage);
  localStorage.setItem = function (key, value) {
    const result = originalSetItem(key, value);
    if (key === savedKey || key === autosaveKey) {
      scheduleSync();
    }
    return result;
  };

  window.addEventListener("DOMContentLoaded", async function () {
    await hydrateFromLocalToml();
    const items = readSavedList();
    await loadIntoSavedEntries(items);

    try {
      const latest = await fetchJson(
        "/api/legacy_lora_configs/load?page=" + encodeURIComponent(pageId)
        + "&name=latest&source=legacy-local"
      );
      if (latest && latest.config && Object.keys(latest.config).length) {
        localStorage.setItem(autosaveKey, JSON.stringify(latest.config));
      }
    } catch (error) {
      // latest is optional
    }

    initialized = true;
  });
})();
