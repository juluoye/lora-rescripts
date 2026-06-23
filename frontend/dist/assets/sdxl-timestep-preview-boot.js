import {
  mountAnimaTimestepPreviewWidget,
  mountSdxlTimestepPreviewWidget,
} from "/assets/sdxl-timestep-preview-widget.js";

const TARGET_ROUTE_TABLE = [
  { re: /\/lora\/sdxl(?:\.html)?\/?$/, mount: mountSdxlTimestepPreviewWidget },
  { re: /\/lora\/anima(?:\.html)?\/?$/, mount: mountAnimaTimestepPreviewWidget },
];
const ROUTE_EVENT = "lulynx:sdxl-timestep-preview-route-change";

let cleanupCurrentMount = null;
let routeTimer = null;
let hooksInstalled = false;
let mountedPathname = "";

function normalizePath(pathname) {
  return String(pathname || "").replace(/\/+$/, "");
}

function resolveTargetMount() {
  const pathname = normalizePath(window.location.pathname);
  const match = TARGET_ROUTE_TABLE.find((item) => item.re.test(pathname));
  return match ? match.mount : null;
}

function clearMountedWidget() {
  if (typeof cleanupCurrentMount === "function") {
    cleanupCurrentMount();
  }
  cleanupCurrentMount = null;
  mountedPathname = "";
}

function syncRoute() {
  const pathname = normalizePath(window.location.pathname);
  const mountWidget = resolveTargetMount();
  if (!mountWidget) {
    clearMountedWidget();
    return;
  }

  if (cleanupCurrentMount && mountedPathname === pathname) {
    return;
  }

  clearMountedWidget();
  cleanupCurrentMount = mountWidget();
  mountedPathname = pathname;
}

function queueSync() {
  if (routeTimer) {
    clearTimeout(routeTimer);
  }
  routeTimer = window.setTimeout(syncRoute, 80);
}

function emitRouteEvent() {
  window.dispatchEvent(new Event(ROUTE_EVENT));
}

function patchHistoryMethod(methodName) {
  const original = window.history[methodName];
  if (typeof original !== "function") {
    return;
  }

  window.history[methodName] = function patchedHistoryMethod(...args) {
    const result = original.apply(this, args);
    emitRouteEvent();
    return result;
  };
}

function installRouteHooks() {
  if (hooksInstalled) {
    return;
  }
  hooksInstalled = true;

  patchHistoryMethod("pushState");
  patchHistoryMethod("replaceState");

  window.addEventListener("popstate", queueSync);
  window.addEventListener(ROUTE_EVENT, queueSync);
  window.addEventListener("pageshow", queueSync);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      queueSync();
    }
  });
}

installRouteHooks();
queueSync();
