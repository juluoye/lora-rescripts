const LEGACY_SCHEMA_PRELOAD_PATHS = new Set([
    "/lora/anima.html",
    "/lora/sdxl.html",
]);

function shouldWarmLegacySchemas() {
    if (typeof window === "undefined") return false;
    const path = String(window.location && window.location.pathname || "");
    return LEGACY_SCHEMA_PRELOAD_PATHS.has(path);
}

function readCachedSchemas() {
    if (typeof window === "undefined") return [];
    try {
        const raw = window.localStorage.getItem("schemas");
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        console.error("Failed to read cached schemas:", error);
        try {
            window.localStorage.removeItem("schemas");
        } catch {
        }
        return [];
    }
}

function needsSchemaRefresh(cachedSchemas, serverHashes) {
    if (!Array.isArray(cachedSchemas) || cachedSchemas.length === 0) return true;
    if (!Array.isArray(serverHashes) || serverHashes.length === 0) return false;
    const cachedHashMap = new Map();
    for (const item of cachedSchemas) {
        if (!item || typeof item.name !== "string") continue;
        cachedHashMap.set(item.name, item.hash);
    }
    for (const item of serverHashes) {
        if (!item || typeof item.name !== "string") continue;
        if (cachedHashMap.get(item.name) !== item.hash) return true;
    }
    return false;
}

async function fetchSchemaHashes() {
    const response = await fetch("/api/schemas/hashes", { cache: "no-store" });
    if (!response.ok) return [];
    const payload = await response.json();
    const schemas = payload && payload.data ? payload.data.schemas : [];
    return Array.isArray(schemas) ? schemas : [];
}

async function fetchAllSchemas() {
    const response = await fetch("/api/schemas/all", { cache: "no-store" });
    if (!response.ok) return null;
    const payload = await response.json();
    const schemas = payload && payload.data ? payload.data.schemas : [];
    return Array.isArray(schemas) ? schemas : null;
}

async function warmLegacySchemas() {
    if (!shouldWarmLegacySchemas()) return;
    const cachedSchemas = readCachedSchemas();
    const serverHashes = await fetchSchemaHashes();
    if (!needsSchemaRefresh(cachedSchemas, serverHashes)) return;
    const serverSchemas = await fetchAllSchemas();
    if (!serverSchemas) return;
    try {
        window.localStorage.setItem("schemas", JSON.stringify(serverSchemas));
    } catch (error) {
        console.error("Failed to persist warmed schemas:", error);
    }
}

try {
    await warmLegacySchemas();
} catch (error) {
    console.error("Legacy schema preload boot failed:", error);
}
