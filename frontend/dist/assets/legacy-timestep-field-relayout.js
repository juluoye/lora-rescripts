const LEGACY_TIMESTEP_LAYOUTS = [
    {
        route: /\/lora\/anima(?:\.html)?\/?$/,
        anchorKeys: ["discrete_flow_shift", "timestep_sampling"],
        movedKeys: ["min_timestep", "max_timestep"],
    },
    {
        route: /\/lora\/sdxl(?:\.html)?\/?$/,
        anchorKeys: ["timestep_sampling", "timestep_loss_weighting"],
        movedKeys: ["min_timestep", "max_timestep"],
    },
];

function normalizeText(value) {
    return String(value || "").trim().toLowerCase();
}

function findFieldBlocks() {
    const blocks = [];
    const headings = document.querySelectorAll(".schema-container h3");
    for (const heading of headings) {
        const key = normalizeText(heading.textContent);
        if (!key) continue;
        const block = heading.closest(".schema-container > div .el-scrollbar__view > div, .schema-container .el-scrollbar__view > div");
        if (!block) continue;
        blocks.push({ key, block });
    }
    return blocks;
}

function moveLegacyTimestepFields() {
    const pathname = String(window.location && window.location.pathname || "");
    const layout = LEGACY_TIMESTEP_LAYOUTS.find((item) => item.route.test(pathname));
    if (!layout) return false;

    const blocks = findFieldBlocks();
    if (!blocks.length) return false;

    const blockMap = new Map(blocks.map((item) => [item.key, item.block]));
    const anchorBlock = layout.anchorKeys.map((key) => blockMap.get(key)).find(Boolean);
    const movedBlocks = layout.movedKeys.map((key) => blockMap.get(key)).filter(Boolean);

    if (!anchorBlock || movedBlocks.length !== layout.movedKeys.length) {
        return false;
    }

    const anchorParent = anchorBlock.parentElement;
    if (!anchorParent) return false;

    let insertAfter = anchorBlock;
    for (const block of movedBlocks) {
        if (block === insertAfter.nextElementSibling) {
            insertAfter = block;
            continue;
        }
        anchorParent.insertBefore(block, insertAfter.nextElementSibling);
        insertAfter = block;
    }
    return true;
}

function installLegacyTimestepRelayout() {
    let observer = null;
    let settledTimer = null;

    const tryRelayout = () => {
        if (!moveLegacyTimestepFields()) return;
        if (settledTimer) window.clearTimeout(settledTimer);
        settledTimer = window.setTimeout(() => {
            if (observer) {
                observer.disconnect();
                observer = null;
            }
        }, 1500);
    };

    const startObserver = () => {
        const container = document.querySelector(".schema-container");
        if (!container || observer) return;
        observer = new MutationObserver(() => {
            tryRelayout();
        });
        observer.observe(container, { childList: true, subtree: true });
        tryRelayout();
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", startObserver, { once: true });
    } else {
        startObserver();
    }

    window.addEventListener("pageshow", tryRelayout);
    window.addEventListener("focus", tryRelayout);
}

installLegacyTimestepRelayout();
