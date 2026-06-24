import asyncio
import json
import os
from typing import Awaitable, Callable

import httpx
import websockets
from fastapi import APIRouter, Request, WebSocket
from httpx import ConnectError
from starlette.background import BackgroundTask
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.websockets import WebSocketDisconnect

from mikazuki.log import log

router = APIRouter()
TAGEDITOR_STATUS_FILE = os.environ.get("MIKAZUKI_TAGEDITOR_STATUS_FILE", "")
DEFAULT_UNAVAILABLE_MESSAGE = (
    "The requested service not started yet or service started fail. This may cost a while when you first time startup\n"
    "请求的服务尚未启动或启动失败。若是第一次启动，可能需要等待一段时间后再刷新网页。"
)
PROXY_TARGETS = {
    "tensorboard": ("MIKAZUKI_TENSORBOARD_HOST", "127.0.0.1", "MIKAZUKI_TENSORBOARD_PORT", "6006"),
    "tageditor": ("MIKAZUKI_TAGEDITOR_HOST", "127.0.0.1", "MIKAZUKI_TAGEDITOR_PORT", "28001"),
}
STREAMING_RESPONSE_EXCLUDED_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
TAGEDITOR_UNAVAILABLE_MESSAGES = {
    "disabled": (
        "Tag Editor is disabled for this launch. Restart without --disable-tageditor to use it.\n"
        "本次启动已禁用标签编辑器。请去掉 --disable-tageditor 后重新启动。"
    ),
    "missing_gradio": (
        "Tag Editor is unavailable because gradio is not installed in the current environment.\n"
        "标签编辑器当前不可用，因为当前环境没有安装 gradio。"
    ),
    "missing_launcher": (
        "Tag Editor files are missing. Check mikazuki/dataset-tag-editor and try again.\n"
        "标签编辑器文件缺失。请检查 mikazuki/dataset-tag-editor 目录后重试。"
    ),
    "missing_dependencies": (
        "Tag Editor dependencies are not installed. Run install_tageditor.ps1 (Windows) or install_tageditor.sh (Linux) first. If main Python is 3.13, prepare a separate python_tageditor or venv-tageditor (Python 3.12) environment.\n"
        "标签编辑器依赖尚未安装。请先运行 install_tageditor.ps1（Windows）或 install_tageditor.sh（Linux）。如果主环境是 Python 3.13，请准备单独的 python_tageditor 或 venv-tageditor（Python 3.12）环境。"
    ),
    "dedicated_runtime_required": (
        "Tag Editor is not auto-started in AMD ROCm / Intel XPU experimental runtimes. Prepare a separate python_tageditor or venv-tageditor with install_tageditor.ps1 (Windows) or install_tageditor.sh (Linux) if you need it.\n"
        "AMD ROCm / Intel XPU 实验运行时下不会自动启动标签编辑器。如需使用，请先运行 install_tageditor.ps1（Windows）或 install_tageditor.sh（Linux），准备单独的 python_tageditor 或 venv-tageditor。"
    ),
    "starting": (
        "Tag Editor is still starting or failed to start. If this is the first launch, wait a moment and refresh.\n"
        "标签编辑器正在启动，或者启动失败。如果是第一次启动，请稍等片刻后刷新页面。"
    ),
}


def read_tageditor_status() -> dict:
    fallback = {"status": os.environ.get("MIKAZUKI_TAGEDITOR_STATUS", "unknown"), "detail": ""}
    status_file = os.environ.get("MIKAZUKI_TAGEDITOR_STATUS_FILE", TAGEDITOR_STATUS_FILE)
    if not status_file or not os.path.exists(status_file):
        return fallback
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": data.get("status", fallback["status"]),
            "detail": data.get("detail", ""),
        }
    except Exception:
        return fallback


def build_tageditor_progress_page() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tag Editor Starting</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f1f1f1; color: #213547; }
    .wrap { min-height: 100vh; display: grid; place-items: center; padding: 24px; }
    .card { width: min(720px, 100%); background: #ffffff; border: 1px solid #d9e3f0; border-radius: 16px; padding: 24px; box-sizing: border-box; box-shadow: 0 14px 40px rgba(64, 158, 255, .08); }
    h1 { font-size: 22px; margin: 0 0 8px; }
    p { margin: 0 0 16px; color: #6b7280; }
    .bar { width: 100%; height: 12px; background: #edf4ff; border-radius: 999px; overflow: hidden; margin: 16px 0; }
    .fill { width: 15%; height: 100%; background: linear-gradient(90deg, #409eff, #6aa8ff); transition: width .35s ease; }
    .status { font-size: 15px; margin: 10px 0 6px; }
    .detail { font-size: 13px; color: #409eff; white-space: pre-wrap; }
    .hint { margin-top: 18px; font-size: 13px; color: #6b7280; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Tag Editor is starting</h1>
      <p>标签编辑器正在启动，准备好后会自动刷新进入。</p>
      <div class="bar"><div class="fill" id="fill"></div></div>
      <div class="status" id="status">Initializing...</div>
      <div class="detail" id="detail">Waiting for startup status...</div>
      <div class="hint">If this stays on failed, the message below should explain which stage broke.</div>
    </div>
  </div>
  <script>
    const progressMap = {
      queued: 8, starting: 12, preparing_environment: 22, loading_interface: 35, importing_interface: 45,
      launching_ui: 58, loading_settings: 68, loading_interrogators: 78, building_ui: 86, starting_server: 93, ready: 100
    };
    const labelMap = {
      queued: "Queued",
      starting: "Launching subprocess",
      preparing_environment: "Preparing runtime",
      loading_interface: "Runtime ready",
      importing_interface: "Importing interface",
      launching_ui: "Launching UI",
      loading_settings: "Loading settings",
      loading_interrogators: "Loading interrogators",
      building_ui: "Building page",
      starting_server: "Starting local service",
      ready: "Ready",
      failed: "Startup failed",
      missing_dependencies: "Dependencies missing",
      dedicated_runtime_required: "Dedicated runtime required",
      missing_launcher: "Launcher missing",
      disabled: "Disabled"
    };
    async function poll() {
      try {
        const res = await fetch('/api/tageditor_status', { cache: 'no-store' });
        const data = await res.json();
        const pct = progressMap[data.status] ?? (data.status === 'failed' ? 100 : 15);
        document.getElementById('fill').style.width = pct + '%';
        document.getElementById('status').textContent = labelMap[data.status] || data.status;
        document.getElementById('detail').textContent = data.detail || '';
        if (data.status === 'ready') {
          location.reload();
          return;
        }
      } catch (err) {
        document.getElementById('detail').textContent = String(err);
      }
      setTimeout(poll, 1200);
    }
    poll();
  </script>
</body>
</html>"""


def get_unavailable_message(url_type: str) -> str:
    if url_type != "tageditor":
        return DEFAULT_UNAVAILABLE_MESSAGE

    status = read_tageditor_status()["status"]
    return TAGEDITOR_UNAVAILABLE_MESSAGES.get(status, DEFAULT_UNAVAILABLE_MESSAGE)


def resolve_proxy_target(url_type: str) -> tuple[str, str]:
    host_key, default_host, port_key, default_port = PROXY_TARGETS[url_type]
    host = os.environ.get(host_key, default_host)
    port = os.environ.get(port_key, default_port)
    return host, port


def build_proxy_url(request: Request, full_path: bool) -> httpx.URL:
    path = request.url.path if full_path else request.path_params.get("path", "")
    return httpx.URL(path=path, query=request.url.query.encode("utf-8"))


def build_proxy_request(client: httpx.AsyncClient, request: Request, url: httpx.URL) -> httpx.Request:
    return client.build_request(
        request.method,
        url,
        headers=request.headers.raw,
        content=request.stream() if request.method != "GET" else None,
    )


def build_proxy_error_response(url_type: str, request_method: str):
    if url_type == "tageditor" and request_method == "GET":
        return HTMLResponse(content=build_tageditor_progress_page(), status_code=503)

    return PlainTextResponse(content=get_unavailable_message(url_type), status_code=502)


def build_proxy_streaming_response(response: httpx.Response) -> StreamingResponse:
    proxy_response = StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        background=BackgroundTask(response.aclose),
    )
    for header_name, header_value in response.headers.multi_items():
        if header_name.lower() in STREAMING_RESPONSE_EXCLUDED_HEADERS:
            continue
        proxy_response.headers.append(header_name, header_value)
    return proxy_response


def reverse_proxy_maker(url_type: str, full_path: bool = False):
    host, port = resolve_proxy_target(url_type)
    client = httpx.AsyncClient(base_url=f"http://{host}:{port}/", trust_env=False, timeout=360)

    async def _reverse_proxy(request: Request):
        url = build_proxy_url(request, full_path)
        rp_req = build_proxy_request(client, request, url)
        try:
            rp_resp = await client.send(rp_req, stream=True)
        except ConnectError:
            return build_proxy_error_response(url_type, request.method)
        return build_proxy_streaming_response(rp_resp)

    return _reverse_proxy


@router.get("/api/tageditor_status")
async def get_tageditor_status():
    return JSONResponse(read_tageditor_status())


async def relay_websocket_messages(
    receiver: Callable[[], Awaitable[object]],
    sender: Callable[[object], Awaitable[object]],
    disconnect_exceptions: tuple[type[BaseException], ...],
    error_message: str,
):
    while True:
        try:
            data = await receiver()
            await sender(data)
        except disconnect_exceptions:
            break
        except Exception as exc:
            log.error(f"{error_message}: {exc}")
            break


async def proxy_ws_forward(ws_a: WebSocket, ws_b: websockets.WebSocketClientProtocol):
    await relay_websocket_messages(
        ws_a.receive_text,
        ws_b.send,
        (WebSocketDisconnect,),
        "Error when proxy data client -> backend",
    )


async def proxy_ws_reverse(ws_a: WebSocket, ws_b: websockets.WebSocketClientProtocol):
    await relay_websocket_messages(
        ws_b.recv,
        ws_a.send_text,
        (websockets.exceptions.ConnectionClosedOK,),
        "Error when proxy data backend -> client",
    )


def build_tageditor_websocket_uri(path: str) -> str:
    host, port = resolve_proxy_target("tageditor")
    return f"ws://{host}:{port}/{path.lstrip('/')}"


@router.websocket("/proxy/tageditor/queue/join")
async def websocket_a(ws_a: WebSocket):
    ws_b_uri = build_tageditor_websocket_uri("queue/join")
    await ws_a.accept()
    async with websockets.connect(ws_b_uri, timeout=360, ping_timeout=None) as ws_b_client:
        fwd_task = asyncio.create_task(proxy_ws_forward(ws_a, ws_b_client))
        rev_task = asyncio.create_task(proxy_ws_reverse(ws_a, ws_b_client))
        await asyncio.gather(fwd_task, rev_task)

router.add_route("/proxy/tensorboard/{path:path}", reverse_proxy_maker("tensorboard"), ["GET", "POST"])
router.add_route("/font-roboto/{path:path}", reverse_proxy_maker("tensorboard", full_path=True), ["GET", "POST"])
router.add_route("/proxy/tageditor/{path:path}", reverse_proxy_maker("tageditor"), ["GET", "POST"])
