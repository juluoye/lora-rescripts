from __future__ import annotations

import socket
import shutil
import subprocess
from typing import Optional

from mikazuki.utils.train_utils import parse_boolish


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def list_local_network_interfaces() -> list[str]:
    try:
        raw_items = socket.if_nameindex()
    except Exception:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for _, name in raw_items:
        normalized = str(name or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def parse_ifname_candidates(value: str) -> list[str]:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for token in normalized_value.split(","):
        name = token.strip()
        if not name or name.startswith("^") or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def is_loopback_interface(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return True
    return normalized in {"lo", "loopback", "loopback pseudo-interface 1"} or "loopback" in normalized


def pick_training_mesh_iface(nccl_socket_ifname: str, gloo_socket_ifname: str, main_process_ip: str) -> str:
    interfaces = list_local_network_interfaces()
    if not interfaces:
        return ""

    interface_set = set(interfaces)
    for name in parse_ifname_candidates(nccl_socket_ifname) + parse_ifname_candidates(gloo_socket_ifname):
        if name in interface_set:
            return name

    if main_process_ip and ":" not in str(main_process_ip) and shutil.which("ip") is not None:
        try:
            route = subprocess.run(
                ["ip", "-4", "route", "get", str(main_process_ip)],
                text=True,
                capture_output=True,
                check=False,
            )
            if route.returncode == 0:
                parts = route.stdout.strip().split()
                if "dev" in parts:
                    dev_idx = parts.index("dev") + 1
                    if dev_idx < len(parts):
                        route_iface = parts[dev_idx]
                        if route_iface in interface_set:
                            return route_iface
        except Exception:
            pass

    for iface in interfaces:
        if not is_loopback_interface(iface):
            return iface

    return interfaces[0] if interfaces else ""


def validate_socket_ifname(name: str, env_key: str) -> tuple[bool, str]:
    normalized = str(name or "").strip()
    if not normalized:
        return True, ""

    interfaces = list_local_network_interfaces()
    if not interfaces or normalized in interfaces:
        return True, ""

    return False, (
        f"{env_key} 配置为 '{normalized}'，但本机不存在该网卡。"
        f"可用网卡: {', '.join(interfaces)}。"
        f"请改成正确网卡名，或留空让系统自动选择。"
    )


def build_distributed_summary(runtime: dict) -> str:
    if runtime.get("is_multi_machine"):
        return (
            f"跨机分布式训练：{runtime['num_machines']} 台机器 × 每机 {runtime['num_processes_per_machine']} 进程，"
            f"总计 {runtime['total_num_processes']} 进程；"
            f"当前节点 rank={runtime['machine_rank']}，主节点 {runtime['main_process_ip']}:{runtime['main_process_port']}。"
        )

    if runtime.get("num_processes_per_machine", 1) > 1:
        return (
            f"本机多进程训练：每机 {runtime['num_processes_per_machine']} 进程，"
            f"总计 {runtime['total_num_processes']} 进程。"
        )

    return "单进程训练。"


def resolve_distributed_runtime(config: dict, gpu_ids: Optional[list[str]] = None) -> dict:
    config = config if isinstance(config, dict) else {}
    gpu_ids = list(gpu_ids or [])

    raw_num_processes = config.get("num_processes")
    num_processes_explicit = True
    if raw_num_processes is None:
        num_processes_explicit = False
    elif isinstance(raw_num_processes, str) and not raw_num_processes.strip():
        num_processes_explicit = False

    if num_processes_explicit:
        num_processes_per_machine = safe_int(raw_num_processes, 0)
    elif gpu_ids:
        num_processes_per_machine = len(gpu_ids)
    else:
        num_processes_per_machine = 1

    if num_processes_per_machine < 1:
        raise ValueError("num_processes 必须 >= 1。")

    if gpu_ids and num_processes_explicit and num_processes_per_machine != len(gpu_ids):
        raise ValueError(
            "显式填写的 num_processes 与所选 GPU 数量不一致。"
            f"当前 num_processes={num_processes_per_machine}，所选 GPU 数量={len(gpu_ids)}。"
            "请让两者保持一致，或留空 num_processes 让系统按所选 GPU 数量自动处理。"
        )

    enable_distributed_training = parse_boolish(config.get("enable_distributed_training", False))
    num_machines = safe_int(config.get("num_machines", 1), 1) if enable_distributed_training else 1
    machine_rank = safe_int(config.get("machine_rank", 0), 0) if enable_distributed_training else 0
    main_process_ip = str(config.get("main_process_ip", "") or "").strip() if enable_distributed_training else ""
    main_process_port = safe_int(config.get("main_process_port", 29500), 29500) if enable_distributed_training else 29500
    nccl_socket_ifname = str(config.get("nccl_socket_ifname", "") or "").strip() if enable_distributed_training else ""
    gloo_socket_ifname = str(config.get("gloo_socket_ifname", "") or "").strip() if enable_distributed_training else ""

    if num_machines < 1:
        raise ValueError("num_machines 必须 >= 1。")

    if machine_rank < 0 or machine_rank >= num_machines:
        raise ValueError("machine_rank 超出范围，请检查 machine_rank 与 num_machines。")

    if main_process_port <= 0 or main_process_port > 65535:
        raise ValueError("main_process_port 必须在 1 到 65535 之间。")

    if enable_distributed_training and num_machines > 1 and not main_process_ip:
        raise ValueError("多机训练时 main_process_ip 不能为空。")

    if enable_distributed_training and num_machines > 1:
        for env_key, ifname in (
            ("NCCL_SOCKET_IFNAME", nccl_socket_ifname),
            ("GLOO_SOCKET_IFNAME", gloo_socket_ifname),
        ):
            ok, message = validate_socket_ifname(ifname, env_key)
            if not ok:
                raise ValueError(message)

    total_num_processes = num_processes_per_machine * num_machines
    warnings: list[str] = []
    if enable_distributed_training and num_machines > 1 and not num_processes_explicit and not gpu_ids:
        warnings.append("已启用跨机分布式训练，但未填写每机进程数，也未显式选择 GPU；当前按每机 1 进程处理。")

    notes: list[str] = []
    if enable_distributed_training and num_machines > 1:
        notes.append(build_distributed_summary({
            "is_multi_machine": True,
            "num_machines": num_machines,
            "num_processes_per_machine": num_processes_per_machine,
            "total_num_processes": total_num_processes,
            "machine_rank": machine_rank,
            "main_process_ip": main_process_ip,
            "main_process_port": main_process_port,
        }))
        notes.append("当前构建支持最小 worker 配置/缺失资源同步；数据集一致性与路径布局仍建议提前人工核对。")
    elif total_num_processes > 1:
        notes.append(build_distributed_summary({
            "is_multi_machine": False,
            "num_processes_per_machine": num_processes_per_machine,
            "total_num_processes": total_num_processes,
        }))

    env_overrides = {}
    if nccl_socket_ifname:
        env_overrides["NCCL_SOCKET_IFNAME"] = nccl_socket_ifname
    if gloo_socket_ifname:
        env_overrides["GLOO_SOCKET_IFNAME"] = gloo_socket_ifname

    return {
        "enabled": enable_distributed_training,
        "is_multi_machine": enable_distributed_training and num_machines > 1,
        "num_processes_explicit": num_processes_explicit,
        "num_processes_per_machine": num_processes_per_machine,
        "num_machines": num_machines,
        "machine_rank": machine_rank,
        "main_process_ip": main_process_ip,
        "main_process_port": main_process_port,
        "nccl_socket_ifname": nccl_socket_ifname,
        "gloo_socket_ifname": gloo_socket_ifname,
        "total_num_processes": total_num_processes,
        "env_overrides": env_overrides,
        "warnings": warnings,
        "notes": notes,
        "summary": build_distributed_summary({
            "is_multi_machine": enable_distributed_training and num_machines > 1,
            "num_processes_per_machine": num_processes_per_machine,
            "num_machines": num_machines,
            "total_num_processes": total_num_processes,
            "machine_rank": machine_rank,
            "main_process_ip": main_process_ip,
            "main_process_port": main_process_port,
        }),
    }
