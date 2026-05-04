from __future__ import annotations

from dataclasses import dataclass, field

from .config import NewbieRuntimeConfig
from .dataset import NewbieDatasetReport, build_newbie_dataset_report


@dataclass(slots=True)
class NewbieExecutionPhase:
    name: str
    enabled: bool
    reason: str
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NewbiePreparationResult:
    config: NewbieRuntimeConfig
    dataset: NewbieDatasetReport
    phases: list[NewbieExecutionPhase]
    warnings: list[str]
    notes: list[str]


class NewbieTrainer:
    def __init__(self, config: NewbieRuntimeConfig) -> None:
        self.config = config

    def prepare(self, initial_warnings: list[str] | None = None) -> NewbiePreparationResult:
        warnings = list(initial_warnings or [])
        notes: list[str] = []

        dataset_report = build_newbie_dataset_report(
            train_data_dir=self.config.train_data_dir,
            caption_extension=self.config.caption_extension,
            max_resolution=self.config.model_resolution,
            min_bucket_reso=self.config.min_bucket_reso,
            max_bucket_reso=self.config.max_bucket_reso,
            bucket_reso_step=self.config.bucket_reso_step,
            caption_length_bucket_size=self.config.newbie_caption_length_bucket_size,
            long_caption_threshold=self.config.newbie_gemma_max_token_length,
        )

        if dataset_report.missing_caption_count > 0:
            warnings.append(
                f"检测到 {dataset_report.missing_caption_count} 张图片缺少 caption；Newbie 联合序列训练建议尽量补齐文本标签。"
            )
        if dataset_report.long_caption_count > 0:
            warnings.append(
                "检测到部分 caption 估计长度超过当前 Gemma token 上限；后续缓存/编码阶段应主动裁剪或分桶，避免被长文本拖大联合序列。"
            )
        if self.config.enable_preview and not self.config.sample_prompts:
            warnings.append("已启用 Newbie 训练预览，但当前没有 sample_prompts；未提供预览提示词时不会生成预览图。")
        if not self.config.use_cache:
            notes.append("已请求 use_cache=false：stable wrapper 会为本次训练生成临时 cache 文件，并在训练完成后清理本次新增 cache。")
        if not self.config.newbie_two_phase_execution:
            notes.append("已请求关闭两阶段执行：stable wrapper 会在同一进程内连续执行 cache 与 train。")
        if self.config.peak_vram_control_enabled:
            notes.append(
                "已启用显存峰值控制："
                f"target_effective={self.config.peak_vram_target_effective_batch or 'off'}，"
                f"realized={self.config.peak_vram_effective_batch_realized or self.config.effective_batch_size}。"
            )
            if self.config.peak_vram_micro_batch_enabled:
                notes.append(
                    f"已启用 micro-batch 拆分：train_batch_size={self.config.train_batch_size}，"
                    f"micro_batch_size={self.config.peak_vram_micro_batch_size}。"
                )
            if self.config.peak_vram_diagnostics_enabled:
                notes.append(
                    f"已启用显存阶段诊断：每 {self.config.peak_vram_diagnostics_interval} 个优化 step 输出一次。"
                )
            if self.config.peak_vram_startup_guard_enabled:
                guard_duration = (
                    "整段训练"
                    if self.config.peak_vram_startup_guard_steps <= 0
                    else f"前 {self.config.peak_vram_startup_guard_steps} 个优化 step"
                )
                notes.append(
                    f"已启用启动峰值保护：{guard_duration} 使用 {self.config.peak_vram_startup_guard_resolved_mode} 档。"
                )

        if self.config.blocks_to_swap > 0:
            notes.append(f"blocks_to_swap={self.config.blocks_to_swap}，后续正式训练实现会优先按安全模式接入该省显存策略。")
            if self.config.newbie_auto_swap_release:
                notes.append("已启用自动 swap 释放：显存占用持续偏低时会逐步减少 blocks_to_swap。")
        if self.config.cpu_offload_checkpointing:
            notes.append("已启用 cpu_offload_checkpointing 规划标记。")
        if self.config.pytorch_cuda_expandable_segments:
            notes.append("已计划使用 PyTorch CUDA expandable_segments 以降低碎片化 OOM 风险。")

        phases = self._build_execution_plan(dataset_report, notes)
        return NewbiePreparationResult(
            config=self.config,
            dataset=dataset_report,
            phases=phases,
            warnings=warnings,
            notes=notes,
        )

    def _build_execution_plan(
        self,
        dataset_report: NewbieDatasetReport,
        notes: list[str],
    ) -> list[NewbieExecutionPhase]:
        phases: list[NewbieExecutionPhase] = []

        transient_cache_mode = not self.config.use_cache
        cache_needed = self.config.newbie_rebuild_cache or transient_cache_mode or not dataset_report.cache_complete
        phases.append(
            NewbieExecutionPhase(
                name="cache",
                enabled=cache_needed,
                reason=(
                    "use_cache=false 已启用兼容模式；仍会先生成本次训练所需的临时 cache。"
                    if transient_cache_mode
                    else (
                        "cache 缺失或显式要求重建，先单独执行编码缓存阶段。"
                        if cache_needed
                        else "缓存完整，跳过独立 cache 阶段。"
                    )
                ),
                notes=[
                    "将 Gemma / Jina CLIP / VAE 与正式训练阶段解耦，避免把首次缓存峰值叠到训练峰值上。"
                ] + (["本次新增 cache 文件会在训练结束后清理。"] if transient_cache_mode else []),
            )
        )

        train_notes = [
            "正式训练阶段优先只保留 NextDiT + adapter。",
            "caption 长度分桶会与分辨率 bucket 共同决定 batch 分组，减少长文本 padding 浪费。",
        ]
        if self.config.newbie_refiner_checkpointing:
            train_notes.append("context_refiner / noise_refiner 规划纳入额外 checkpointing。")
        if self.config.newbie_force_cache_only:
            train_notes.append("force_cache_only 已开启；正式训练阶段不应回退到 no-cache 编码路径。")
        if not self.config.newbie_two_phase_execution:
            train_notes.append("two_phase 已关闭；cache 与 train 会在同一次脚本执行中串行完成。")

        phases.append(
            NewbieExecutionPhase(
                name="train",
                enabled=True,
                reason="执行 Newbie LoRA / LoKr 正式训练阶段。",
                notes=train_notes,
            )
        )

        if self.config.enable_preview:
            phases.append(
                NewbieExecutionPhase(
                    name="preview",
                    enabled=True,
                    reason="训练中预览已启用，将按 sample_at_first / sample_every_n_steps / sample_every_n_epochs 规则尝试生成。",
                    notes=["当前预览走 Newbie stable 最小采样链路，会额外占用一定显存与时间。"],
                )
            )

        notes.append(f"execution phases: {', '.join(phase.name for phase in phases if phase.enabled)}")
        return phases

    def format_preparation_summary(self, result: NewbiePreparationResult) -> list[str]:
        lines = [
            "========================================",
            "Lulynx Newbie Trainer Preparation",
            "========================================",
        ]
        lines.extend(result.config.describe())
        lines.append(
            "dataset="
            f"{result.dataset.total_images} images, "
            f"repeated={result.dataset.total_repeated_images}, "
            f"cache_complete={'yes' if result.dataset.cache_complete else 'no'}, "
            f"missing_cache={result.dataset.missing_cache_count}, "
            f"missing_caption={result.dataset.missing_caption_count}"
        )
        lines.append(
            "caption_stats="
            f"avg {result.dataset.average_caption_length:.1f}, "
            f"max {result.dataset.max_caption_length}, "
            f"over_limit {result.dataset.long_caption_count}"
        )

        if result.dataset.resolution_buckets:
            lines.append(
                "resolution_buckets="
                + ", ".join(f"{key}:{value}" for key, value in sorted(result.dataset.resolution_buckets.items()))
            )
        if result.dataset.caption_buckets:
            lines.append(
                "caption_buckets="
                + ", ".join(
                    f"{key}:{value}" for key, value in sorted(
                        result.dataset.caption_buckets.items(),
                        key=lambda item: int(item[0]),
                    )
                )
            )

        for phase in result.phases:
            status = "enabled" if phase.enabled else "skipped"
            lines.append(f"[phase] {phase.name}: {status} - {phase.reason}")
            for note in phase.notes:
                lines.append(f"  - {note}")

        if result.warnings:
            lines.append("[warnings]")
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        if result.notes:
            lines.append("[notes]")
            for note in result.notes:
                lines.append(f"  - {note}")

        return lines
