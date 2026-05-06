from __future__ import annotations

import bitsandbytes
import bitsandbytes.functional as F
import torch

from library.optimizer_offload_util import normalize_optimizer_offload_mode, should_offload_optimizer_tensor


def _stochastic_round_bf16_(source_fp32: torch.Tensor, target_bf16: torch.Tensor) -> None:
    """Stochastically round an fp32 tensor into a bf16 target via mantissa dithering."""
    source_fp32 = source_fp32.contiguous()
    bits = source_fp32.view(torch.int32)
    rand = torch.randint_like(bits, 0, 1 << 16)
    rounded = ((bits + rand) & ~0xFFFF).view(torch.float32)
    target_bf16.copy_(rounded)


class AdamW8bitKahan(bitsandbytes.optim.AdamW8bit):
    def __init__(
        self,
        *args,
        stabilize: bool = True,
        kahan_buffer_offload: bool = False,
        optimizer_offload_mode: str = "ndim_ge_2",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stabilize = stabilize
        self.kahan_buffer_offload = bool(kahan_buffer_offload)
        self.optimizer_offload_mode = normalize_optimizer_offload_mode(optimizer_offload_mode)

    def _should_offload_kahan_buffer(self, p: torch.Tensor) -> bool:
        return self.kahan_buffer_offload and should_offload_optimizer_tensor(p, mode=self.optimizer_offload_mode)

    def _get_shift_runtime(self, p: torch.Tensor) -> tuple[torch.Tensor, bool]:
        shift = self.state[p]["shift"]
        if shift.device == p.device:
            return shift, False
        return shift.to(device=p.device, non_blocking=True), True

    def _restore_shift_storage(self, p: torch.Tensor, shift_runtime: torch.Tensor, *, copied_from_offload: bool) -> torch.Tensor:
        if copied_from_offload:
            restored = shift_runtime.to(device="cpu", non_blocking=True)
            self.state[p]["shift"] = restored
            return restored
        self.state[p]["shift"] = shift_runtime
        return shift_runtime

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        super().init_state(group, p, gindex, pindex)
        shift = self.get_state_buffer(p, dtype=p.dtype)
        if self._should_offload_kahan_buffer(p):
            shift = shift.to(device="cpu")
        self.state[p]["shift"] = shift

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        # bitsandbytes kernels are happier on contiguous buffers.
        p.data = p.data.contiguous()
        if p.grad is not None:
            p.grad = p.grad.contiguous()

        state = self.state[p]
        grad = p.grad
        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]

        if config["percentile_clipping"] < 100:
            _, _, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"],
                step,
                config["percentile_clipping"],
            )
        else:
            gnorm_scale = 1.0

        shift, shift_was_offloaded = self._get_shift_runtime(p)

        if self.stabilize:
            exp_avg_sq = state["state2"]
            eps_sq = torch.tensor(config["eps"] ** 2, dtype=exp_avg_sq.dtype, device=exp_avg_sq.device)
            rms = grad.pow(2).div_(torch.maximum(exp_avg_sq, eps_sq)).mean().sqrt()
            lr = config["lr"] / max(1.0, rms.item())
        else:
            lr = config["lr"]

        if state["state1"].dtype == torch.float:
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                lr,
                state["state2"],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config["alpha"],
                0.0,
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )
        elif state["state1"].dtype == torch.uint8 and not config["block_wise"]:
            F.optimizer_update_8bit(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                lr,
                state["qmap1"],
                state["qmap2"],
                state["max1"],
                state["max2"],
                state["new_max1"],
                state["new_max2"],
                0.0,
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
            )
            state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
            state["max2"], state["new_max2"] = state["new_max2"], state["max2"]
        elif state["state1"].dtype == torch.uint8 and config["block_wise"]:
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config["alpha"],
                config["eps"],
                step,
                lr,
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                0.0,
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )

        # Apply decoupled weight decay against the real parameter rather than the compensation buffer.
        wd = config["weight_decay"]
        if wd > 0.0:
            wd_update = p.data.float().mul_(lr * wd)
            shift_fp32 = shift.float().sub_(wd_update)
            if shift.dtype == torch.bfloat16:
                _stochastic_round_bf16_(shift_fp32, shift)
            else:
                shift.copy_(shift_fp32.to(dtype=shift.dtype))

        buffer = p.clone()
        p.add_(shift)
        shift.add_(buffer.sub_(p))
        self._restore_shift_storage(p, shift, copied_from_offload=shift_was_offloaded)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                shift = state.get("shift")
                if torch.is_tensor(shift) and self._should_offload_kahan_buffer(p):
                    state["shift"] = shift.to(device="cpu")


__all__ = ["AdamW8bitKahan"]
