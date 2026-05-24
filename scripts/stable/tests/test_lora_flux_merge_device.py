import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.lora_flux import LoRAInfModule


def _lora_weights(in_features=4, out_features=3, rank=2):
    torch.manual_seed(7)
    return {
        "lora_down.weight": torch.randn(rank, in_features),
        "lora_up.weight": torch.randn(out_features, rank),
    }


def test_lora_inf_merge_matches_cpu_reference_with_explicit_cpu_device():
    linear = torch.nn.Linear(4, 3, bias=False)
    original_weight = linear.weight.detach().clone()
    module = LoRAInfModule("lora_unet_block_linear", linear, multiplier=1.0, lora_dim=2, alpha=2)
    weights_sd = _lora_weights()

    module.merge_to(weights_sd, dtype=torch.float32, device="cpu")

    expected = original_weight.float() + (
        weights_sd["lora_up.weight"].float() @ weights_sd["lora_down.weight"].float()
    )
    torch.testing.assert_close(linear.weight, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lora_inf_merge_accepts_cpu_merge_device_for_cuda_module():
    linear = torch.nn.Linear(4, 3, bias=False).cuda()
    original_weight = linear.weight.detach().cpu().clone()
    module = LoRAInfModule("lora_unet_block_linear", linear, multiplier=1.0, lora_dim=2, alpha=2)
    weights_sd = _lora_weights()

    module.merge_to(weights_sd, dtype=torch.float32, device="cpu")

    expected = original_weight.float() + (
        weights_sd["lora_up.weight"].float() @ weights_sd["lora_down.weight"].float()
    )
    assert linear.weight.device.type == "cuda"
    torch.testing.assert_close(linear.weight.detach().cpu(), expected)
