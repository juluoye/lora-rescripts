import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.lora_anima import LoKrInfModule, LoKrModule, LoRANetwork, create_network, create_network_from_weights


class Block(torch.nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.k_norm = torch.nn.LayerNorm(in_features)
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)


class DummyAnima(torch.nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.block = Block(in_features, out_features)


class DummyAnimaWithNorm(torch.nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.block = Block(in_features, out_features)


def _lokr_keys(module):
    return sorted(key for key in module.state_dict().keys() if "lokr" in key)


def _comfyui_lokr_weight(state_dict, prefix, shape):
    alpha = float(state_dict[f"{prefix}.alpha"].detach().float().cpu().item())
    lokr_w1 = state_dict.get(f"{prefix}.lokr_w1")
    lokr_w2 = state_dict.get(f"{prefix}.lokr_w2")
    lokr_w1_a = state_dict.get(f"{prefix}.lokr_w1_a")
    lokr_w1_b = state_dict.get(f"{prefix}.lokr_w1_b")
    lokr_w2_a = state_dict.get(f"{prefix}.lokr_w2_a")
    lokr_w2_b = state_dict.get(f"{prefix}.lokr_w2_b")

    if lokr_w1 is None:
        lokr_w1 = (lokr_w1_a @ lokr_w1_b) * (alpha / lokr_w1_b.shape[0])
    if lokr_w2 is None:
        lokr_w2 = (lokr_w2_a @ lokr_w2_b) * (alpha / lokr_w2_b.shape[0])

    return torch.kron(lokr_w1, lokr_w2).reshape(shape)


def _assert_forward_backward(module, linear, in_features):
    module.apply_to()
    output = linear(torch.randn(2, in_features))
    output.sum().backward()
    assert torch.isfinite(output).all()


def test_lokr_default_uses_native_direct_keys():
    linear = torch.nn.Linear(16, 16, bias=False)
    module = LoKrModule("direct", linear, lora_dim=4, alpha=4, factor=4)

    assert _lokr_keys(module) == ["lokr_rank", "lokr_w1", "lokr_w2"]
    _assert_forward_backward(module, linear, 16)


def test_lokr_decompose_both_low_rank_uses_native_factor_keys():
    linear = torch.nn.Linear(16, 16, bias=False)
    module = LoKrModule("decomposed", linear, lora_dim=1, alpha=1, factor=4, decompose_both=True)

    assert _lokr_keys(module) == ["lokr_rank", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b"]
    _assert_forward_backward(module, linear, 16)


def test_lokr_decomposed_keeps_alpha_rank_scale():
    linear = torch.nn.Linear(16, 16, bias=False)
    module = LoKrModule("decomposed_scaled", linear, lora_dim=1, alpha=0.25, factor=4, decompose_both=True)

    assert _lokr_keys(module) == ["lokr_rank", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b"]
    assert module.scale == 0.25


def test_lokr_full_matrix_and_high_rank_use_native_direct_keys():
    full_linear = torch.nn.Linear(16, 16, bias=False)
    full_module = LoKrModule("full", full_linear, lora_dim=1_000_000, alpha=1, factor=4, full_matrix=True)
    assert _lokr_keys(full_module) == ["lokr_rank", "lokr_w1", "lokr_w2"]
    assert full_module.scale == 1.0

    high_rank_linear = torch.nn.Linear(16, 16, bias=False)
    high_rank_module = LoKrModule(
        "high_rank",
        high_rank_linear,
        lora_dim=10000,
        alpha=1,
        factor=4,
        decompose_both=True,
    )
    assert _lokr_keys(high_rank_module) == ["lokr_rank", "lokr_w1", "lokr_w2"]
    assert high_rank_module.scale == 1.0


def test_lokr_native_decomposed_roundtrip_from_weights():
    unet = DummyAnima()
    network = LoRANetwork(
        None,
        unet,
        lora_dim=1,
        alpha=1,
        module_class=LoKrModule,
        use_lokr=True,
        adapter_type="lokr",
        lokr_factor=4,
        lokr_decompose_both=True,
    )
    network.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    weights_sd = {key: value.detach().clone() for key, value in network.state_dict().items()}

    assert any(key.endswith(".lokr_w1_a") for key in weights_sd)
    assert any(key.endswith(".lokr_w2_b") for key in weights_sd)

    loaded_unet = DummyAnima()
    loaded_network, loaded_weights = create_network_from_weights(
        1.0,
        "",
        None,
        None,
        loaded_unet,
        weights_sd=weights_sd,
    )
    loaded_network.apply_to(None, loaded_unet, apply_text_encoder=False, apply_unet=True)
    info = loaded_network.load_state_dict(loaded_weights, False)

    assert info.missing_keys == []
    assert info.unexpected_keys == []
    assert set(loaded_network.state_dict().keys()) == set(weights_sd.keys())


def test_lokr_native_direct_roundtrip_from_weights():
    unet = DummyAnima()
    network = LoRANetwork(
        None,
        unet,
        lora_dim=4,
        alpha=4,
        module_class=LoKrModule,
        use_lokr=True,
        adapter_type="lokr",
        lokr_factor=4,
    )
    network.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    weights_sd = {key: value.detach().clone() for key, value in network.state_dict().items()}

    assert any(key.endswith(".lokr_w1") for key in weights_sd)
    assert any(key.endswith(".lokr_w2") for key in weights_sd)

    loaded_unet = DummyAnima()
    loaded_network, loaded_weights = create_network_from_weights(
        1.0,
        "",
        None,
        None,
        loaded_unet,
        weights_sd=weights_sd,
    )
    loaded_network.apply_to(None, loaded_unet, apply_text_encoder=False, apply_unet=True)
    info = loaded_network.load_state_dict(loaded_weights, False)

    assert info.missing_keys == []
    assert info.unexpected_keys == []
    assert set(loaded_network.state_dict().keys()) == set(weights_sd.keys())


def test_lokr_create_network_accepts_native_shape_options():
    unet = DummyAnima()
    network = create_network(
        1.0,
        1,
        1,
        None,
        None,
        unet,
        anima_adapter_type="lokr",
        lokr_factor=4,
        decompose_both=True,
        full_matrix=False,
        unbalanced_factorization=True,
    )

    assert network.adapter_type == "lokr"
    assert network.lokr_factor == 4
    assert network.lokr_decompose_both is True
    assert network.lokr_full_matrix is False
    assert network.lokr_unbalanced_factorization is True


def test_lokr_create_network_uses_full_matrix_dim_sentinel():
    unet = DummyAnima()
    network = create_network(
        1.0,
        100_000,
        1,
        None,
        None,
        unet,
        anima_adapter_type="lokr",
        lokr_factor=4,
    )
    network.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)

    assert network.adapter_type == "lokr"
    assert network.lokr_full_matrix is True
    assert _lokr_keys(network.unet_loras[0]) == ["lokr_rank", "lokr_w1", "lokr_w2"]
    assert network.unet_loras[0].scale == 1.0


def test_lokr_native_save_keeps_lokr_keys_without_lora_compatible_weights():
    unet = DummyAnima()
    network = LoRANetwork(
        None,
        unet,
        lora_dim=1,
        alpha=1,
        module_class=LoKrModule,
        use_lokr=True,
        adapter_type="lokr",
        lokr_factor=4,
        lokr_decompose_both=True,
    )
    network.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)

    prepared_sd, metadata = network._prepare_lokr_export_for_save(
        network.state_dict(),
        {"ss_network_args": '["anima_adapter_type=lokr","lokr_export_mode=native","decompose_both=True"]'},
    )

    assert metadata["ss_lokr_export_mode"] == "native"
    assert metadata["ss_lokr_native_export"] == "true"
    assert metadata["ss_lokr_rank_exported"] == "false"
    assert metadata["ss_lokr_scale_export_format"] == "comfyui_baked_single_scale"
    assert not any(".lora_down.weight" in key or ".lora_up.weight" in key for key in prepared_sd)
    assert not any(key.endswith(".lokr_rank") for key in prepared_sd)
    assert any(key.endswith(".lokr_w1_a") for key in prepared_sd)
    assert any(key.endswith(".lokr_w2_b") for key in prepared_sd)


def test_lokr_native_export_bakes_scale_for_comfyui_loader():
    cases = [
        dict(lora_dim=4, alpha=1, lokr_factor=4, lokr_decompose_both=False),
        dict(lora_dim=2, alpha=1, lokr_factor=2, lokr_decompose_both=True),
        dict(lora_dim=1, alpha=1, lokr_factor=4, lokr_decompose_both=True),
        dict(lora_dim=1_000_000, alpha=1, lokr_factor=4, lokr_full_matrix=True),
    ]

    for index, kwargs in enumerate(cases):
        torch.manual_seed(index + 10)
        unet = DummyAnima()
        network = LoRANetwork(
            None,
            unet,
            module_class=LoKrModule,
            use_lokr=True,
            adapter_type="lokr",
            **kwargs,
        )
        network.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
        lora = network.unet_loras[0]
        for parameter in lora.parameters():
            if parameter.requires_grad:
                parameter.data.normal_(mean=0.0, std=0.1)

        prepared_sd, metadata = network._prepare_lokr_export_for_save(network.state_dict(), {})
        prefix = lora.lora_name

        assert metadata["ss_lokr_rank_exported"] == "false"
        assert f"{prefix}.lokr_rank" not in prepared_sd

        actual = _comfyui_lokr_weight(prepared_sd, prefix, (lora.out_features, lora.in_features))
        expected = lora._compute_weight(device=torch.device("cpu"), dtype=torch.float32) * float(lora.scale)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_lokr_native_from_weights_uses_alpha_when_rank_key_is_absent():
    unet = DummyAnima()
    network = LoRANetwork(
        None,
        unet,
        lora_dim=8,
        alpha=1,
        module_class=LoKrModule,
        use_lokr=True,
        adapter_type="lokr",
        lokr_factor=4,
    )
    network.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    prepared_sd, _metadata = network._prepare_lokr_export_for_save(network.state_dict(), {})
    alpha_items = [(key, value) for key, value in prepared_sd.items() if key.endswith(".alpha")]
    delayed_alpha_sd = {key: value for key, value in prepared_sd.items() if not key.endswith(".alpha")}
    delayed_alpha_sd.update(alpha_items)

    loaded_unet = DummyAnima()
    loaded_network, _loaded_weights = create_network_from_weights(
        1.0,
        "",
        None,
        None,
        loaded_unet,
        weights_sd=delayed_alpha_sd,
    )

    assert loaded_network.unet_loras[0].lora_dim == 8


def test_lokr_train_norm_export_writes_comfyui_diff():
    unet = DummyAnimaWithNorm()
    network = LoRANetwork(
        None,
        unet,
        lora_dim=4,
        alpha=4,
        module_class=LoKrModule,
        use_lokr=True,
        adapter_type="lokr",
        lokr_factor=4,
        train_norm=True,
    )
    network.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    norm_ref = network.unet_norms[0]
    params = dict(norm_ref.named_parameters())
    base_weight = norm_ref.base_params["weight"]
    params["weight"].data.copy_(base_weight.to(params["weight"]) + 0.25)

    prepared_sd, metadata = network._prepare_train_norm_comfyui_export_for_save(network.state_dict(), {})
    old_weight_key = f"{norm_ref.lora_name}.weight"
    new_weight_key = f"{norm_ref.lora_name.replace('train_norm_unet', 'lora_unet', 1)}.diff"
    new_bias_key = f"{norm_ref.lora_name.replace('train_norm_unet', 'lora_unet', 1)}.diff_b"

    assert metadata["ss_train_norm_export_format"] == "comfyui_diff"
    assert metadata["ss_train_norm_exported_count"] == "2"
    assert old_weight_key not in prepared_sd
    assert new_weight_key in prepared_sd
    assert new_bias_key in prepared_sd
    torch.testing.assert_close(prepared_sd[new_weight_key], torch.full_like(prepared_sd[new_weight_key], 0.25))


def test_lokr_train_norm_loads_comfyui_diff_and_old_weight_format():
    source_unet = DummyAnimaWithNorm()
    source_network = LoRANetwork(
        None,
        source_unet,
        lora_dim=4,
        alpha=4,
        module_class=LoKrModule,
        use_lokr=True,
        adapter_type="lokr",
        lokr_factor=4,
        train_norm=True,
    )
    source_network.apply_to(None, source_unet, apply_text_encoder=False, apply_unet=True)
    source_norm_ref = source_network.unet_norms[0]
    source_params = dict(source_norm_ref.named_parameters())
    source_params["weight"].data.add_(0.25)
    source_params["bias"].data.add_(0.5)

    prepared_sd, metadata = source_network._prepare_lokr_export_for_save(source_network.state_dict(), {})
    prepared_sd, metadata = source_network._prepare_train_norm_comfyui_export_for_save(prepared_sd, metadata)

    loaded_unet = DummyAnimaWithNorm()
    loaded_network, loaded_weights = create_network_from_weights(
        1.0,
        "",
        None,
        None,
        loaded_unet,
        weights_sd=prepared_sd,
    )
    loaded_network.apply_to(None, loaded_unet, apply_text_encoder=False, apply_unet=True)
    loaded_network.load_state_dict(loaded_weights, False)
    loaded_params = dict(loaded_network.unet_norms[0].named_parameters())

    torch.testing.assert_close(loaded_params["weight"], source_params["weight"])
    torch.testing.assert_close(loaded_params["bias"], source_params["bias"])

    old_format_weights = {
        f"{source_norm_ref.lora_name}.weight": source_params["weight"].detach().clone(),
        f"{source_norm_ref.lora_name}.bias": source_params["bias"].detach().clone(),
    }
    old_unet = DummyAnimaWithNorm()
    old_network, old_weights = create_network_from_weights(
        1.0,
        "",
        None,
        None,
        old_unet,
        weights_sd=old_format_weights,
    )
    old_network.apply_to(None, old_unet, apply_text_encoder=False, apply_unet=True)
    old_network.load_state_dict(old_weights, False)
    old_params = dict(old_network.unet_norms[0].named_parameters())

    torch.testing.assert_close(old_params["weight"], source_params["weight"])
    torch.testing.assert_close(old_params["bias"], source_params["bias"])


def test_lokr_inf_merge_accepts_native_decomposed_weights():
    torch.manual_seed(1)
    unet = DummyAnima()
    network = LoRANetwork(
        None,
        unet,
        lora_dim=1,
        alpha=1,
        module_class=LoKrModule,
        use_lokr=True,
        adapter_type="lokr",
        lokr_factor=4,
        lokr_decompose_both=True,
    )
    network.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    for parameter in network.parameters():
        if parameter.requires_grad:
            parameter.data.normal_(mean=0.0, std=0.1)
    weights_sd = {key: value.detach().clone() for key, value in network.state_dict().items()}

    merge_unet = DummyAnima()
    original_weight = merge_unet.block.linear.weight.detach().clone()
    merge_network, merge_weights = create_network_from_weights(
        1.0,
        "",
        None,
        None,
        merge_unet,
        weights_sd=weights_sd,
        for_inference=True,
    )

    assert all(isinstance(lora, LoKrInfModule) for lora in merge_network.unet_loras)
    merge_network.merge_to(None, merge_unet, merge_weights, dtype=torch.float32, device="cpu")

    assert torch.isfinite(merge_unet.block.linear.weight).all()
    assert not torch.allclose(original_weight, merge_unet.block.linear.weight)
