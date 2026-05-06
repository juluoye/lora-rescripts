from unittest.mock import patch
import logging
from library.train_util import get_optimizer
from library.optimizer_util import parse_optimizer_kwargs
from library.full_bf16_stochastic_util import FullBf16StochasticOptimizer
from library.lulynx_optimizer_compat import Compass, CompassPlus, FARMScrop, FCompass, FishMonger
from train_network import setup_parser
import torch
from torch.nn import Parameter

# Optimizer libraries
import bitsandbytes as bnb
from lion_pytorch import lion_pytorch
import schedulefree

import dadaptation
import dadaptation.experimental as dadapt_experimental

import prodigyopt
import schedulefree as sf
import transformers

from library.adamw_8bit_kahan import AdamW8bitKahan
from library.optimizer_offload_util import should_offload_optimizer_tensor

def test_default_get_optimizer():
    with patch("sys.argv", [""]):
        parser = setup_parser()
        args = parser.parse_args()
        params_t = torch.tensor([1.5, 1.5])

        param = Parameter(params_t)
        optimizer_name, optimizer_args, optimizer = get_optimizer(args, [param])
        assert optimizer_name == "torch.optim.adamw.AdamW"
        assert optimizer_args == ""
        assert isinstance(optimizer, torch.optim.AdamW)


def test_get_schedulefree_optimizer():
    with patch("sys.argv", ["", "--optimizer_type", "AdamWScheduleFree"]):
        parser = setup_parser()
        args = parser.parse_args()
        params_t = torch.tensor([1.5, 1.5])

        param = Parameter(params_t)
        optimizer_name, optimizer_args, optimizer = get_optimizer(args, [param])
        assert optimizer_name == "schedulefree.adamw_schedulefree.AdamWScheduleFree"
        assert optimizer_args == ""
        assert isinstance(optimizer, schedulefree.adamw_schedulefree.AdamWScheduleFree)


def test_all_supported_optimizers():
    optimizers = [
        {
            "name": "bitsandbytes.optim.adamw.AdamW8bit",
            "alias": "AdamW8bit",
            "instance": bnb.optim.AdamW8bit,
        },
        {
            "name": "library.adamw_8bit_kahan.AdamW8bitKahan",
            "alias": "AdamW8bitKahan",
            "instance": AdamW8bitKahan,
        },
        {
            "name": "lion_pytorch.lion_pytorch.Lion",
            "alias": "Lion",
            "instance": lion_pytorch.Lion,
        },
        {
            "name": "torch.optim.adamw.AdamW",
            "alias": "AdamW",
            "instance": torch.optim.AdamW,
        },
        {
            "name": "bitsandbytes.optim.lion.Lion8bit",
            "alias": "Lion8bit",
            "instance": bnb.optim.Lion8bit,
        },
        {
            "name": "bitsandbytes.optim.adamw.PagedAdamW8bit",
            "alias": "PagedAdamW8bit",
            "instance": bnb.optim.PagedAdamW8bit,
        },
        {
            "name": "bitsandbytes.optim.lion.PagedLion8bit",
            "alias": "PagedLion8bit",
            "instance": bnb.optim.PagedLion8bit,
        },
        {
            "name": "bitsandbytes.optim.adamw.PagedAdamW",
            "alias": "PagedAdamW",
            "instance": bnb.optim.PagedAdamW,
        },
        {
            "name": "bitsandbytes.optim.adamw.PagedAdamW32bit",
            "alias": "PagedAdamW32bit",
            "instance": bnb.optim.PagedAdamW32bit,
        },
        {"name": "torch.optim.sgd.SGD", "alias": "SGD", "instance": torch.optim.SGD},
        {
            "name": "dadaptation.experimental.dadapt_adam_preprint.DAdaptAdamPreprint",
            "alias": "DAdaptAdamPreprint",
            "instance": dadapt_experimental.DAdaptAdamPreprint,
        },
        {
            "name": "dadaptation.dadapt_adagrad.DAdaptAdaGrad",
            "alias": "DAdaptAdaGrad",
            "instance": dadaptation.DAdaptAdaGrad,
        },
        {
            "name": "dadaptation.dadapt_adan.DAdaptAdan",
            "alias": "DAdaptAdan",
            "instance": dadaptation.DAdaptAdan,
        },
        {
            "name": "dadaptation.experimental.dadapt_adan_ip.DAdaptAdanIP",
            "alias": "DAdaptAdanIP",
            "instance": dadapt_experimental.DAdaptAdanIP,
        },
        {
            "name": "dadaptation.dadapt_lion.DAdaptLion",
            "alias": "DAdaptLion",
            "instance": dadaptation.DAdaptLion,
        },
        {
            "name": "dadaptation.dadapt_sgd.DAdaptSGD",
            "alias": "DAdaptSGD",
            "instance": dadaptation.DAdaptSGD,
        },
        {
            "name": "prodigyopt.prodigy.Prodigy",
            "alias": "Prodigy",
            "instance": prodigyopt.Prodigy,
        },
        {
            "name": "transformers.optimization.Adafactor",
            "alias": "Adafactor",
            "instance": transformers.optimization.Adafactor,
        },
        {
            "name": "schedulefree.adamw_schedulefree.AdamWScheduleFree",
            "alias": "AdamWScheduleFree",
            "instance": sf.AdamWScheduleFree,
        },
        {
            "name": "schedulefree.sgd_schedulefree.SGDScheduleFree",
            "alias": "SGDScheduleFree",
            "instance": sf.SGDScheduleFree,
        },
    ]

    for opt in optimizers:
        with patch("sys.argv", ["", "--optimizer_type", opt.get("alias")]):
            parser = setup_parser()
            args = parser.parse_args()
            params_t = torch.tensor([1.5, 1.5])

            param = Parameter(params_t)
            optimizer_name, _, optimizer = get_optimizer(args, [param])
            assert optimizer_name == opt.get("name")

            instance = opt.get("instance")
            assert instance is not None
            assert isinstance(optimizer, instance)


def test_parse_optimizer_kwargs_sanitizes_invalid_numeric_values():
    with patch(
        "sys.argv",
        [
            "",
            "--optimizer_type",
            "pytorch_optimizer.Compass",
            "--optimizer_args",
            "eps=0",
            "weight_decay=-0.1",
            "betas=(1.2, 0.999)",
        ],
    ):
        parser = setup_parser()
        args = parser.parse_args()
        kwargs = parse_optimizer_kwargs(args, logging.getLogger("test"))
        assert kwargs["eps"] > 0
        assert kwargs["weight_decay"] == 0.0
        assert "betas" not in kwargs
        assert kwargs["amp_fac"] == 2.0


def test_compat_optimizer_aliases_can_be_created():
    compat_optimizers = [
        ("pytorch_optimizer.Compass", Compass),
        ("pytorch_optimizer.FCompass", FCompass),
        ("pytorch_optimizer.FishMonger", FishMonger),
        ("pytorch_optimizer.FARMScrop", FARMScrop),
        ("pytorch_optimizer.CompassPlus", CompassPlus),
    ]

    for alias, instance_type in compat_optimizers:
        with patch("sys.argv", ["", "--optimizer_type", alias, "--max_train_steps", "32"]):
            parser = setup_parser()
            args = parser.parse_args()
            param = Parameter(torch.tensor([1.5, 1.5]))
            optimizer_name, _, optimizer = get_optimizer(args, [param])
            assert optimizer_name.endswith(instance_type.__name__)
            assert isinstance(optimizer, instance_type)


def test_full_bf16_optimizer_wraps_master_params():
    with patch("sys.argv", ["", "--full_bf16", "--mixed_precision", "bf16"]):
        parser = setup_parser()
        args = parser.parse_args()
        param = Parameter(torch.tensor([1.5, 1.5], dtype=torch.bfloat16))
        optimizer_name, _, optimizer = get_optimizer(args, [param])
        assert optimizer_name == "torch.optim.adamw.AdamW"
        assert isinstance(optimizer, FullBf16StochasticOptimizer)
        assert optimizer.param_groups[0]["params"][0].dtype == torch.float32


def test_adamw8bitkahan_accepts_kahan_offload_args():
    with patch(
        "sys.argv",
        [
            "",
            "--optimizer_type",
            "AdamW8bitKahan",
            "--optimizer_args",
            "kahan_buffer_offload=True",
            "optimizer_offload_mode='ndim_ge_2'",
        ],
    ):
        parser = setup_parser()
        args = parser.parse_args()
        param = Parameter(torch.tensor([[1.5, 1.5], [1.0, 1.0]]))
        optimizer_name, optimizer_args, optimizer = get_optimizer(args, [param])
        assert optimizer_name == "library.adamw_8bit_kahan.AdamW8bitKahan"
        assert "kahan_buffer_offload=True" in optimizer_args
        assert isinstance(optimizer, AdamW8bitKahan)
        assert optimizer.kahan_buffer_offload is True
        assert optimizer.optimizer_offload_mode == "ndim_ge_2"


def test_optimizer_offload_helper_uses_ndim_heuristic():
    vector = Parameter(torch.tensor([1.0, 2.0]))
    matrix = Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert should_offload_optimizer_tensor(vector, mode="ndim_ge_2") is False
    assert should_offload_optimizer_tensor(matrix, mode="ndim_ge_2") is True
    assert should_offload_optimizer_tensor(vector, mode="all") is True
