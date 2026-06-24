import pytest
import torch

from library.train_loss_util import get_timesteps


def test_get_timesteps_uniform_range():
    timesteps = get_timesteps(100, 200, 512, torch.device("cpu"), "uniform", 1.0, 1.0)
    assert timesteps.shape == (512,)
    assert torch.all(timesteps >= 100)
    assert torch.all(timesteps < 200)


def test_get_timesteps_sigmoid_range():
    timesteps = get_timesteps(0, 1000, 512, torch.device("cpu"), "sigmoid", 1.0, 1.0)
    assert torch.all(timesteps >= 0)
    assert torch.all(timesteps < 1000)


def test_get_timesteps_shift_biases_lower_with_small_shift():
    torch.manual_seed(1234)
    low_bias = get_timesteps(0, 1000, 20000, torch.device("cpu"), "shift", 1.0, 0.5).float().mean().item()
    torch.manual_seed(1234)
    neutral = get_timesteps(0, 1000, 20000, torch.device("cpu"), "shift", 1.0, 1.0).float().mean().item()
    assert low_bias < neutral


def test_get_timesteps_shift_biases_higher_with_large_shift():
    torch.manual_seed(1234)
    high_bias = get_timesteps(0, 1000, 20000, torch.device("cpu"), "shift", 1.0, 2.0).float().mean().item()
    torch.manual_seed(1234)
    neutral = get_timesteps(0, 1000, 20000, torch.device("cpu"), "shift", 1.0, 1.0).float().mean().item()
    assert high_bias > neutral


def test_get_timesteps_invalid_shift_raises():
    with pytest.raises(ValueError):
        get_timesteps(0, 1000, 8, torch.device("cpu"), "shift", 1.0, 0.0)


def test_get_timesteps_invalid_sampling_raises():
    with pytest.raises(ValueError):
        get_timesteps(0, 1000, 8, torch.device("cpu"), "unknown", 1.0, 1.0)
