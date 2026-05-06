import torch

from library.train_loss_util import apply_wavelet_loss, compute_wavelet_loss


def test_compute_wavelet_loss_returns_zero_for_identical_tensors():
    tensor = torch.randn(2, 4, 8, 8)
    loss = compute_wavelet_loss(tensor, tensor.clone(), levels=2, reduction="none")
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_compute_wavelet_loss_detects_detail_difference():
    pred = torch.zeros(1, 1, 8, 8)
    target = pred.clone()
    target[:, :, 2, 2] = 1.0
    loss = compute_wavelet_loss(pred, target, levels=1, reduction="mean")
    assert float(loss.item()) > 0.0


def test_compute_wavelet_loss_can_include_low_frequency_difference():
    pred = torch.zeros(1, 1, 8, 8)
    target = pred.clone()
    target[:, :, 2:4, 2:4] = 1.0
    loss = compute_wavelet_loss(pred, target, levels=1, approx_weight=1.0, reduction="mean")
    assert float(loss.item()) > 0.0


def test_apply_wavelet_loss_respects_disabled_flag():
    base_loss = torch.ones(1, 1, 8, 8)
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)
    result = apply_wavelet_loss(
        base_loss,
        pred,
        target,
        enabled=False,
        weight=0.5,
        levels=1,
        approx_weight=0.0,
    )
    assert torch.equal(result, base_loss)
