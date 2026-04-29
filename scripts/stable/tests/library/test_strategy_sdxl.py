import torch

from library.strategy_sdxl import SdxlTextEncodingStrategy


def test_encode_tokens_with_weights_preserves_hidden_state_mean():
    strategy = SdxlTextEncodingStrategy()

    hidden_states1 = torch.ones((1, 77, 4), dtype=torch.float32)
    hidden_states2 = torch.full((1, 77, 6), 2.0, dtype=torch.float32)
    pool2 = torch.zeros((1, 6), dtype=torch.float32)

    strategy.encode_tokens = lambda tokenize_strategy, models, tokens_list: [  # type: ignore[method-assign]
        hidden_states1.clone(),
        hidden_states2.clone(),
        pool2.clone(),
    ]

    weights1 = torch.ones((1, 1, 77), dtype=torch.float32)
    weights2 = torch.ones((1, 1, 77), dtype=torch.float32)
    weights1[:, :, 10:20] = 1.5
    weights2[:, :, 30:40] = 0.5

    out_hidden_states1, out_hidden_states2, out_pool2 = strategy.encode_tokens_with_weights(
        tokenize_strategy=None,
        models=[],
        tokens_list=[],
        weights_list=[weights1, weights2],
    )

    assert torch.allclose(out_hidden_states1.float().mean(), hidden_states1.float().mean(), atol=1e-5)
    assert torch.allclose(out_hidden_states2.float().mean(), hidden_states2.float().mean(), atol=1e-5)
    assert torch.equal(out_pool2, pool2)
