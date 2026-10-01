import torch

from src.models.global_lstm import GlobalVolatilityLSTM


def _model() -> GlobalVolatilityLSTM:
    return GlobalVolatilityLSTM(
        input_size=6,
        num_tickers=3,
        embedding_dim=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )


def test_global_lstm_returns_one_log_volatility_per_sample():
    model = _model()
    features = torch.randn(5, 30, 6)
    ticker_ids = torch.tensor([0, 1, 2, 0, 1])

    output = model(features, ticker_ids)

    assert output.shape == (5,)
    assert torch.isfinite(output).all()


def test_public_volatility_prediction_is_strictly_positive():
    model = _model()

    forecast = model.predict_volatility(
        torch.randn(3, 30, 6),
        torch.tensor([0, 1, 2]),
    )

    assert (forecast > 0).all()


def test_ticker_embedding_can_change_forecast_for_same_history():
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.ticker_embedding.weight[1].fill_(1.0)
        first_linear = model.head[1]
        first_linear.weight[0, 8:] = 1.0
        final_linear = model.head[3]
        final_linear.weight[0, 0] = 1.0

    features = torch.zeros(2, 30, 6)
    output = model(features, torch.tensor([0, 1]))

    assert output[0].item() == 0.0
    assert output[1].item() > output[0].item()
