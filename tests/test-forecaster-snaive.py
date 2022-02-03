import numpy as np
from metats.forecasters.statistical import SeasonalNaiveForecaster

def test_forecaster():
    data = np.random.rand(10, 55, 5)
    forecaster = SeasonalNaiveForecaster()
    predictions = forecaster.predict(data, horizon=7)
    assert predictions.shape == (10, 7, 5)
    assert np.all(predictions[:, :, :] == data[:, -7:, :])