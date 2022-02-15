import pytest
import numpy as np
from sktime.forecasting.ets import AutoETS
from metats.forecasters.utils import is_sktime_forecaster, generate_sktime_prediction

def test_sktime_wrapper():
    y = 10 + np.random.normal(size=(2, 55, 5))
    forecaster = AutoETS(auto=True, sp=4, n_jobs=-1)
    assert is_sktime_forecaster(forecaster)
    predictions = generate_sktime_prediction(forecaster, y, fh=4, forecast_dim=0)
    assert predictions.shape == (2, 4)