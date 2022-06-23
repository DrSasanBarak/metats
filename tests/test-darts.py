import pytest
import numpy as np

from metats.forecasters.utils import is_darts_forecaster
from metats.forecasters.wrappers import DartsForecasterWrapper
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing as ETS
from darts.models import Theta, AutoARIMA, TBATS

def test_darts():
    ets_model = Theta()
    assert is_darts_forecaster(ets_model)
    wrapped_model = DartsForecasterWrapper(ets_model)
    data = np.random.uniform(size=(4, 55, 5))
    predictions = wrapped_model.predict(data, fh=7, forecast_dim=1)
    assert predictions.shape == (4, 7)