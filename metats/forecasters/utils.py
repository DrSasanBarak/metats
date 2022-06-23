import sktime
import numpy as np
from sktime.base import BaseEstimator
from darts.models.forecasting.forecasting_model import ForecastingModel

def is_sktime_forecaster(forecaster):
    """
    Check if a given object is a sktime forecaster
    """
    return isinstance(forecaster, BaseEstimator)


def is_darts_forecaster(forecaster):
    """
    Check if a given object is a darts forecaster
    """
    return isinstance(forecaster, ForecastingModel)

    