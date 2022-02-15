import sktime
import numpy as np
from sktime.base import BaseEstimator

def is_sktime_forecaster(forecaster):
    """
    Check if a given object is a sktime forecaster
    """
    return isinstance(forecaster, BaseEstimator)

def generate_sktime_prediction(forecaster, Y, fh, forecast_dim=0):
    """
    Generate a sktime prediction for each series in Y
    inputs:
        forecaster: a sktime forecaster
        Y: the timeseries (numpy array) (num_series x series_length x covariates_dim)
        fh: forecast horizon
        forecast_dim: the dimension of the forecast to be generated
    """
    if is_sktime_forecaster(forecaster):
        if isinstance(fh, int):
            fh = 1 + np.arange(fh)

        predictions = np.zeros((Y.shape[0], len(fh)))
        # forecast for each series
        for i in range(Y.shape[0]):
            forecaster.fit(y=Y[i, :, forecast_dim], fh=fh)
            predictions[i, :] = np.squeeze(forecaster.predict())
        
        return predictions
    else:
        raise ValueError('Not a sktime forecaster')
    