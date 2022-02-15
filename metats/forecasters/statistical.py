import numpy as np
from metats.forecasters import BaseForecaster

# Naive Forecaster
class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal Naive Forecaster
    """
    def __init__(self):
        self.name = 'Seasonal Naive Forecaster'
    
    def predict(self, y, fh):
        """
        forecasting using the naive method
        inputs:
            y: the time series to be forecasted numpy array (num_series x series_length x covariates_dim)
            fh: the number of steps to forecast
        """
        if y.ndim < 3:
            z = y[:, :, np.newaxis]
        predictions = z[:, -fh:, :]
        return predictions