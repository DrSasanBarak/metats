import numpy as np
import pandas as pd
from metats.forecasters import BaseForecaster
from metats.forecasters.utils import is_sktime_forecaster, is_darts_forecaster



class DartsForecasterWrapper(BaseForecaster):
    """
    A Wrapper class for darts models
    """
    def __init__(self, darts_model):
        """
        inputs:
            dart_model: a darts models which must be checked using is_darts_forecaster
        """
        if not is_darts_forecaster(darts_model):
            raise ValueError('The input is not a valid darts forecaster')

        self.darts_model = darts_model
    
    def predict(self, Y, fh, forecast_dim):
        """
        inputs:
            Y: the timeseries (numpy array) (num_series x series_length x covariates_dim)
            fh: forecast horizon
            forecast_dim: the dimension of the forecast to be generated
        """
        from darts import TimeSeries as DTS
        
        predictions = []
        # forecast for each series
        for i in range(Y.shape[0]):
            frame = DTS.from_values(Y[i, :, forecast_dim:forecast_dim+1])
            self.darts_model.fit(frame)
            single_pred = np.squeeze(self.darts_model.predict(fh))
            predictions.append(single_pred.reshape((1, -1)))
        predictions = np.concatenate(predictions, axis=0)
        return predictions

class SKTimeForecasterWrapper(BaseForecaster):
    """
    A Wrapper class for sktime models
    """
    def __init__(self, sktime_model):
        """
        inputs:
            dart_model: a sktime models which must be checked using is_sktime_forecaster
        """
        if not is_sktime_forecaster(sktime_model):
            raise ValueError('The input is not a valid sktime forecaster')
        self.sktime_model = sktime_model
    
    def predict(self, Y, fh, forecast_dim):
        """
        inputs:
            Y: the timeseries (numpy array) (num_series x series_length x covariates_dim)
            fh: forecast horizon
            forecast_dim: the dimension of the forecast to be generated
        """
        if isinstance(fh, int):
            fh = 1 + np.arange(fh)
        
        predictions = np.zeros((Y.shape[0], len(fh)))
        # forecast for each series
        for i in range(Y.shape[0]):
            self.sktime_model.fit(y=Y[i, :, forecast_dim], fh=fh)
            predictions[i, :] = np.squeeze(self.sktime_model.predict())
        
        return predictions