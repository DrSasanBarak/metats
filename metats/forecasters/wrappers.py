import numpy as np
import pandas as pd
from metats.forecasters import BaseForecaster
from metats.forecasters.utils import is_sktime_forecaster, is_darts_forecaster, is_nixtla_stats_forecaster



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

class NixtlaStatsForecasterWrapper(BaseForecaster):
    """
    A Wrapper class for Nixtla statistical forecasters
    """
    def __init__(self, nixtla_model):
        """
        inputs:
            nixtla_model: a darts models which must be checked using is_nixtla_stats_forecaster
        """
        if not is_nixtla_stats_forecaster(nixtla_model):
            raise ValueError('The input is not a valid nixtla forecaster')

        self.nixtla_model = nixtla_model
    
    def predict(self, Y, fh, forecast_dim):
        """
        inputs:
            Y: the timeseries (numpy array) (num_series x series_length x covariates_dim)
            fh: forecast horizon
            forecast_dim: the dimension of the series to be generated
        """

        # prepare data for nixtla
        data_df = pd.DataFrame(Y[:, :, forecast_dim])
        data_df['unique_id'] = data_df.index
        nixtla_df = data_df.melt(var_name='ds', value_name='y', id_vars=['unique_id'])
        # fit the nixtla frecaster
        self.nixtla_model.fit(nixtla_df)
        # generting and collecting the predictions
        pred_df = self.nixtla_model.predict(h=fh)
        predictions = []
        for i in range(Y.shape[0]):
            single_pred = pred_df[pred_df.index==i][pred_df.columns[-1]].values
            predictions.append(single_pred.reshape((1, -1)))
        predictions = np.concatenate(predictions, axis=0)
        return predictions