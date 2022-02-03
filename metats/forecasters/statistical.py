from metats.forecasters import BaseForecaster

# Naive Forecaster
class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal Naive Forecaster
    """
    def __init__(self):
        self.name = 'Seasonal Naive Forecaster'
    
    def predict(self, timeseries, horizon):
        """
        forecasting using the naive method
        inputs:
            timeseries: the time series to be forecasted numpy array (num_series x series_length x covariates_dim)
            horizon: the number of steps to forecast
        """
        predictions = timeseries[:, -horizon:, :]
        return predictions