
class FeatureGenerator():
    """
    Base class for feature generators
    """
    def fit(self, timeseries):
        """
        Fit the feature generator
        inputs:
            timeseries: the timeseries (numpy array) (num_series x series_length)
        """
        raise NotImplementedError('FeatureGenerator.fit() not implemented')
    
    def transform(self, timeseries):
        """
        Generate features
        inputs:
            timeseries: the timeseries (numpy array) (num_series x series_length)
        """
        raise NotImplementedError('FeatureGenerator.transform() not implemented')