import numpy as np
import pandas as pd
import tsfresh
from metats.features import FeatureGenerator


class TsFresh(FeatureGenerator):
    """
    Generate features using TsFresh
    """
    def fit(self, Y):
        pass
    
    def reshape_for_tsfresh(self, Y):
        """
        Reshape the timeseries for tsfresh
        inputs:
            Y: the timeseries (numpy array) (num_series x series_length x covariates_dim)
        """
        reshaped = []
        for id in range(Y.shape[0]):
            for time in range(Y.shape[1]):
                row = np.array([id, time])
                row = np.hstack([row, Y[id, time, :]])
                reshaped.append(row)
        reshaped = np.vstack(reshaped)
        return reshaped

    def transform(self, Y):
        """
        Extract features using tsfresh
        inputs:
            Y: the timeseries (numpy array) (num_series x series_length x covariates_dim)
        """
        reshaped = self.reshape_for_tsfresh(Y)
        covariates_dim = Y.shape[2]
        column_names = ['id', 'time'] + ['v_'+ str(i) for i in range(covariates_dim)]
        tsf = pd.DataFrame(reshaped, columns=column_names)
        del reshaped
        features = tsfresh.extract_features(tsf, column_id='id', column_sort='time')
        # remove the Nan values
        features.dropna(axis=1, inplace=True)
        features = features.values[:, 2:]
        return features

