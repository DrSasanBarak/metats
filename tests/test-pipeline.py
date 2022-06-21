import pytest
import numpy as np
import metats
from sktime.forecasting.naive import NaiveForecaster
# from metats.features.statistical import TsFresh
from metats.pipeline import MetaLearning
from metats.features import FeatureGenerator
from sklearn.ensemble import RandomForestClassifier
from sktime.forecasting.compose import make_reduction
from sklearn.neighbors import KNeighborsRegressor

class CustomFeature(FeatureGenerator):
    def fit(self, Y):
        pass
    
    def transform(self, Y):
        return np.random.normal(size=(Y.shape[0], 3))

def custom_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred), axis=2)


def test_basic_pipeline():
    data = np.random.normal(size=(2, 55, 5))
    regressor = KNeighborsRegressor(n_neighbors=1)
    forecaster1 = make_reduction(regressor, window_length=15, strategy="recursive")
    forecaster2 = NaiveForecaster()    


    feature1 = CustomFeature()

    pipeline = MetaLearning(method='selection', loss=custom_loss)
    pipeline.add_forcecaster(forecaster1)
    pipeline.add_forcecaster(forecaster2)
    assert len(pipeline.base_forecasters) == 2
    pipeline.add_feature(feature1)
    assert len(pipeline.feature_generators) == 1
    pipeline.add_metalearner(RandomForestClassifier())
    assert isinstance(pipeline.meta_learner, RandomForestClassifier)
    pipeline.fit(data, fh=7)
    weights, predictions = pipeline.predict(data, fh=7)
    print(weights.shape)
    print(weights)
    print(predictions.shape)
    print(predictions)


