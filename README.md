# MetaTS | Meta-Learning for Global Time Series Forecasting
![example workflow](https://github.com/amirabbasasadi/metats/actions/workflows/main.yml/badge.svg)
[![PyPI version fury.io](https://badge.fury.io/py/metats.svg)](https://pypi.python.org/pypi/metats/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/amirabbasasadi/metats.svg)](https://github.com/amirabbasasadi/metats/blob/master/LICENSE)
![image](https://user-images.githubusercontent.com/8543469/176514410-bf8efea2-fb54-4903-a0ee-169c9595958a.png)

## Features:
- Generating meta features
    - Statistical features : TsFresh, User defined features
    - Automated feature extraction using Deep Unsupervised Learning : Deep AutoEncoder (MLP, LSTM, GRU, ot custom model)
- Supporting sktime and darts libraries for base-forecasters
- Providing a Meta-Learning pipeline

## Quick Start

### Installing the package
```
pip install metats
```

### Generating a toy dataset by sampling from two different processes
```python
from metats.datasets import ETSDataset

ets_generator = ETSDataset({'A,N,N': 512,
                            'M,M,M': 512}, length=30, freq=4)

data, labels = ets_generator.load(return_family=True)
colors = list(map(lambda x: (x=='A,N,N')*1, labels))
```

### Normalizing the time series
```python
from sklearn.preprocessing import StandardScaler

scaled_data = StandardScaler().fit_transform(data.T)
data = scaled_data.T[:, :, None]
```
### Checking How data looks like
```python
import matplotlib.pyplot as plt
_ = plt.plot(data[10, :, 0])
```
![image](https://user-images.githubusercontent.com/8543469/176520933-64be6613-c64b-4a6c-baa7-d1c0ca13a7b2.png)

### Generating the meta-features
#### Statistical features using TsFresh
```python
from metats.features.statistical import TsFresh

stat_features = TsFresh().transform(data)
```
#### Deep Unsupervised Features
##### Training an AutoEncoder
```python
from metats.features.unsupervised import DeepAutoEncoder
from metats.features.deep import AutoEncoder, MLPEncoder, MLPDecoder

enc = MLPEncoder(input_size=1, input_length=30, latent_size=8, hidden_layers=(16,))
dec = MLPDecoder(input_size=1, input_length=30, latent_size=8, hidden_layers=(16,))

ae = AutoEncoder(encoder=enc, decoder=dec)
ae_feature = DeepAutoEncoder(auto_encoder=ae, epochs=150, verbose=True)

ae_feature.fit(data)
```
##### Generating features using the auto-encoder
```python
deep_features = ae_feature.transform(data)
```

#### Visualizing both statistical and deep meta-features
Dimensionality reduction using UMAP for visualization
```python
from umap import UMAP
deep_reduced = UMAP().fit_transform(deep_features)
stat_reduced = UMAP().fit_transform(stat_features)
```
Visualizing the statistical features:
```python
plt.scatter(stat_reduced[:512, 0], stat_reduced[:512, 1], c='#e74c3c', label='ANN')
plt.scatter(stat_reduced[512:, 0], stat_reduced[512:, 1], c='#9b59b6', label='MMM')
plt.legend()
plt.title('TsFresh Meta-Features')
_ = plt.show()
```
And similarly the auto encoder's features
```python
plt.scatter(deep_reduced[:512, 0], deep_reduced[:512, 1], c='#e74c3c', label='ANN')
plt.scatter(deep_reduced[512:, 0], deep_reduced[512:, 1], c='#9b59b6', label='MMM')
plt.legend()
plt.title('Deep Unsupervised Meta-Features')
_ = plt.show()
```
![image](https://user-images.githubusercontent.com/8543469/176526565-e26cbd0c-2b20-4848-995e-e12632bde8e3.png)
![image](https://user-images.githubusercontent.com/8543469/176526711-989e1ac3-2af8-4d27-a90d-ea6007594f36.png)



## Meta-Learning Pipeline
Creating a meta-learning pipeline with selection strategy:
```python
from metats.pipeline import MetaLearning

pipeline = MetaLearning(method='selection', loss='mse')
```
Adding AutoEncoder features:
```python
from metats.features.unsupervised import DeepAutoEncoder
from metats.features.deep import AutoEncoder, MLPEncoder, MLPDecoder

enc = MLPEncoder(input_size=1, input_length=23, latent_size=8, hidden_layers=(16,))
dec = MLPDecoder(input_size=1, input_length=23, latent_size=8, hidden_layers=(16,))

ae = AutoEncoder(encoder=enc, decoder=dec)
ae_features = DeepAutoEncoder(auto_encoder=ae, epochs=200, verbose=True)

pipeline.add_feature(ae_features)
```
You can add as many features as you like:
```python
from metats.features.statistical import TsFresh

stat_features = TsFresh()
pipeline.add_feature(stat_features)
```
Adding two sktime forecaster as base-forecasters
```python
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import make_reduction
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors=1)
forecaster1 = make_reduction(regressor, window_length=15, strategy="recursive")

forecaster2 = NaiveForecaster() 

pipeline.add_forecaster(forecaster1)
pipeline.add_forecaster(forecaster2)
```
Specify some meta-learner
```python
from sklearn.ensemble import RandomForestClassifier

pipeline.add_metalearner(RandomForestClassifier())
```

Training the pipeline
```python
pipeline.fit(data, fh=7)
```
Prediction for another set of data
```python
pipeline.predict(data, fh=7)
```

## About the package
### Contributors
- Sasan Barak
- Amirabbas Asadi


We wish to see your name in the list of contributors, So we are waiting for pull requests!

## Papers using MetaTS
Below is a list of projects using MetaTS. If you have used MetaTS in a project, you're welcome to make a PR to add it to this list.
* [Retail time series forecasting using an automated deep meta-learning framework](https://github.com/mohammadjoshaghani/retail_metalearning)
