import sys
import os
sys.path.insert(0, os.getcwd())

def main():
    """ - First, we create synthetic dataset.
        - We add meta-learning pipeline.
        - Then, we extract meta-features of time series with three
            deep learning methods, i.e., EDMLP, EDLSTM, EDTCN.
        - We also extract statistical features of time series with TsFresh.
        - Then, We add 2 base_forecasters
        - Finally, we fit and predict time series.    
    """

    ##########
    # dataset
    ##########    
    
    LEN_TS = 39 # time series length
    FH = 7      # forecasting horizon
    
    # create synthetic data    
    from metats.datasets import ETSDataset
    ets_generator = ETSDataset({'A,N,N': 512,
                                'M,M,M': 512}, length=LEN_TS, freq=4)
    data, labels = ets_generator.load(return_family=True)
    colors = list(map(lambda x: (x=='A,N,N')*1, labels))
    
    # scaling data
    from sklearn.preprocessing import StandardScaler
    scaled_data = StandardScaler().fit_transform(data.T)
    data = scaled_data.T[:, :, None]    # size: batch_dim x seires_length x series_dim
    print('data shape is:',data.shape)
    
    # creats MetaLearning pipeline
    from metats.pipeline import MetaLearning
    pipeline = MetaLearning(method='selection', loss='mse')
    
    ##########
    # MLP
    ##########
    
    from metats.features.unsupervised import DeepAutoEncoder
    from metats.features.deep import AutoEncoder, MLPEncoder, MLPDecoder

    enc = MLPEncoder(input_size=data.shape[2], input_length=LEN_TS-FH, latent_size=8, hidden_layers=(16,))
    dec = MLPDecoder(input_size=data.shape[2], input_length=LEN_TS-FH, latent_size=8, hidden_layers=(16,))
    ae = AutoEncoder(encoder=enc, decoder=dec)
    ae_features = DeepAutoEncoder(auto_encoder=ae, epochs=100, verbose=False)
    
    pipeline.add_feature(ae_features)
    
    ##########
    # LSTM
    ##########
    
    from metats.features.unsupervised import DeepAutoEncoder
    from metats.features.deep import AutoEncoder, LSTMDecoder, LSTMEncoder

    H = 5   # hidden size
    l = 3   # latent size
    NL= 2   # number of layers

    enc = LSTMEncoder(input_size=data.shape[2], latent_size=l,
                    hidden_size=H, num_layers=NL, directions=1)
    dec = LSTMDecoder(output_length=LEN_TS-FH, output_size=data.shape[2],
                    latent_size=l, hidden_size=H, num_layers=NL, directions=1)
    ae = AutoEncoder(encoder=enc, decoder=dec)
    ae_features = DeepAutoEncoder(auto_encoder=ae, epochs=100, verbose=False)
    pipeline.add_feature(ae_features)

    ##########
    # TCN
    ##########

    from metats.features.unsupervised import DeepAutoEncoder
    from metats.features.deep import AutoEncoder, Encoder_Decoder_TCN
    
    # construct model
    EDTCN = Encoder_Decoder_TCN(input_size=data.shape[2], input_length=LEN_TS-FH,             
                                hidden_layers=(4,1), kernel_size =4, dilation=2)
    enc = EDTCN.encoder
    dec = EDTCN.decoder
    ae = AutoEncoder(encoder=enc, decoder=dec)
    ae_features = DeepAutoEncoder(auto_encoder=ae, epochs=100, verbose=True)
    
    pipeline.add_feature(ae_features)
    
    ##########
    # TsFresh
    ##########

    from metats.features.statistical import TsFresh  

    # adding TsFresh as statistical features extractor
    stat_features = TsFresh()
    pipeline.add_feature(stat_features)
    
    ##########
    # Forecasters
    ##########
    
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.compose import make_reduction
    from sklearn.neighbors import KNeighborsRegressor
    
    # creating two base-forecasters
    regressor = KNeighborsRegressor(n_neighbors=1)
    forecaster1 = make_reduction(regressor, window_length=15, strategy="recursive")
    forecaster2 = NaiveForecaster() 
    
    # adding base-forecasters to pipline
    pipeline.add_forecaster(forecaster1)
    pipeline.add_forecaster(forecaster2)
    
    ##########
    # MetaLearner
    ##########

    from sklearn.ensemble import RandomForestClassifier
    # to adopt random foreset as meta-learner:
    # meta-learner makes connection between features and 
    # each base-forecaster's weights 
    pipeline.add_metalearner(RandomForestClassifier())
    pipeline.fit(data, fh=FH)
    predict = pipeline.predict(data, fh=FH)
    print(predict.shape)

if __name__ == "__main__":
    main()