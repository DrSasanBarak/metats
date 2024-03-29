import numpy as np

class MetaLearning():
    """
    Base Class for Meta-Learning Pipelines
    """
    def __init__(self, method='averaging', loss='mse', reduction='none', reduction_dim=64):
        """
        Args:
            method: can be either 'averaging' or 'selection'. in the case of averagin,
                    the predictions of base-forecaster will be combined by weighting
                    based on classification weights. In the 'selection' mode, the best
                    forecaster will be selected and others will be ingored
            loss: loss function for generating meta-learning labels, can be 'mse', 'mae'.
                  to use a custom loss function you can also pass a callable object.
            reduction: you can apply dimensionality reduction on generated features.
                       can be 'none', 'pca'. use reduction dim to specify
                       final dimension.
            reduction_dim: will be ignored if reduction='none'.
                   
        """
        self.method = method
        self.feature_generators = []
        self.base_forecasters = []
        self.base_forecasters_name = []
        self.features_prefix_name = []
        self.features_name = []
        self.meta_learner = None
        self.loss = loss
        self.reduction = reduction
        self.reduction_dim = reduction_dim

        if self.reduction == 'pca':
            from sklearn.decomposition import PCA
            self.reducer = PCA(n_components=self.reduction_dim)
    
    def add_feature(self, feature_generator, feature_name=""):
        """
        Add a feature generator to the pipeline
        """
        self.features_prefix_name.append(feature_name)
        self.feature_generators.append(feature_generator)
    
    def add_forecaster(self, forecaster, forecaster_name=""):
        """
        Add a base forecaster to the pipeline
        """
        from metats.forecasters.utils import is_sktime_forecaster, is_darts_forecaster, is_nixtla_stats_forecaster
        # checking for dart forecasters
        wrapped = forecaster
        if is_darts_forecaster(forecaster):
            from metats.forecasters.wrappers import DartsForecasterWrapper
            wrapped = DartsForecasterWrapper(forecaster)
        elif is_sktime_forecaster(forecaster):
            from metats.forecasters.wrappers import SKTimeForecasterWrapper
            wrapped = SKTimeForecasterWrapper(forecaster)
        elif is_nixtla_stats_forecaster(forecaster):
            from metats.forecasters.wrappers import NixtlaStatsForecasterWrapper
            wrapped = NixtlaStatsForecasterWrapper(forecaster)

        # append the wrapped forecaster to the stack
        self.base_forecasters.append(wrapped)
        # generate a name for forecater
        name = "Forecaster {}".format(len(self.base_forecasters))
        if len(forecaster_name) > 0:
            name = forecaster_name
        self.base_forecasters_name.append(name)

        

    
    def loss_fn(self, y_true, y_pred):
        """
        Loss function for the meta-learner
        Args:
            y_true: true values (numpy array) (num_series x horizon)
            Y_pred: the matrix of predictions (num_series x num_models x horizon)
        """
        if self.loss == 'mse':
            return np.mean(np.square(y_true - y_pred), axis=2)
        elif self.loss == 'mae':
            return np.mean(np.abs(y_true - y_pred), axis=2)
        elif callable(self.loss):
            return self.loss(y_true, y_pred)
        else:
            raise ValueError('Loss function not supported')

    def labels_selection(self, y_true, Y_pred, return_one_hot=False):
        """
        Generate labels for the meta-learner using selection
        Args:
            y_true: true values (numpy array) (num_series x horizon)
            Y_pred: the matrix of predictions (num_series x num_models x horizon)
        """
        error = self.loss_fn(y_true, np.transpose(Y_pred, (1, 0, 2)))
        error = error.T
        label_indices = np.argmin(error, axis=1)
        if return_one_hot:    
            # convert to one-hot encoding
            labels = np.zeros(Y_pred.shape[1:])
            for i in range(Y_pred.shape[1]):
                labels[i, label_indices[i]] = 1.0
            return labels
        else:
            return label_indices

    def generate_labels(self, y_true, Y_pred):
        """
        Generate labels for the meta-learner
        Args:
            y_true: true values
            Y_pred: the matrix of predictions 
        """
        if self.method == 'selection':
            return self.labels_selection(y_true, Y_pred)
        elif self.method == 'averaging':
            return self.labels_selection(y_true, Y_pred)
        else:
            raise ValueError('Method not supported')
        
    def generate_features(self, Y, prediction=False):
        # generate meta features
        meta_features = []
        for feature_generator, feature_prefix in zip(self.feature_generators, self.features_prefix_name):
            if (not prediction) or (not feature_generator.is_trainable()):
                feature_generator.fit(Y)
            single_meta_features = feature_generator.transform(Y)
            features_dim = single_meta_features.shape[1] 
            meta_features.append(single_meta_features)
            # generate the labels for meta-features
            if feature_prefix == "":
                feature_prefix = f"F_{len(self.features_name)}"
            if features_dim == 1:
                self.features_name.append(feature_prefix)
            else:
                names = [ '{}_{}'.format(feature_prefix, i) for i in range(features_dim) ]
                self.features_name.extend(names)

        return np.hstack(meta_features)
    
    def generate_prediction(self, forecaster, Y, fh, forecast_dim):
        """
        Generate predictions for a single forecaster
        Args:
            forecaster: the forecaster to be used
            Y: the timeseries (numpy array) (num_series x series_length x covariates_dim)
            fh: forecast horizon
            forecast_dim: the dimension of the variable to be forecasted
        """
        return forecaster.predict(Y, fh, forecast_dim)
    
    def generate_predictions(self, Y, fh, forecast_dim=0):
        # generate predictions
        predictions = []
        for base_forecaster in self.base_forecasters:
            prediction = self.generate_prediction(base_forecaster, Y, fh, forecast_dim) 
            if prediction.ndim == 3:
                prediction = prediction[:, :, forecast_dim]
            prediction = np.expand_dims(prediction, axis=1)
            predictions.append(prediction)
        predictions = np.concatenate(predictions, axis=1)
        return predictions
    
    def add_metalearner(self, metalearner):
        """
        Add a meta-learner to the pipeline
        Args:
            metalearner: the meta-learner to be added a scikit-learn estimator
        """
        self.meta_learner = metalearner

    def reduce_meta_features(self, meta_features, prediction=False):
        """
        applying dimensionality reduction on the generated features 
        Args:
            meta_features: numpy array of meta-features (num_series, features_dim)
            prediction: if True the stored features from training stage will be used
        """
        num_training_series = 0
        if prediction:
            num_training_series = self.training_features.shape[0] 
            # concatenating the training features
            meta_features = np.vstack([self.training_features, meta_features])
        reduced = self.reducer.fit_transform(meta_features)[num_training_series:, :]
        return reduced

    def fit(self, Y, fh, forecast_dim=0):
        """
        Fit the meta-learner
        Args:
            Y: the timeseries (numpy array) (num_series x series_length x covariates_dim)
            fh: forecasting horizon
            forecast_dim: forecasting dimension
        """
        # data spliting for meta-learning
        Y_true = Y[:, -fh:, forecast_dim]

        self.features_fh = fh
        meta_features = self.generate_features(Y[:, :-fh, :], prediction=False)
        predictions = self.generate_predictions(Y[:, :-fh, :], fh, forecast_dim=forecast_dim)
        labels = self.generate_labels(Y_true, predictions)
        
        # saving the generated features for using in prediction phase
        if self.reduction != 'none':
            self.training_features = np.copy(meta_features)
            meta_features = self.reduce_meta_features(meta_features)

        self.meta_learner.fit(X=meta_features, y=labels)

    def predict_generate_weights(self, meta_features):
        """
        Predict using the meta-learner
        Args:
            metafeatures: the extracted meta-features (numpy array) (num_series x features_dim)
        """
        weights = None
        if self.method == 'selection':
            weights = self.meta_learner.predict(meta_features)
        elif self.method == 'averaging':
            weights = self.meta_learner.predict_proba(meta_features)
        
        return weights

    def averaging_predictions(self, weights, predictions):
        """
        Generating the predictions by averaging each base-forecaster
        """
        weighted_predictions = []
        for series in range(predictions.shape[0]):
            p = weights[series, :].reshape(1, -1) @ predictions[series, :, :]
            weighted_predictions.append(p)
        result = np.vstack(weighted_predictions)
        return result
    
    def selection_predictions(self, weights, predictions):
        """
        Generating the predictions by selecting the best-forecaster
        """
        selected_predictions = []
        for series in range(predictions.shape[0]):
            p = predictions[series, weights[series], :].reshape(1,-1)
            selected_predictions.append(p)
        result = np.vstack(selected_predictions)
        return result

    def predict(self, Y, fh, forecast_dim=0, return_metalearning_data=False, explain=None):
        """
        Predict using the meta-learner
        Args:
            Y: the timeseries (numpy array) (num_series x series_length x covariates_dim)
            fh: forecast horizon for predicting
            forecast_dim: the dimension of the variable to be forecasted
            return_metalearning_data: if True, the meta-features, weights and base forecasts will be returned
            explain: currently only 'shap' for tree-based meta-learners is supported
        Return:
            result: (numpy array) (num_series x forecast_horizon)
            if return_metalearning_data is True:
                predictions: (numpy array) (num_series x number_of_base_forecasters x forecast_horizon)
                weights: (numpy array) (num_series x number_of_base_forecasters)
                meta_features: (numpy array) (num_series x features_dim)
        """
        meta_features = self.generate_features(Y[:, :-self.features_fh, :], prediction=True)

        if self.reduction != 'none':
            meta_features = self.reduce_meta_features(meta_features, prediction=True)

        predictions = self.generate_predictions(Y, fh, forecast_dim=forecast_dim)
        weights = self.predict_generate_weights(meta_features)

        if self.method == 'averaging':
            result = self.averaging_predictions(weights, predictions)
        else:
            result = self.selection_predictions(weights, predictions)

        metalearning_data = None

        if return_metalearning_data:
            metalearning_data = [predictions, weights, meta_features]

        explanation_data = None
        # explain the result
        if explain == 'shap':
            import shap
            shap_explainer = shap.Explainer(self.meta_learner)
            shap_values = shap_explainer(meta_features)
            shap_values.feature_names = self.features_name
            explanation_data = shap_values
        
        return result, metalearning_data, explanation_data
