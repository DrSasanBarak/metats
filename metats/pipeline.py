
class MetaLearning():
    """
    Base Class for Meta-Learning Pipelines
    """
    def __init__(self, method='averaging', loss='mse'):
        self.method = method
        self.feature_generators = []
        self.base_forecasters = []
        self.meta_learner = None
        self.loss = loss
    
    def add_feature(self, feature_generator):
        """
        Add a feature generator to the pipeline
        """
        self.feature_generators.append(feature_generator)
    
    def add_forcecaster(self, forcecaster):
        """
        Add a base forecaster to the pipeline
        """
        self.base_forecasters.append(forcecaster)
    
    def loss_fn(self, y_true, y_pred):
        """
        Loss function for the meta-learner
        inputs:
            y_true: true values (numpy array) (num_series x horizon)
            Y_pred: the matrix of predictions (num_series x num_models x horizon)
        """
        if self.loss == 'mse':
            return np.mean(np.square(y_true - y_pred), axis=2)
        elif self.loss == 'mae':
            return np.mean(np.abs(y_true - y_pred), axis=2)
        else:
            raise ValueError('Loss function not supported')

    def labels_selection(self, y_true, Y_pred, return_one_hot=False):
        """
        Generate labels for the meta-learner using selection
        inputs:
            y_true: true values (numpy array) (num_series x horizon)
            Y_pred: the matrix of predictions (num_series x num_models x horizon)
        """
        error = self.loss_fn(y_true, Y_pred)
        label_indices = np.argmin(error, axis=1)
        if return_one_hot:    
            # convert to one-hot encoding
            labels = np.zeros(Y_pred.shape[1:])
            for i in range(Y_pred.shape[1]):
                labels[i, label_indices[i]] = 1
            return labels
        else:
            return label_indices

    def generate_labels(self, y_true, Y_pred):
        """
        Generate labels for the meta-learner
        inputs:
            y_true: true values
            Y_pred: the matrix of predictions 
        """
        if self.method == 'selection':
            return self.labels_selection(y_true, Y_pred)
        else:
            raise ValueError('Method not supported')
        
    def generate_features(self):
        # generate meta features
        meta_features = []
        for feature_generator in self.feature_generators:
            meta_features.append(feature_generator.generate_features())