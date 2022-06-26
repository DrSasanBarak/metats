import numpy as np
import torch
import torch.nn as nn
from metats.features import FeatureGenerator
from metats.features.deep import PyTorchTrainer


class DeepAETrainer(PyTorchTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_mini_batch(self):
        indices = np.random.choice(self.Y.shape[0], number_of_samples, replace=False)
        mini_batch = self.Y[indices]
        mini_batch_tensor = torch.from_numpy(mini_batch).float()
        return {'Y' : minibatch_tensor}

class DeepAutoEncoder(FeatureGenerator):
    """
    A wrapper class for feature extraction using deep auto encoders
    """
    def __init__(self, auto_encoder, epochs=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_encoder = auto_encoder
        self.epochs = epochs

    def fit(self, Y):
        trainer = DeepAETrainer(model=self.auto_encoder)
        trainer.Y = Y
        for epoch in range(self.epochs):
            trainer.step()
    
    def transform(self, Y):
        Y_tensor = torch.from_numpy(Y).float()
        latent = self.auto_encoder.encode(Y_tensor, inference=True)
        return latent

class Contrastive(FeatureGenerator):
    """
    Base Class for Contrastive feature extractors
    """
    def fit(self, Y):
        pass
    
    def transform(self, Y):
        pass