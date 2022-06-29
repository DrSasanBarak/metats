import numpy as np
import torch
import torch.nn as nn
from metats.features import FeatureGenerator
from metats.features.deep import PyTorchTrainer


class DeepAETrainer(PyTorchTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_mini_batch(self):
        data_size = self.Y.shape[0]
        sample_size = min(self.batch_size, data_size)
        indices = np.random.choice(data_size, sample_size, replace=False)
        mini_batch = self.Y[indices]
        mini_batch_tensor = torch.from_numpy(mini_batch).float()
        return {'Y' : mini_batch_tensor}

class DeepAutoEncoder(FeatureGenerator):
    """
    A wrapper class for feature extraction using deep auto encoders
    """
    def __init__(self, auto_encoder, verbose=False, epochs=64, learning_rate=0.006, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_encoder = auto_encoder
        self.epochs = epochs
        self.verbose = verbose
        self.learning_rate = learning_rate
    
    def print_loss_callback(self, loss):
        print('Deep AE Reconstruction Loss: {:.3f}'.format(loss))


    def fit(self, Y):
        trainer = DeepAETrainer(model=self.auto_encoder, learning_rate=self.learning_rate)
        if self.verbose:
            trainer.register_loss_callback(self.print_loss_callback)

        trainer.Y = Y
        for epoch in range(self.epochs):
            trainer.step()
    
    def transform(self, Y):
        Y_tensor = torch.from_numpy(Y).float()
        latent = self.auto_encoder.encode(Y_tensor, inference=True)
        return latent.numpy()
    
    def is_trainable(self):
        return True

class Contrastive(FeatureGenerator):
    """
    Base Class for Contrastive feature extractors
    """
    def fit(self, Y):
        pass
    
    def transform(self, Y):
        pass