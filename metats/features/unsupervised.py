import numpy as np
from metats.features import FeatureGenerator

class AutoEncoder(FeatureGenerator):
    """
    Base Class for Endeor-Decoder feature extractors
    """
    def fit(self, Y):
        pass
    
    def transform(self, Y):
        pass

class Contrastive(FeatureGenerator):
    """
    Base Class for Contrastive feature extractors
    """
    def fit(self, Y):
        pass
    
    def transform(self, Y):
        pass