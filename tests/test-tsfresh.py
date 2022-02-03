import pytest
import numpy as np
from metats.features.statistical import TsFresh

def test_tsfrsh_transform():
    data = np.random.rand(10, 55, 5)
    try:
        tsfresh = TsFresh()
    except:
        pytest.fail("Failed to instantiate TsFresh")
    
    tsfresh.fit(data)
    try:
        features = tsfresh.transform(data)
    except:
        pytest.fail("TSFresh.transform() failed")

    assert features.shape[0] == 10
