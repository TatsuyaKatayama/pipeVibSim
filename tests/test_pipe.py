import pytest
import numpy as np
from pipeVibSim.pipe import Pipe

@pytest.fixture
def sample_pipe():
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [2, 1, 0]
    ]) * 0.1
    radius = 0.05
    step = 0.01
    return Pipe(points, radius, step)

def test_pipe_creation(sample_pipe):
    assert sample_pipe is not None
    assert isinstance(sample_pipe.node_positions, np.ndarray)
    assert isinstance(sample_pipe.node_connectivity, np.ndarray)
    assert isinstance(sample_pipe.bend_direction, np.ndarray)

def test_node_connectivity(sample_pipe):
    assert sample_pipe.node_connectivity.shape[1] == 2
    assert sample_pipe.node_connectivity.shape[0] == sample_pipe.node_positions.shape[0] - 1
