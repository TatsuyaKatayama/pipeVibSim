import pytest
import numpy as np
from pipeVibSim.pipe_path import PipePath

@pytest.fixture
def sample_pipe_path():
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [2, 1, 0]
    ]) * 0.1
    radius = 0.05
    step = 0.01
    return PipePath(points, radius, step)

def test_pipe_creation(sample_pipe_path):
    assert sample_pipe_path is not None
    assert isinstance(sample_pipe_path.node_positions, np.ndarray)
    assert isinstance(sample_pipe_path.node_connectivity, np.ndarray)
    assert isinstance(sample_pipe_path.bend_direction, np.ndarray)

def test_node_connectivity(sample_pipe_path):
    assert sample_pipe_path.node_connectivity.shape[1] == 2
    assert sample_pipe_path.node_connectivity.shape[0] == sample_pipe_path.node_positions.shape[0] - 1

def test_pipe_path_addition():
    # 1つ目のPipePath
    points1 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
        ],
        dtype=float
    )
    path1 = PipePath(points=points1, radius=0.1, step=0.1)

    # 2つ目のPipePath
    points2 = np.array(
        [
            [0, 0, 0], # この始点がpath1の終点 [1, 0, 0] に移動するはず
            [1, 1, 0],
        ],
        dtype=float
    )
    path2 = PipePath(points=points2, radius=0.1, step=0.1)

    # PipePathを結合
    combined_path = path1 + path2

    # 期待されるpoints
    expected_points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 1, 0],
        ]
    )

    # 結合後のPipePathのpointsが期待通りかチェック
    assert np.allclose(combined_path.points, expected_points)

    # radiusとstepが引き継がれているかもチェック
    assert combined_path.radius == path1.radius
    assert combined_path.step == path1.step