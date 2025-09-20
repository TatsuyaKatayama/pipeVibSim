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

def test_u_bend_x():
    points = np.array([[-1, 0, 0], [0, 0, 0], [1, 1e-9, 0]], dtype=float)
    path = PipePath(points, radius=0.5, step=0.05)
    assert np.allclose(path.node_positions[0], points[0])
    assert np.allclose(path.node_positions[-1], points[-1])
    # Uベンドの頂点が(0, 0.5, 0)の近くにあることを確認
    center_point = np.array([0, 0.5, 0])
    distances = np.linalg.norm(path.node_positions - center_point, axis=1)
    assert np.min(distances) <= 0.5 + 1e-6

def test_u_bend_y():
    points = np.array([[0, -1, 0], [0, 0, 0], [1e-9, 1, 0]], dtype=float)
    path = PipePath(points, radius=0.5, step=0.05)
    assert np.allclose(path.node_positions[0], points[0])
    assert np.allclose(path.node_positions[-1], points[-1])
    # Uベンドの頂点が(-0.5, 0, 0)の近くにあることを確認
    center_point = np.array([-0.5, 0, 0])
    distances = np.linalg.norm(path.node_positions - center_point, axis=1)
    assert np.min(distances) <= 0.5 + 1e-6

def test_u_bend_z():
    points = np.array([[0, 0, -1], [0, 0, 0], [1e-9, 0, 1]], dtype=float)
    path = PipePath(points, radius=0.5, step=0.05)
    assert np.allclose(path.node_positions[0], points[0])
    assert np.allclose(path.node_positions[-1], points[-1])
    # Uベンドの頂点が(-0.5, 0, 0)の近くにあることを確認
    center_point = np.array([-0.5, 0, 0])
    distances = np.linalg.norm(path.node_positions - center_point, axis=1)
    assert np.min(distances) <= 0.5 + 1e-6

def test_u_bend_diagonal():
    p_prev = np.array([1, 1, 0])
    p_curr = np.array([0, 0, 0])
    p_next = np.array([-1, -1, 1e-9])
    points = np.array([p_prev, p_curr, p_next], dtype=float)
    path = PipePath(points, radius=0.5, step=0.05)
    assert np.allclose(path.node_positions[0], points[0])
    assert np.allclose(path.node_positions[-1], points[-1])
    # Uベンドの頂点の近くを通るか確認
    v_in = (p_prev - p_curr)
    v_in_norm = v_in / np.linalg.norm(v_in)
    # This logic should match the one in the source code to get the correct axis
    if abs(v_in_norm[0]) < 0.9:
        axis_ref = np.array([1.0, 0.0, 0.0])
    else:
        axis_ref = np.array([0.0, 1.0, 0.0])
    axis = np.cross(v_in_norm, axis_ref)
    axis /= np.linalg.norm(axis)

    center_dir = np.cross(axis, v_in_norm)
    center_point = p_curr + 0.5 * center_dir
    distances = np.linalg.norm(path.node_positions - center_point, axis=1)
    assert np.min(distances) <= 0.5 + 1e-6