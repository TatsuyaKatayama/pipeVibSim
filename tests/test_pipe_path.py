import numpy as np
import pytest

from pipeVibSim.pipe_path import PipePath


@pytest.fixture
def sample_pipe_path():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]]) * 0.1
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
    assert sample_pipe_path.node_connectivity.shape[
        0] == sample_pipe_path.node_positions.shape[0] - 1


def test_pipe_path_addition():
    # 1つ目のPipePath
    points1 = np.array([
        [0, 0, 0],
        [1, 0, 0],
    ], dtype=float)
    path1 = PipePath(points=points1, radius=0.1, step=0.1)

    # 2つ目のPipePath
    points2 = np.array(
        [
            [0, 0, 0],  # この始点がpath1の終点 [1, 0, 0] に移動するはず
            [1, 1, 0],
        ],
        dtype=float)
    path2 = PipePath(points=points2, radius=0.1, step=0.1)

    # PipePathを結合
    combined_path = path1 + path2

    # 期待されるpoints
    expected_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 1, 0],
    ])

    # 結合後のPipePathのpointsが期待通りかチェック
    assert np.allclose(combined_path.points, expected_points)

    # radiusとstepが引き継がれているかもチェック
    assert combined_path.radius == path1.radius
    assert combined_path.step == path1.step


def test_curvatures():
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],  # 90度曲げ
        ],
        dtype=float)
    radius = 0.1
    path = PipePath(points, radius=radius, step=0.1)

    # 総要素数と曲率配列の長さが一致することを確認
    assert path.node_connectivity.shape[0] == len(path.curvatures)

    # 曲げ部の曲率が 1/radius になっていることを確認
    expected_curvature = 1.0 / radius

    # 最初のセグメントは直線なので曲率は0のはず
    assert np.isclose(path.curvatures[0], 0.0)

    # どこかに 1/radius に近い曲率があるはず
    assert np.any(np.isclose(path.curvatures, expected_curvature))

    # 0 と 1/radius 以外の値がないことを確認（丸め誤差を考慮）
    unique_curvatures = np.unique(np.round(path.curvatures, 5))
    expected_unique_values = np.round([0.0, expected_curvature], 5)
    assert np.all(np.isin(unique_curvatures, expected_unique_values))


def test_u_bend_reversibility_and_shape():
    """Uベンドの可逆性と形状をテストする"""
    points_forward = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.4, 0], [0, 0.4, 0]], dtype=float)
    radius = 0.2
    center = np.array([0, 0.2, 0])  # Corrected center

    path_forward = PipePath(points_forward, radius=radius, step=0.02)
    nodes_forward = path_forward.node_positions
    points_backward = np.flip(points_forward, axis=0)
    path_backward = PipePath(points_backward, radius=radius, step=0.02)
    nodes_backward = path_backward.node_positions

    # 可逆性の検証
    assert np.allclose(nodes_forward[0], nodes_backward[-1])
    assert np.allclose(nodes_forward[-1], nodes_backward[0])
    len_forward = np.sum(np.linalg.norm(np.diff(nodes_forward, axis=0), axis=1))
    len_backward = np.sum(np.linalg.norm(np.diff(nodes_backward, axis=0), axis=1))
    assert np.isclose(len_forward, len_backward, rtol=1e-2)

    # 形状の検証
    arc_element_indices = np.where(path_forward.curvatures > 0)[0]
    arc_node_indices = np.unique(path_forward.node_connectivity[arc_element_indices].flatten())
    arc_nodes = nodes_forward[arc_node_indices]
    distances_from_center = np.linalg.norm(arc_nodes - center, axis=1)
    expected_distances = np.full_like(distances_from_center, radius)
    assert np.allclose(distances_from_center, expected_distances,
                       atol=1e-5), "U-bend nodes are not on the arc"


def test_3d_bend_reversibility_and_shape():
    """3Dベンドの可逆性と形状をテストする"""
    inv_sqrt2 = 1 / np.sqrt(2)
    points_forward = np.array([[0, 0, 0], [inv_sqrt2, 0, 0], [inv_sqrt2, 1, 1], [0, 1, 1]],
                              dtype=float)
    radius = inv_sqrt2
    center = np.array([0., 0.5, 0.5])  # As specified

    path_forward = PipePath(points_forward, radius=radius, step=0.1)
    nodes_forward = path_forward.node_positions
    points_backward = np.flip(points_forward, axis=0)
    path_backward = PipePath(points_backward, radius=radius, step=0.1)
    nodes_backward = path_backward.node_positions

    # 可逆性の検証
    assert np.allclose(nodes_forward[0], nodes_backward[-1])
    assert np.allclose(nodes_forward[-1], nodes_backward[0])
    len_forward = np.sum(np.linalg.norm(np.diff(nodes_forward, axis=0), axis=1))
    len_backward = np.sum(np.linalg.norm(np.diff(nodes_backward, axis=0), axis=1))
    assert np.isclose(len_forward, len_backward, rtol=1e-2)

    # 形状の検証
    arc_element_indices = np.where(path_forward.curvatures > 0)[0]
    arc_node_indices = np.unique(path_forward.node_connectivity[arc_element_indices].flatten())
    arc_nodes = nodes_forward[arc_node_indices]
    distances_from_center = np.linalg.norm(arc_nodes - center, axis=1)
    expected_distances = np.full_like(distances_from_center, radius)
    assert np.allclose(distances_from_center, expected_distances,
                       atol=1e-5), "3D-bend nodes are not on the arc"
