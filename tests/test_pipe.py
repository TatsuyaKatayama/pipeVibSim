import pytest
import numpy as np
from pipeVibSim.pipe_path import PipePath
from pipeVibSim.pipe import Pipe
from pipeVibSim.materials import get_material_properties

@pytest.fixture
def sample_pipe_segment1():
    """最初の配管セグメントを提供するフィクスチャ"""
    points = np.array([[0, 0, 0], [1, 0, 0]]) * 0.1
    pipe_path = PipePath(points, radius=0.03, step=0.01)
    n_elements = pipe_path.node_positions.shape[0] - 1
    material_properties = get_material_properties(
        E=200e9, rho=7850, nu=0.3, D_out=0.02, D_in=0.015, n_elements=n_elements
    )
    return pipe_path, material_properties

@pytest.fixture
def sample_pipe_segment2():
    """2番目の配管セグメントを提供するフィクスチャ"""
    points = np.array([[0, 0, 0], [0, 1, 0]]) * 0.1 # Y方向に伸びる
    pipe_path = PipePath(points, radius=0.03, step=0.01)
    n_elements = pipe_path.node_positions.shape[0] - 1
    material_properties = get_material_properties(
        E=110e9, rho=8960, nu=0.34, D_out=0.02, D_in=0.015, n_elements=n_elements
    )
    return pipe_path, material_properties

def test_pipe_single_segment(sample_pipe_segment1):
    """単一セグメントでPipeオブジェクトが正しく初期化されるかテスト"""
    pipe_path, material_properties = sample_pipe_segment1
    pipe = Pipe(pipe_path, material_properties)

    assert pipe is not None
    np.testing.assert_array_equal(pipe.node_positions, pipe_path.node_positions)
    np.testing.assert_array_equal(pipe.node_connectivity, pipe_path.node_connectivity)
    assert pipe.material_properties['E'] == material_properties['E']

def test_pipe_add_segment(sample_pipe_segment1, sample_pipe_segment2):
    """add_pipe_segmentでセグメントが追加され、正しく結合されるかテスト"""
    pipe_path1, material_properties1 = sample_pipe_segment1
    pipe_path2, material_properties2 = sample_pipe_segment2

    # 最初のセグメントでPipeを初期化
    pipe = Pipe(pipe_path1, material_properties1)
    n_nodes1 = pipe_path1.node_positions.shape[0]
    n_elements1 = pipe_path1.node_connectivity.shape[0]

    # 2番目のセグメントを追加
    pipe.add_pipe_segment(pipe_path2, material_properties2)
    n_nodes2 = pipe_path2.node_positions.shape[0]
    n_elements2 = pipe_path2.node_connectivity.shape[0]

    # 結合後のノード数と要素数を確認
    assert pipe.node_positions.shape[0] == n_nodes1 + n_nodes2 - 1
    assert pipe.node_connectivity.shape[0] == n_elements1 + n_elements2

    # 接続性を確認
    # 2番目のセグメントの接続情報がオフセットされているか
    expected_connectivity_start_node = n_nodes1 - 1
    assert pipe.node_connectivity[n_elements1, 0] == expected_connectivity_start_node

    # 材料特性が結合されているか確認
    assert len(pipe.material_properties['E']) == n_elements1 + n_elements2
    assert pipe.material_properties['E'][0] == material_properties1['E']
    assert pipe.material_properties['E'][-1] == material_properties2['E']
