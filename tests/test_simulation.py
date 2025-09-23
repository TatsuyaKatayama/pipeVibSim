import pytest
import numpy as np
import sys
import os

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeVibSim.pipe_path import PipePath
from pipeVibSim.pipe import Pipe
from pipeVibSim.simulation import VibrationAnalysis

@pytest.fixture
def material_props_base():
    return {
        'young_modulus': 2.06e11,
        'poisson_ratio': 0.3,
        'density': 7850,
        'outer_diameter': 0.1143
    }

@pytest.fixture
def straight_pipe_path():
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    # step=0.5 -> 2 elements
    return PipePath(points, radius=0.1, step=0.5)

def test_uniform_vs_variable_thickness(straight_pipe_path, material_props_base):
    """均一な肉厚と可変な肉厚で、生成されるシステムが異なることを確認するテスト。"""
    # Case 1: 均一な肉厚
    material_uniform = material_props_base.copy()
    material_uniform['thickness'] = 0.01
    pipe_uniform = Pipe(straight_pipe_path, material_uniform)
    analysis_uniform = VibrationAnalysis(pipe_uniform)

    # Case 2: 可変な肉厚
    material_variable = material_props_base.copy()
    n_elements = straight_pipe_path.node_connectivity.shape[0]
    thickness_list = np.linspace(0.01, 0.005, n_elements).tolist()
    material_variable['thickness'] = thickness_list
    pipe_variable = Pipe(straight_pipe_path, material_variable)
    analysis_variable = VibrationAnalysis(pipe_variable)

    # 両者のシステムが構築されていることを確認
    assert analysis_uniform.system is not None
    assert analysis_variable.system is not None

    # 質量行列が異なることを確認 (肉厚が違えば質量も違うはず)
    mass_matrix_uniform = analysis_uniform.system.mass
    mass_matrix_variable = analysis_variable.system.mass
    assert not np.allclose(mass_matrix_uniform, mass_matrix_variable)

    # 剛性行列も異なることを確認
    stiffness_matrix_uniform = analysis_uniform.system.stiffness
    stiffness_matrix_variable = analysis_variable.system.stiffness
    assert not np.allclose(stiffness_matrix_uniform, stiffness_matrix_variable)

def test_thickness_list_length_mismatch(straight_pipe_path, material_props_base):
    """肉厚リストの長さが要素数と一致しない場合にエラーを送出するかテスト。"""
    material_props = material_props_base.copy()
    # 要素数(2)と異なる長さのリスト
    thickness_list = [0.01, 0.008, 0.006]
    material_props['thickness'] = thickness_list

    pipe = Pipe(straight_pipe_path, material_props)
    
    with pytest.raises(ValueError, match="Length of thickness list must match the number of elements."):
        VibrationAnalysis(pipe)

@pytest.fixture
def analysis_setup(straight_pipe_path, material_props_base):
    """解析オブジェクトと基本的なセットアップを提供するフィクスチャ"""
    material_props = material_props_base.copy()
    material_props['thickness'] = 0.01
    pipe = Pipe(straight_pipe_path, material_props)
    analysis = VibrationAnalysis(pipe)
    # 始点を固定
    constraints = [(pipe.node_positions[0], None)]
    analysis.substructure_by_coordinate(constraints)
    return analysis

def test_eigensolution_storage(analysis_setup):
    """run_eigensolutionが結果をインスタンスに保存することをテスト"""
    analysis = analysis_setup
    assert analysis.eigensolution is None
    shapes = analysis.run_eigensolution(maximum_frequency=1000)
    assert analysis.eigensolution is not None
    assert analysis.eigensolution is shapes

def test_frf_modal_without_eigensolution(analysis_setup):
    """eigensolutionなしでrun_frf_modalを呼ぶとエラーになることをテスト"""
    analysis = analysis_setup
    frequencies = np.linspace(1, 100, 100)
    with pytest.raises(RuntimeError, match="モード重ね合わせ法を使用するには、先に `run_eigensolution` を実行してください。"):
        analysis.run_frf_modal(frequencies, load_dof_indices=[-1], response_dof_indices=[-1])

def test_frf_methods(analysis_setup):
    """run_frf_directとrun_frf_modalが実行でき、結果が近いことをテスト"""
    analysis = analysis_setup
    analysis.run_eigensolution(maximum_frequency=5000)
    
    frequencies = np.linspace(1, 500, 200)
    load_dof = [-4]
    resp_dof = [-4]

    # モード法
    frf_modal = analysis.run_frf_modal(frequencies, load_dof_indices=load_dof, response_dof_indices=resp_dof)
    
    # 直接法
    frf_direct = analysis.run_frf_direct(frequencies, load_dof_indices=load_dof, response_dof_indices=resp_dof)

    assert frf_modal is not None
    assert frf_direct is not None
    assert frf_modal.ordinate.shape == frf_direct.ordinate.shape

    # 低周波数域で結果が近いことを確認（許容誤差を大きめに設定）
    # ピーク周波数付近では差が大きくなる可能性があるため、平均的な差で比較
    avg_diff = np.mean(np.abs(frf_modal.ordinate - frf_direct.ordinate))
    avg_mag = np.mean(np.abs(frf_direct.ordinate))
    assert avg_diff / avg_mag < 0.1 # 平均して10%以下の差異であること