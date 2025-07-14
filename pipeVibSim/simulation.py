import sdynpy as sdpy
import numpy as np
from .pipe import Pipe
from .materials import get_material_properties

class VibrationAnalysis:
    """
    配管の振動解析を実行するクラス。

    Args:
        pipe (Pipe): 解析対象のPipeオブジェクト。
        material_properties (dict): 材料特性の辞書。
    """
    def __init__(self, pipe, material_properties):
        self.pipe = pipe
        self.material_properties = material_properties
        self.system, self.geometry = self._setup_system()

    def _setup_system(self):
        """sdynpyシステムをセットアップします。"""
        return sdpy.System.beam_from_arrays(
            self.pipe.node_positions,
            self.pipe.node_connectivity,
            self.pipe.bend_direction,
            self.material_properties
        )

    def run_eigensolution(self, maximum_frequency):
        """
        固有値解析を実行します。

        Args:
            maximum_frequency (float): 解析する最大周波数。

        Returns:
            eigensolution: sdynpyの固有値解析結果。
        """
        return self.system.eigensolution(maximum_frequency=maximum_frequency)

    def run_frf(self, frequencies, load_dof_indices, response_dof_indices):
        """
        周波数応答解析（FRF）を実行します。

        Args:
            frequencies (np.ndarray): 解析する周波数の配列。
            load_dof_indices (int or list): 荷重をかける自由度のインデックス。
            response_dof_indices (int or list): 応答を観測する自由度のインデックス。

        Returns:
            frf: sdynpyの周波数応答解析結果。
        """
        load_dof = self.system.coordinate[load_dof_indices]
        response_dof = self.system.coordinate[response_dof_indices]
        fixed_dofs = self.system.coordinate[:6]
        constrained_system = self.system.substructure_by_coordinate([(fixed_dofs, None)])
        return constrained_system.frequency_response(
            frequencies=frequencies,
            references=load_dof,
            responses=response_dof,
            displacement_derivative=2
        )