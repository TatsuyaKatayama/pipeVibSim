import numpy as np

import sdynpy as sdpy

from .pipe import Pipe


class VibrationAnalysis:
    """
    配管の振動解析を実行するクラス。

    Args:
        pipe (Pipe): 解析対象のPipeオブジェクト。
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.init_system, self.geometry = self._setup_system()
        self.system = self.init_system

    def _setup_system(self):
        """sdynpyシステムをセットアップします。"""
        
        # 材料定数を取得
        E = self.pipe.material_properties['young_modulus']
        G = E / (2 * (1 + self.pipe.material_properties['poisson_ratio']))
        rho = self.pipe.material_properties['density']
        D_o = self.pipe.material_properties['outer_diameter']
        thickness = self.pipe.material_properties['thickness']

        n_elements = self.pipe.node_connectivity.shape[0]
        
        # thicknessがリストかスカラーかによって処理を分ける
        if isinstance(thickness, (list, np.ndarray)):
            if len(thickness) != n_elements:
                raise ValueError("Length of thickness list must match the number of elements.")
            thickness_arr = np.array(thickness)
        else: # スカラーの場合
            thickness_arr = np.full(n_elements, thickness)

        # 要素ごとの断面特性を計算
        D_i_arr = D_o - 2 * thickness_arr
        A_arr = np.pi / 4 * (D_o**2 - D_i_arr**2)
        I_arr = np.pi / 64 * (D_o**4 - D_i_arr**4)
        J_arr = 2 * I_arr

        props = {
            'ae': E * A_arr,
            'jg': G * J_arr,
            'ei1': E * I_arr,
            'ei2': E * I_arr,
            'mass_per_length': rho * A_arr,
            'tmmi_per_length': rho * J_arr
        }

        return sdpy.System.beam_from_arrays(self.pipe.node_positions, self.pipe.node_connectivity,
                                            self.pipe.bend_direction, props)

    def reset_system(self):
        """システムを初期状態に戻します。"""
        self.system = self.init_system

    def substructure_by_coordinate(self, constraints):
        """
        座標に基づいて部分構造を作成し、self.systemを更新します。

        Args:
            constraints (list): 拘束条件のリスト。各要素は (coordinates, fixed_dofs) のタプル。
                                coordinatesは拘束する節点の座標、fixed_dofsは拘束する自由度。
        """
        fixed_dofs_list = []
        for coords, fixed_dof_indices in constraints:
            node_index = np.argmin(np.linalg.norm(self.pipe.node_positions - coords, axis=1))
            fixed_dofs = self.system.coordinate[node_index * 6:node_index * 6 + 6]
            if fixed_dof_indices is not None:
                fixed_dofs = fixed_dofs[fixed_dof_indices]
            fixed_dofs_list.append((fixed_dofs, None))

        self.system = self.system.substructure_by_coordinate(fixed_dofs_list)

    def run_eigensolution(self, maximum_frequency):
        """
        固有値解析を実行します。

        Args:
            maximum_frequency (float): 解析する最大周波数。

        Returns:
            eigensolution: sdynpyの固有値解析結果。
        """
        return self.system.eigensolution(maximum_frequency=maximum_frequency)

    def run_frf(self,
                frequencies,
                load_dof_indices,
                response_dof_indices=slice(None),
                displacement_derivative=0):
        """
        周波数応答解析（FRF）を実行します。

        Args:
            frequencies (np.ndarray): 解析する周波数の配列。
            load_dof_indices (int or list): 荷重をかける自由度のインデックス。
            response_dof_indices (int, list, or slice, optional): 応答を観測する自由度のインデックス。デフォルトは全自由度。
            displacement_derivative (int, optional): 変位の導関数の次数。デフォルトは0は変位。1は速度、2は加速度。
        Returns:
            frf: sdynpyの周波数応答解析結果。
        """
        load_dof = self.system.coordinate[load_dof_indices]
        response_dof = self.system.coordinate[response_dof_indices]
        return self.system.frequency_response(frequencies=frequencies,
                                              references=load_dof,
                                              responses=response_dof,
                                              displacement_derivative=displacement_derivative)
