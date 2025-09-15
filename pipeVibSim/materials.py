import numpy as np

import sdynpy as sdpy


def get_material_properties(E, rho, nu, D_out, D_in, n_elements):
    """
    円筒管の材料特性と断面特性を返します。

    Args:
        E (float or np.ndarray): ヤング率 (Pa)。スカラーまたは(n_elements,)配列。
        rho (float or np.ndarray): 密度 (kg/m^3)。スカラーまたは(n_elements,)配列。
        nu (float or np.ndarray): ポアソン比。スカラーまたは(n_elements,)配列。
        D_out (float or np.ndarray): 外径 (m)。スカラーまたは(n_elements,)配列。
        D_in (float or np.ndarray): 内径 (m)。スカラーまたは(n_elements,)配列。
        n_elements (int): 要素数。

    Returns:
        dict: 材料特性と断面特性の辞書。
    """
    # 配列かどうか判定
    arrs = [E, rho, nu, D_out, D_in]
    is_array = any(isinstance(x, (np.ndarray, list)) for x in arrs)
    # すべて1要素ならまとめて
    if not is_array or (np.asarray(E).shape == () or np.asarray(E).shape == (1,)):
        mat_prop = sdpy.fem.sdynpy_beam.cylindrical_pipe_props(E, rho, nu, D_out, D_in, n_elements)
    else:
        # 個別に
        mat_prop_list = []
        for i in range(n_elements):
            mat_prop_i = sdpy.fem.sdynpy_beam.cylindrical_pipe_props(
                np.asarray(E)[i],
                np.asarray(rho)[i],
                np.asarray(nu)[i],
                np.asarray(D_out)[i],
                np.asarray(D_in)[i], 1)
            mat_prop_list.append(mat_prop_i)
        # dictのリストをマージ
        mat_prop = {}
        for k in mat_prop_list[0].keys():
            mat_prop[k] = np.array([mp[k] for mp in mat_prop_list]).flatten()
    mat_prop['E'] = E
    mat_prop['rho'] = rho
    mat_prop['nu'] = nu
    mat_prop['D_out'] = D_out
    mat_prop['D_in'] = D_in
    return {k: v for k, v in mat_prop.items() if k not in ['dD_outer', 'dD_inner']}
