import numpy as np

import sdynpy as sdpy


def get_material_properties(E, rho, nu, D_out, D_in, n_elements, thickness=None):
    """
    円筒管の材料特性と断面特性を返します。

    Args:
        E (float or np.ndarray): ヤング率 (Pa)。スカラーまたは(n_elements,)配列。
        rho (float or np.ndarray): 密度 (kg/m^3)。スカラーまたは(n_elements,)配列。
        nu (float or np.ndarray): ポアソン比。スカラーまたは(n_elements,)配列。
        D_out (float or np.ndarray): 外径 (m)。スカラーまたは(n_elements,)配列。
        D_in (float or np.ndarray): 内径 (m)。スカラーまたは(n_elements,)配列。
        n_elements (int): 要素数。
        thickness (float or np.ndarray, optional): 肉厚 (m)。D_inの代わりに指定可能。

    Returns:
        dict: 材料特性と断面特性の辞書。
    """
    if thickness is not None:
        D_in = D_out - 2 * np.asarray(thickness)

    # 返却する辞書を作成
    props = {
        'young_modulus': E,
        'density': rho,
        'poisson_ratio': nu,
        'outer_diameter': D_out,
        'inner_diameter': D_in,
    }
    # thicknessが指定されていればそれも追加
    if thickness is not None:
        props['thickness'] = thickness
    else:
        props['thickness'] = (np.asarray(D_out) - np.asarray(D_in)) / 2

    return props