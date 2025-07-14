import sdynpy as sdpy

def get_material_properties(E, rho, nu, D_out, D_in, n_elements, D_out_std=0.0):
    """
    円筒管の材料特性と断面特性を返します。

    Args:
        E (float): ヤング率 (Pa)。
        rho (float): 密度 (kg/m^3)。
        nu (float): ポアソン比。
        D_out (float): 外径 (m)。
        D_in (float): 内径 (m)。
        n_elements (int): 要素数。
        D_out_std (float, optional): 外径の標準偏差。デフォルトは0.0。

    Returns:
        dict: 材料特性と断面特性の辞書。
    """
    mat_prop = sdpy.fem.sdynpy_beam.cylindrical_pipe_props(E, rho, nu, D_out, D_in, n_elements, D_out_std)
    return {k: v for k, v in mat_prop.items() if k not in ['dD_outer', 'dD_inner']}
