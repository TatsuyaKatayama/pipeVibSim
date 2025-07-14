import matplotlib.pyplot as plt
import numpy as np

def plot_node_path(node_positions, points):
    """
    ノードパスと元の座標点をプロットします。

    Args:
        node_positions (np.ndarray): プロットするノードの座標。
        points (np.ndarray): 元の座標点。
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2], label='node_positions', color='b')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', s=50, label='Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def plot_mode_shapes(geometry, shapes):
    """
    モード形状をプロットします。

    Args:
        geometry: sdynpyのジオメトリオブジェクト。
        shapes: sdynpyの固有値解析結果。
    """
    geometry.plot_shape(shapes)

def plot_frf(frf):
    """
    周波数応答関数（FRF）をプロットします。

    Args:
        frf: sdynpyの周波数応答解析結果。
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.semilogy(frf.abscissa.flatten(), np.abs(frf.ordinate.squeeze()))
    plt.subplot(2, 1, 2)
    plt.plot(frf.abscissa.flatten(), np.angle(frf.ordinate.squeeze(), deg=True))
    plt.tight_layout()
    plt.show()
