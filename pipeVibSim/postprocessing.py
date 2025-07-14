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
    ax.plot(node_positions[:, 0],
            node_positions[:, 1],
            node_positions[:, 2],
            label='node_positions',
            color='b')
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


def plot_frf(frf, fig=None, axes=None):
    """
    周波数応答関数（FRF）をプロットします。

    Args:
        frf: sdynpyの周波数応答解析結果。
        fig (matplotlib.figure.Figure, optional): プロットするFigureオブジェクト。Noneの場合、新しいFigureを作成します。
        axes (list of matplotlib.axes.Axes, optional): プロットするAxesオブジェクトのリスト。Noneの場合、新しいAxesを作成します。

    Returns:
        tuple: (fig, axes) FigureオブジェクトとAxesオブジェクトのリスト。
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].semilogy(frf.abscissa.flatten(), np.abs(frf.ordinate.squeeze()))
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Frequency Response Function (FRF)')
    axes[0].grid(True)

    axes[1].plot(frf.abscissa.flatten(), np.angle(frf.ordinate.squeeze(), deg=True))
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].grid(True)

    plt.tight_layout()
    return fig, axes
