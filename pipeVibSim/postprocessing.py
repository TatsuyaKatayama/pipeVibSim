import matplotlib.pyplot as plt
import numpy as np
from qtpy.QtWidgets import QApplication
from sdynpy.core.sdynpy_geometries import MultipleShapePlotter, MultipleDeflectionShapePlotter


def plot_node_path(node_positions, points=None, fig=None, ax=None, color='b'):
    """
    ノードパスと元の座標点をプロットします。

    Args:
        node_positions (np.ndarray): プロットするノードの座標。
        points (np.ndarray, optional): 元の座標点。 Defaults to None.
        fig (matplotlib.figure.Figure, optional): プロットするFigureオブジェクト。Noneの場合、新しいFigureを作成します。 Defaults to None.
        ax (matplotlib.axes.Axes, optional): プロットするAxesオブジェクト。Noneの場合、新しいAxesを作成します。 Defaults to None.
        color (str, optional): プロットの色。 Defaults to 'b'.

    Returns:
        tuple: (fig, ax) FigureオブジェクトとAxesオブジェクト。
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    ax.plot(node_positions[:, 0],
            node_positions[:, 1],
            node_positions[:, 2],
            label='node_positions',
            color=color)
    if points is not None:
        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2],
                   color=color,
                   s=50,
                   alpha=0.5,
                   label='Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    return fig, ax


def plot_mode_shapes(geometry, shapes):
    """
    モード形状をプロットします。

    Args:
        geometry: sdynpyのジオメトリオブジェクト。
        shapes: sdynpyの固有値解析結果。
    """
    geometry.plot_shape(shapes)


def plot_deflection_shape(geometry, frf):
    """
    FRFの変形形状をプロットします。

    Args:
        geometry: sdynpyのジオメトリオブジェクト。
        frf: sdynpyの周波数応答解析結果。
    """
    geometry.plot_deflection_shape(frf)


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


def plot_pipe_geometry(pipe, fig=None, ax=None, cmap_name='Greys'):
    """
    配管システムのジオメトリをプロットします。セグメントごとに色分けします。

    Args:
        pipe (Pipe): プロットするPipeオブジェクト。
        fig (matplotlib.figure.Figure, optional): プロットするFigureオブジェクト。 Defaults to None.
        ax (matplotlib.axes.Axes, optional): プロットする3D Axesオブジェクト。 Defaults to None.
        cmap_name (str, optional): セグメントの色分けに使用するMatplotlibのカラーマップ名。Sequentialカラーマップがおすすめ
                                 'Blues', 'Reds', 'Greys'などが使用可能です。 Defaults to 'Greys'.

    Returns:
        tuple: (fig, ax) FigureオブジェクトとAxesオブジェクト。
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    if not pipe.pipe_paths:
        return fig, ax

    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"Warning: Colormap '{cmap_name}' not found. Using 'viridis' instead.")
        cmap = plt.get_cmap('viridis')

    num_segments = len(pipe.pipe_paths)

    # Sequentialカラーマップの薄い色を避けるため、色の範囲を調整
    color_start = 0.4
    color_end = 1.0

    def get_color(index, total):
        if total == 1:
            return cmap(color_start + (color_end - color_start) * 0.5)  # 1つの場合は中間色

        # 0から1の範囲を、color_startからcolor_endの範囲にマッピング
        normalized_index = index / (total - 1)
        color_val = color_start + normalized_index * (color_end - color_start)
        return cmap(color_val)

    # 最初のセグメント
    path0 = pipe.pipe_paths[0]
    color0 = get_color(0, num_segments)

    is_first_in_segment = True
    for conn in path0.node_connectivity:
        nodes = path0.node_positions[conn]
        label = f'Segment 1' if is_first_in_segment else None
        ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], color=color0, label=label)
        is_first_in_segment = False

    last_node_of_previous_segment = path0.node_positions[-1]

    # 2つ目以降のセグメント
    for i in range(1, num_segments):
        path = pipe.pipe_paths[i]
        color = get_color(i, num_segments)

        translation = last_node_of_previous_segment - path.node_positions[0]
        translated_nodes = path.node_positions + translation

        is_first_in_segment = True
        for conn in path.node_connectivity:
            nodes = translated_nodes[conn]
            label = f'Segment {i+1}' if is_first_in_segment else None
            ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], color=color, label=label)
            is_first_in_segment = False

        last_node_of_previous_segment = translated_nodes[-1]

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Pipe Geometry')
    if num_segments > 0:
        ax.legend()

    return fig, ax

def plot_multiple_mode_shapes(geometries, shapes_list, **kwargs):
    """
    複数のジオメトリにモード形状をインタラクティブにプロットします。
    
    sdynpyのMultipleShapePlotterを呼び出します。
    GUIウィンドウが立ち上がり、操作がブロックされる可能性があります。

    Args:
        geometries (list): sdynpyのGeometryオブジェクトのリスト。
        shapes_list (list): sdynpyのShapeArrayオブジェクトのリスト。
        **kwargs: MultipleShapePlotterに渡される追加のキーワード引数。
    """
    # QApplicationのインスタンスが存在するか確認し、なければ作成
    if QApplication.instance() is None:
        QApplication([])
        
    plotter = MultipleShapePlotter(geometries, shapes_list, **kwargs)
    plotter.show()
    return plotter

def plot_multiple_deflection_shapes(geometries, deflection_shape_data_list, **kwargs):
    """
    複数のジオメトリにたわみ形状をインタラクティブにプロットします。
    
    sdynpyのMultipleDeflectionShapePlotterを呼び出します。
    GUIウィンドウが立ち上がり、操作がブロックされる可能性があります。

    Args:
        geometries (list): sdynpyのGeometryオブジェクトのリスト。
        deflection_shape_data_list (list): NDDataArrayオブジェクトのリスト。
        **kwargs: MultipleDeflectionShapePlotterに渡される追加のキーワード引数。
    """
    # QApplicationのインスタンスが存在するか確認し、なければ作成
    if QApplication.instance() is None:
        QApplication([])
        
    plotter = MultipleDeflectionShapePlotter(geometries, deflection_shape_data_list, **kwargs)
    plotter.show()
    return plotter