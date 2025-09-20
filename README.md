# pipeVibSim

`pipeVibSim` は、複雑な3D配管系の振動解析を簡単に行うためのPythonライブラリです。
`sdynpy`ライブラリをバックエンドとして使用し、配管形状の定義から固有値解析、周波数応答解析（FRF）までをシームレスに実行します。

## 主な機能

- 仮想的な点群から滑らかな配管の3Dモデル（ノードと要素）を自動生成
- 材質や断面形状（円管）を簡単に設定
- 固有値解析によるモード形状と固有振動数の計算
- 指定した点における周波数応答（FRF）の計算と可視化

## インストール

リポジトリをクローンし、`pip` を使ってインストールします。

```bash
git clone https://github.com/your-username/pipeVibSim.git
cd pipeVibSim
pip install .
```

## 使い方

基本的な使い方の例です。詳細は `notebooks/example.ipynb` を参照してください。

```python
import numpy as np
from pipeVibSim.pipe import Pipe
from pipeVibSim.materials import get_material_properties
from pipeVibSim.simulation import VibrationAnalysis
import pipeVibSim.postprocessing as post

# 1. 配管形状の定義
# 仮想的な制御点
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 1]
], dtype=float) * 0.1

# 配管オブジェクトの作成
pipe = Pipe(points, radius=0.03, step=0.01)

# 2. 材料特性の設定
material_properties = get_material_properties(
    E=110.0e9,      # ヤング率 (Pa)
    rho=8960.0,     # 密度 (kg/m^3)
    nu=0.34,        # ポアソン比
    D_out=0.01,     # 外径 (m)
    D_in=0.008,     # 内径 (m)
    n_elements=pipe.node_positions.shape[0] - 1
)

# 3. 解析の実行
# 解析オブジェクトの作成
analysis = VibrationAnalysis(pipe, material_properties)

# 固有値解析
shapes = analysis.run_eigensolution(maximum_frequency=4000)
print(shapes)

# モード形状のプロット
post.plot_mode_shapes(analysis.geometry, shapes)

# 周波数応答解析
frequencies = np.linspace(0., 500, 1000)
frf = analysis.run_frf(frequencies, load_dof_indices=-4, response_dof_indices=-4)

# FRFのプロット
post.plot_frf(frf)
```
