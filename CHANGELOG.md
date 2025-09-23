# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-09-23

### Added
- モード重ね合わせ法による周波数応答解析 (`run_frf_modal`) を追加。
- `VibrationAnalysis`クラスに固有値解析結果を保持する機能を追加。
- 複数のジオメトリや変形形状を同時にプロットする機能 (`post.plot_multiple_mode_shapes`, `post.plot_multiple_deflection_shapes`) を追加。
- パイプの曲率に応じた肉厚の変化を考慮した解析機能を追加。
- 180度曲げを含む複雑な配管形状の作成に対応。
- 複数の配管セグメントを結合する機能 (`Pipe.add_pipe_segment`, `PipePath.__add__`) を追加。

### Changed
- `VibrationAnalysis.run_frf` を `run_frf_direct` に名称変更し、直接法によるFRF計算専用とした。
- `get_material_properties` の戻り値のキーをスネークケース (`young_modulus` 等) に統一。
- `post.plot_node_path` のプロット機能を改善。

### Fixed
- `PipePath` のUベンドや3Dベンドにおける形状生成の不具合を修正。
- `materials.py` と `simulation.py` 間の材料特性のキーの不整合を修正。
- テストコードにおける `einsum` の次元不整合エラーを修正。
