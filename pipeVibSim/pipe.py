import numpy as np

class Pipe:
    """
    配管システム全体を管理するクラス。
    複数のPipePathとそれに対応する材料特性を保持し、結合することができます。
    """
    def __init__(self, pipe_path=None, material_properties=None):
        self.pipe_paths = []
        self.material_properties_list = []
        if pipe_path is not None and material_properties is not None:
            self.add_pipe_segment(pipe_path, material_properties)

    def add_pipe_segment(self, pipe_path, material_properties):
        """
        新しい配管セグメント（PipePathと材料特性）を追加します。
        """
        self.pipe_paths.append(pipe_path)
        self.material_properties_list.append(material_properties)
        self._combine_segments()

    def _combine_segments(self):
        """
        保持しているすべての配管セグメントを結合し、
        システム全体としてのジオメトリと材料特性を構築します。
        """
        if not self.pipe_paths:
            self.node_positions = np.array([])
            self.node_connectivity = np.array([])
            self.bend_direction = np.array([])
            self.material_properties = {}
            return

        # 複数のセグメントを結合するロジック
        if len(self.pipe_paths) > 1:
            all_node_positions = [self.pipe_paths[0].node_positions]
            all_node_connectivity = [self.pipe_paths[0].node_connectivity]
            all_bend_directions = [self.pipe_paths[0].bend_direction]
            
            # 材料特性のキーをすべて集める
            all_mat_keys = set()
            for props in self.material_properties_list:
                all_mat_keys.update(props.keys())

            # 結合された材料特性を準備
            combined_material_props = {key: [] for key in all_mat_keys}

            for i in range(len(self.material_properties_list)):
                props = self.material_properties_list[i]
                n_elements = self.pipe_paths[i].node_connectivity.shape[0]
                for key in all_mat_keys:
                    if key in props:
                        # スカラー値の場合は要素数分だけ繰り返す
                        if np.isscalar(props[key]):
                            combined_material_props[key].extend([props[key]] * n_elements)
                        else:
                            combined_material_props[key].extend(props[key])
                    else:
                        # もしプロパティが存在しない場合はデフォルト値（例：0やNaN）で埋めるか、エラーを出す
                        # ここではnp.nanで埋める
                        combined_material_props[key].extend([np.nan] * n_elements)


            node_offset = self.pipe_paths[0].node_positions.shape[0]

            for i in range(1, len(self.pipe_paths)):
                # 座標の結合（前のセグメントの終点と次のセグメントの始点を一致させる）
                prev_last_node = all_node_positions[-1][-1]
                current_first_node = self.pipe_paths[i].node_positions[0]
                translation = prev_last_node - current_first_node
                
                translated_nodes = self.pipe_paths[i].node_positions + translation
                all_node_positions.append(translated_nodes[1:]) # 始点は重複するので除く

                # 接続情報の結合
                connectivity = self.pipe_paths[i].node_connectivity + node_offset -1 # 1つ前の最後のnodeに接続
                all_node_connectivity.append(connectivity)

                # 曲げ方向の結合
                all_bend_directions.append(self.pipe_paths[i].bend_direction)
                
                node_offset += self.pipe_paths[i].node_positions.shape[0] -1

            self.node_positions = np.vstack(all_node_positions)
            self.node_connectivity = np.vstack(all_node_connectivity)
            self.bend_direction = np.vstack(all_bend_directions)
            self.material_properties = {key: np.array(val) for key, val in combined_material_props.items()}

        else: # セグメントが1つの場合
            self.node_positions = self.pipe_paths[0].node_positions
            self.node_connectivity = self.pipe_paths[0].node_connectivity
            self.bend_direction = self.pipe_paths[0].bend_direction
            self.material_properties = self.material_properties_list[0]

