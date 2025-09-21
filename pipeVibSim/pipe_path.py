import numpy as np

class PipePath:
    """
    配管の形状を定義し、計算に必要なジオメトリ情報を生成するクラス。

    Args:
        points (np.ndarray): 配管の経路を定義する3D座標点の配列。
        radius (float or np.ndarray): 配管の曲げ半径。スカラーまたは角ごとの半径の配列。
        step (float): 配管の離散化ステップサイズ。
    """
    def __init__(self, points, radius, step):
        self.points = points
        
        num_corners = len(points) - 2
        if num_corners > 0:
            if np.isscalar(radius):
                self.radius = np.full(num_corners, float(radius))
            else:
                self.radius = np.asarray(radius)
                if self.radius.shape != (num_corners,):
                    raise ValueError(f"The length of the radius array must be equal to the number of corners ({num_corners}).")
        else:
            self.radius = np.array([])

        self.step = step
        self.node_positions, self.curvatures = self._create_node_path()
        self.node_connectivity = self._get_node_connectivity()
        self.bend_direction = self._get_bend_direction()

    def _rotation_matrix(self, axis, theta):
        """ロドリゲスの回転公式を用いて回転行列を計算します。"""
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        return np.array([
            [a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
            [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
            [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]
        ])

    def _fillet_3d(self, p0, p1, p2, radius, step=0.1):
        """3D空間で2つのセグメント間にフィレット（円弧）を作成するヘルパー関数。"""
        v1 = p0 - p1
        v2 = p2 - p1
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        axis = np.cross(v1, v2)
        axis /= np.linalg.norm(axis)
        tan_len = radius / np.tan(angle / 2)
        pt1 = p1 + v1 * tan_len
        pt2 = p1 + v2 * tan_len
        bisector = (v1 + v2)
        bisector /= np.linalg.norm(bisector)
        center = p1 + bisector * (radius / np.sin(angle / 2))
        start_vec = pt1 - center
        end_vec = pt2 - center
        arc_angle = np.arccos(np.clip(np.dot(start_vec/np.linalg.norm(start_vec), end_vec/np.linalg.norm(end_vec)), -1.0, 1.0))
        n_steps = max(2, int(arc_angle * radius / step))
        arc_points = []
        for i in range(n_steps + 1):
            theta = arc_angle * i / n_steps
            rot_matrix = self._rotation_matrix(axis, theta)
            arc_vec = np.dot(rot_matrix, start_vec)
            arc_points.append(center + arc_vec)
        return np.array(arc_points), pt1, pt2

    def _create_node_path(self):
        """完全なノードパス（節点座標）と各要素の曲率を構築します。"""
        if len(self.points) < 2:
            return np.array(self.points), np.array([])

        node_positions = [self.points[0]]
        curvatures = []

        for i in range(1, len(self.points) - 1):
            p_prev = self.points[i - 1]
            p_curr = self.points[i]
            p_next = self.points[i + 1]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 < 1e-9 or norm_v2 < 1e-9:
                continue

            v1_norm = v1 / norm_v1
            v2_norm = v2 / norm_v2

            dot_product = np.dot(v1_norm, v2_norm)

            line_vec_sq_norm = np.sum((p_next - p_prev)**2)
            is_collinear = False
            if line_vec_sq_norm > 1e-12:
                cross_product = np.cross(p_next - p_prev, p_curr - p_prev)
                dist_sq = np.dot(cross_product, cross_product) / line_vec_sq_norm
                if dist_sq < 1e-12:
                    is_collinear = True

            current_radius = self.radius[i-1]

            if np.isclose(dot_product, -1.0) and not is_collinear:
                axis = np.cross(v1_norm, v2_norm)
                if np.linalg.norm(axis) < 1e-9:
                    if abs(v1_norm[0]) < 0.9:
                        axis = np.cross(v1_norm, np.array([1.0, 0.0, 0.0]))
                    else:
                        axis = np.cross(v1_norm, np.array([0.0, 1.0, 0.0]))
                axis /= np.linalg.norm(axis)

                pt1 = p_curr
                seg = np.linspace(node_positions[-1], pt1, int(np.linalg.norm(pt1 - node_positions[-1]) / self.step) + 1)[1:]
                node_positions.extend(seg)
                curvatures.extend([0.0] * len(seg))

                center_dir = np.cross(axis, v1_norm)
                center = p_curr + center_dir * current_radius

                n_steps = max(2, int(np.pi * current_radius / self.step))
                arc_points = []
                start_vec = p_curr - center
                for j in range(n_steps + 1):
                    theta = np.pi * j / n_steps
                    rot_matrix = self._rotation_matrix(axis, theta)
                    arc_vec = np.dot(rot_matrix, start_vec)
                    arc_points.append(center + arc_vec)
                
                node_positions.extend(arc_points[1:])
                curvatures.extend([1.0 / current_radius] * (len(arc_points) - 1))
            
            elif np.isclose(dot_product, 1.0) or is_collinear:
                seg = np.linspace(node_positions[-1], p_curr, int(np.linalg.norm(p_curr - node_positions[-1]) / self.step) + 1)[1:]
                node_positions.extend(seg)
                curvatures.extend([0.0] * len(seg))

            else:
                arc, pt1, pt2 = self._fillet_3d(p_prev, p_curr, p_next, current_radius, self.step)
                seg = np.linspace(node_positions[-1], pt1, int(np.linalg.norm(pt1 - node_positions[-1]) / self.step) + 1)[1:]
                node_positions.extend(seg)
                curvatures.extend([0.0] * len(seg))
                
                node_positions.extend(arc[1:])
                curvatures.extend([1.0 / current_radius] * (len(arc) - 1))
                
                node_positions[-1] = pt2

        seg = np.linspace(node_positions[-1], self.points[-1], int(np.linalg.norm(self.points[-1] - node_positions[-1]) / self.step) + 1)[1:]
        node_positions.extend(seg)
        curvatures.extend([0.0] * len(seg))
        return np.array(node_positions), np.array(curvatures)

    def _get_node_connectivity(self):
        """節点接続情報を作成します。"""
        num_nodes = self.node_positions.shape[0]
        return np.array((np.arange(num_nodes - 1), np.arange(1, num_nodes))).T

    def _get_bend_direction(self):
        """各要素の接線ベクトルから法線方向（ローカルz軸）を決定します。"""
        bend_direction_1 = []
        for i in range(self.node_connectivity.shape[0]):
            start, end = self.node_connectivity[i]
            tangent = self.node_positions[end] - self.node_positions[start]
            tangent /= np.linalg.norm(tangent)
            if i == 0:
                bend_dir = np.array([0, 0, 1])
            else:
                prev_start, prev_end = self.node_connectivity[i-1]
                prev_tangent = self.node_positions[prev_end] - self.node_positions[prev_start]
                prev_tangent /= np.linalg.norm(prev_tangent)
                bend_dir = np.cross(prev_tangent, tangent)
                if np.linalg.norm(bend_dir) < 1e-8:
                    bend_dir = bend_direction_1[-1]
                else:
                    bend_dir /= np.linalg.norm(bend_dir)
            bend_direction_1.append(bend_dir)
        return np.array(bend_direction_1)

    def __add__(self, other):
        """
        2つのPipePathオブジェクトを結合します。
        2つ目のPipePathの始点を1つ目の終点にオフセットして結合します。
        """
        if not isinstance(other, PipePath):
            return NotImplemented

        new_step = self.step

        offset = self.points[-1] - other.points[0]
        offset_other_points = other.points + offset

        new_points = np.vstack((self.points, offset_other_points[1:]))

        self_radii = self.radius
        other_radii = other.radius

        if len(self_radii) > 0:
            joint_radius = self_radii[-1]
        elif len(other_radii) > 0:
            joint_radius = other_radii[0]
        else:
            joint_radius = 0.1

        new_radius = np.concatenate((self_radii, [joint_radius], other_radii))

        return PipePath(new_points, new_radius, new_step)
