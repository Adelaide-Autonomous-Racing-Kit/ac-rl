import numpy as np

MAX_RAY_LEN = 200


class LimitsLIDAR:
    def __init__(self):
        pass

    def get_intersections_and_distances_to_wall(self, x, y, angle):
        """
        point: shape (2,) -> (x, y)
        angle: shape (1,) -> angle in radians

        Note: ~155us for 10 rays on a laptop
        """
        # number_of_rays must be odd
        assert self.number_of_rays % 2 != 0, "number of rays must be odd"

        scene = self.scene
        point = np.array([x, y])

        # Create an array of angles covering 360 degrees
        angles = (
            np.linspace(0, np.pi, num=self.number_of_rays, endpoint=True)
            + angle
            - np.pi / 2
        )

        # Tile the point to match the size of the angles array
        points = np.tile(point, (len(angles), 1))

        rectangle_width = MAX_RAY_LEN * 2
        filtered_walls, rectangle = scene.get_filtered_walls_in_rectangle(
            point, rectangle_width
        )

        if len(angles) * len(filtered_walls) > 2000:
            if not self.warning_shown:
                self.warning_shown = True
                print(
                    f"warning will calculate {len(angles) * len(filtered_walls)} rays. (> 10000 points is too much. 2000 is recommended)"
                )

        intersections, vector_lengths = self.ray_casting_vectorized_v2(
            points,
            angles,
            filtered_walls,
            MAX_RAY_LEN,
        )
        return intersections, vector_lengths, filtered_walls, points, rectangle

    def ray_casting_vectorized_v2(points, angles, walls, max_distance):
        """
        points: shape (n, 2) -> [(x1, y1), (x2, y2), ...]
        angles: shape (m,) -> [angle1, angle2, ...]
        walls: shape (k, 4) -> [(x1, y1, x2, y2), ...]
        max_distance: float
        """
        walls = np.array(walls, dtype=float)
        x1, y1, x2, y2 = walls.T
        x2_x1 = x2 - x1
        y2_y1 = y2 - y1

        x, y = points.T
        cos_angles = np.cos(angles)[:, None]
        sin_angles = np.sin(angles)[:, None]

        det_A = cos_angles * y2_y1 - sin_angles * x2_x1
        det_A[det_A == 0] = 1e-8  # Set zero elements to a small tolerance value

        valid_indices = np.where(det_A != 0)
        if valid_indices[0].size == 0:
            return points, np.ones(len(points)) * max_distance

        x_x1 = x1 - x[:, None]
        y_y1 = y1 - y[:, None]

        t = (x_x1 * y2_y1 - y_y1 * x2_x1) / det_A
        u = (x_x1 * sin_angles - y_y1 * cos_angles) / det_A

        intersections = np.logical_and(u >= 0, np.logical_and(u <= 1, t >= 0))
        if not intersections.any():
            return points, np.ones(len(points)) * max_distance

        # Set t to inf for rays that do not intersect any wall
        t[~intersections] = max_distance  # float('inf')

        min_distance = np.min(t, axis=-1)

        intersections = np.stack(
            [
                x + cos_angles.squeeze() * min_distance,
                y + sin_angles.squeeze() * min_distance,
            ],
            axis=-1,
        )

        vector = intersections - points
        vector_lengths = np.linalg.norm(vector, axis=1)
        return intersections, vector_lengths
