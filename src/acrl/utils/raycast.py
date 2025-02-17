import numpy as np
import matplotlib.pyplot as plt


class Raycast2D:
    def __init__(self, ray_length: float):
        self._max_distance = ray_length

    def distance_to_walls(
        self,
        origins: np.array,
        ray_angles: np.array,
        walls: np.array,
    ) -> np.array:
        """
        origins: shape (n, 2) -> [(x1, y1), (x2, y2), ...]
        ray_angles: shape (m,) -> [angle1, angle2, ...]
        walls: shape (k, 4) -> [(x1, y1, x2, y2), ...]

        Return:
        distances: (n, 1) -> [distance, ...]
        """
        x1, y1, x2, y2 = walls.T
        x2_x1 = x2 - x1
        y2_y1 = y2 - y1

        x, y = origins.T
        cos_angles = np.cos(ray_angles)[:, None]
        sin_angles = np.sin(ray_angles)[:, None]

        det_A = cos_angles * y2_y1 - sin_angles * x2_x1
        det_A[det_A == 0] = 1e-8

        valid_indices = np.where(det_A != 0)
        if valid_indices[0].size == 0:
            return origins, np.ones(len(origins)) * self._max_distance

        x_x1 = x1 - x[:, None]
        y_y1 = y1 - y[:, None]

        t = (x_x1 * y2_y1 - y_y1 * x2_x1) / det_A
        u = (x_x1 * sin_angles - y_y1 * cos_angles) / det_A

        intersections = np.logical_and(u >= 0, np.logical_and(u <= 1, t >= 0))
        if not intersections.any():
            return np.ones(len(origins)) * self._max_distance

        t[~intersections] = self._max_distance

        min_distance = np.min(t, axis=-1)

        intersections = np.stack(
            [
                x + cos_angles.squeeze() * min_distance,
                y + sin_angles.squeeze() * min_distance,
            ],
            axis=-1,
        )
        vector = intersections - origins
        distances = np.linalg.norm(vector, axis=1)
        return distances


def plot_LiDAR(
    left_limits,
    right_limits,
    local_limits,
    points,
    intersections,
    number_of_rays,
):
    # Plot the walls
    plt.figure(figsize=(8, 8))

    x1, y1, x2, y2 = left_limits.T
    plt.plot([x1, x2], [y1, y2], color="blue")
    x1, y1, x2, y2 = right_limits.T
    plt.plot([x1, x2], [y1, y2], color="red")

    x1, y1, x2, y2 = local_limits.T
    plt.plot([x1, x2], [y1, y2], "k-", color="orange")

    # Plot the rays
    for i, (point, intersection) in enumerate(zip(points, intersections)):
        if i == int(number_of_rays / 2):
            color = "blue"
        else:
            color = "green"
        plt.plot(*intersection, ".", markersize=2.0, color="blue")
        plt.plot([point[0], intersection[0]], [point[1], intersection[1]], color)

    # Plot the starting points of the rays
    plt.plot(points[:, 0], points[:, 1], "bo", markersize=1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
