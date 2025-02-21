from dataclasses import dataclass
from typing import Dict, Tuple, Union

from acrl.utils.curvature import curvature_splines
from acrl.utils.raycast import Raycast2D
import numpy as np
from scipy.spatial import KDTree as KDTreeBase


class KDTree(KDTreeBase):
    """
    Adds some list-like properties
    """

    def __getitem__(self, index: Union[int, slice]) -> np.array:
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]


@dataclass
class Window:
    ahead: float
    behind: float


class LookAhead:
    def __init__(self, config: Dict):
        self._setup(config)

    def __call__(self, observation: Dict) -> Tuple[np.array, float, np.array]:
        location, yaw = self._get_ego_location(observation)
        index = self._get_index_of_closest_raceline_point(location)
        distance_to_raceline = self._calculate_distance_to_raceline(index, location)

        # Look ahead curvature
        indices = self._get_windowed_indices(index, self._curvature_window)
        indices = self._downsample(indices)
        local_curvature = np.ravel(self._curvatures[indices])

        # Track limits LiDAR
        indices = self._get_windowed_indices(index, self._LiDAR_window)
        distances_to_limits = self._LiDAR_readings(location, yaw, indices)

        return local_curvature, distance_to_raceline, distances_to_limits

    def _get_ego_location(self, observation: Dict) -> Tuple[np.array, float]:
        """
        This transform comes from the export of the raceline from .ai files by
            AC Gym plugin sensor_par.structures
        """
        x = observation["ego_location_z"]
        y = observation["ego_location_x"]
        return np.array([x, y]), observation["heading"]

    def _get_index_of_closest_raceline_point(self, point: np.array) -> int:
        _, index = self._raceline.query(point)
        return index

    def _calculate_distance_to_raceline(self, index: int, point: np.array) -> float:
        indices = np.arange(index - 1, index + 2)
        points = np.take(self._raceline, indices, axis=0, mode="wrap")
        behind, closest, ahead = points[0], points[1], points[2]
        distance_1 = distance_to_line(behind, closest, point)
        distance_2 = distance_to_line(closest, ahead, point)
        return min(distance_1, distance_2)

    def _get_windowed_indices(self, start_index: int, window: Window) -> np.array:
        """
        Retrieves the indices of points inside the given window
        """
        current_distance = self._cum_distance[start_index]
        start_distance = current_distance - window.behind
        end_distance = current_distance + window.ahead
        indices = self._get_segment_indices(start_distance, end_distance)
        if self._is_wrapping(end_distance):
            remaining = end_distance - self._track_length
            indices = np.hstack([indices, self._get_segment_indices(0.0, remaining)])
        return indices

    def _get_segment_indices(self, start: float, end: float) -> np.array:
        return np.nonzero((self._cum_distance >= start) & (self._cum_distance <= end))[
            0
        ]

    def _is_wrapping(self, end_distance: float) -> bool:
        return self._track_length < end_distance

    def _downsample(self, indices: np.array) -> np.array:
        sampling_interval = len(indices) // self._n_curvature_points
        return indices[0::sampling_interval][0 : self._n_curvature_points]

    def _LiDAR_readings(
        self,
        location: np.array,
        yaw: float,
        indices: np.array,
    ) -> np.array:
        points = np.tile(location, (self._n_LiDAR_rays, 1))
        ray_angles = self._ray_angles.copy() - yaw
        local_left_limits = self._left_limits[indices]
        local_right_limits = self._right_limits[indices]
        local_limits = np.vstack([local_right_limits, local_left_limits])
        distance = self._raycaster.distance_to_walls(points, ray_angles, local_limits)
        return distance

    def _setup(self, config: Dict):
        self._config = config
        self._track_path = config["track_path"]
        self._setup_track()
        self._setup_curvature()
        self._setup_cum_distance()
        self._setup_limits_LiDAR()

    def _setup_track(self):
        track = np.load(self._track_path, allow_pickle=True).item()
        self._raceline = KDTree(track["raceline"])
        self._left_limits = points_to_segments(track["left_track"])
        self._right_limits = points_to_segments(track["right_track"])

    def _setup_cum_distance(self):
        x_diff = np.diff(self._raceline[:, 0])
        y_diff = np.diff(self._raceline[:, 1])
        cum_distance = np.cumsum(np.hypot(x_diff, y_diff))
        self._cum_distance = np.insert(cum_distance, 0, 0)
        self._track_length = self._cum_distance[-1]

    def _setup_curvature(self):
        look_ahead_distance = self._config["curvature"]["distance_m"]
        self._n_curvature_points = self._config["curvature"]["n_points"]
        curvatures = curvature_splines(self._raceline[:, 0], self._raceline[:, 1])
        self._curvatures = curvatures
        self._curvature_window = Window(ahead=look_ahead_distance, behind=0.0)

    def _setup_limits_LiDAR(self):
        LiDAR_distance = self._config["limits_LiDAR"]["distance_m"]
        self._LiDAR_distance = LiDAR_distance
        self._LiDAR_window = Window(ahead=LiDAR_distance, behind=LiDAR_distance)
        n_rays = self._config["limits_LiDAR"]["n_rays"]
        self._n_LiDAR_rays = n_rays
        self._ray_angles = np.linspace(-np.pi / 2, np.pi / 2, num=n_rays, endpoint=True)
        self._raycaster = Raycast2D(LiDAR_distance)


def distance_to_line(
    line_point_1: np.array,
    line_point_2: np.array,
    point: np.array,
) -> float:
    line = line_point_2 - line_point_1
    return np.cross(line, line_point_1 - point) / np.linalg.norm(line)


def points_to_segments(points: np.array) -> np.array:
    x, y = points[:, 0], points[:, 1]
    segments = np.vstack((x[:-1], y[:-1], x[1:], y[1:])).T
    final_segment = np.vstack((x[-1], y[-1], x[0], y[0])).T
    return np.vstack([segments, final_segment])
