from typing import Dict, Tuple

from acrl.utils.curvature import curvature_splines
import numpy as np
from scipy.spatial import KDTree as KDTreeBase


class KDTree(KDTreeBase):
    """
    Adds some list-like properties
    """

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]


class LookAhead:
    def __init__(self, config: Dict):
        self._setup(config)

    def __call__(self, observation: Dict) -> Tuple[np.array, float]:
        location = self._get_ego_location(observation)
        index = self._get_index_of_closest_raceline_point(location)
        distance_to_raceline = self._calculate_distance_to_raceline(index, location)
        indices = self._get_look_ahead_indices(index)
        indices = self._downsample(indices)
        return np.ravel(self._curvatures[indices]), distance_to_raceline

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

    def _get_look_ahead_indices(self, start_index: int) -> np.array:
        start_distance = self._cum_distance[start_index]
        end_distance = start_distance + self._look_ahead_distance
        indices = self._get_segment_indices(start_distance, end_distance)
        if self._is_wrapping(end_distance):
            remaining = end_distance - self._track_length
            indices = np.hstack([indices, self._get_segment_indices(0, remaining)])
        return indices

    def _get_segment_indices(self, start: float, end: float) -> np.array:
        return np.where((self._cum_distance >= start) & (self._cum_distance <= end))[0]

    def _is_wrapping(self, end_distance: float) -> bool:
        return self._track_length < end_distance

    def _downsample(self, indices: np.array) -> np.array:
        sampling_interval = len(indices) // self._n_curvature_points
        return indices[0::sampling_interval][0 : self._n_curvature_points]

    def _get_ego_location(self, observation: Dict) -> np.array:
        # This transform comes from the export of the raceline from .ai files by
        #   AC Gym plugin sensor_par.structures
        x = observation["ego_location_z"]
        y = observation["ego_location_x"]
        return np.array([x, y])

    def _setup(self, config: Dict):
        self._config = config
        self._look_ahead_distance = config["distance_m"]
        self._raceline_path = config["raceline_path"]
        self._n_curvature_points = config["n_points"]
        self._setup_raceline()
        self._setup_curvature()
        self._setup_cum_distance()

    def _setup_raceline(self):
        raceline = np.genfromtxt(self._raceline_path, delimiter=",")[1:]
        self._raceline = KDTree(raceline)

    def _setup_cum_distance(self):
        x_diff = np.diff(self._raceline[:, 0])
        y_diff = np.diff(self._raceline[:, 1])
        cum_distance = np.cumsum(np.hypot(x_diff, y_diff))
        self._cum_distance = np.insert(cum_distance, 0, 0)
        self._track_length = self._cum_distance[-1]

    def _setup_curvature(self):
        curvatures = curvature_splines(self._raceline[:, 0], self._raceline[:, 1])
        self._curvatures = curvatures


def distance_to_line(
    line_point_1: np.array,
    line_point_2: np.array,
    point: np.array,
) -> float:
    line = line_point_2 - line_point_1
    return np.cross(line, line_point_1 - point) / np.linalg.norm(line)
