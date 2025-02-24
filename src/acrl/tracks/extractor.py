from argparse import ArgumentParser, Namespace
from io import BufferedReader
from pathlib import Path
import struct
from typing import Dict, Generator, Tuple

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt

INVALID_TRACKS = {
    "silverstone-international",
    "drift",
}


class InvalidTrackFile(Exception):
    pass


def parse_arguments() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--content-path", type=str, required=True)
    arg_parser.add_argument("--output-path", type=str, required=True)
    return arg_parser.parse_args()


def main():
    args = parse_arguments()
    input_path = Path(args.content_path)
    output_path = Path(args.output_path)
    file_paths = get_track_files(input_path)
    extract_tracks(output_path, file_paths)


def get_track_files(content_path: str) -> Generator[Path, None, None]:
    tracks_path = Path(content_path).joinpath("tracks")
    return tracks_path.rglob("fast_lane.ai")


def extract_tracks(output_folder: Path, path_glob: Generator[Path, None, None]):
    for track_path in path_glob:
        output_name = get_output_filename(track_path)
        if output_name in INVALID_TRACKS:
            continue
        output_path = output_folder.joinpath(output_name)
        tracks = extract_track(track_path)
        write_track_file(tracks, output_path)
        plot_saved_track(output_path)


def get_output_filename(track_path: Path) -> str:
    track_name = track_path.parents[2].name
    layout_name = track_path.parents[1].name
    if track_name == "tracks":
        output_name = layout_name
    else:
        output_name = f"{track_name}-{layout_name}"
    return output_name.replace("ks_", "")


def extract_track(track_path: Path) -> Dict:
    limits, raceline = read_track_data(track_path)
    return format_tracks(limits, raceline)


def read_track_data(track_path: Path) -> Tuple[np.array, np.array]:
    with track_path.open("rb") as file:
        header = file.read(4 * 4)
        _, n_values, _, _ = struct.unpack("4i", header)
        raceline = unpack_raceline_buffer(file, n_values)
        limits = unpack_limits_buffer(file, n_values)
    return limits, raceline


def unpack_raceline_buffer(file: BufferedReader, n_values: int) -> np.array:
    # Raceline buffer unpack
    # x: float, y: float, z: float, distance: float, index: int
    np_dtype = np.dtype([("floats", np.float32, (4,)), ("ints", np.int32)])
    raceline = np.frombuffer(file.read(4 * 5 * n_values), dtype=np_dtype)
    raceline = raceline["floats"].reshape(-1, 4)
    # Raceline 3D coordinates: x -> y, z -> x
    return np.vstack([raceline[:, 2], raceline[:, 0]]).T


def unpack_limits_buffer(file: BufferedReader, n_values: int) -> np.array:
    # Track details buffer unpack (All floats)
    # _, speed, throttle, brake, direction, right, left
    limits = np.frombuffer(file.read(4 * 18 * n_values), dtype=np.float32)
    limits = limits.reshape((-1, 18))
    # Right limit, Left limit
    return np.vstack([limits[:, 6], limits[:, 7]]).T


def format_tracks(limits: np.array, raceline: np.array) -> Dict:
    angles = get_angle_between_points(raceline)
    left = get_left_limit_coordinates(angles, limits[:, 1], raceline)
    right = get_right_limit_coordinates(angles, limits[:, 0], raceline)
    return {"raceline": raceline, "left_track": left, "right_track": right}


def get_angle_between_points(raceline: np.array) -> np.array:
    x, y = raceline[:, 0], raceline[:, 1]
    x_diff, y_diff = x[:-1] - x[1:], y[1:] - y[:-1]
    final_x, final_y = x[-1] - x[0], y[0] - y[-1]
    x_diff, y_diff = np.hstack([final_x, x_diff]), np.hstack([final_y, y_diff])
    return np.atan2(x_diff, y_diff)


def get_left_limit_coordinates(
    angles: np.array,
    left_limits: np.array,
    raceline: np.array,
) -> np.array:
    delta_y = np.sin(angles) * left_limits
    delta_x = np.cos(angles) * left_limits
    deltas = np.vstack([delta_x, delta_y]).T
    return raceline + deltas


def get_right_limit_coordinates(
    angles: np.array,
    right_limits: np.array,
    raceline: np.array,
) -> np.array:
    delta_y = -np.sin(angles) * right_limits
    delta_x = -np.cos(angles) * right_limits
    deltas = np.vstack([delta_x, delta_y]).T
    return raceline + deltas


def write_track_file(tracks: Dict, output_path: Path):
    with output_path.with_suffix(".npy").open("wb") as file:
        np.save(file, tracks, allow_pickle=True)


def plot_saved_track(output_path: Path):
    logger.info(f"Plot of: {output_path.stem}")
    with output_path.with_suffix(".npy").open("rb") as file:
        tracks = np.load(file, allow_pickle=True).item()
    plot_tracks(tracks)


def plot_tracks(tracks: Dict):
    plt.scatter(tracks["raceline"][:, 0], tracks["raceline"][:, 1], c="r")
    plt.scatter(tracks["left_track"][:, 0], tracks["left_track"][:, 1], c="b")
    plt.scatter(tracks["right_track"][:, 0], tracks["right_track"][:, 1], c="g")
    plt.show()


if __name__ == "__main__":
    main()
