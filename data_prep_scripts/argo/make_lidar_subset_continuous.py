import argparse
from pathlib import Path
import json
from bucketed_scene_flow_eval.utils import load_json

def make_data_sequence_folder(
    input_data_sequence_folder: Path, data_feathers: list[Path], output_data_sequence_folder: Path
):
    # Ensure output folder exists
    output_data_sequence_folder.mkdir(parents=True, exist_ok=True)

    symlink_objects = ["annotations.feather", "calibration", "city_SE3_egovehicle.feather", "map"]
    for item in symlink_objects:
        src = input_data_sequence_folder / item
        dst = output_data_sequence_folder / item
        if src.is_dir():
            dst.symlink_to(src, target_is_directory=True)
        elif src.is_file():
            dst.symlink_to(src)

    data_feathers_parent = output_data_sequence_folder / "sensors/lidar"
    data_feathers_parent.mkdir(parents=True, exist_ok=True)
    for data_feather in data_feathers:
        dst = data_feathers_parent / data_feather.name
        dst.symlink_to(data_feather)


def process_sequence(
    input_data_sequence_folder: Path,
    sequence_length: int,
    output_data_sequence_parent_folder: Path,
):

    data_feather_files = sorted(
        (input_data_sequence_folder / "sensors" / "lidar").glob("*.feather")
    )

    num_chunks = len(data_feather_files) // sequence_length

    # Iterate over each chunk
    for chunk_idx in range(num_chunks):
        chunk_data_feather_files = data_feather_files[
            chunk_idx * sequence_length : (chunk_idx + 1) * sequence_length
        ]

        chunk_sequence_name = (
            input_data_sequence_folder.name
        )

        chunk_output_data_sequence_folder = output_data_sequence_parent_folder / chunk_sequence_name

        make_data_sequence_folder(
            input_data_sequence_folder, chunk_data_feather_files, chunk_output_data_sequence_folder
        )

def process_sequence_2(
    input_data_sequence_folder: Path,
    sequence_length: int,
    output_data_sequence_parent_folder: Path,
):

    data_feather_files = sorted(
        (input_data_sequence_folder / "sensors" / "lidar").glob("*.feather")
    )


    chunk_data_feather_files = data_feather_files[3:]

    chunk_sequence_name = (
        input_data_sequence_folder.name
    )

    chunk_output_data_sequence_folder = output_data_sequence_parent_folder / chunk_sequence_name

    make_data_sequence_folder(
        input_data_sequence_folder, chunk_data_feather_files, chunk_output_data_sequence_folder
    )


def main(input_data: Path, sequence_length: int, output_root_folder: Path, seq_name_path: Path):
    sequence_id_to_length = load_json(seq_name_path)

    input_data_subfolders = [
        x for x in input_data.iterdir() 
        if x.is_dir() and x.name in sequence_id_to_length
    ]

    output_data = output_root_folder / input_data.name

    for input_data_sequence_folder in input_data_subfolders:
        process_sequence_2(
            input_data_sequence_folder,
            sequence_length,
            output_data,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process data sequences.")

    parser.add_argument("input_data", type=Path, help="Path to the input data folder.")
    parser.add_argument("sequence_length", type=int, help="Length of the sequences.")
    parser.add_argument("output_root_folder", type=Path, help="Path to the output root folder.")
    parser.add_argument("seq_name_path", type=Path, help="Path to the json file containing the sequence names.")

    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.input_data, args.sequence_length, args.output_root_folder, args.seq_name_path)

            # "/bigdata/argoverse_lidar/train",
            # "160",
            # "/bigdata/argoverse_lidar/train_3X",
            # "/bigdata/euler_filtered_sequence_length_3X.json"

            # "/efs/argoverse2/val",
            # "160",
            # "/efs/argoverse2/val_minus_3",
            # "/efs/argoverse2/val_sceneflow_feather_counts.json"