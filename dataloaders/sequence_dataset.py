import torch
import numpy as np
from tqdm import tqdm
import enum
from typing import Union
from pointclouds import to_fixed_array


class OriginMode(enum.Enum):
    FIRST_ENTRY = 0
    LAST_ENTRY = 1


class SubsequenceDataset(torch.utils.data.Dataset):

    def __init__(self,
                 sequence_loader,
                 subsequence_length: int,
                 max_sequence_length: int,
                 origin_mode: Union[OriginMode, str],
                 max_pc_points: int = 90000,
                 shuffle=False):
        self.sequence_loader = sequence_loader
        assert subsequence_length > 0, f"subsequence_length must be > 0, got {subsequence_length}"
        assert max_sequence_length > 0, f"max_sequence_length must be > 0, got {max_sequence_length}"

        self.subsequence_length = subsequence_length
        self.max_sequence_length = max_sequence_length

        if isinstance(origin_mode, str):
            origin_mode = OriginMode[origin_mode]
        self.origin_mode = origin_mode
        self.max_pc_points = max_pc_points

        self.subsequence_id_index = []
        for id in tqdm(sequence_loader.get_sequence_ids(),
                       desc="Preprocessing sequences",
                       leave=False):
            # sequence = sequence_loader.load_sequence(id)
            num_subsequences = max_sequence_length // subsequence_length
            assert num_subsequences > 0, f"num_subsequences must be > 0, got {num_subsequences}"
            self.subsequence_id_index.extend([
                (id, i) for i in range(num_subsequences)
            ])

        self.subsequence_id_shuffled_index = list(
            range(len(self.subsequence_id_index)))
        if shuffle:
            random_state = np.random.RandomState(len(
                self.subsequence_id_index))
            random_state.shuffle(self.subsequence_id_shuffled_index)

    def __len__(self):
        return len(self.subsequence_id_index)

    def _get_subsequence(self, index):
        shuffled_index = self.subsequence_id_shuffled_index[index]
        id, subsequence_index = self.subsequence_id_index[shuffled_index]
        sequence = self.sequence_loader.load_sequence(id)

        offset = subsequence_index * self.subsequence_length
        if self.origin_mode == OriginMode.FIRST_ENTRY:
            origin_idx = offset
        elif self.origin_mode == OriginMode.LAST_ENTRY:
            origin_idx = offset + self.subsequence_length - 1
        else:
            raise ValueError(f"Unknown origin mode {self.origin_mode}")

        assert origin_idx >= 0 and origin_idx < len(
            sequence
        ), f"origin_idx must be >= 0 and < len(sequence), got {origin_idx} and {len(sequence)}"
        assert offset >= 0 and offset + self.subsequence_length <= len(
            sequence
        ), f"offset must be >= 0 and offset + self.subsequence_length <= len(sequence), got {offset} and {len(sequence)} for max sequence len {self.max_sequence_length} and a subsequence length {self.subsequence_length}"
        subsequence_lst = [
            sequence.load(offset + i, origin_idx)
            for i in range(self.subsequence_length)
        ]
        return subsequence_lst

    def __getitem__(self, index):

        subsequence_lst = self._get_subsequence(index)

        pc_arrays = [
            pc.to_fixed_array(self.max_pc_points) for pc, _ in subsequence_lst
        ]
        pose_arrays = [pose.to_array() for _, pose in subsequence_lst]
        pc_array_stack = np.stack(pc_arrays, axis=0).astype(np.float32)
        pose_array_stack = np.stack(pose_arrays, axis=0).astype(np.float32)

        return {
            "pc_array_stack": pc_array_stack,
            "pose_array_stack": pose_array_stack,
            "data_index": index
        }


class SubsequenceFlowDataset(SubsequenceDataset):

    def __getitem__(self, index):
        subsequence_lst = self._get_subsequence(index)

        pc_arrays = [
            pc.to_fixed_array(self.max_pc_points)
            for pc, _, _, _, _ in subsequence_lst
        ]
        pose_arrays = [pose.to_array() for _, pose, _, _, _ in subsequence_lst]
        flowed_pc_arrays = [
            flowed_pc.to_fixed_array(self.max_pc_points)
            for _, _, flowed_pc, _, _ in subsequence_lst
        ]
        pc_class_masks = [
            to_fixed_array(class_mask, self.max_pc_points)
            for _, _, _, class_mask, _ in subsequence_lst
        ]

        pc_array_stack = np.stack(pc_arrays, axis=0).astype(np.float32)
        pose_array_stack = np.stack(pose_arrays, axis=0).astype(np.float32)
        flowed_pc_array_stack = np.stack(flowed_pc_arrays,
                                         axis=0).astype(np.float32)
        pc_class_mask_stack = np.stack(pc_class_masks,
                                       axis=0).astype(np.float32)

        return {
            "pc_array_stack": pc_array_stack,
            "pose_array_stack": pose_array_stack,
            "flowed_pc_array_stack": flowed_pc_array_stack,
            "pc_class_mask_stack": pc_class_mask_stack,
            "data_index": index
        }