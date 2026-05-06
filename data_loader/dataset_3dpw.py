import os
import pickle as pkl
from glob import glob

import numpy as np

from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


def load_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pkl.load(f, encoding="latin1")
    return data


class Dataset3DPW(Dataset):
    """
    3DPW loader for two-person, single-scene training.

    - Input files: <data_path>/sequenceFiles/{train|validation|test}/*.pkl
    - We keep only sequences that contain exactly 2 persons.
    - Joint layout: 24 SMPL joints per person, concatenated as [P1(24), P2(24)].
    """

    def __init__(
        self,
        mode,
        t_his=25,
        t_pred=100,
        actions="all",
        use_vel=False,
        data_path="/data/user/qkh/datasets/3DPW",
        scene_filter=None,
        require_two_person=True,
        use_data_aug=False,
        aug_rotate_prob=0.5,
        aug_reverse_prob=0.3,
    ):
        self.use_vel = use_vel
        self.data_path = data_path
        self.scene_filter = scene_filter
        self.require_two_person = require_two_person
        self.use_data_aug = use_data_aug and mode == "train"
        self.aug_rotate_prob = aug_rotate_prob
        self.aug_reverse_prob = aug_reverse_prob
        super().__init__(mode, t_his, t_pred, actions)

        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        # 24-joint SMPL kinematic tree.
        smpl_parents = [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
        ]
        smpl_links = [(j, p) for j, p in enumerate(smpl_parents) if p != -1]
        p2_shift = 24
        all_parents = smpl_parents + [p + p2_shift if p >= 0 else -1 for p in smpl_parents]
        all_links = smpl_links + [(a + p2_shift, b + p2_shift) for a, b in smpl_links]
        joints_left = [1, 4, 7, 10, 13, 16, 18, 20, 22, 1 + p2_shift, 4 + p2_shift, 7 + p2_shift, 10 + p2_shift, 13 + p2_shift, 16 + p2_shift, 18 + p2_shift, 20 + p2_shift, 22 + p2_shift]
        joints_right = [2, 5, 8, 11, 14, 17, 19, 21, 23, 2 + p2_shift, 5 + p2_shift, 8 + p2_shift, 11 + p2_shift, 14 + p2_shift, 17 + p2_shift, 19 + p2_shift, 21 + p2_shift, 23 + p2_shift]

        self.skeleton = Skeleton(
            parents=all_parents,
            joints_left=joints_left,
            joints_right=joints_right,
            links=all_links,
        )
        self.kept_joints = np.arange(48)
        self.removed_joints = set()
        self.total_joints = 48

        split_map = {"train": "train", "val": "validation", "test": "test"}
        if self.mode not in split_map:
            raise ValueError(f"Unsupported mode '{self.mode}' for 3DPW loader.")
        split_dir = os.path.join(self.data_path, "sequenceFiles", split_map[self.mode])
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"3DPW split folder not found: {split_dir}")

        if isinstance(self.scene_filter, dict):
            raw_filter = self.scene_filter.get(self.mode, None)
        else:
            raw_filter = self.scene_filter
        if raw_filter is None:
            scene_set = None
        elif isinstance(raw_filter, (list, tuple, set)):
            scene_set = {str(x).strip().lower() for x in raw_filter if str(x).strip()}
        else:
            scene_set = {str(raw_filter).strip().lower()}

        pkl_files = sorted(glob(os.path.join(split_dir, "*.pkl")))
        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No 3DPW sequence files found in: {split_dir}")

        self.data = {}
        for pkl_file in pkl_files:
            stem = os.path.splitext(os.path.basename(pkl_file))[0]
            scene_name = stem.split("_")[0].lower()
            if scene_set is not None and scene_name not in scene_set:
                continue

            seq_dict = load_pkl(pkl_file)
            joint_positions = seq_dict.get("jointPositions", None)
            if joint_positions is None or len(joint_positions) < 2:
                continue
            if self.require_two_person and len(joint_positions) != 2:
                continue

            p1 = np.asarray(joint_positions[0], dtype=np.float32).reshape(-1, 24, 3)
            p2 = np.asarray(joint_positions[1], dtype=np.float32).reshape(-1, 24, 3)
            n_frames = min(p1.shape[0], p2.shape[0])
            if n_frames <= self.t_total:
                continue
            p1 = p1[:n_frames]
            p2 = p2[:n_frames]
            seq = np.concatenate([p1, p2], axis=1)

            if self.use_vel:
                v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                v = np.append(v, v[[-1]], axis=0)

            seq[:, 1:] -= seq[:, :1]

            if self.use_vel:
                seq = np.concatenate((seq, v), axis=1)

            if scene_name not in self.data:
                self.data[scene_name] = {}
            self.data[scene_name][stem] = seq

        self.subjects = sorted(self.data.keys())
        if len(self.subjects) == 0:
            raise RuntimeError(
                f"No valid 3DPW sequences loaded for mode={self.mode}. "
                f"Check scene_filter={self.scene_filter} and two-person constraint."
            )

    def _apply_scene_rotation(self, sample):
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=sample.dtype,
        )
        return np.matmul(sample, rot.T)

    def _apply_sequence_reverse(self, sample):
        return sample[:, ::-1].copy()

    def augment_sample(self, sample):
        if np.random.uniform() < self.aug_rotate_prob:
            sample = self._apply_scene_rotation(sample)
        if np.random.uniform() < self.aug_reverse_prob:
            sample = self._apply_sequence_reverse(sample)
        return sample

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        for _ in range(num_samples // batch_size):
            sample = []
            for _ in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            if aug and self.use_data_aug:
                sample = self.augment_sample(sample)
            yield sample
