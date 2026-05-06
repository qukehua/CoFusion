import os
from glob import glob

import numpy as np

from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


class DatasetCMUMocap(Dataset):
    """
    CMU mocap loader with synthetic two-person pairing.

    Raw CMU files are single-person clips in txt format. To satisfy the
    two-character setting, we synthesize Person_2 from Person_1 via mirror +
    rotation + translation.
    """

    def __init__(
        self,
        mode,
        t_his=25,
        t_pred=100,
        actions="all",
        use_vel=False,
        data_path="/data/user/qkh/datasets/cmu_mocap",
        scene_filter=None,
        file_filter=None,
        use_data_aug=False,
        aug_rotate_prob=0.5,
        aug_reverse_prob=0.3,
    ):
        self.use_vel = use_vel
        self.data_path = data_path
        self.scene_filter = scene_filter
        self.file_filter = file_filter
        self.use_data_aug = use_data_aug and mode == "train"
        self.aug_rotate_prob = aug_rotate_prob
        self.aug_reverse_prob = aug_reverse_prob
        super().__init__(mode, t_his, t_pred, actions)

        if use_vel:
            self.traj_dim += 3

    @staticmethod
    def _build_skeleton(num_joints):
        # Use a simple chain topology for visualization compatibility.
        parent_single = [-1] + list(range(num_joints - 1))
        links_single = [(j, p) for j, p in enumerate(parent_single) if p != -1]
        shift = num_joints
        all_parents = parent_single + [p + shift if p >= 0 else -1 for p in parent_single]
        all_links = links_single + [(a + shift, b + shift) for a, b in links_single]
        # Keep equal L/R counts for Skeleton checks; center/root joints can stay unassigned.
        left_single = list(range(1, num_joints, 2))
        right_single = list(range(2, num_joints, 2))
        joints_left = left_single + [j + shift for j in left_single]
        joints_right = right_single + [j + shift for j in right_single]
        return all_parents, all_links, joints_left, joints_right

    @staticmethod
    def _augment_second_person(person_a):
        person_b = person_a.copy()
        person_b[..., 0] *= -1.0
        deg = 35.0
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=person_b.dtype)
        person_b = person_b @ rot.T
        person_b[..., 0] += 0.6
        return person_b

    def prepare_data(self):
        split_dir = os.path.join(self.data_path, self.mode)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"CMU mocap split folder not found: {split_dir}")

        txt_files = sorted(glob(os.path.join(split_dir, "*", "*.txt")))
        if len(txt_files) == 0:
            raise FileNotFoundError(f"No CMU txt files found in: {split_dir}")

        scene_set = None
        if self.scene_filter is not None:
            if isinstance(self.scene_filter, (list, tuple, set)):
                scene_set = {str(x).strip().lower() for x in self.scene_filter if str(x).strip()}
            else:
                scene_set = {str(self.scene_filter).strip().lower()}

        file_set = None
        if self.file_filter is not None:
            if isinstance(self.file_filter, (list, tuple, set)):
                file_set = {str(x).strip().lower() for x in self.file_filter if str(x).strip()}
            else:
                file_set = {str(self.file_filter).strip().lower()}

        self.data = {}
        joint_num_single = None
        for txt_file in txt_files:
            scene_name = os.path.basename(os.path.dirname(txt_file)).lower()
            file_stem = os.path.splitext(os.path.basename(txt_file))[0].lower()
            if scene_set is not None and scene_name not in scene_set:
                continue
            if file_set is not None and file_stem not in file_set:
                continue

            arr = np.loadtxt(txt_file, delimiter=",", dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] % 3 != 0:
                continue
            n_frames = arr.shape[0]
            j = arr.shape[1] // 3
            if n_frames <= self.t_total:
                continue
            person_a = arr.reshape(n_frames, j, 3)
            person_b = self._augment_second_person(person_a)
            seq = np.concatenate([person_a, person_b], axis=1)

            if self.use_vel:
                v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                v = np.append(v, v[[-1]], axis=0)

            seq[:, 1:] -= seq[:, :1]

            if self.use_vel:
                seq = np.concatenate((seq, v), axis=1)

            if scene_name not in self.data:
                self.data[scene_name] = {}
            self.data[scene_name][file_stem] = seq
            joint_num_single = j

        if not self.data:
            raise RuntimeError(
                f"No valid CMU sequences loaded for mode={self.mode}. "
                f"Check scene_filter={self.scene_filter} and file_filter={self.file_filter}."
            )

        self.subjects = sorted(self.data.keys())
        self.total_joints = 2 * joint_num_single
        all_parents, all_links, joints_left, joints_right = self._build_skeleton(joint_num_single)
        self.skeleton = Skeleton(
            parents=all_parents,
            joints_left=joints_left,
            joints_right=joints_right,
            links=all_links,
        )
        self.kept_joints = np.arange(self.total_joints)
        self.removed_joints = set()

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
