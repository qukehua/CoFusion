"""
Preprocess CMU mocap dataset to generate multimodal evaluation files.

This script generates:
1. data_candi_*.npz - Candidate trajectories for multimodal evaluation
2. t_his*_filtered.npz - Multimodal indices (same history, different future)

Notes:
- Raw CMU txt clips are single-person.
- We synthesize Person_2 from Person_1 (mirror + rotation + translation),
  consistent with DatasetCMUMocap used in training.

Usage:
    python preprocess_cmu_mocap.py --data_path /data/user/qkh/datasets/cmu_mocap --output_dir /data/user/qkh/datasets/cmu_mocap/multimodal
"""

import os
import argparse
from glob import glob

import numpy as np
import torch
from tqdm import tqdm


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")


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


def load_cmu_sequences(data_path, split="test", scene_filter=None, file_filter=None):
    """
    Load CMU txt clips from:
      data_path/{train|test}/*/*.txt
    """
    split_dir = os.path.join(data_path, split)
    txt_files = sorted(glob(os.path.join(split_dir, "*", "*.txt")))
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No CMU txt files found in {split_dir}")

    scene_set = None
    if scene_filter is not None and str(scene_filter).strip() != "":
        scene_set = {x.strip().lower() for x in str(scene_filter).split(",") if x.strip()}
    file_set = None
    if file_filter is not None and str(file_filter).strip() != "":
        file_set = {x.strip().lower() for x in str(file_filter).split(",") if x.strip()}

    all_sequences = []
    sequence_info = []
    for txt_file in tqdm(txt_files, desc=f"Loading CMU {split}"):
        scene_name = os.path.basename(os.path.dirname(txt_file)).lower()
        stem = os.path.splitext(os.path.basename(txt_file))[0].lower()
        if scene_set is not None and scene_name not in scene_set:
            continue
        if file_set is not None and stem not in file_set:
            continue

        arr = np.loadtxt(txt_file, delimiter=",", dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] % 3 != 0:
            continue
        n_frames = arr.shape[0]
        n_joints = arr.shape[1] // 3
        person_a = arr.reshape(n_frames, n_joints, 3)
        person_b = _augment_second_person(person_a)
        seq = np.concatenate([person_a, person_b], axis=1)

        # Match DatasetCMUMocap root-relative representation.
        seq[:, 1:] -= seq[:, :1]

        all_sequences.append(seq)
        sequence_info.append(
            {
                "scene": scene_name,
                "file": os.path.basename(txt_file),
                "length": n_frames,
                "joints_per_person": n_joints,
            }
        )
    return all_sequences, sequence_info


def extract_windows(sequences, t_his, t_pred, skip_rate):
    t_total = t_his + t_pred
    windows = []
    window_origins = []
    for seq_idx, seq in enumerate(tqdm(sequences, desc="Extracting windows")):
        seq_len = seq.shape[0]
        for i in range(0, seq_len - t_total, skip_rate):
            windows.append(seq[i : i + t_total])
            window_origins.append((seq_idx, i))
    return np.array(windows), window_origins


def compute_multimodal_indices(windows, t_his, thre_his=0.5, thre_pred=0.1):
    n_windows = len(windows)
    if n_windows == 0:
        print("ERROR: No windows to process.")
        return {}

    history = windows[:, t_his - 1 : t_his, 1:].reshape(n_windows, -1)
    future = windows[:, t_his:, 1:].reshape(n_windows, -1)
    print(f"Computing pairwise distances for {n_windows} windows using {DEVICE}...")

    multimodal_dict = {}
    if USE_CUDA:
        history_t = torch.tensor(history, dtype=torch.float32, device=DEVICE)
        future_t = torch.tensor(future, dtype=torch.float32, device=DEVICE)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        estimated_mem_per_batch = n_windows * future.shape[1] * 4 * 2
        max_batch = max(1, int(gpu_mem * 0.3 / max(estimated_mem_per_batch, 1)))
        batch_size = min(200, max_batch)
        print(f"Using batch_size={batch_size} for {n_windows} windows")

        for i in tqdm(range(0, n_windows, batch_size), desc="Computing multimodal indices (CUDA)"):
            batch_end = min(i + batch_size, n_windows)
            try:
                hist_i = history_t[i:batch_end]
                fut_i = future_t[i:batch_end]
                dist_his = torch.norm(hist_i[:, None, :] - history_t[None, :, :], dim=2)
                dist_pred = torch.norm(fut_i[:, None, :] - future_t[None, :, :], dim=2)
                mask = (dist_his <= thre_his) & (dist_pred >= thre_pred)
                for j in range(batch_end - i):
                    idx = i + j
                    mask[j, idx] = False
                    idx_multi = torch.where(mask[j])[0].cpu().numpy().tolist()
                    if len(idx_multi) > 0:
                        multimodal_dict[idx] = idx_multi
                del dist_his, dist_pred, mask, hist_i, fut_i
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA OOM at batch {i}, switching this batch to CPU.")
                    torch.cuda.empty_cache()
                    hist_i = history[i:batch_end]
                    fut_i = future[i:batch_end]
                    dist_his = np.linalg.norm(hist_i[:, None, :] - history[None, :, :], axis=2)
                    dist_pred = np.linalg.norm(fut_i[:, None, :] - future[None, :, :], axis=2)
                    for j in range(batch_end - i):
                        idx = i + j
                        mask = (dist_his[j] <= thre_his) & (dist_pred[j] >= thre_pred)
                        mask[idx] = False
                        idx_multi = np.where(mask)[0].tolist()
                        if len(idx_multi) > 0:
                            multimodal_dict[idx] = idx_multi
                else:
                    raise e
        del history_t, future_t
        torch.cuda.empty_cache()
    else:
        batch_size = 500
        for i in tqdm(range(0, n_windows, batch_size), desc="Computing multimodal indices (CPU)"):
            batch_end = min(i + batch_size, n_windows)
            hist_i = history[i:batch_end]
            fut_i = future[i:batch_end]
            dist_his = np.linalg.norm(hist_i[:, None, :] - history[None, :, :], axis=2)
            dist_pred = np.linalg.norm(fut_i[:, None, :] - future[None, :, :], axis=2)
            for j in range(batch_end - i):
                idx = i + j
                mask = (dist_his[j] <= thre_his) & (dist_pred[j] >= thre_pred)
                mask[idx] = False
                idx_multi = np.where(mask)[0]
                if len(idx_multi) > 0:
                    multimodal_dict[idx] = idx_multi.tolist()
    return multimodal_dict


def main():
    parser = argparse.ArgumentParser(description="Preprocess CMU mocap for multimodal evaluation")
    parser.add_argument("--data_path", type=str, default="/data/user/qkh/datasets/cmu_mocap", help="Path to CMU mocap root")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/user/qkh/datasets/cmu_mocap/multimodal",
        help="Output directory for multimodal npz files",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Split to preprocess")
    parser.add_argument("--scene_filter", type=str, default="walking", help="Comma-separated action folders, e.g. walking,running")
    parser.add_argument("--file_filter", type=str, default="walking_1", help="Comma-separated clip stems, e.g. walking_1,walking_2")
    parser.add_argument("--t_his", type=int, default=25, help="History frames")
    parser.add_argument("--t_pred", type=int, default=100, help="Prediction frames")
    parser.add_argument("--skip_rate", type=int, default=20, help="Skip rate for extracting windows")
    parser.add_argument("--thre_his", type=float, default=0.5, help="History similarity threshold")
    parser.add_argument("--thre_pred", type=float, default=0.1, help="Future difference threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scene_tag = "all" if args.scene_filter.strip() == "" else args.scene_filter.replace(",", "_")
    file_tag = "all" if args.file_filter.strip() == "" else args.file_filter.replace(",", "_")

    print("=" * 60)
    print("CMU mocap Preprocessing for TransFusion")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Split: {args.split}")
    print(f"scene_filter: {args.scene_filter}")
    print(f"file_filter: {args.file_filter}")
    print(f"t_his: {args.t_his}, t_pred: {args.t_pred}, skip_rate: {args.skip_rate}")
    print(f"thre_his: {args.thre_his}, thre_pred: {args.thre_pred}")
    print("=" * 60)

    print("\n[1/4] Loading sequences...")
    sequences, seq_info = load_cmu_sequences(
        args.data_path,
        split=args.split,
        scene_filter=args.scene_filter,
        file_filter=args.file_filter,
    )
    print(f"Loaded {len(sequences)} sequences")

    print("\n[2/4] Extracting sliding windows...")
    windows, _ = extract_windows(sequences, args.t_his, args.t_pred, args.skip_rate)
    print(f"Extracted {len(windows)} windows")
    print(f"Window shape: {windows.shape}")

    print("\n[3/4] Saving candidate trajectories...")
    candi_file = os.path.join(
        args.output_dir,
        f"data_candi_cmu_mocap_{args.split}_{scene_tag}_{file_tag}_t_his{args.t_his}_t_pred{args.t_pred}_skiprate{args.skip_rate}.npz",
    )
    np.savez_compressed(candi_file, **{"data_candidate.npy": windows})
    print(f"Saved: {candi_file}")

    print("\n[4/4] Computing multimodal indices...")
    multimodal_dict = compute_multimodal_indices(windows, args.t_his, args.thre_his, args.thre_pred)
    multi_file = os.path.join(
        args.output_dir,
        f"t_his{args.t_his}_cmu_mocap_{args.split}_{scene_tag}_{file_tag}_thre{args.thre_his:.3f}_t_pred{args.t_pred}_thre{args.thre_pred:.3f}_filtered.npz",
    )
    np.savez_compressed(multi_file, data_multimodal=multimodal_dict)
    print(f"Saved: {multi_file}")

    n_multi = len(multimodal_dict)
    avg_multi = np.mean([len(v) for v in multimodal_dict.values()]) if n_multi > 0 else 0
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    if len(windows) > 0:
        print(f"Windows with multimodal futures: {n_multi}/{len(windows)} ({100 * n_multi / len(windows):.1f}%)")
    print(f"Average multimodal count: {avg_multi:.1f}")
    if len(seq_info) > 0:
        scenes = sorted(list(set([x["scene"] for x in seq_info])))
        print(f"Scenes used: {scenes}")
    print("=" * 60)

    print("\nUse these paths in cfg/cmu_mocap.yml:")
    print(f"  multimodal_path: {multi_file}")
    print(f"  data_candi_path: {candi_file}")


if __name__ == "__main__":
    main()
