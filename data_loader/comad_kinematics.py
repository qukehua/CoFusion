"""
CoMaD Person_1 joint order matches official mapping.json indices 0..24:

 0 BackLeft, 1 BackRight, 2 BackTop, 3 Chest, 4 HeadFront, 5 HeadSide, 6 HeadTop,
 7 LElbowOut, 8 LHandOut, 9 LShoulderBack, 10 LShoulderTop, 11 LUArmHigh,
 12 LWristIn, 13 LWristOut, 14 RElbowOut, 15 RHandOut, 16 RShoulderBack,
 17 RShoulderTop, 18 RUArmHigh, 19 RWristIn, 20 RWristOut,
 21 WaistLBack, 22 WaistLFront, 23 WaistRBack, 24 WaistRFront

Parent array for visualization / Skeleton.links (root = index 0, consistent with root-relative code).
"""

import numpy as np

# parent[i] = parent joint index of i, or -1 for root
COMAD_P1_PARENTS = [
    -1,  # 0 BackLeft (dataset root-relative anchor)
    0,  # 1 BackRight
    0,  # 2 BackTop
    2,  # 3 Chest
    6,  # 4 HeadFront
    6,  # 5 HeadSide
    3,  # 6 HeadTop
    11,  # 7 LElbowOut
    7,  # 8 LHandOut
    3,  # 9 LShoulderBack
    9,  # 10 LShoulderTop
    10,  # 11 LUArmHigh
    11,  # 12 LWristIn
    11,  # 13 LWristOut
    18,  # 14 RElbowOut
    14,  # 15 RHandOut
    3,  # 16 RShoulderBack
    16,  # 17 RShoulderTop
    17,  # 18 RUArmHigh
    18,  # 19 RWristIn
    18,  # 20 RWristOut
    3,  # 21 WaistLBack
    21,  # 22 WaistLFront
    3,  # 23 WaistRBack
    23,  # 24 WaistRFront
]

# Left / right coloring in render_animation (anatomical left vs right arm chains)
COMAD_P1_JOINTS_LEFT = [7, 8, 9, 10, 11, 12, 13]
COMAD_P1_JOINTS_RIGHT = [14, 15, 16, 17, 18, 19, 20]


def comad_p1_links():
    return [(j, p) for j, p in enumerate(COMAD_P1_PARENTS) if p >= 0]


def comad_fix_orientation_motive_to_interact(x):
    """
    Match InteRACT interact/utils/comad_hr.py::fix_orientation (Motive -> plotting frame).

    https://github.com/portal-cornell/interact/blob/release/interact/utils/comad_hr.py
    tensor[:, :, [0, 1, 2]] = tensor[:, :, [0, 2, 1]]; tensor[:, :, 0] *= -1

    Same as interact/utils/read_json_data.py::transform_coords for Motive (x,y,z).
    Supports any leading batch dims; last dim is xyz.
    """
    a = np.asarray(x, dtype=np.float32)
    y = np.empty_like(a, dtype=np.float32)
    y[..., 0] = -a[..., 0]
    y[..., 1] = a[..., 2]
    y[..., 2] = a[..., 1]
    return y
