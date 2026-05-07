from __future__ import annotations

import re
from pathlib import Path


URDF_PATH = Path("/home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/JC01-URDF.urdf")

LIMITS = {
    "waist_roll": ("330.000", "2.094"),
    "waist_yaw": ("150.000", "2.618"),
    "right_shoulder_pitch": ("90.000", "2.000"),
    "right_shoulder_roll": ("90.000", "2.000"),
    "right_shoulder_yaw": ("60.000", "2.000"),
    "right_elbow_pitch": ("60.000", "2.000"),
    "right_elbow_yaw": ("36.000", "2.000"),
    "right_wrist_pitch": ("36.000", "2.000"),
    "right_wrist_roll": ("36.000", "2.000"),
    "left_shoulder_pitch": ("90.000", "2.000"),
    "left_shoulder_roll": ("90.000", "2.000"),
    "left_shoulder_yaw": ("60.000", "2.000"),
    "left_elbow_pitch": ("60.000", "2.000"),
    "left_elbow_yaw": ("36.000", "2.000"),
    "left_wrist_pitch": ("36.000", "2.000"),
    "left_wrist_roll": ("36.000", "2.000"),
}


def _replace_joint_limit(text: str, joint_name: str, effort: str, velocity: str) -> str:
    pattern = re.compile(
        rf'<joint\b(?:(?!</joint>).)*?\bname="{re.escape(joint_name)}"(?:(?!</joint>).)*?</joint>',
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Missing joint block: {joint_name}")

    block = match.group(0)
    if 'effort="' not in block or 'velocity="' not in block:
        raise ValueError(f"Joint block has no effort/velocity attributes: {joint_name}")
    updated = re.sub(r'effort="[^"]*"', f'effort="{effort}"', block, count=1)
    updated = re.sub(r'velocity="[^"]*"', f'velocity="{velocity}"', updated, count=1)
    return text[: match.start()] + updated + text[match.end() :]


def main() -> None:
    text = URDF_PATH.read_text()
    for joint_name, (effort, velocity) in LIMITS.items():
        text = _replace_joint_limit(text, joint_name, effort, velocity)
    URDF_PATH.write_text(text)
    print(f"Updated {len(LIMITS)} joint limits in {URDF_PATH}")


if __name__ == "__main__":
    main()
