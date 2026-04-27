import re
import os
import numpy as np
import pandas as pd


def extract_fields(prompt: str) -> list[str]:
    return re.findall(
        r"\{(.*?)\}", prompt.replace("{{", "").replace("}}", ""), re.DOTALL
    )


def byte_to_gb(bytes: int) -> float:
    return bytes / (1024**3)


def gb_to_byte(gb: float) -> int:
    return int(gb * (1024**3))


def partition_list(data: list, chunk_size: int = 100) -> list[list]:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def comb_matrix(root_dir: str, cname: str = "job") -> np.ndarray:
    mat: np.ndarray = None
    mats: list[str] = []  # files.find_files(root_dir, f"{cname}_mat.csv")
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(f"{cname}_mat.csv"):
                mats.append(os.path.join(root, filename))

    if not mats:
        raise ValueError(f"No matrices found in {root_dir} with cname {cname}")

    for mat_path in mats:
        if mat is None:
            mat = np.array(pd.read_csv(mat_path).iloc[:, 1:])
        else:
            # add them together
            mat += np.array(pd.read_csv(mat_path).iloc[:, 1:])
    mat = mat.astype(float)
    mat *= 1 / len(mats)
    return mat
