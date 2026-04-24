import os
import pathlib
import json
import tomllib
from datetime import datetime
import pandas as pd
import shutil


# directory operations


def os_path(file_path: str) -> str:
    path = os.path.normpath(file_path)
    return path


def exists(file_path: str) -> bool:
    my_file = pathlib.Path(file_path).resolve()

    if my_file.is_file():
        return True
    else:
        return False


def create_folder(folder_path: str) -> str:
    path = os_path(folder_path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def append_directory(dir1: str, dir2: str) -> str:
    create_folder(os.path.join(dir1, dir2))
    return os_path(os.path.join(dir1, dir2))


def copy_file(src: str, dst: str) -> None:
    src_path = os_path(src)
    dst_path = os_path(dst)
    if not exists(src_path):
        raise FileNotFoundError(f"Source file {src_path} does not exist.")
    shutil.copy2(src_path, dst_path)


def split_path(filepath: str) -> tuple[str, str]:
    # return absolute directory path and filename
    path = os.path.abspath(os_path(filepath))
    return os.path.split(path)


# read files


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text: str = f.read()
    return text


def read_lines(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines: list[str] = f.readlines()
    return lines


def read_csv(file_path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(os_path(file_path), low_memory=False)
    df = df.fillna("")
    return df


def read_json(file_path: str) -> dict:
    if exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    else:
        return {}


def read_csv_dict(file_path: str) -> list[dict]:
    try:
        data = pd.read_csv(os_path(file_path)).to_dict(orient="records")
        return data
    except Exception as e:
        return []


def read_toml(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        config_data = tomllib.load(f)
    return config_data


# write files


def write_csv(df: pd.DataFrame, file_path: str) -> None:
    df.to_csv(file_path, index=False)


def write_json(data: dict, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def write_lines(lines: list[str], file_path: str) -> None:
    with open(file_path, "w") as f:
        f.writelines(lines)


# find files


def find_files(directory: str, extension: str, sort_desc: bool = False) -> list[str]:
    # use os.walk and endswith to find files
    files: list[str] = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os_path(os.path.join(root, filename)))
    if sort_desc:
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files


def list_files(directory: str, start_str: str, extension: str) -> list[str]:
    files = find_files(directory, extension)
    files = [
        file.replace("\\", "/").replace(f"{directory}/", "")
        for file in files
        if file.startswith(os_path(directory) + os.sep + start_str)
    ]
    if files == []:
        return None
    return files


# other


def clean_filename(filename: str) -> str:
    invalid_charmap: dict[str, str] = {
        "<": "#lt#",
        ">": "#gt#",
        ":": "#col#",
        '"': "#quot#",
        "/": "#slsh#",
        "\\": "#bslsh#",
        "|": "#pipe#",
        "?": "#ques#",
        "*": "#ast#",
        ".": "#dot#",
    }
    for char, replacement in invalid_charmap.items():
        filename = filename.replace(char, replacement)
    return filename


def revert_filename(filename: str) -> str:
    invalid_charmap: dict[str, str] = {
        "#lt#": "<",
        "#gt#": ">",
        "#col#": ":",
        "#quot#": '"',
        "#slsh#": "/",
        "#bslsh#": "\\",
        "#pipe#": "|",
        "#ques#": "?",
        "#ast#": "*",
        "#dot#": ".",
    }
    for replacement, char in invalid_charmap.items():
        filename = filename.replace(replacement, char)
    return filename


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")
