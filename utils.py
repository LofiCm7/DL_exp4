import json
import random
from csv import DictWriter
from pathlib import Path

import numpy as np
import torch


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path, must_exist: bool = False) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path, REPO_ROOT / path, CODE_DIR / path]
    if must_exist:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    if path.parts and path.parts[0] == "DL_exp4":
        return REPO_ROOT / path
    return Path.cwd() / path


def ensure_dir(path: str | Path) -> Path:
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_ratio_tag(name: str, ratio: float) -> str:
    percent = ratio * 100
    if abs(percent - round(percent)) < 1e-8:
        value = str(int(round(percent)))
    else:
        value = f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{name}{value}"


def build_run_name(*parts) -> str:
    tokens = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip().replace("/", "-").replace(" ", "")
        if text:
            tokens.append(text)
    return "_".join(tokens)


def resolve_output_dir(base_dir: str | Path, run_name: str | None = None) -> Path:
    base_dir = ensure_dir(base_dir)
    if run_name:
        return ensure_dir(base_dir / run_name)
    return base_dir


def infer_run_name_from_checkpoint(checkpoint_path: str | Path) -> str | None:
    checkpoint_path = resolve_path(checkpoint_path, must_exist=True)
    if checkpoint_path.parent.name != "weights":
        return None
    return checkpoint_path.parent.parent.name


def save_json(data: dict, path: str | Path) -> None:
    output_path = resolve_path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_history_csv(history: dict, path: str | Path) -> None:
    output_path = resolve_path(path)
    ensure_dir(output_path.parent)

    keys = list(history)
    length = len(history[keys[0]]) if keys else 0
    rows = [{key: history[key][index] for key in keys} for index in range(length)]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
