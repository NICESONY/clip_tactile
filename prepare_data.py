"""
prepare_data.py

clip_regression/data/ 디렉토리를 구성하는 스크립트.
이 폴더만 떼어 가져갔을 때 데이터를 연결하기 위해 사용.

사용법:
    # 심볼릭 링크 (기본, 빠름, 디스크 절약)
    python prepare_data.py --source /path/to/tactile_coordinate_project/data

    # 실제 복사 (폴더를 완전히 독립시킬 때)
    python prepare_data.py --source /path/to/tactile_coordinate_project/data --copy

    # 같은 프로젝트 안에 있을 때 (자동 감지)
    python prepare_data.py
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


REQUIRED = [
    "train/images",
    "val/images",
    "test/images",
    "processed/train_labels_normalized.csv",
    "processed/val_labels_normalized.csv",
    "processed/test_labels_normalized.csv",
    "processed/label_stats.json",
]


def find_source_auto():
    """현재 위치 기준으로 상위 프로젝트의 data/ 자동 탐색."""
    candidates = [
        Path(__file__).resolve().parent / ".." / "data",       # ../data
        Path(__file__).resolve().parent.parent / "data",       # ../../data
    ]
    for c in candidates:
        c = c.resolve()
        if c.is_dir() and (c / "processed").is_dir():
            return c
    return None


def validate_source(source: Path):
    missing = []
    for r in REQUIRED:
        if not (source / r).exists():
            missing.append(r)
    if missing:
        print(f"[ERROR] source 경로에 필요한 파일이 없습니다: {source}")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


def symlink_data(source: Path, dest: Path):
    """data/ 하위 디렉토리별로 심볼릭 링크 생성."""
    dest.mkdir(parents=True, exist_ok=True)

    for item in ["train", "val", "test", "processed"]:
        src = source / item
        dst = dest / item
        if dst.exists() or dst.is_symlink():
            print(f"  [SKIP] {dst} already exists")
            continue
        os.symlink(src.resolve(), dst)
        print(f"  [LINK] {dst} -> {src.resolve()}")


def copy_data(source: Path, dest: Path):
    """data/ 실제 복사."""
    dest.mkdir(parents=True, exist_ok=True)

    for item in ["processed"]:
        src = source / item
        dst = dest / item
        if dst.exists():
            print(f"  [SKIP] {dst} already exists")
            continue
        shutil.copytree(src, dst)
        print(f"  [COPY] {dst}")

    for item in ["train", "val", "test"]:
        src = source / item
        dst = dest / item
        if dst.exists():
            print(f"  [SKIP] {dst} already exists")
            continue
        shutil.copytree(src, dst)
        print(f"  [COPY] {dst}")


def main():
    parser = argparse.ArgumentParser(description="clip_regression 데이터 준비")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="원본 data/ 디렉토리 경로. 미지정 시 자동 탐색.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="심볼릭 링크 대신 실제 복사.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dest = project_root / "data"

    # source 결정
    if args.source:
        source = Path(args.source).resolve()
    else:
        source = find_source_auto()
        if source is None:
            print("[ERROR] 자동 탐색 실패. --source 옵션으로 원본 data/ 경로를 지정하세요.")
            print("  예: python prepare_data.py --source /path/to/tactile_coordinate_project/data")
            sys.exit(1)

    print(f"Source: {source}")
    print(f"Dest:   {dest}")
    print()

    validate_source(source)

    if args.copy:
        print("[MODE] 실제 복사")
        copy_data(source, dest)
    else:
        print("[MODE] 심볼릭 링크")
        symlink_data(source, dest)

    print()
    print("Done! data/ 준비 완료.")
    print("확인: ls data/processed/ data/train/images/ data/val/images/ data/test/images/")


if __name__ == "__main__":
    main()
