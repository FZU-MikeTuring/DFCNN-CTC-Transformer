import argparse
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CODE_ROOT = os.path.join(REPO_ROOT, "code")
DFCNN_DIR = os.path.join(CODE_ROOT, "DFCNN")

if DFCNN_DIR not in sys.path:
    sys.path.insert(0, DFCNN_DIR)

os.chdir(REPO_ROOT)

from data_process import compute_fbank


DATA_DIR = os.path.join(REPO_ROOT, "data_thchs30", "data")
DATA_TXT = os.path.join(REPO_ROOT, "data_thchs30", "data.txt")
SAVE_DIR = os.path.join(REPO_ROOT, "data_thchs30", "fbank")


def load_wav_names(data_txt, data_length=None):
    wav_names = []

    with open(data_txt, "r", encoding="utf8") as f:
        lines = f.readlines()

    for line in lines:
        wav_name = line.strip().split("\t")[0]
        wav_names.append(wav_name)

    if data_length:
        wav_names = wav_names[:data_length]

    return wav_names


def process_one(args):
    wav_name, overwrite = args

    wav_path = os.path.join(DATA_DIR, wav_name)
    save_path = os.path.join(SAVE_DIR, wav_name + ".npy")

    if os.path.exists(save_path) and not overwrite:
        return "skip", wav_name

    try:
        fbank = compute_fbank(wav_path)
        np.save(save_path, fbank.astype(np.float32))
        return "ok", wav_name

    except Exception as e:
        return "fail", f"{wav_name}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_length", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    wav_names = load_wav_names(DATA_TXT, args.data_length)

    print(f"Total wav files: {len(wav_names)}")
    print(f"Save dir: {SAVE_DIR}")
    print(f"Workers: {args.workers}")
    print("-" * 80)

    tasks = [(wav_name, args.overwrite) for wav_name in wav_names]

    ok_count = 0
    skip_count = 0
    fail_count = 0

    with Pool(processes=args.workers) as pool:
        for idx, (status, message) in enumerate(
            pool.imap_unordered(process_one, tasks),
            1,
        ):
            if status == "ok":
                ok_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                fail_count += 1
                print(f"[FAIL] {message}")

            if idx == 1 or idx % 100 == 0 or idx == len(tasks):
                print(
                    f"[{idx}/{len(tasks)}] "
                    f"ok={ok_count} | "
                    f"skip={skip_count} | "
                    f"fail={fail_count}"
                )

    print("-" * 80)
    print("FBANK precompute finished.")
    print(f"OK: {ok_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()