import argparse
import copy
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_ROOT = os.path.join(REPO_ROOT, "code")
DFCNN_DIR = os.path.join(CODE_ROOT, "DFCNN")

if DFCNN_DIR not in sys.path:
    sys.path.insert(0, DFCNN_DIR)

os.chdir(REPO_ROOT)

from data_process import get_data
from dfcnn_model import DFCNN_CTC


FBANK_DIR = os.path.join(REPO_ROOT, "data_thchs30", "fbank")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_loader(loader, train_ratio=0.9):
    indices = list(range(len(loader.wav_lst)))
    random.shuffle(indices)

    split = max(1, int(len(indices) * train_ratio))

    train_loader = copy.deepcopy(loader)
    val_loader = copy.deepcopy(loader)

    train_idx = indices[:split]
    val_idx = indices[split:] or indices[:1]

    def apply_subset(target_loader, subset):
        target_loader.wav_lst = [loader.wav_lst[i] for i in subset]
        target_loader.pin_lst = [loader.pin_lst[i] for i in subset]
        target_loader.han_lst = [loader.han_lst[i] for i in subset]

    apply_subset(train_loader, train_idx)
    apply_subset(val_loader, val_idx)

    return train_loader, val_loader


def shuffle_loader(loader):
    indices = list(range(len(loader.wav_lst)))
    random.shuffle(indices)

    loader.wav_lst = [loader.wav_lst[i] for i in indices]
    loader.pin_lst = [loader.pin_lst[i] for i in indices]
    loader.han_lst = [loader.han_lst[i] for i in indices]


def load_fbank(wav_name):
    fbank_path = os.path.join(FBANK_DIR, wav_name + ".npy")

    if not os.path.exists(fbank_path):
        raise FileNotFoundError(
            f"Missing FBANK file: {fbank_path}\n"
            f"请先运行 precompute_fbank.py"
        )

    return np.load(fbank_path)


def flatten_targets(labels, label_lengths):
    targets = []

    for i, length in enumerate(label_lengths.tolist()):
        targets.extend(labels[i, :length].tolist())

    return torch.tensor(targets, dtype=torch.long)


def split_targets(labels, label_lengths):
    refs = []

    for i, length in enumerate(label_lengths.tolist()):
        refs.append([int(x) for x in labels[i, :length].tolist()])

    return refs


def ctc_greedy_decode(pred_ids, blank_index, input_lengths=None):
    results = []

    for i, row in enumerate(pred_ids):
        if input_lengths is not None:
            row = row[: int(input_lengths[i])]

        prev = None
        decoded = []

        for idx in row:
            idx = int(idx)

            if idx != prev and idx != blank_index:
                decoded.append(idx)

            prev = idx

        results.append(decoded)

    return results


def edit_distance(pred, ref):
    n, m = len(pred), len(ref)

    if n == 0:
        return m
    if m == 0:
        return n

    dp = [list(range(m + 1))]

    for i in range(1, n + 1):
        row = [i] + [0] * m

        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == ref[j - 1] else 1

            row[j] = min(
                dp[i - 1][j] + 1,
                row[j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

        dp.append(row)

    return dp[n][m]


def sequence_metrics(preds, refs):
    total_err = 0
    total_ref = 0

    for pred, ref in zip(preds, refs):
        total_err += edit_distance(pred, ref)
        total_ref += max(1, len(ref))

    wer = total_err / max(1, total_ref)
    acc = max(0.0, 1.0 - wer)

    return wer, acc


def make_acoustic_batch(loader, start, end, vocab):
    wav_data_lst = []
    label_data_lst = []

    for index in range(start, end):
        fbank = load_fbank(loader.wav_lst[index])

        pad_len = fbank.shape[0] // 8 * 8 + 8
        pad_fbank = np.zeros((pad_len, fbank.shape[1]), dtype=np.float32)
        pad_fbank[:fbank.shape[0], :] = fbank

        label = loader.pin2id(loader.pin_lst[index], vocab)
        label_ctc_len = loader.ctc_len(label)

        if pad_fbank.shape[0] // 8 >= label_ctc_len:
            wav_data_lst.append(pad_fbank)
            label_data_lst.append(label)

    if len(wav_data_lst) == 0:
        return None

    wav_lens = np.array([len(data) // 8 for data in wav_data_lst], dtype=np.int64)
    wav_max_len = max(len(data) for data in wav_data_lst)

    batch_x = np.zeros(
        (len(wav_data_lst), wav_max_len, 200, 1),
        dtype=np.float32,
    )

    for i, data in enumerate(wav_data_lst):
        batch_x[i, :data.shape[0], :, 0] = data

    label_lens = np.array([len(label) for label in label_data_lst], dtype=np.int64)
    max_label_len = max(label_lens)

    batch_y = np.zeros(
        (len(label_data_lst), max_label_len),
        dtype=np.int64,
    )

    for i, label in enumerate(label_data_lst):
        batch_y[i, :len(label)] = label

    return {
        "the_inputs": batch_x,
        "the_labels": batch_y,
        "input_length": wav_lens,
        "label_length": label_lens,
    }


def decode_batch(log_probs, batch_inputs, blank_index, vocab_size):
    pred_ids = log_probs.argmax(dim=-1).permute(1, 0).cpu().numpy()

    preds = ctc_greedy_decode(
        pred_ids,
        blank_index,
        input_lengths=batch_inputs["input_length"],
    )
    refs = split_targets(
        batch_inputs["the_labels"],
        batch_inputs["label_length"],
    )

    preds = [[idx for idx in seq if idx < vocab_size] for seq in preds]

    return preds, refs


def compute_metrics(model, loader, criterion, blank_index, vocab_size, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_preds = []
    all_refs = []

    steps = len(loader.wav_lst) // loader.batch_size

    if steps == 0:
        return 0.0, 1.0, 0.0

    with torch.no_grad():
        for batch_idx in range(steps):
            start = batch_idx * loader.batch_size
            end = start + loader.batch_size

            batch_inputs = make_acoustic_batch(
                loader,
                start,
                end,
                loader.acoustic_vocab,
            )

            if batch_inputs is None:
                continue

            inputs = torch.from_numpy(batch_inputs["the_inputs"]).float()
            inputs = inputs.permute(0, 3, 1, 2).to(device)

            labels = torch.from_numpy(batch_inputs["the_labels"]).long()
            input_length = torch.from_numpy(batch_inputs["input_length"]).long()
            label_length = torch.from_numpy(batch_inputs["label_length"]).long()

            targets = flatten_targets(labels, label_length).to(device)

            log_probs = model(inputs).permute(1, 0, 2)

            loss = criterion(
                log_probs,
                targets,
                input_length,
                label_length,
            )

            preds, refs = decode_batch(
                log_probs,
                batch_inputs,
                blank_index,
                vocab_size,
            )

            all_preds.extend(preds)
            all_refs.extend(refs)

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    wer, acc = sequence_metrics(all_preds, all_refs)

    return avg_loss, wer, acc


def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CTC Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_wer"], label="train")
    plt.plot(epochs, history["val_wer"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.title("Pinyin Error Rate")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Pinyin Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_length", type=int, default=13388)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_model", type=str, default="dfcnn_ctc_best.pt")
    parser.add_argument("--save_fig", type=str, default="dfcnn_ctc_history.png")

    args = parser.parse_args()

    set_seed(args.seed)

    data_args = argparse.Namespace(
        data_path=os.path.join(REPO_ROOT, "data_thchs30", "data") + os.sep,
        data_length=args.data_length,
        batch_size=args.batch_size,
    )

    base_loader = get_data(data_args)
    train_loader, val_loader = split_loader(base_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = len(base_loader.acoustic_vocab)
    blank_index = vocab_size - 1

    model = DFCNN_CTC(num_classes=vocab_size).to(device)

    criterion = nn.CTCLoss(
        blank=blank_index,
        zero_infinity=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_wer": [],
        "val_wer": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_wer = float("inf")

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.wav_lst)}")
    print(f"Val samples: {len(val_loader.wav_lst)}")
    print(f"Acoustic vocab size: {vocab_size}")
    print(f"Blank index: {blank_index}")
    print(f"FBANK dir: {FBANK_DIR}")
    print("-" * 100)

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffle_loader(train_loader)

        steps = len(train_loader.wav_lst) // args.batch_size

        if steps == 0:
            print("No training batches. Please reduce batch_size.")
            return

        total_loss = 0.0
        total_batches = 0

        for batch_idx in range(steps):
            start = batch_idx * args.batch_size
            end = start + args.batch_size

            batch_inputs = make_acoustic_batch(
                train_loader,
                start,
                end,
                base_loader.acoustic_vocab,
            )

            if batch_inputs is None:
                continue

            inputs = torch.from_numpy(batch_inputs["the_inputs"]).float()
            inputs = inputs.permute(0, 3, 1, 2).to(device)

            labels = torch.from_numpy(batch_inputs["the_labels"]).long()
            input_length = torch.from_numpy(batch_inputs["input_length"]).long()
            label_length = torch.from_numpy(batch_inputs["label_length"]).long()

            targets = flatten_targets(labels, label_length).to(device)

            log_probs = model(inputs).permute(1, 0, 2)

            loss = criterion(
                log_probs,
                targets,
                input_length,
                label_length,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            running_loss = total_loss / total_batches

            display_batch = batch_idx + 1

            if (
                display_batch == 1
                or display_batch % args.log_interval == 0
                or display_batch == steps
            ):
                print(
                    f"[Train] "
                    f"Epoch {epoch:02d}/{args.epochs} "
                    f"Batch {display_batch:04d}/{steps} | "
                    f"loss={loss.item():.4f} | "
                    f"avg_loss={running_loss:.4f}"
                )

        train_loss, train_wer, train_acc = compute_metrics(
            model=model,
            loader=train_loader,
            criterion=criterion,
            blank_index=blank_index,
            vocab_size=vocab_size,
            device=device,
        )

        val_loss, val_wer, val_acc = compute_metrics(
            model=model,
            loader=val_loader,
            criterion=criterion,
            blank_index=blank_index,
            vocab_size=vocab_size,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_wer"].append(train_wer)
        history["val_wer"].append(val_wer)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch Summary] "
            f"{epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"train_wer={train_wer:.4f} | "
            f"val_wer={val_wer:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_wer < best_val_wer:
            best_val_wer = val_wer

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_wer": val_wer,
                    "val_acc": val_acc,
                    "acoustic_vocab": base_loader.acoustic_vocab,
                    "blank_index": blank_index,
                    "args": vars(args),
                },
                args.save_model,
            )

            print(
                f"Saved best acoustic model "
                f"(val_wer={val_wer:.4f}) "
                f"-> {args.save_model}"
            )

        print("=" * 100)

    plot_history(history, args.save_fig)

    print("Training finished.")
    print(f"Best val WER: {best_val_wer:.4f}")
    print(f"Saved figure: {args.save_fig}")


if __name__ == "__main__":
    main()
