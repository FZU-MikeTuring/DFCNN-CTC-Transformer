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
from transformer_model import TransformerModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_loader(loader, train_ratio=0.9):
    indices = list(range(len(loader.pin_lst)))
    random.shuffle(indices)

    split = max(1, int(len(indices) * train_ratio))

    train_loader = copy.deepcopy(loader)
    val_loader = copy.deepcopy(loader)

    train_idx = indices[:split]
    val_idx = indices[split:] or indices[:1]

    train_loader.pin_lst = [loader.pin_lst[i] for i in train_idx]
    train_loader.han_lst = [loader.han_lst[i] for i in train_idx]

    val_loader.pin_lst = [loader.pin_lst[i] for i in val_idx]
    val_loader.han_lst = [loader.han_lst[i] for i in val_idx]

    return train_loader, val_loader


def levenshtein(a, b):
    n, m = len(a), len(b)

    if n == 0:
        return m
    if m == 0:
        return n

    dp = [list(range(m + 1))]

    for i in range(1, n + 1):
        row = [i] + [0] * m

        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            row[j] = min(
                dp[i - 1][j] + 1,
                row[j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

        dp.append(row)

    return dp[n][m]


def ids_to_tokens(ids, vocab, pad_id=0, ignore_ids=None):
    if ignore_ids is None:
        ignore_ids = set()

    tokens = []

    for x in ids:
        x = int(x)

        if x == pad_id or x in ignore_ids:
            continue

        if 0 <= x < len(vocab):
            tokens.append(vocab[x])

    return tokens


def compute_metrics(preds, refs):
    total_err = 0
    total_ref = 0

    for pred, ref in zip(preds, refs):
        total_err += levenshtein(pred, ref)
        total_ref += max(1, len(ref))

    wer = total_err / max(1, total_ref)
    char_acc = max(0.0, 1.0 - wer)

    return wer, char_acc


def collect_predictions(pred_ids, target_ids, vocab, pad_id, ignore_ids=None):
    preds = []
    refs = []

    for pred_row, ref_row in zip(pred_ids, target_ids):
        pred_seq = ids_to_tokens(pred_row, vocab, pad_id, ignore_ids)
        ref_seq = ids_to_tokens(ref_row, vocab, pad_id)

        preds.append(pred_seq)
        refs.append(ref_seq)

    return preds, refs


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    batch_size,
    sos_id,
    epoch,
    epochs,
    log_interval=10,
):
    model.train()

    steps = len(loader.pin_lst) // batch_size

    if steps == 0:
        return 0.0

    gen = loader.get_language_model_batch()

    total_loss = 0.0
    total_batches = 0

    for batch_idx in range(1, steps + 1):
        src_batch, tgt_batch = next(gen)

        src = torch.from_numpy(src_batch).long().to(device)
        tgt = torch.from_numpy(tgt_batch).long().to(device)

        tgt_in = torch.cat(
            [
                torch.full(
                    (tgt.size(0), 1),
                    sos_id,
                    dtype=torch.long,
                    device=device,
                ),
                tgt[:, :-1],
            ],
            dim=1,
        )

        logits = model(src, tgt_in)
        vocab_size = logits.size(-1)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        avg_loss = total_loss / total_batches

        if (
            batch_idx == 1
            or batch_idx % log_interval == 0
            or batch_idx == steps
        ):
            print(
                f"[Train] "
                f"Epoch {epoch:02d}/{epochs} "
                f"Batch {batch_idx:04d}/{steps} | "
                f"loss={loss.item():.4f} | "
                f"avg_loss={avg_loss:.4f}"
            )

    return total_loss / max(1, total_batches)


def evaluate_model(
    model,
    loader,
    criterion,
    device,
    batch_size,
    sos_id,
    han_vocab,
    pad_id,
    stage="Val",
):
    model.eval()

    steps = len(loader.pin_lst) // batch_size

    if steps == 0:
        return 0.0, 1.0, 0.0

    gen = loader.get_language_model_batch()

    total_loss = 0.0
    total_batches = 0

    all_preds = []
    all_refs = []

    with torch.no_grad():
        for _ in range(steps):
            src_batch, tgt_batch = next(gen)

            src = torch.from_numpy(src_batch).long().to(device)
            tgt = torch.from_numpy(tgt_batch).long().to(device)

            tgt_in = torch.cat(
                [
                    torch.full(
                        (tgt.size(0), 1),
                        sos_id,
                        dtype=torch.long,
                        device=device,
                    ),
                    tgt[:, :-1],
                ],
                dim=1,
            )

            logits = model(src, tgt_in)
            vocab_size = logits.size(-1)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt.reshape(-1),
            )

            pred_ids = logits.argmax(dim=-1).detach().cpu().numpy()

            batch_preds, batch_refs = collect_predictions(
                pred_ids,
                tgt_batch,
                han_vocab,
                pad_id,
                ignore_ids={sos_id},
            )

            all_preds.extend(batch_preds)
            all_refs.extend(batch_refs)

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    wer, char_acc = compute_metrics(all_preds, all_refs)

    print(
        f"[{stage} Eval] "
        f"loss={avg_loss:.4f} | "
        f"wer={wer:.4f} | "
        f"char_acc={char_acc:.4f}"
    )

    return avg_loss, wer, char_acc


def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross Entropy Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_wer"], label="train")
    plt.plot(epochs, history["val_wer"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.title("Character Error Rate")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Character Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_length", type=int, default=13388)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--save_model", type=str, default="transformer_lm_best.pt")
    parser.add_argument("--save_fig", type=str, default="transformer_lm_history.png")

    args = parser.parse_args()

    set_seed(args.seed)

    data_args = argparse.Namespace(
        data_path=os.path.join(REPO_ROOT, "data_thchs30", "data") + os.sep,
        data_length=args.data_length,
        batch_size=args.batch_size,
    )

    base_loader = get_data(data_args)
    train_loader, val_loader = split_loader(base_loader)

    src_vocab_size = len(base_loader.pin_vocab)
    pad_id = 0
    sos_id = len(base_loader.han_vocab)
    tgt_vocab_size = len(base_loader.han_vocab) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=4,
        d_model=256,
        dropout=0.3,
        pad_id=pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

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
    print(f"Train samples: {len(train_loader.pin_lst)}")
    print(f"Val samples: {len(val_loader.pin_lst)}")
    print(f"Pinyin vocab size: {src_vocab_size}")
    print(f"Hanzi vocab size: {tgt_vocab_size}")
    print(f"PAD id: {pad_id}")
    print(f"SOS id: {sos_id}")
    print("-" * 100)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            sos_id=sos_id,
            epoch=epoch,
            epochs=args.epochs,
            log_interval=args.log_interval,
        )

        train_loss, train_wer, train_acc = evaluate_model(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            sos_id=sos_id,
            han_vocab=base_loader.han_vocab,
            pad_id=pad_id,
            stage="Train",
        )

        val_loss, val_wer, val_acc = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            sos_id=sos_id,
            han_vocab=base_loader.han_vocab,
            pad_id=pad_id,
            stage="Val",
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
                    "han_vocab": base_loader.han_vocab,
                    "pin_vocab": base_loader.pin_vocab,
                    "pad_id": pad_id,
                    "sos_id": sos_id,
                    "args": vars(args),
                },
                args.save_model,
            )

            print(
                f"Saved best language model "
                f"(val_wer={val_wer:.4f}) "
                f"-> {args.save_model}"
            )

        print("=" * 100)

    plot_history(history, args.save_fig)

    print("Training finished.")
    print(f"Best val WER: {best_val_wer:.4f}")
    print(f"Saved figure: {args.save_fig}")


if __name__ == "__main__":
    train()