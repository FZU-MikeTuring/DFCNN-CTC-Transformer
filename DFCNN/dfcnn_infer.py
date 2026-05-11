import os
import sys
import argparse
import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_ROOT = os.path.join(REPO_ROOT, "code")
DFCNN_DIR = os.path.join(CODE_ROOT, "DFCNN")
if DFCNN_DIR not in sys.path:
    sys.path.insert(0, DFCNN_DIR)
os.chdir(REPO_ROOT)

from data_process import compute_fbank
from ctc_decoder import ctc_beam_decode_ids, ctc_greedy_decode_ids
from dfcnn_model import DFCNN_CTC


def pad_to_multiple_of_8(fbank, freq=200):
    T = fbank.shape[0]
    target_T = (T // 8) * 8 + 8
    pad = np.zeros((target_T, freq), dtype=float)
    pad[:T, :] = fbank
    return pad, target_T // 8


def decode_tokens(log_probs, acoustic_vocab, blank_index, input_length, method, beam_size):
    if log_probs.dim() == 3:
        batch_log_probs = log_probs.permute(1, 0, 2)
    else:
        batch_log_probs = log_probs.unsqueeze(1)

    input_lengths = [int(input_length)]

    if method == "beam":
        pred_ids = ctc_beam_decode_ids(
            batch_log_probs,
            blank_index=blank_index,
            input_lengths=input_lengths,
            beam_size=beam_size,
        )[0]
    else:
        greedy_ids = torch.argmax(batch_log_probs, dim=-1).cpu().numpy()
        pred_ids = ctc_greedy_decode_ids(
            greedy_ids,
            blank_index=blank_index,
            input_lengths=input_lengths,
        )[0]

    return [acoustic_vocab[idx] for idx in pred_ids]


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (pt file)")
    parser.add_argument("--wav", required=True, help="Path to wav file to infer")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--decode_method", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam_size", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = load_checkpoint(args.ckpt, device)
    acoustic_vocab = ckpt.get("acoustic_vocab")
    if acoustic_vocab is None:
        raise RuntimeError("Checkpoint does not contain 'acoustic_vocab' list")

    model = DFCNN_CTC(num_classes=len(acoustic_vocab)).to(device)
    model.load_state_dict(ckpt["model_state_dict"]) if "model_state_dict" in ckpt else model.load_state_dict(ckpt)
    model.eval()

    # compute FBANK
    fbank = compute_fbank(args.wav)
    pad_fbank, input_length = pad_to_multiple_of_8(fbank, freq=fbank.shape[1])

    # prepare input: [B, 1, T, F]
    np_input = np.zeros((1, pad_fbank.shape[0], pad_fbank.shape[1], 1), dtype=float)
    np_input[0, : pad_fbank.shape[0], :, 0] = pad_fbank
    inputs = torch.from_numpy(np_input).float().permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        log_probs = model(inputs)  # [B, T', C]

    blank_index = len(acoustic_vocab) - 1
    tokens = decode_tokens(
        log_probs,
        acoustic_vocab,
        blank_index=blank_index,
        input_length=int(input_length),
        method=args.decode_method,
        beam_size=args.beam_size,
    )

    print("===== Inference result (pinyin tokens) =====")
    print(" ".join(tokens))


if __name__ == "__main__":
    main()
