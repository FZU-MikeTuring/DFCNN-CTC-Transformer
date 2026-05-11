import math

import torch


NEG_INF = float("-inf")


def logsumexp(*values):
    valid = [v for v in values if v != NEG_INF]
    if not valid:
        return NEG_INF
    if len(valid) == 1:
        return valid[0]

    max_val = max(valid)
    total = sum(math.exp(v - max_val) for v in valid)
    return max_val + math.log(total)


def ctc_greedy_decode_ids(pred_ids, blank_index, input_lengths=None):
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


def ctc_prefix_beam_search(log_probs, blank_index, beam_size=10):
    # log_probs: Tensor [T, C]
    beams = {(): (0.0, NEG_INF)}

    for frame in log_probs:
        next_beams = {}

        top_k = min(int(beam_size), frame.shape[-1])
        top_scores, top_indices = torch.topk(frame, k=top_k)
        top_scores = top_scores.tolist()
        top_indices = top_indices.tolist()

        for prefix, (p_blank, p_non_blank) in beams.items():
            for score, token in zip(top_scores, top_indices):
                if token == blank_index:
                    n_blank, n_non_blank = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    n_blank = logsumexp(n_blank, p_blank + score, p_non_blank + score)
                    next_beams[prefix] = (n_blank, n_non_blank)
                    continue

                end_token = prefix[-1] if prefix else None
                extended = prefix + (token,)

                if token == end_token:
                    n_blank, n_non_blank = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    n_non_blank = logsumexp(n_non_blank, p_non_blank + score)
                    next_beams[prefix] = (n_blank, n_non_blank)

                    e_blank, e_non_blank = next_beams.get(extended, (NEG_INF, NEG_INF))
                    e_non_blank = logsumexp(e_non_blank, p_blank + score)
                    next_beams[extended] = (e_blank, e_non_blank)
                else:
                    e_blank, e_non_blank = next_beams.get(extended, (NEG_INF, NEG_INF))
                    e_non_blank = logsumexp(
                        e_non_blank,
                        p_blank + score,
                        p_non_blank + score,
                    )
                    next_beams[extended] = (e_blank, e_non_blank)

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: logsumexp(item[1][0], item[1][1]),
                reverse=True,
            )[:beam_size]
        )

    best_prefix, _ = max(
        beams.items(),
        key=lambda item: logsumexp(item[1][0], item[1][1]),
    )
    return list(best_prefix)


def ctc_beam_decode_ids(log_probs, blank_index, input_lengths=None, beam_size=10):
    # log_probs: Tensor [T, B, C]
    results = []

    for i in range(log_probs.shape[1]):
        sample = log_probs[:, i, :]
        if input_lengths is not None:
            sample = sample[: int(input_lengths[i])]
        results.append(
            ctc_prefix_beam_search(
                sample,
                blank_index=blank_index,
                beam_size=beam_size,
            )
        )

    return results
