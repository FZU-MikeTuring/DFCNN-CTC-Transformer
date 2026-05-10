import os
import wave

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data_thchs30", "data")
DATA_TXT = os.path.join(REPO_ROOT, "data_thchs30", "data.txt")


def getTXT():
    dir_list = os.listdir(DATA_DIR)
    temp = []

    for item in dir_list:
        if item.endswith(".trn"):
            temp.append(item)

    with open(DATA_TXT, "w", encoding="utf8") as f:
        for item in temp:
            with open(os.path.join(DATA_DIR, item), "r", encoding="utf8") as f_:
                f_str = f_.readlines()
                f.write(
                    item[:-4]
                    + "\t"
                    + f_str[1].strip()
                    + "\t"
                    + f_str[0].strip().replace(" ", "")
                    + "\n"
                )


def _hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(sample_rate, n_fft, n_mels, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sample_rate / 2.0

    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1

        left_end = min(center, filterbank.shape[1])
        right_end = min(right, filterbank.shape[1])

        for j in range(left, left_end):
            filterbank[i - 1, j] = (j - left) / max(1, center - left)

        for j in range(center, right_end):
            filterbank[i - 1, j] = (right - j) / max(1, right - center)

    return filterbank


def _load_wav(file):
    with wave.open(file, "rb") as wf:
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(num_frames)

    if sample_width == 2:
        dtype = np.int16
        scale = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    wavsignal = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    if num_channels > 1:
        wavsignal = wavsignal.reshape(-1, num_channels).mean(axis=1)

    wavsignal = wavsignal / scale

    return sample_rate, wavsignal


def compute_fbank(file):
    fs, wavsignal = _load_wav(file)

    frame_length = int(fs * 25 / 1000)
    frame_step = int(fs * 10 / 1000)
    n_mels = 200
    n_fft = 1

    while n_fft < frame_length:
        n_fft *= 2

    if len(wavsignal) < frame_length:
        wavsignal = np.pad(
            wavsignal,
            (0, frame_length - len(wavsignal)),
            mode="constant",
        )

    num_frames = 1 + max(0, (len(wavsignal) - frame_length) // frame_step)

    frame_index = (
        np.arange(frame_length)[None, :]
        + np.arange(num_frames)[:, None] * frame_step
    )
    frames = wavsignal[frame_index]
    frames = frames * np.hamming(frame_length)[None, :]

    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power_spec = (np.abs(spectrum) ** 2) / n_fft

    mel_filterbank = _build_mel_filterbank(fs, n_fft, n_mels)
    mel_spec = np.matmul(power_spec, mel_filterbank.T)
    mel_spec = np.maximum(mel_spec, 1e-10)

    log_mel = np.log(mel_spec).astype(np.float32)

    mean = np.mean(log_mel, axis=0, keepdims=True)
    std = np.std(log_mel, axis=0, keepdims=True)
    log_mel = (log_mel - mean) / np.maximum(std, 1e-5)

    return log_mel


class get_data():
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.source_init()

    def source_init(self):
        self.wav_lst = []
        self.pin_lst = []
        self.han_lst = []

        with open(DATA_TXT, "r", encoding="utf8") as f:
            data = f.readlines()

        for line in data:
            wav_file, pin, han = line.split("\t")
            self.wav_lst.append(wav_file)
            self.pin_lst.append(pin.split(" "))
            self.han_lst.append(han.strip("\n"))

        if self.data_length:
            self.wav_lst = self.wav_lst[: self.data_length]
            self.pin_lst = self.pin_lst[: self.data_length]
            self.han_lst = self.han_lst[: self.data_length]

        self.acoustic_vocab = self.acoustic_model_vocab(self.pin_lst)
        self.pin_vocab = self.language_model_pin_vocab(self.pin_lst)
        self.han_vocab = self.language_model_han_vocab(self.han_lst)

    def get_acoustic_model_batch(self):
        _list = [i for i in range(len(self.wav_lst))]

        while 1:
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []
                label_data_lst = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = _list[begin:end]

                for index in sub_list:
                    fbank = compute_fbank(self.data_path + self.wav_lst[index])
                    pad_fbank = np.zeros(
                        (fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]),
                        dtype=np.float32,
                    )
                    pad_fbank[: fbank.shape[0], :] = fbank
                    label = self.pin2id(self.pin_lst[index], self.acoustic_vocab)
                    label_ctc_len = self.ctc_len(label)

                    if pad_fbank.shape[0] // 8 >= label_ctc_len:
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)

                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {
                    "the_inputs": pad_wav_data,
                    "the_labels": pad_label_data,
                    "input_length": input_length,
                    "label_length": label_length,
                }
                outputs = {"ctc": np.zeros(pad_wav_data.shape[0],)}
                yield inputs, outputs

    def get_language_model_batch(self):
        batch_num = len(self.pin_lst) // self.batch_size

        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            input_batch = self.pin_lst[begin:end]
            label_batch = self.han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array(
                [
                    self.pin2id(line, self.pin_vocab) + [0] * (max_len - len(line))
                    for line in input_batch
                ]
            )
            label_batch = np.array(
                [
                    self.han2id(line, self.han_vocab) + [0] * (max_len - len(line))
                    for line in label_batch
                ]
            )
            yield input_batch, label_batch

    def pin2id(self, line, vocab):
        return [vocab.index(pin) for pin in line]

    def han2id(self, line, vocab):
        return [vocab.index(han) for han in line]

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1), dtype=np.float32)

        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, : wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]

        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))

        for i in range(len(label_data_lst)):
            new_label_data_lst[i][: len(label_data_lst[i])] = label_data_lst[i]

        return new_label_data_lst, label_lens

    def acoustic_model_vocab(self, data):
        vocab = []

        for line in data:
            for pin in line:
                if pin not in vocab:
                    vocab.append(pin)

        vocab.append("_")
        return vocab

    def language_model_pin_vocab(self, data):
        vocab = ["<PAD>"]

        for line in data:
            for pin in line:
                if pin not in vocab:
                    vocab.append(pin)

        return vocab

    def language_model_han_vocab(self, data):
        vocab = ["<PAD>"]

        for line in data:
            line = "".join(line.split(" "))

            for han in line:
                if han not in vocab:
                    vocab.append(han)

        return vocab

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)

        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1

        return label_len + add_len
