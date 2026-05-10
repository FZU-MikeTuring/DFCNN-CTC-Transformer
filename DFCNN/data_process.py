import os 


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data_thchs30", "data")
DATA_TXT = os.path.join(REPO_ROOT, "data_thchs30", "data.txt")

# 获取data.txt文件，包含音频名称、拼音和汉字
def getTXT(): # 若data.txt已存在，则忽略此函数 
    dir_list = os.listdir(DATA_DIR) # 获取文件夹下的所有数据 
    # print((dir_list)) 
    temp = [] 
    for i in dir_list:# 遍历音频数据名称 
        if i.endswith(".trn"): # 获取文件名以.trn结尾的文件 
            temp.append(i) # 将文件添加至列表 
    # len(temp) 
    with open(DATA_TXT, "w", encoding="utf8") as f: 
        for i in temp: # 遍历所有trn文件 
            with open(os.path.join(DATA_DIR, i), "r", encoding="utf8") as f_: 
                f_str = f_.readlines() 
                f.write(i[:-4]+"\t"+f_str[1].strip()+"\t"+f_str[0].strip().replace(" ","")+"\n") # 将音频名称和trn文件内容写入data.txt中 
 
if __name__ =="__main__": 
    getTXT() 

if os.path.exists(DATA_TXT):
    with open(DATA_TXT, "r", encoding='UTF-8') as f:    #设置文件对象 
        f_ = f.readlines() 
        if __name__ == "__main__":
            for i in range(min(10, len(f_))): 
                for j in range(3): 
                    print(f_[i].split('\t')[j]) 
            print('语音总数量：',len(f_), '\n')

from scipy.fftpack import fft 
import numpy as np 
import scipy.io.wavfile as wav 
import matplotlib.pyplot as plt 
 
# 获取信号的时频图 
def compute_fbank(file): 
    fs, wavsignal = wav.read(file)
    # window: 25ms, step: 10ms
    time_window = 25
    window_length = int(fs * time_window / 1000)
    window_step = int(fs * 10 / 1000)

    # Hamming window
    x = np.arange(window_length, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * x / (window_length - 1))

    wav_arr = np.array(wavsignal)
    range0_end = max(0, (len(wavsignal) - window_length) // window_step + 1)

    desired_freq = 200
    data_input = np.zeros((range0_end, desired_freq), dtype=float)

    for i in range(range0_end):
        p_start = i * window_step
        p_end = p_start + window_length
        frame = wav_arr[p_start:p_end]
        if frame.shape[0] < window_length:
            frame = np.pad(frame, (0, window_length - frame.shape[0]), mode="constant")
        frame = frame * w
        spec = np.abs(fft(frame))
        # take first frequency bins and normalize to desired_freq
        if spec.shape[0] >= desired_freq:
            data_input[i] = spec[:desired_freq]
        else:
            data_input[i, : spec.shape[0]] = spec

    data_input = np.log(data_input + 1)
    return data_input
 
 
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
        with open(DATA_TXT, 'r', encoding='utf8') as f: 
            data = f.readlines() 
        for line in data: 
            wav_file, pin, han = line.split('\t') 
            self.wav_lst.append(wav_file) 
            self.pin_lst.append(pin.split(' ')) 
            self.han_lst.append(han.strip('\n')) 
        if self.data_length: 
            self.wav_lst = self.wav_lst[:self.data_length] 
            self.pin_lst = self.pin_lst[:self.data_length] 
            self.han_lst = self.han_lst[:self.data_length] 
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
                sub_list = _list[begin:end]# 获取一个批次的索引列表 
                for index in sub_list: 
                    fbank = compute_fbank(self.data_path + self.wav_lst[index])  #（时间帧，频率）
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1])) 
                    pad_fbank[:fbank.shape[0], :] = fbank 
                    label = self.pin2id(self.pin_lst[index], self.acoustic_vocab) 
                    label_ctc_len = self.ctc_len(label) 
                    if pad_fbank.shape[0] // 8 >= label_ctc_len: 
                        wav_data_lst.append(pad_fbank) 
                        label_data_lst.append(label) 
                pad_wav_data, input_length = self.wav_padding(wav_data_lst) 
                pad_label_data, label_length = self.label_padding(label_data_lst) 
                inputs = {'the_inputs': pad_wav_data, 
                          'the_labels': pad_label_data, 
                          'input_length': input_length, 
                          'label_length': label_length, 
                          } 
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )} 
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
                [self.pin2id(line, self.pin_vocab) + [0] * (max_len - len(line)) for line in input_batch]) 
            label_batch = np.array( 
                [self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch]) 
            yield input_batch, label_batch 
 
    def pin2id(self, line, vocab): 
        return [vocab.index(pin) for pin in line] 
 
    def han2id(self, line, vocab): 
        return [vocab.index(han) for han in line] 
 
    def wav_padding(self, wav_data_lst): 
        wav_lens = [len(data) for data in wav_data_lst] 
        wav_max_len = max(wav_lens) 
        wav_lens = np.array([leng // 8 for leng in wav_lens]) 
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1)) # （文件个数，最长时间帧，频率，通道）
        for i in range(len(wav_data_lst)): 
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i] 
        return new_wav_data_lst, wav_lens 
 
    def label_padding(self, label_data_lst): 
        label_lens = np.array([len(label) for label in label_data_lst]) 
        max_label_len = max(label_lens) 
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len)) # 让所有等于0
        for i in range(len(label_data_lst)): 
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i] # 将原来的标签数据放入新的标签数据中，剩余部分为0
        return new_label_data_lst, label_lens 

    def acoustic_model_vocab(self, data): 
        vocab = [] 
        for line in data: 
            line = line 
            for pin in line: 
                if pin not in vocab: 
                    vocab.append(pin) 
        vocab.append('_') #空白符号
        return vocab 

    def language_model_pin_vocab(self, data): 
        vocab = ['<PAD>'] #掩码符号
        for line in data: 
            for pin in line: 
                if pin not in vocab: 
                    vocab.append(pin) 
        return vocab

    def language_model_han_vocab(self, data): 
        vocab = ['<PAD>'] #掩码符号
        for line in data: 
            line = ''.join(line.split(' ')) 
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