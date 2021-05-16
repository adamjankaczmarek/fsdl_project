import os
import soundfile as sf
from fairseq.data.audio.raw_audio_dataset import * 
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
import librosa
import numpy as np
import random
#from speech_commands.input_data import AudioProcessor
import torch


CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go, zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')


class SpeechCommandsDataset(RawAudioDataset):
    
    #sil=0.1, np=0.5, nl=0.7, sp=0.5, mp=0.5
    def __init__(self, mode='train', root='', 
                 sample_rate=16000, loudest_section=True, silence_percentage=0.1, noise_prob=0.5, noise_level=0.7, shift_prob=0.5, mask_prob=0.5, mask_len=0.1, tf_audio_processor=None):
        super(SpeechCommandsDataset, self).__init__(
            sample_rate,
            pad=False
        )
        self.mode = mode
        self.root = root
        self.mode_root = os.path.join(root,self.mode)
        self.ap = tf_audio_processor
        self.loudest_section = loudest_section
        self.sample_rate = sample_rate
        self.data = list()
        self.prep_dataset()

        if self.mode=='training':
            self.noise_data = list()
            self.prep_noise_dataset()
            self.noise_prob = noise_prob
            self.noise_level = noise_level
            self.shift_prob = shift_prob
            self.mask_prob = mask_prob
            self.mask_len = mask_len
        
            
    def prep_dataset(self):
        if self.ap is None:
            self.id = 0
            for c in CLASSES:
                for root, dir, files in os.walk(os.path.join(self.mode_root,c)):
                    for file in files:
                        f_path, cmd = os.path.join(root, file), c
                        self.data.append((f_path, cmd, self.id))
                        self.id += 1
        else:
            self.id = 0
            tf_data = self.ap.data_index[self.mode]
            for td in tf_data:
                f_path, cmd = td['file'], td['label']
                if cmd=='_silence_':
                    self.data += [('','silence',self.id)]
                elif cmd in CLASSES:
                    self.data.append((f_path, cmd, self.id))
                elif not cmd in CLASSES:
                    self.data.append((f_path, 'unknown', self.id))
                self.id += 1
        print(f"{self.mode} data number: {len(self.data)}")
    
    def prep_noise_dataset(self):
        noise_path = os.path.join(self.root,'_background_noise_')
        samples = []
        for root, dir, files in os.walk(noise_path):
            for file in files:
                f_path = os.path.join(root,file)
                wav, _ = sf.read(f_path)
                samples.append(wav)
        samples = np.hstack(samples)
        c = int(self.sample_rate)
        r = len(samples) // c
        self.noise_data = samples[:r*c].reshape(-1, c)
    
    def __getitem__(self, idx):
        f_path, cmd, id = self.data[idx]
        
        if f_path:
            wav, curr_sample_rate = sf.read(f_path)
            if curr_sample_rate!=self.sample_rate:
                wav, curr_sample_rate = librosa.resample(wav, curr_sample_rate, self.sample_rate), self.sample_rate
                
            if len(wav.shape)==2:
                wav = librosa.to_mono(wav.transpose(1,0))
                
            if self.loudest_section:
                wav = self.extract_loudest_section(wav)
            
            wav_len = len(wav)
            if wav_len < self.sample_rate:
                pad_size = self.sample_rate - wav_len
                wav = np.pad(wav, (round(pad_size/2)+1,round(pad_size/2)+1), 'constant', constant_values=0)
        else:
            wav, curr_sample_rate = np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate
        wav_len = len(wav)
        mid = int(len(wav)/2)
        cut_off = int(self.sample_rate/2)
        wav = wav[mid-cut_off:mid+cut_off]
        if self.mode=='training':
            if random.random()<self.shift_prob:
                percentage = random.uniform(-self.shift_prob, self.shift_prob)
                d = int(self.sample_rate*percentage)
                wav = np.roll(wav, d)
                if d>0:
                    wav[:d] = 0
                else:
                    wav[d:] = 0
            
            if random.random()<self.mask_prob:
                t = int(self.mask_len*self.sample_rate)
                t0 = random.randint(0, self.sample_rate - t)
                wav[t0:t+t0] = 0
            
            if random.random()<self.noise_prob:
                noise = random.choice(self.noise_data)
                if cmd=='silence':
                    percentage = random.uniform(0, 1)
                    wav = wav * (1 - percentage) + noise * percentage
                else:
                    percentage = random.uniform(0, self.noise_level)
                    wav = wav * (1 - percentage) + noise * percentage
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        y = CLASSES.index(cmd)
        return {"id": id, "target": y, "source": feats}
    
    def extract_loudest_section(self, wav, win_len=30):
        wav_len = len(wav)
        temp = abs(wav)

        st,et = 0,0
        max_dec = 0

        for ws in range(0, wav_len, win_len):
            cur_dec = temp[ws:ws+16000].sum()
            if cur_dec >= max_dec:
                max_dec = cur_dec
                st,et = ws, ws+16000
            if ws+16000 > wav_len:
                break

        return wav[st:et]
    
    def __len__(self):
        return len(self.data)
    
    def make_weights_for_balanced_classes(self):
        nclasses = len(CLASSES)
        count = np.zeros(nclasses)
        for item in self.data:
            count[CLASSES.index(item[1])] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[CLASSES.index(item[1])]
        return weight
    
def _collate_fn(collater):
    def collate_fn(samples):
        sub_samples = [s for s in samples if s["source"] is not None]
        if len(sub_samples) == 0:
            return {}
    
        batch = collater(samples)
        batch['target'] = torch.LongTensor([s["target"] for s in sub_samples])
        return batch
    return collate_fn
