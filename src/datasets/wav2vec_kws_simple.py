# return type: {"id": id, "source": feats, "target": label}
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch


class Wav2VecKWS_SC(SPEECHCOMMANDS):

    CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go, zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')

    def __init__(self, subset: str = None):
        super(Wav2VecKWS_SC, self).__init__("./resources/data/", download=False, subset=subset)
        self.labels = sorted(Wav2VecKWS_SC.CLASSES)

    def label_to_index(self, word):
        return torch.tensor(self.labels.index(word))

    def index_to_label(self, index):
        return self.labels[index]

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = super(Wav2VecKWS_SC, self).__getitem__(idx)
        return {"id": idx, "source": waveform, "target": label, "label": self.label_to_index(label)}


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    tensors, targets = [], []

    for x in batch:
        tensors.append(x["source"])
        targets.append(x["label"])

    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

