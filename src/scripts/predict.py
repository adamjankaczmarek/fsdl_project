
import argparse
import pytorch_lightning as pl
from src.models.wav2keyword import KWS
#from src.datasets.wav2vec_kws_simple import Wav2VecKWS_SC, collate_fn
from src.datasets.wav2vec_kws import SpeechCommandsDataset, CLASSES, _collate_fn
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import MetricCollection, Accuracy, Precision, Recall


class ModelHandler:

    def __init__(self, config):
        pass
