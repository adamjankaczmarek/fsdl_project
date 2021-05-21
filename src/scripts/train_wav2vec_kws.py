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
from pytorch_lightning.loggers import MLFlowLogger
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from ray.tune.integration.pytorch_lightning import TuneReportCallback

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


num_workers = 12#32
pin_memory = True


cfg = {
    "batch_size": 64,
    "w2v_lr": 6.2e-5,
    "decoder_lr": 0.00123,
    "weight_decay": 8.8e-5
}

dataset_root = "/home/akaczmarek/workspace/fsdl_project/resources/data/SpeechCommandsSplit"


class Wav2VecKWS(pl.LightningModule):

    def __init__(self, config, w2v_sd):
        super(Wav2VecKWS, self).__init__()
        self.config = config
        self.num_labels = len(CLASSES)
        self.model = KWS(n_class=self.num_labels, w2v_sd=w2v_sd)
        metrics = MetricCollection([
            Accuracy(),
            Precision(num_classes=self.num_labels, average='macro'),
            Recall(num_classes=self.num_labels, average='macro')
        ])
        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.w2v_encoder.parameters(), 'lr': self.config.w2v_lr},
            {'params': self.model.decoder.parameters(), 'lr': self.config.decoder_lr},
        ], weight_decay=self.config.weight_decay)
        return optimizer

    def forward(x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['net_input'], batch['target']
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['net_input'], batch['target'] 
        logits = self.model(x)
        output = self.valid_metrics(torch.argmax(logits, dim=-1), y)
        self.log_dict(output)

    def validation_epoch_end(self, outputs):
        self.log_dict(self.valid_metrics.compute()) 

    def test_step(self, batch, batch_idx):
        x, y = batch['net_input'], batch['target']
        logits = self.model(x)
        output = self.test_metrics(torch.argmax(logits, dim=-1), batch)
        self.log_dict(output)

    def test_end(self):
        self.log_dict(self.test_metrics.compute())
        return self.test_metrics.compute()['test_Accuracy']


    def train_dataloader(self):
        dataset = SpeechCommandsDataset("training", dataset_root)
        return DataLoader(dataset,
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          collate_fn=_collate_fn(dataset.collater),
                          num_workers=num_workers,
                          pin_memory=pin_memory)

    def val_dataloader(self):
        dataset = SpeechCommandsDataset("validation", dataset_root)
        return DataLoader(dataset,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          drop_last=False,
                          collate_fn=_collate_fn(dataset.collater),
                          num_workers=num_workers,
                          pin_memory=pin_memory)

    def test_dataloader(self):
        dataset = SpeechCommandsDataset("test", dataset_root)
        return DataLoader(dataset,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          drop_last=False,
                          collate_fn=_collate_fn(dataset.collater),
                          num_workers=num_workers,
                          pin_memory=pin_memory)


def train_model(config, gpus, w2v):
    early_stop_callback = EarlyStopping(
        monitor="val_Accuracy", 
        min_delta=0.0, 
        patience=5, 
        verbose=True, 
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        "models/wav2vec_kws/", 
        save_top_k=1, 
        verbose=True,  
        monitor='val_Accuracy',
        mode='min',  
    )
    #tune_callback = TuneReportCallback({"acc": "val_Accuracy"}, on="validation_end")
    logger = TensorBoardLogger("tb_logs", name="wav2vec_kws")
    #mlf_logger = MLFlowLogger(experiment_name="wav2vec_kws", tracking_uri="http://192.168.0.32")
    trainer = pl.Trainer(
        gpus=gpus,
        accelerator='ddp',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[logger],
        accumulate_grad_batches=4,
        amp_level="O0",
        max_epochs=10,
        progress_bar_refresh_rate=1,
    )
    if trainer.global_rank == 0:
        mlflow.set_experiment("wav2vec_kws")
        mlflow.set_tracking_uri("http://192.168.0.32")
        mlflow.pytorch.autolog()
    #with mlflow.start_run() as run:
        mlflow.log_params(cfg)
    
    model = Wav2VecKWS(config, w2v)
    
    trainer.fit(model)
    #trainer.test()
    delattr(model, "trainer")
    if trainer.global_rank == 0:
        mlflow.pytorch.log_model(model, "wav2vec")    
    #    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    #trainer.save_checkpoint(config["model"]["out"])


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", type=str)
    parser.add_argument("--gpus", type=int) 
    parser.add_argument("--w2v", type=str)
    args = parser.parse_args()
    
    w2v_sd = torch.load(args.w2v)

    # config = load_config_from_yaml(args.config)
    config = argparse.Namespace(**cfg)
    train_model(config, args.gpus, w2v_sd)
 
