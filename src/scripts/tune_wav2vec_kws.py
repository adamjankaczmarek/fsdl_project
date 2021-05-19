import argparse
import torch
import mlflow
from ray import tune
from src.scripts.train_wav2vec_kws import Wav2VecKWS 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.integration.pytorch_lightning import TuneReportCallback


config = {
    "w2v_lr": tune.loguniform(1e-5, 1e-1),
    "decoder_lr": tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    "batch_size": tune.choice([32, 64, 128, 256]),
    "mlflow": {
        "experiment_name": "wav2vec_kws",
        "tracking_uri": "http://192.168.0.32"
    },
}

@mlflow_mixin
def train_model(config, gpus, w2v, num_epochs=10):
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
    tune_callback = TuneReportCallback({"acc": "val_Accuracy"}, on="validation_end")
    logger = TensorBoardLogger("tb_logs", name="wav2vec_kws_tune")
    mlf_logger = MLFlowLogger(experiment_name="wav2vec_kws", tracking_uri="http://192.168.0.32")
    mlflow.pytorch.autolog()
    trainer = pl.Trainer(
        gpus=gpus, 
        callbacks=[checkpoint_callback, early_stop_callback, tune_callback],
        logger=[logger, mlf_logger],
        accumulate_grad_batches=4,
        amp_level="O0",
        max_epochs=num_epochs,
        progress_bar_refresh_rate=1,
    )
    config = argparse.Namespace(**config)
    model = Wav2VecKWS(config, w2v)
    trainer.fit(model)
    trainer.test()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus")
    parser.add_argument("--gpus-per-trial", type=float)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--w2v", type=str)
    args = parser.parse_args()

    w2v_sd = torch.load(args.w2v)
    gpus_per_trial = args.gpus_per_trial
    trainable = tune.with_parameters(
        train_model,
        gpus=args.gpus,
        w2v=w2v_sd,
        num_epochs=args.num_epochs,
    )

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 4,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=args.num_samples,
        name="tune_w2v_lr"
    )

    print(analysis.best_config)

if __name__ == "__main__":
    main()
