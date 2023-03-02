import hashlib
import os
import sys
import gin
import pytorch_lightning as pl
import torch
import shutil
from absl import flags
from torch.utils.data import DataLoader
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from torch import LongTensor, Tensor, nn
import dotenv
from huggingface_hub import Repository

from einops import rearrange
import wandb

import rave
import rave.core
import rave.dataset

dotenv.load_dotenv(override=True)

wandb_logger = pl.loggers.WandbLogger(
    project="rave",
    entity="tvergho1",
    job_type="train",
    group="",
    save_dir="/logs"
)

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run', required=True)
flags.DEFINE_multi_string('config',
                          default='v2.gin',
                          help='RAVE configuration to use')
flags.DEFINE_string('db_path',
                    None,
                    help='Preprocessed dataset path',
                    required=True)
flags.DEFINE_integer('max_steps',
                     6000000,
                     help='Maximum number of training steps')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('n_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('batch', 8, help='Batch size')
flags.DEFINE_string('ckpt',
                    None,
                    help='Path to previous checkpoint of the run')
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_integer('workers',
                     default=8,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_multi_integer('gpu', default=None, help='GPU to use')
flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_bool('normalize',
                  default=False,
                  help='Train RAVE on normalized signals')
flags.DEFINE_bool('progress',
                  default=True,
                  help='Display training progress bar')


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def main(argv):
    torch.backends.cudnn.benchmark = True
    gin.parse_config_files_and_bindings(
        map(add_gin_extension, FLAGS.config),
        FLAGS.override,
    )

    model = rave.RAVE()

    if FLAGS.derivative:
        model.integrator = rave.dataset.get_derivator_integrator(model.sr)[1]

    dataset = rave.dataset.get_dataset(FLAGS.db_path,
                                       model.sr,
                                       FLAGS.n_signal,
                                       derivative=FLAGS.derivative,
                                       normalize=FLAGS.normalize)
    train, val = rave.dataset.split_dataset(dataset, 98)
    num_workers = FLAGS.workers

    if os.name == "nt" or sys.platform == "darwin":
        num_workers = 0

    train = DataLoader(train,
                       FLAGS.batch,
                       True,
                       drop_last=True,
                       num_workers=num_workers)
    val = DataLoader(val, FLAGS.batch, False, num_workers=num_workers)

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(monitor="validation",
                                                         filename="best", every_n_train_steps=FLAGS.val_every)
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last", every_n_train_steps=FLAGS.val_every)

    val_check = {}
    if len(train) >= FLAGS.val_every:
        val_check["val_check_interval"] = FLAGS.val_every
    else:
        nepoch = FLAGS.val_every // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    gin_hash = hashlib.md5(
        gin.operative_config_str().encode()).hexdigest()[:10]

    RUN_NAME = f'{FLAGS.name}_{gin_hash}'

    os.makedirs(os.path.join("runs", RUN_NAME), exist_ok=True)

    if FLAGS.gpu == [-1]:
        gpu = 0
    else:
        gpu = FLAGS.gpu or rave.core.setup_gpu()

    print('selected gpu:', gpu)

    accelerator = None
    devices = None
    if FLAGS.gpu == [-1]:
        pass
    elif torch.cuda.is_available():
        accelerator = "cuda"
        devices = FLAGS.gpu or rave.core.setup_gpu()
    elif torch.backends.mps.is_available():
        print(
            "Training on mac is not available yet. Use --gpu -1 to train on CPU (not recommended)."
        )
        exit()
        accelerator = "mps"
        devices = 1

    trainer = pl.Trainer(
        logger=[
            pl.loggers.TensorBoardLogger(
                "runs",
                name=RUN_NAME,
            ),
            wandb_logger
        ],
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            last_checkpoint,
            validation_checkpoint,
            rave.model.WarmupCallback(),
            rave.model.QuantizeCallback(),
            rave.core.LoggerCallback(rave.core.ProgressLogger(RUN_NAME)),
            SampleLogger(),
            # HuggingFaceHubCallback('tvergho/rave-maestro')
        ],
        max_epochs=100000,
        max_steps=FLAGS.max_steps,
        profiler="simple",
        enable_progress_bar=FLAGS.progress,
        **val_check,
        log_every_n_steps=FLAGS.val_every,
    )

    run = rave.core.search_for_run(FLAGS.ckpt)
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    with open(os.path.join("runs", RUN_NAME, "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    trainer.fit(model, train, val, ckpt_path=run)


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    return wandb_logger


def log_wandb_audio_batch(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{id}": wandb.Audio(
                samples[idx],
                caption=caption,
                sample_rate=sampling_rate,
            )
            for idx in range(num_items)
        }
    )

class SampleLogger(Callback):
    def __init__(
        self,
    ) -> None:
        self.num_items = 1
        self.channels = 2
        self.sampling_rate = 44100
        self.num_steps = 20
        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment

        model = pl_module
        k = model.encode(torch.randn(1, 1, 2**18).to(model.device))
        x = torch.randn_like(k)
        z = model.decode(x)
        
        log_wandb_audio_batch(
            logger=wandb_logger,
            id="sample_distribution",
            samples=z,
            sampling_rate=self.sampling_rate,
            caption=f"Sampled in {self.num_steps} steps",
        )

        sample = batch[0].unsqueeze(0).unsqueeze(0)
        log_wandb_audio_batch(
            logger=wandb_logger,
            id="original",
            samples=sample,
            sampling_rate=self.sampling_rate,
            caption=f"Sampled in {self.num_steps} steps",
        )

        x = model.encode(sample)
        y = model.decode(x)

        log_wandb_audio_batch(
            logger=wandb_logger,
            id="decoded",
            samples=y,
            sampling_rate=self.sampling_rate,
            caption=f"Sampled in {self.num_steps} steps",
        )

        if is_train:
            pl_module.train()

# class HuggingFaceHubCallback(Callback):
#     def __init__(self, model_id: str):
#         self.model_id = model_id
#         self.local_dir = os.path.join(os.getcwd(), "rave-hub")
#         self.repo = Repository(local_dir=self.local_dir, clone_from=model_id)
#         self.repo.git_pull()
    
#     def on_validation_end(self, trainer, pl_module):
#         path = trainer.checkpoint_callback.best_model_path
#         print("Path", path)
#         if os.path.exists(path):
#             shutil.copy(path, self.local_dir)
#             self.repo.push_to_hub(commit_message="Latest model", blocking=True)