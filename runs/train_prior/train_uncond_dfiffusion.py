import os
import sys
sys.path.append("./")
import os.path as osp

from glob import glob

import torch
import hydra

import omegaconf
from omegaconf import OmegaConf, DictConfig
import wandb
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
import numpy as np

torch.utils.data._utils.worker.IS_DAEMON = False
os.environ["PYTHONFAULTHANDLER"] = "1"
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar as ProgressBar
from loguru import logger
os.environ["WANDB_API_KEY"] = "966c253d5fa55824f75c84f91d90a64d44a509ae"

from configs import constants as _C
from llib.utils.util import init_random_seed, set_random_seed
from llib.trainer.diffusion.ldmks_prior import DiffusionTrainer as ModelCls
# from llib.models import build_model

try:
    torch.set_float32_matmul_precision('high')
except:
    pass
torch.multiprocessing.set_sharing_strategy('file_system')

CUR_PATH = os.getcwd()
@hydra.main(version_base=None, config_path=os.path.join(CUR_PATH, "configs/train/prior"), config_name="config.yaml")
def main(cfg: DictConfig):
    
    OmegaConf.register_new_resolver("mult", lambda x,y: x*y)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x))
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x))

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_name = cfg.exp_name + '_' + timestamp if not cfg.debug_mode else "debug"
    work_dir = osp.join(CUR_PATH, cfg.work_dir, 'train', 'prior', exp_name)
    log_dir = osp.join(work_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    run_dir = HydraConfig.get().runtime.output_dir
    os.system(f"cp {run_dir}/.hydra/config.yaml {work_dir}/config.yaml")
    
    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    # set cudnn_benchmark
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    seed = init_random_seed(cfg.seed)
    set_random_seed(seed, deterministic=cfg.deterministic)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))

    experiment_loggers = []
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        log_graph=False,
    )

    experiment_loggers.append(tb_logger)
    project = "Debug" if cfg.debug_mode else "DeepGaitLab"
    wandb_logger = WandbLogger(project=project, tags=exp_name)
    wandb_logger.log_hyperparams(omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    ))
    experiment_loggers.append(wandb_logger)

    ckpt_callback = ModelCheckpoint(
        dirpath=f"{work_dir}/checkpoints",
        monitor='train/loss',
        verbose=True,
        save_top_k=3,
        mode='min',
        every_n_train_steps=5000,
    )

    model = ModelCls(cfg, device=device).to(device)

    trainer = pl.Trainer(
        overfit_batches=1 if cfg.debug_mode else 0,
        devices=int(cfg.gpus_n),
        max_steps=cfg.max_steps,
        strategy=cfg.strategy,
        logger=experiment_loggers,
        callbacks=[ckpt_callback, ProgressBar(refresh_rate=1)],
        default_root_dir=work_dir,
        check_val_every_n_epoch=9999,
        num_sanity_val_steps=0,
        accumulate_grad_batches=2,
        profiler='simple',
    )

    logger.info('*** Started training ***')
    trainer.fit(model)


if __name__ == '__main__':
    main()