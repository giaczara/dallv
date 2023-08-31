import pyrootutils

import lovely_tensors as lt


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` above is optional line to make environment more convenient
# should be placed at the top of each entry file
#
# main advantages:
# - allows you to keep all entry files in "src/" without installing project as a package
# - launching python file works no matter where is your current work dir
# - automatically loads environment variables from ".env" if exists
#
# how it works:
# - `setup_root()` above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to project root
# - loads environment variables from ".env" in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import hydra
import torch
import sys

import pytorch_lightning as pl
import torch.nn.functional as F

from os.path import isfile, join
from pathlib import Path
from typing import List, Optional, Tuple
from omegaconf import DictConfig, open_dict, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src.models.components.modules import Adapter

from src.models.components.modules import ImageCLIP, TextCLIP, VisualPrompt
from src.utils.utils import process_state_dict, convert_32, process_run_name
from src import utils

log = utils.get_pylogger(__name__)


# @utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.lovely_tensors:
        lt.monkey_patch()

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    if not cfg.datamodule.get("im2vid"):
        im2vid = False
    else:
        im2vid = cfg.datamodule.im2vid
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule,
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    if cfg.model.network.clip_version == "actionclip":
        import src.models.components.clip as clip

        clip_model, clip_state_dict = clip.load(
            cfg.model.network.arch,
            device="cuda",
            jit=False,
            tsm=cfg.model.network.tsm,
            T=cfg.datamodule.num_segments,
            dropout=cfg.model.network.dropout,
            emb_dropout=cfg.model.network.emb_dropout,
            pretrain=cfg.model.network.init,
            joint=cfg.model.network.joint,
        )
    else:
        from clip import clip

        clip_model, clip_state_dict = clip.load(
            cfg.model.network.arch,
            device="cuda",
            jit=False,
        )

        if cfg.model.distillation:
            clip_model_student, clip_state_dict = clip.load(
                cfg.model.network.arch_student,
                device="cuda",
                jit=False,
            )
            image_model_student = ImageCLIP(clip_model_student)
            text_model_student = TextCLIP(clip_model_student)

    if cfg.model.network.backbone == "clip_encoder":
        image_model = ImageCLIP(clip_model)
    else:
        raise NotImplementedError(
            "Backbone {} not implemented!".format(cfg.model.network.backbone)
        )

    text_model = TextCLIP(clip_model)
    fusion_model = VisualPrompt(
        cfg.model.network.sim_header, clip_state_dict, cfg.datamodule.num_segments
    )
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224).cuda()
        dim = image_model.cuda()(x).shape[1]
    adapter = {
        "source": Adapter(c_in=dim),
        "target": Adapter(c_in=dim),
    }

    if cfg.model.distillation:
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224).cuda()
            dim = image_model_student.cuda()(x).shape[1]
        adapter["final"] = Adapter(c_in=dim)


    if cfg.model.network.pretrained_model != "none":
        assert isfile(
            cfg.model.network.pretrained_model
        ), "Pretrained model not found at {}!".format(
            cfg.model.network.pretrained_model
        )
        log.info("Loading ActionClip pretrained model")
        checkpoint = torch.load(cfg.model.network.pretrained_model)
        model_state_dict = process_state_dict(checkpoint["model_state_dict"])
        fusion_model_state_dict = process_state_dict(
            checkpoint["fusion_model_state_dict"]
        )
        fusion_model.load_state_dict(fusion_model_state_dict)
        if not cfg.model.network.fusion_only:
            clip_model.load_state_dict(model_state_dict)
        del checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        if cfg.model.network.backbone == "clip_encoder":
            image_model.float()
            if cfg.model.distillation:
                image_model_student.float()
                text_model_student.float()
        else:
            for model in image_model.values():
                model.float()
        text_model.float()
    else:
        if cfg.model.network.backbone == "clip_encoder":
            convert_32(image_model)
            if cfg.model.distillation:
                convert_32(image_model_student)
                convert_32(text_model_student)
        else:
            for model in image_model.values():
                convert_32(model)
        convert_32(text_model)

    # config file
    run_name = process_run_name(cfg, im2vid=im2vid)

    # generic logging
    Path(cfg.paths.working_dir).mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        log_file_path = join(cfg.paths.working_dir, "log.txt")

    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        clip_model=clip_model,
        image_model=image_model,
        text_model=text_model,
        fusion_model=fusion_model,
        image_model_student=None,
        text_model_student=None,
        adapter=adapter if (cfg.model.network.use_adapter or cfg.model.classifier_baseline) else None,
        classifier=None,
        extra_args={
            "dataset": cfg.datamodule.dataset,
            "train_file": cfg.datamodule.source_train_file
            if cfg.model.domain_shift
            else cfg.datamodule.train_file,
            "epochs": cfg.trainer.max_epochs,
            "num_segments": cfg.datamodule.num_segments,
            "log_file_path": log_file_path,
            "working_dir": cfg.paths.working_dir,
            "save": cfg.save,
            "data_folder": cfg.datamodule.data_folder
            if cfg.datamodule.dataset == "hmdb51"
            else None,
            "classes_limit": cfg.datamodule.classes_limit,
            "command": " ".join(sys.argv),
            "prototypes_extraction": False,
            "subtasks_analysis": False,
            "im2vid": im2vid,
        },
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(
        cfg.get("logger"), run_name=run_name
    )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
