from pathlib import Path

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from typing import Tuple

root = pyrootutils.setup_root(
    search_from=__file__, indicator="pyproject.toml", pythonpath=True
)


import torch.nn.functional as F
import sys
from os.path import isfile, join
from src import utils
from src.models.components.modules import ImageCLIP, TextCLIP, VisualPrompt, Adapter
from src.utils.utils import (
    convert_32,
    process_run_name,
    process_state_dict,
    collate_results,
)

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[dict, dict]:
    """Predicts given checkpoint on a dataset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multi-runs, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    log.info(f"Instantiating data <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule,
        domain_shift=cfg.model.domain_shift,
        prototype_extraction=cfg.prototypes_extraction,
        subtasks_analysis=cfg.subtasks_analysis.enabled,
    )

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

    image_model = ImageCLIP(clip_model)
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
    classifier = {
        "source": F.linear(dim, cfg.datamodule.num_classes),
        "target": F.linear(dim, cfg.datamodule.num_classes),
    }
    if cfg.model.distillation:
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224).cuda()
            dim = image_model_student.cuda()(x).shape[1]
        adapter["final"] = Adapter(c_in=dim)
        classifier["final"] = F.linear(dim, cfg.datamodule.num_classes)

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
        text_model.float()
        image_model.float()
        if cfg.model.distillation:
            image_model_student.float()
            text_model_student.float()
    else:
        convert_32(text_model)
        convert_32(image_model)
        if cfg.model.distillation:
            convert_32(image_model_student)
            convert_32(text_model_student)

    # config file
    run_name = process_run_name(cfg, test=True)


    Path(cfg.paths.working_dir).mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        log_file_path = join(cfg.paths.working_dir, "log.txt")
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        clip_model=clip_model,
        image_model=image_model,
        text_model=text_model,
        fusion_model=fusion_model,
        image_model_student=image_model_student if cfg.model.distillation else None,
        text_model_student=text_model_student if cfg.model.distillation else None,
        adapter=adapter if cfg.model.network.use_adapter else None,
        extra_args={
            "dataset": cfg.datamodule.dataset,
            "source_train_file": cfg.datamodule.source_train_file,
            "epochs": cfg.trainer.max_epochs,
            "num_segments": cfg.datamodule.num_segments,
            "log_file_path": log_file_path,
            "working_dir": cfg.paths.working_dir,
            "data_folder": cfg.datamodule.data_folder
            if cfg.datamodule.dataset == "hmdb51"
            else None,
            "classes_limit": cfg.datamodule.classes_limit,
            "command": " ".join(sys.argv),
            "prototypes_extraction": cfg.prototypes_extraction,
            "subtasks_analysis": cfg.subtasks_analysis.enabled,
            "subtasks_analysis_class": cfg.subtasks_analysis.class_,
        },
    )

    if model.HAS_CUSTOM_CKPT_HANDLER:
        cfg.ckpt_path = None
    else:
        assert cfg.ckpt_path

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting predicting!")
    preds = trainer.predict(
        model=model, dataloaders=datamodule, ckpt_path=cfg.ckpt_path
    )

    save_path = Path(cfg.paths.output_dir, "predictions.pt")
    if cfg.artifact_path:
        save_path = Path(cfg.paths.artifact_dir, cfg.artifact_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving predictions to {save_path}")
    torch.save(collate_results(preds, pivot=cfg.get("pivot", None)), save_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    predict(cfg)


if __name__ == "__main__":
    main()
