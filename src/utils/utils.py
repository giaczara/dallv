import time
import warnings
import torch
import hydra
import io

import numpy as np
import torch.nn as nn
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
import src.models.components.clip as clip
import torch.nn.functional as F

from importlib.util import find_spec
from pathlib import Path
from json import load
from typing import Callable, List
from os import uname
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from src.utils import pylogger, rich_utils
from src.models.components.text_prompt import hierarchical_text_prompt
from random import choice
from string import ascii_lowercase, digits
from os import listdir
from collections import OrderedDict

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = (
                f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            )
            save_file(
                path, content
            )  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig, run_name) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            if "mlflow" in lg_conf._target_:
                logger.append(hydra.utils.instantiate(lg_conf, run_name=run_name))
            elif "wandb" in lg_conf._target_:
                logger.append(hydra.utils.instantiate(lg_conf, name=run_name))
            else:
                logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


def get_random_string(length):
    # choose from all lowercase letter
    characters = ascii_lowercase + digits
    result_str = "".join(choice(characters) for _ in range(length))
    return result_str


def process_run_name(config, test=False, im2vid=False):
    run_name = "[{}]-[{}]-[{}]-{}-{}-s={}-t={}".format(
        "TRAIN" if not test else "TEST",
        config.tags[0] if len(list(config.tags)) else "exp",
        uname().nodename,
        "{}_split_{}".format(config.datamodule.dataset, config.datamodule.split)
        if im2vid
        else config.datamodule.dataset,
        "CDA",
        config.model.loss.source.weight,
        config.model.loss.target.weight,
    )
    return run_name


def get_classes(data):
    ek_map = {
        "opening": 2,
        "taking": 0,
        "closing": 3,
        "putting": 1,
        "washing": 4,
        "pouring": 7,
        "mixing": 6,
        "cutting": 5,
    }

    hu_map = {
        "climb": 0,
        "fencing": 1,
        "golf": 2,
        "kick ball": 3,
        "pullup": 4,
        "punch": 5,
    }

    daily_da_map = {
        "drink": 0,
        "jump": 1,
        "pick": 2,
        "pour": 3,
        "push": 4,
        "run": 5,
        "walk": 6,
        "wave": 7,
    }

    hmdb_ucf_map = {
        "climb": 0,
        "fencing": 1,
        "golf": 2,
        "kick ball": 3,
        "pullup": 4,
        "punch": 5,
        "pushup": 6,
        "ride bike": 7,
        "ride horse": 8,
        "shoot ball": 9,
        "shoot bow": 10,
        "walk": 11,
    }

    sports_da_map = {
        "archery": 0,
        "baseball": 1,
        "basketball": 2,
        "biking": 3,
        "bowling": 4,
        "breast stroke": 5,
        "diving": 6,
        "fencing": 7,
        "hockey": 8,
        "floor gymnastics": 9,
        "golfing": 10,
        "horseback riding": 11,
        "kayaking": 12,
        "rock climbing": 13,
        "rope climbing": 14,
        "skateboarding": 15,
        "skiing": 16,
        "sumo wrestling": 17,
        "surfing": 18,
        "taichi": 19,
        "tennis": 20,
        "trampoline jumping": 21,
        "volleyball": 22,
    }

    hmdb51_map = None
    if data["data_folder"] is not None:
        hmdb51_map = {
            c.replace("_", " "): i
            for i, c in enumerate(sorted(listdir(data["data_folder"])))
        }

    uo_map = {"basketball": 0, "clean and jerk": 1, "throw discus": 2}

    if "ek" in data["dataset"]:
        res = [[i, c] for c, i in sorted(ek_map.items(), key=lambda x: x[1])]
    elif data["dataset"] in [
        "hmdb_ucf",
        "ucf_hmdb",
        "hmdb_ucf_im2vid",
        "ucf_hmdb_im2vid",
    ]:
        res = [[i, c] for c, i in hmdb_ucf_map.items()]
    elif data["dataset"] in [
        "kinetics_hmdb",
        "hmdb_kinetics",
        "kinetics_arid",
        "arid_kinetics",
        "hmdb_arid",
        "arid_hmdb",
        "hmdb_mit",
        "mit_hmdb",
        "kinetics_mit",
        "arid_mit",
        "mit_kinetics",
        "mit_arid",
    ]:
        res = [[i, c] for c, i in daily_da_map.items()]
    elif data["dataset"] in [
        "ucf_kinetics",
        "ucf_sports",
        "kinetics_ucf",
        "kinetics_sports",
        "sports_kinetics",
        "sports_ucf",
    ]:
        res = [[i, c] for c, i in sports_da_map.items()]
    elif "olympic" in data["dataset"]:
        res = [[i, c] for c, i in uo_map.items()]
    elif data["dataset"] == "hmdb51":
        res = [[i, c] for c, i in sorted(hmdb51_map.items(), key=lambda x: x[1])]
    else:
        folder = data["train_file"]
        classes = sorted(listdir(folder))
        res = [[i, c] for i, c in enumerate(classes)]

    if data["classes_limit"] > 0:
        res = res[: data["classes_limit"]]

    return res


def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i, k] = 1
    return gt


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


def process_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def convert_32(model: nn.Module):
    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)


def epoch_saving(epoch, model, fusion_model, optimizer, filename):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "fusion_model_state_dict": fusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )  # just change to your preferred folder/filename


def best_saving(working_dir, epoch, model, fusion_model, optimizer):
    best_name = "{}/model_best.pt".format(working_dir)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "fusion_model_state_dict": fusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        best_name,
    )  # just change to your preferred folder/filename


class LabelsManager:
    def __init__(self, data):

        dataset = data["dataset"]
        if "ek" in dataset:
            self.label_map = {
                2: "opening",
                0: "taking",
                3: "closing",
                1: "putting",
                4: "washing",
                7: "pouring",
                6: "mixing",
                5: "cutting",
            }
        elif dataset in [
            "kinetics_hmdb",
            "hmdb_kinetics",
            "kinetics_arid",
            "arid_kinetics",
            "hmdb_arid",
            "arid_hmdb",
            "hmdb_mit",
            "mit_hmdb",
            "kinetics_mit",
            "arid_mit",
            "mit_kinetics",
            "mit_arid",
        ]:
            self.label_map = {
                0: "drink",
                1: "jump",
                2: "pick",
                3: "pour",
                4: "push",
                5: "run",
                6: "walk",
                7: "wave",
            }
        elif dataset in [
            "ucf_kinetics",
            "ucf_sports",
            "kinetics_ucf",
            "kinetics_sports",
            "sports_kinetics",
            "sports_ucf",
        ]:
            self.label_map = {
                0: "archery",
                1: "baseball",
                2: "basketball",
                3: "biking",
                4: "bowling",
                5: "breast stroke",
                6: "diving",
                7: "fencing",
                8: "hockey",
                9: "floor gymnastics",
                10: "golfing",
                11: "horseback riding",
                12: "kayaking",
                13: "rock climbing",
                14: "rope climbing",
                15: "skateboarding",
                16: "skiing",
                17: "sumo wrestling",
                18: "surfing",
                19: "taichi",
                20: "tennis",
                21: "trampoline jumping",
                22: "volleyball",
            }
        elif dataset in ["hmdb_ucf", "ucf_hmdb", "hmdb_ucf_im2vid", "ucf_hmdb_im2vid"]:
            self.label_map = {
                0: "climb",
                1: "fencing",
                2: "golf",
                3: "kick_ball",
                4: "pullup",
                5: "punch",
                6: "pushup",
                7: "ride bike",
                8: "ride horse",
                9: "shoot ball",
                10: "shoot bow",
                11: "walk",
            }
        elif dataset == "hmdb51":
            assert data["data_folder"] is not None
            self.label_map = {
                i: c for i, c in enumerate(sorted(listdir(data["data_folder"])))
            }
        else:
            self.label_map = {
                0: "basketball",
                1: "clean and jerk",
                2: "throw discus",
            }

        self.rev_label_map = {v: k for k, v in self.label_map.items()}

    def convert(self, labels, reverse=False):
        if reverse:
            return [self.rev_label_map[label] for label in labels]
        return [self.label_map[label] for label in labels]

    def convert_single_label(self, label, reverse=False):
        if reverse:
            return self.rev_label_map[label]
        return self.label_map[label]

    def index_to_example(self, index):
        return self.label_map[index]


def get_decomposition_matrix(
    num_classes, text_embedding, middle_dim, subtask_weight=None, version="standard"
):
    if version == "standard":
        res = torch.eye(num_classes, device=text_embedding.device).repeat(middle_dim, 1)
    elif version == "weighted":
        assert (
            subtask_weight is not None
        ), "Weights must be provided for weighted decomposition"
        weights = [middle_dim - ((middle_dim - 1) * subtask_weight)] + [
            subtask_weight for _ in range(middle_dim - 1)
        ]
        res = torch.cat([torch.eye(num_classes) * w for w in weights]).to(
            text_embedding.device
        )
    else:
        raise ValueError("Unknown decomposition version")
    return res


def apply_moving_average(model, similarity):
    _, n_frames, _ = similarity.size()
    if model.hparams.network.moving_average_weight == "uniform":
        moving_average_weights = torch.tensor(
            [1 for _ in range(model.hparams.network.moving_average_size)]
        ).softmax(dim=0)
    else:
        moving_average_weights = [
            model.hparams.network.moving_average_weight
            for _ in range(model.hparams.network.moving_average_size)
        ] + [
            1
            - (
                model.hparams.network.moving_average_size
                * model.hparams.network.moving_average_weight
            )
        ]
    for f in range(model.hparams.network.moving_average_size, n_frames):
        window = [
            similarity[:, j, :]
            * moving_average_weights[
                j - (f - model.hparams.network.moving_average_size)
            ]
            for j in range(f - model.hparams.network.moving_average_size, f)
        ]
        similarity[:, f, :] = torch.stack(window).mean(dim=0)
    return similarity


def apply_moving_average2(model, similarity):
    output = torch.zeros_like(similarity)
    moving_average_weights = None
    if model.hparams.network.moving_average_weight != "uniform":
        moving_average_weights = [
            model.hparams.network.moving_average_weight
            for _ in range(model.hparams.network.moving_average_size - 1)
        ] + [
            1
            - (
                (model.hparams.network.moving_average_size - 1)
                * model.hparams.network.moving_average_weight
            )
        ]
        moving_average_weights = (
            torch.tensor(moving_average_weights).view(1, -1, 1).to(similarity.device)
        )
    for i in range(similarity.shape[1]):
        window = similarity[
            :, max(0, i - model.hparams.network.moving_average_size + 1) : i + 1, :
        ]
        if (
            window.size(1) >= model.hparams.network.moving_average_size
            and model.hparams.network.moving_average_weight != "uniform"
        ):
            window = window * moving_average_weights
        output[:, i, :] = window.mean(dim=1)
        # output[:, i, :] = similarity[
        #     :, max(0, i - model.hparams.network.moving_average_size + 1) : i + 1, :
        # ].mean(dim=1)
    return output


def frame_level_loss(
    labels,
    text_model,
    num_segments,
    clip_model,
    fusion_model,
    image_model,
    videos,
    num_text_aug,
    dataset,
    labels_manager,
    loss,
    adapter,
    backbone,
    device,
):
    json_file = "data/hierarchical_prompts.json"
    with open(json_file, "r") as json_file:
        class_maps = load(json_file)
    if dataset in ["hmdb_ucf", "ucf_hmdb"]:
        class_map = class_maps["hmdb_ucf"]
    elif dataset in [
        "kinetics_hmdb",
        "kinetics_arid",
        "hmdb_kinetics",
        "arid_kinetics",
        "hmdb_arid",
        "arid_hmdb",
        "hmdb_mit",
        "mit_hmdb",
        "kinetics_mit",
        "arid_mit",
        "mit_kinetics",
        "mit_arid",
    ]:
        class_map = class_maps["daily_da"]["v3"]
    elif "ek" in dataset:
        class_map = class_maps["epic_kitchens"]["v3"]
    elif dataset == "hmdb51":
        class_map = class_maps["hmdb51"]
    else:
        raise ValueError("Hierarchical prompts for {} not available!".format(dataset))

    text_aug = [
        f"a photo of action {{}}",
        f"a picture of action {{}}",
        f"Human action of {{}}",
        f"{{}}, an action",
        f"{{}} this is an action",
        f"{{}}, a video of action",
        f"Playing action of {{}}",
        f"{{}}",
        f"Playing a kind of action, {{}}",
        f"Doing a kind of action, {{}}",
        f"Look, the human is {{}}",
        f"Can you recognize the action of {{}}?",
        f"Video classification of {{}}",
        f"A video of {{}}",
        f"The man is {{}}",
        f"The woman is {{}}",
    ]

    video_loss = 0.0
    for i in range(videos.size(0)):
        with torch.no_grad():
            label_string = labels_manager.convert_single_label(labels[i].item())
            classes = class_map[label_string]
            text_dict = {}
            for ii, txt in enumerate(text_aug):
                text_dict[ii] = torch.cat(
                    [clip.tokenize(txt.format(c)) for c in classes]
                )
            classes = torch.cat([v for k, v in text_dict.items()])
            text_inputs = classes.to(device)
            text_embedding = text_model(text_inputs)
            if backbone == "clip_encoder":
                video = (
                    videos[i]
                    .unsqueeze(0)
                    .view((-1, num_segments, 3) + videos.size()[-2:])
                )
                b, t, c, h, w = video.size()
                video = video.view(-1, c, h, w)
                video_embedding = clip_model.encode_image(video).view(
                    1, num_segments, -1
                )
            else:
                video = (
                    videos[i]
                    .unsqueeze(0)
                    .view((-1, num_segments, 3) + videos.size()[-2:])
                ).permute(0, 2, 1, 3, 4)
                b, c, t, h, w = video.size()
                video_feat, scale_feats = image_model["feat"](video)
                video_feat, scale_feats = image_model["fc"](video_feat, scale_feats)
                video_embedding, scale_feats, pred = image_model["cls"](
                    video_feat, scale_feats
                )
            if adapter is not None:
                video_embedding = adapter(video_embedding)
            video_embedding /= video_embedding.norm(dim=-1, keepdim=True)
            similarity = 100.0 * video_embedding @ text_embedding.T
            similarity = similarity.view(b, num_segments, num_text_aug, -1).softmax(
                dim=-1
            )
            similarity = similarity.mean(dim=2, keepdim=False)
            _, pseudo_y = similarity.topk(1, dim=-1)

        # compute logits
        frames_loss = 0.0
        for f in range(num_segments):
            frame_embedding = video_embedding[:, f, :]
            logit_scale = clip_model.logit_scale.exp()
            logits_video_frame, logits_text_frame = create_logits(
                frame_embedding, text_embedding, logit_scale
            )

            # generate ground-truth label
            ground_truth_source = (
                torch.tensor(gen_label(pseudo_y[:, f, :]), dtype=video_embedding.dtype)
                .float()
                .to(device)
            )

            # compute loss
            loss_image = loss["image"](logits_video_frame, ground_truth_source)
            loss_text = loss["text"](logits_text_frame, ground_truth_source)
            frame_loss = (loss_image + loss_text) / 2
            frames_loss += frame_loss

        frames_loss /= num_segments
        video_loss += frames_loss

    video_loss /= videos.size(0)

    return video_loss


def compute_statistics_matrix(
    videos,
    classes_names,
    device,
    text_model,
    num_segments,
    clip_model,
    adapter,
    num_text_aug,
    dataset,
    hierarchical_prompt_version,
):

    matrices = []
    with torch.no_grad():
        for i in range(videos.size(0)):
            text_inputs, middle_dim, _, n_templates, texts = hierarchical_text_prompt(
                classes_names=classes_names,
                dataset=dataset,
                version=hierarchical_prompt_version,
            )
            text_inputs = text_inputs.to(device)
            B, N, _ = text_inputs.shape
            text_inputs = text_inputs.view(B * N, -1)
            text_embedding = text_model(text_inputs)

            video = (
                videos[i].unsqueeze(0).view((-1, num_segments, 3) + videos.size()[-2:])
            )
            b, t, c, h, w = video.size()
            video = video.view(-1, c, h, w)
            video_embedding = clip_model.encode_image(video).view(1, num_segments, -1)
            assert adapter is not None, "Adapter is required for statistics loss"
            video_embedding = adapter(video_embedding)
            video_embedding /= video_embedding.norm(dim=-1, keepdim=True)
            similarity = 100.0 * video_embedding @ text_embedding.T
            similarity = similarity.view(
                b, num_segments, num_text_aug, -1
            )  # .softmax(dim=-1)
            similarity = similarity.mean(dim=2, keepdim=False)
            matrices.append(similarity)
    matrices = torch.cat(matrices, dim=0)
    return matrices


def load_prototypes(source_dataset, labels):
    prototypes_all = torch.load("prototypes/{}.pt".format(source_dataset))
    prototypes = []
    for label in labels:
        prototypes.append(prototypes_all[label])
    prototypes = torch.stack(prototypes, dim=0)
    return prototypes


def compute_similarity(model, video_embedding, text_embedding, middle_dim):
    compute_similarity
    b = video_embedding.size(0)

    # hierarchical text prompts
    if model.hparams.prompts.eval == "hierarchical":

        # condition sub-tasks to tasks at embedding level
        if model.hparams.prompts.conditioned:
            text_embedding = text_embedding.view(middle_dim, model.num_classes, -1)
            text_embedding[1:, :, :] = (
                model.hparams.prompts.alpha * text_embedding[1:, :, :]
            ) + (
                (1 - model.hparams.prompts.alpha)
                * text_embedding[0, :, :].unsqueeze(0).repeat(middle_dim - 1, 1, 1)
            )
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)
            text_embedding = text_embedding.view(middle_dim * model.num_classes, -1)
        else:
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

        # compute similarity for hierarchical prompts
        similarity = 100.0 * video_embedding @ text_embedding.T
        similarity = similarity / model.hparams.loss.temperature

        # get decomposition matrix to condition sub-tasks to tasks
        decomposition_matrix = get_decomposition_matrix(
            model.num_classes,
            text_embedding,
            middle_dim,
            subtask_weight=model.hparams.prompts.subtask_weight,
            version=model.hparams.prompts.decomposition_version,
        )

        # apply decomposition matrix
        scores = similarity @ decomposition_matrix
        similarity = similarity.softmax(dim=-1) @ decomposition_matrix

        # aggregation across logits
        if model.hparams.network.fusion_input == "logits":

            # apply moving average
            if model.hparams.network.moving_average:
                similarity = apply_moving_average(model, similarity)

            # temporal aggregation
            scores = scores.mean(dim=1, keepdim=False)
            similarity = similarity.mean(dim=1, keepdim=False)

    # standard text prompts
    else:

        # compute similarity for standard prompts
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        if model.hparams.network.backbone == "trn":
            text_embedding = text_embedding.float()
        similarity = 100.0 * video_embedding @ text_embedding.T
        similarity = similarity / model.hparams.loss.temperature
        if model.hparams.network.fusion_input == "logits":
            similarity = similarity.view(
                b, middle_dim, model.extra_args["num_segments"], -1
            )
            # temporal aggregation
            similarity = similarity.mean(dim=2, keepdim=False)
        scores = similarity.view(b, middle_dim, -1)
        similarity = scores.softmax(dim=-1)

        # average across prompts
        scores = scores.mean(dim=1, keepdim=False)
        similarity = similarity.mean(dim=1, keepdim=False)

    return similarity, scores


def self_supervised_representation_quality_score(features):
    """Compute the self-supervised representation quality score.

    Args:
        images_z (torch.Tensor): Images embeddings.

    References:
        - Kalibhat et al. Understanding Failure Modes of Self-Supervised Learning. preprint 2022.
    """
    std, mean = torch.std_mean(features, dim=-1, keepdim=True)
    highly_active_mask = features > mean + std
    highly_active_z = features[highly_active_mask]
    quality_score = (highly_active_z - mean).sum(dim=-1) / highly_active_mask.sum(
        dim=-1
    )

    return quality_score


def select_samples_top_k(outputs, confidences, top_k):
    """
    Select the top_k samples according to confidence. Supports classwise balance.
    """

    confidences = torch.tensor(confidences)
    sorted_idxs = np.argsort(-confidences)
    confidences = confidences[sorted_idxs]

    # convert top_k to integer if is has no decimals
    if int(top_k) == top_k:
        top_k = int(top_k)

    # handle classwise balance
    class_idxs = torch.tensor(outputs)

    class_top_k = []
    for cls in list(set(class_idxs.tolist())):
        mask = class_idxs == cls
        mask = torch.tensor(mask).bool()
        threshold = torch.quantile(confidences[mask].to(float), q=(1 - top_k))
        mask_conf = confidences > threshold
        m_final = mask & mask_conf
        indices = m_final.nonzero().squeeze()
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        class_top_k.extend(indices.tolist())

    assert len(class_top_k) <= len(class_idxs), "More samples selected than available!"

    return class_top_k


def compute_adapter_outputs(
    video_embedding, text_embedding, middle_dim, model, adapter, device
):
    adapter = adapter.to(device)
    video_embedding = video_embedding.to(device)
    with torch.no_grad():
        video_embedding = adapter(video_embedding)
    video_embedding /= video_embedding.norm(dim=-1, keepdim=True)
    similarity, scores = compute_similarity(
        model, video_embedding, text_embedding, middle_dim
    )
    return similarity, scores, video_embedding


def compute_ensemble(model, similarities, video_embeddings, scores, scores_no_adapter):

    if model.hparams.network.ensemble.method == "hard_max":
        if "source_model" in model.hparams.network.ensemble.pred_ensemble:
            assert "source" in scores, "Source adapter scores not found!"
            similarity = torch.max(scores["source"], scores_no_adapter)
            if "target_model" in model.hparams.network.ensemble.pred_ensemble:
                assert "target" in scores, "Target adapter scores not found!"
                similarity = torch.max(similarity, scores["target"])
        elif "target_model" in model.hparams.network.ensemble.pred_ensemble:
            assert "target" in scores, "Target adapter scores not found!"
            similarity = torch.max(scores["target"], scores_no_adapter)
        similarity = similarity.softmax(dim=1)

    elif model.hparams.network.ensemble.method == "hard_mean":
        if "source_model" in model.hparams.network.ensemble.pred_ensemble:
            assert "source" in scores, "Source adapter scores not found!"
            numerator = scores["source"] + scores_no_adapter
            denominator = 2
            if "target_model" in model.hparams.network.ensemble.pred_ensemble:
                assert "target" in scores, "Target adapter scores not found!"
                numerator += scores["target"]
                denominator += 1
        elif "target_model" in model.hparams.network.ensemble.pred_ensemble:
            assert "target" in scores, "Target adapter scores not found!"
            numerator = scores["target"] + scores_no_adapter
            denominator = 2
        similarity = numerator / denominator
        similarity = similarity.softmax(dim=1)

    elif model.hparams.network.ensemble.method == "weighted_average":
        similarity = 0.0
        alphas = 0.0
        if "source_model" in model.hparams.network.ensemble.pred_ensemble:
            assert "source" in similarities, "Source adapter similarity not found!"
            similarity += (
                similarities["source"] * model.hparams.network.ensemble.alpha.source
            )
            alphas += model.hparams.network.ensemble.alpha.source
        if "target_model" in model.hparams.network.ensemble.pred_ensemble:
            assert "target" in similarities, "Target adapter similarity not found!"
            similarity += (
                similarities["target"] * model.hparams.network.ensemble.alpha.target
            )
            alphas += model.hparams.network.ensemble.alpha.target
        similarity += (
            similarities["no_adapter"] * model.hparams.network.ensemble.alpha.clip
        )
        alphas += model.hparams.network.ensemble.alpha.clip
        if alphas != 1.0:
            log.warning("Alpha coefficients do not sum to 1.0!")

    elif model.hparams.network.ensemble.method == "quality_score":

        # compute quality score
        quality_scores = {}
        if model.hparams.network.fusion_input == "features":
            if "source_model" in model.hparams.network.ensemble.pred_ensemble:
                assert (
                    "source" in video_embeddings
                ), "Source adapter video embeddings not found!"
                quality_score_source_adapter = (
                    self_supervised_representation_quality_score(
                        video_embeddings["source"]
                    )
                )
                quality_scores["source"] = quality_score_source_adapter
            if "target_model" in model.hparams.network.ensemble.pred_ensemble:
                assert (
                    "target" in video_embeddings
                ), "Target adapter video embeddings not found!"
                quality_score_target_adapter = (
                    self_supervised_representation_quality_score(
                        video_embeddings["target"]
                    )
                )
                quality_scores["target"] = quality_score_target_adapter
            quality_score_no_adapter = self_supervised_representation_quality_score(
                video_embeddings["no_adapter"]
            )
            quality_scores["no_adapter"] = quality_score_no_adapter

        elif model.hparams.network.fusion_input == "logits":
            if "source_model" in model.hparams.network.ensemble.pred_ensemble:
                quality_score_source_adapter = (
                    self_supervised_representation_quality_score(
                        video_embeddings["source"].mean(dim=1)
                    )
                )
                quality_scores["source"] = quality_score_source_adapter
            if "target_model" in model.hparams.network.ensemble.pred_ensemble:
                quality_score_target_adapter = (
                    self_supervised_representation_quality_score(
                        video_embeddings["target"].mean(dim=1)
                    )
                )
                quality_scores["target"] = quality_score_target_adapter
            quality_score_no_adapter = self_supervised_representation_quality_score(
                video_embeddings["no_adapter"].mean(dim=1)
            )
            quality_scores["no_adapter"] = quality_score_no_adapter
        else:
            raise NotImplementedError("Fusion input must be features or logits")

        # normalize quality score
        assert len(list(quality_scores.values())) <= 3, "Too many quality scores!"
        quality_score_merged = torch.stack(list(quality_scores.values()))
        quality_score_merged = quality_score_merged.softmax(dim=0)
        quality_score_no_adapter = quality_score_merged[-1].unsqueeze(1)

        # compute final similarity
        similarity = 0.0
        if "source_model" in model.hparams.network.ensemble.pred_ensemble:
            assert "source" in similarities, "Source adapter similarity not found!"
            quality_score_source = quality_score_merged[0].unsqueeze(1)
            similarity += similarities["source"] * quality_score_source
            if "target_model" in model.hparams.network.ensemble.pred_ensemble:
                assert "target" in similarities, "Target adapter similarity not found!"
                quality_score_target = quality_score_merged[1].unsqueeze(1)
                similarity += similarities["target"] * quality_score_target
        if "target_model" in model.hparams.network.ensemble.pred_ensemble:
            assert "target" in similarities, "Target adapter similarity not found!"
            quality_score_target = quality_score_merged[0].unsqueeze(1)
            similarity += similarities["target"] * quality_score_target
        similarity += similarities["no_adapter"] * quality_score_no_adapter

    else:
        raise NotImplementedError(
            "Ensemble type {} not implemented!".format(
                model.hparams.network.ensemble.method
            )
        )

    return similarity


def collate_results(batches, pivot) -> dict:
    """Collate results from multiple dataloaders into a single dict.

    Supports a list of results from multiple dataloaders, and handle the case where
    the input is only a single dataloader.

    Args:
        batch (Union[list[dict], list[list[dict]]]): List of results.
        pivot (str, optional): Key to pivot the results on. Defaults to None.
    """
    if isinstance(batches, list) and isinstance(batches[0], dict):
        batches = [batches]

    collated = {}
    for dataloader in batches:
        for batch in dataloader:
            batch_size = max(len(v) for v in batch.values() if hasattr(v, "__len__"))
            for k, v in batch.items():
                if not hasattr(v, "__len__"):
                    v = torch.tensor([v] * batch_size)
                collated.setdefault(k, []).append(v)

    collated = {
        k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else list(chain(*v))
        for k, v in collated.items()
    }

    if pivot:
        keys = collated.pop(pivot)
        collated = {
            key: {k: v[i] for k, v in collated.items()} for i, key in enumerate(keys)
        }

    return collated
