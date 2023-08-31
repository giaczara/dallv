from typing import Optional

import torch
import wandb

import pytorch_lightning as pl
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from shutil import copy
from json import dumps
from collections import defaultdict
from os.path import exists, isfile
from os import makedirs
from torch.utils.data import Subset
from rich.progress import track
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.aggregation import MeanMetric

from src.models.components.solver import WarmupMultiStepLR, WarmupCosineAnnealingLR
from src.utils.utils import (
    get_classes,
    gen_label,
    create_logits,
    LabelsManager,
    compute_ensemble,
    frame_level_loss,
    compute_statistics_matrix,
    compute_adapter_outputs,
    compute_similarity,
    select_samples_top_k,
)
from src.models.components.text_prompt import (
    text_prompt,
    manually_enriched_text_prompt,
    gpt_text_prompt,
    merged_text_prompt,
    hierarchical_text_prompt,
)
from src.models.components.modules import KLLoss
from src.models.components.modules import Adapter

from src import utils

log = utils.get_pylogger(__name__)


class VideoModel(pl.LightningModule):
    def __init__(
        self,
        clip_model,
        image_model,
        text_model,
        fusion_model,
        image_model_student,
        text_model_student,
        adapter,
        classifier,
        validation_adapter,
        loss,
        solver,
        network,
        prompts,
        domain_shift,
        distillation,
        distill_on_clip,
        classifier_baseline,
        umap,
        umap_feats,
        extra_args,
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=["clip_model", "image_model", "text_model", "fusion_model"]
        )

        self.new_samples_per_pseudo_labels = defaultdict(list)
        self.samples_per_pseudo_labels = None
        self.extra_args = extra_args

        # models
        self.clip_model = clip_model
        self.image_model = image_model
        self.image_model_student = image_model_student
        self.text_model_student = text_model_student
        self.text_model = text_model
        self.fusion_model = fusion_model
        self.source_adapter = None
        self.target_adapter = None
        self.adapter = adapter
        self.classifier = classifier
        if self.classifier is not None:
            self.source_classifier = classifier["source"]
            self.target_classifier = classifier["target"]
        if self.adapter is not None:
            self.source_adapter = adapter["source"]
            self.target_adapter = adapter["target"]
            if self.hparams.distillation:
                self.final_adapter = adapter["final"]

        # text prompt
        self.classes_names = get_classes(extra_args)
        self.num_classes = len(self.classes_names)
        self.decomposition_dict = {}

        # prompts
        if self.hparams.prompts.type == "original":
            prompt_func = text_prompt
        elif self.hparams.prompts.type == "manually_enriched":
            prompt_func = manually_enriched_text_prompt
        elif self.hparams.prompts.type == "gpt":
            prompt_func = gpt_text_prompt
        elif self.hparams.prompts.type == "merged":
            prompt_func = merged_text_prompt
        elif self.hparams.prompts.type == "hierarchical":
            prompt_func = hierarchical_text_prompt
        else:
            raise ValueError(
                "Prompt function {} not recognized!".format(self.hparams.prompts.type)
            )

        self.classes, self.num_text_aug, self.text_dict = prompt_func(
            self.classes_names,
            dataset=self.extra_args["dataset"],
            n_templates=self.hparams.prompts.n_templates,
            image_templates=self.hparams.prompts.image_templates,
        )

        # metrics
        self.num = 0
        self.corr_1 = 0
        self.corr_5 = 0
        self.correct_per_class = [0 for _ in range(self.num_classes + 1)]
        self.instances_per_class = [0 for _ in range(self.num_classes + 1)]
        self.pred = []
        self.gt = []
        self.best = 0.0

        self.metrics = torch.nn.ModuleDict()
        self.metrics["train/loss"] = MeanMetric()
        self.metrics["train/source/acc"] = MetricCollection(
            {
                f"train/acc@{k}": Accuracy(
                    task="multiclass", num_classes=self.num_classes, top_k=k
                )
                for k in [1, 3, 5]
            }
        )
        self.metrics["train/target/acc"] = MetricCollection(
            {
                f"train/acc@{k}": Accuracy(
                    task="multiclass", num_classes=self.num_classes, top_k=k
                )
                for k in [1, 3, 5]
            }
        )

        # losses
        self.loss_video = KLLoss()
        self.loss_text = KLLoss()

        # prototypes extraction
        if self.extra_args["prototypes_extraction"]:
            self.statistics_matrices = torch.tensor([]).to(self.device)
            self.labels = torch.tensor([]).to(self.device)
        if self.hparams.umap:
            self.umap_features = []
            self.umap_labels = []

    def forward(self, x):

        embedding = self.encoder(x)
        return embedding

    def load_adapter_checkpoints(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224).cuda()
            dim = self.image_model.cuda()(x).shape[1]
        if self.hparams.network.source_adapter_checkpoint != "none":
            source_adapter = Adapter(c_in=dim)
            assert isfile(
                self.hparams.network.source_adapter_checkpoint
            ), "Source supervised checkpoint not found at {}!".format(
                self.hparams.network.source_adapter_checkpoint
            )
            print("Loading source supervised model")
            checkpoint = torch.load(self.hparams.network.source_adapter_checkpoint)
            source_adapter_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("adapter"):
                    source_adapter_state_dict[key.replace("adapter.", "")] = value
                if key.startswith("source_adapter"):
                    source_adapter_state_dict[
                        key.replace("source_adapter.", "")
                    ] = value
            source_adapter.load_state_dict(source_adapter_state_dict)
            self.source_adapter = source_adapter
            print("Source adapter loaded!")
            del checkpoint
        if self.hparams.network.target_adapter_checkpoint != "none":
            target_adapter = Adapter(c_in=dim)
            assert isfile(
                self.hparams.network.target_adapter_checkpoint
            ), "Target unsupervised checkpoint not found at {}!".format(
                self.hparams.network.target_adapter_checkpoint
            )
            print("Loading target unsupervised model")
            checkpoint = torch.load(self.hparams.network.target_adapter_checkpoint)
            target_adapter_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("target_adapter"):
                    target_adapter_state_dict[
                        key.replace("target_adapter.", "")
                    ] = value
            target_adapter.load_state_dict(target_adapter_state_dict)
            self.target_adapter = target_adapter
            print("Target adapter loaded!")
            del checkpoint

    def load_source_adapter(self):
        log.info("Loading source adapter...")
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224).cuda()
            dim = self.image_model.cuda()(x).shape[1]
        source_adapter = Adapter(c_in=dim)
        assert isfile(
            self.hparams.network.source_sup_checkpoint
        ), "Source supervised checkpoint not found at {}!".format(
            self.hparams.network.source_sup_checkpoint
        )
        log.info("Loading source unsupervised model")
        checkpoint = torch.load(self.hparams.network.source_sup_checkpoint)
        source_adapter_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("adapter"):
                source_adapter_state_dict[key.replace("adapter.", "")] = value
        source_adapter.load_state_dict(source_adapter_state_dict)
        self.source_adapter = source_adapter
        log.info("Source adapter loaded!")
        print("Source adapter loaded!")
        del checkpoint

    def configure_optimizers(self):

        if self.classifier is not None:
            assert (
                self.adapter is not None
            ), "Adapter must be defined if classifier is defined!"
            parameters = []
            if self.hparams.distillation:
                parameters = list(self.final_adapter.paramters()) + list(
                    self.final_classifier.parameters()
                )
            else:
                if self.hparams.loss.source.weight:
                    parameters.extend(self.source_adapter.parameters())
                    parameters.extend(self.source_classifier.parameters())
                if self.hparams.loss.target.weight:
                    parameters.extend(self.target_adapter.parameters())
                    parameters.extend(self.target_classifier.parameters())

            # optimizer
            if self.hparams.solver.optim == "adam":
                optimizer = optim.Adam(
                    parameters,
                    lr=self.hparams.solver.lr,
                    betas=(0.9, 0.98),
                    eps=1e-8,
                    weight_decay=0.2,
                )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
            elif self.hparams.solver.optim == "sgd":

                optimizer = optim.SGD(
                    parameters,
                    self.hparams.solver.lr,
                    momentum=self.hparams.solver.momentum,
                    weight_decay=self.hparams.solver.weight_decay,
                )
            elif self.hparams.solver.optim == "adamw":
                optimizer = optim.AdamW(
                    parameters,
                    betas=(0.9, 0.98),
                    lr=self.hparams.solver.lr,
                    eps=1e-8,
                    weight_decay=self.hparams.solver.weight_decay,
                )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
            else:
                raise ValueError(
                    "Unknown optimizer: {}".format(self.hparams.solver.optim)
                )
        elif self.adapter is not None:
            parameters = []
            if self.hparams.distillation:
                if self.hparams.distill_on_clip:
                    parameters = self.image_model_student.parameters()
                else:
                    parameters = self.final_adapter.parameters()
            else:
                if self.hparams.loss.source.weight:
                    parameters.extend(self.source_adapter.parameters())
                if self.hparams.loss.target.weight:
                    parameters.extend(self.target_adapter.parameters())
            # optimizer
            if self.hparams.solver.optim == "adam":
                optimizer = optim.Adam(
                    parameters,
                    lr=self.hparams.solver.lr,
                    betas=(0.9, 0.98),
                    eps=1e-8,
                    weight_decay=0.2,
                )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
            elif self.hparams.solver.optim == "sgd":

                optimizer = optim.SGD(
                    parameters,
                    self.hparams.solver.lr,
                    momentum=self.hparams.solver.momentum,
                    weight_decay=self.hparams.solver.weight_decay,
                )
            elif self.hparams.solver.optim == "adamw":
                optimizer = optim.AdamW(
                    parameters,
                    betas=(0.9, 0.98),
                    lr=self.hparams.solver.lr,
                    eps=1e-8,
                    weight_decay=self.hparams.solver.weight_decay,
                )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
            else:
                raise ValueError(
                    "Unknown optimizer: {}".format(self.hparams.solver.optim)
                )
        else:

            if self.hparams.network.backbone == "trn":
                text_params = self.text_model.parameters()
                # optimizer
                if self.hparams.solver.optim == "adam":
                    optimizer = optim.Adam(
                        [m.parameters() for m in self.image_model.values()]
                        + [text_params],
                        lr=self.hparams.solver.lr,
                        betas=(0.9, 0.98),
                        eps=1e-8,
                        weight_decay=0.2,
                    )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
                elif self.hparams.solver.optim == "sgd":

                    optimizer = optim.SGD(
                        [m.parameters() for m in self.image_model.values()]
                        + [text_params],
                        self.hparams.solver.lr,
                        momentum=self.hparams.solver.momentum,
                        weight_decay=self.hparams.solver.weight_decay,
                    )
                elif self.hparams.solver.optim == "adamw":
                    vision_params = [m.parameters() for m in self.image_model.values()]
                    # text_params = filter(
                    #     lambda p: id(p) not in vision_params, self.clip_model.parameters()
                    # )

                    optimizer = optim.AdamW(
                        [
                            {"params": text_params},
                            {
                                "params": self.clip_model.visual.parameters(),
                                "lr": self.hparams.solver.lr
                                * self.hparams.solver.ratio,
                            },
                        ],
                        betas=(0.9, 0.98),
                        lr=self.hparams.solver.lr,
                        eps=1e-8,
                        weight_decay=self.hparams.solver.weight_decay,
                    )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
                else:
                    raise ValueError(
                        "Unknown optimizer: {}".format(self.hparams.solver.optim)
                    )
            else:
                # optimizer
                if self.hparams.solver.optim == "adam":
                    optimizer = optim.Adam(
                        [
                            {"params": self.clip_model.parameters()},
                            {
                                "params": self.fusion_model.parameters(),
                                "lr": self.hparams.solver.lr
                                * self.hparams.solver.f_ratio,
                            },
                        ],
                        lr=self.hparams.solver.lr,
                        betas=(0.9, 0.98),
                        eps=1e-8,
                        weight_decay=0.2,
                    )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
                elif self.hparams.solver.optim == "sgd":

                    optimizer = optim.SGD(
                        [
                            {"params": self.clip_model.parameters()},
                            {
                                "params": self.fusion_model.parameters(),
                                "lr": self.hparams.solver.lr
                                * self.hparams.solver.f_ratio,
                            },
                        ],
                        self.hparams.solver.lr,
                        momentum=self.hparams.solver.momentum,
                        weight_decay=self.hparams.solver.weight_decay,
                    )
                elif self.hparams.solver.optim == "adamw":
                    vision_params = list(map(id, self.clip_model.visual.parameters()))
                    text_params = filter(
                        lambda p: id(p) not in vision_params,
                        self.clip_model.parameters(),
                    )

                    optimizer = optim.AdamW(
                        [
                            {"params": text_params},
                            {
                                "params": self.clip_model.visual.parameters(),
                                "lr": self.hparams.solver.lr
                                * self.hparams.solver.ratio,
                            },
                            {
                                "params": self.fusion_model.parameters(),
                                "lr": self.hparams.solver.lr
                                * self.hparams.solver.f_ratio,
                            },
                        ],
                        betas=(0.9, 0.98),
                        lr=self.hparams.solver.lr,
                        eps=1e-8,
                        weight_decay=self.hparams.solver.weight_decay,
                    )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
                else:
                    raise ValueError(
                        "Unknown optimizer: {}".format(self.hparams.solver.optim)
                    )

        # scheduler
        if self.hparams.solver.type == "cosine":
            lr_scheduler = WarmupCosineAnnealingLR(
                optimizer,
                self.extra_args["epochs"],
                warmup_epochs=self.hparams.solver.lr_warmup_steps,
            )
        elif self.hparams.solver.type == "multistep":
            if isinstance(self.hparams.solver.lr_decay_step, list):
                milestones = self.hparams.solver.lr_decay_step
            elif isinstance(self.hparams.solver.lr_decay_step, int):
                milestones = [
                    self.hparams.solver.lr_decay_step * (i + 1)
                    for i in range(
                        self.extra_args["epochs"] // self.hparams.solver.lr_decay_step
                    )
                ]
            else:
                raise ValueError(
                    "error learning rate decay step: {}".format(
                        type(self.hparams.solver.lr_decay_step)
                    )
                )
            lr_scheduler = WarmupMultiStepLR(
                optimizer, milestones, warmup_epochs=self.hparams.solver.lr_warmup_steps
            )
        elif self.hparams.solver.type == "none":
            lr_scheduler = None
        else:
            raise ValueError(
                "Unknown lr scheduler: {}".format(self.hparams.solver.type)
            )

        if lr_scheduler is not None:
            res = {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            res = {"optimizer": optimizer}
        return res

    def log_image(self, image, key):
        self.logger.log_image(key, [wandb.Image(image)], step=self.global_step)

    def training_step(self, train_batch, **kwargs):
        if self.hparams.distillation:
            return self.distill_step(train_batch)
        elif self.hparams.classifier_baseline:
            return self.training_step_classifier(train_batch)
        else:
            return self.training_step_normal(train_batch)

    def training_step_classifier(self, train_batch):
        assert self.adapter is not None, "adapter is None"
        (
            video_source,
            y_source,
            index_source,
            video_target,
            y_target,
            index_target,
        ) = train_batch
        if self.hparams.loss.source.weight:
            video_source = video_source.view(
                (-1, self.extra_args["num_segments"], 3) + video_source.size()[-2:]
            )
            b, t, c, h, w = video_source.size()
            video_source = video_source.view(-1, c, h, w)
            video_embedding_source = self.image_model(video_source)
            video_embedding_source = video_embedding_source.view(b, t, -1)
            video_embedding_source = self.fusion_model(video_embedding_source)
            video_embedding_source = self.source_adapter(video_embedding_source)
            output_source = self.classifier(video_embedding_source)
            loss = F.cross_entropy(output_source, y_source)
        elif self.hparams.target.weight:
            video_target = video_target.view(
                (-1, self.extra_args["num_segments"], 3) + video_target.size()[-2:]
            )
            b, t, c, h, w = video_target.size()
            video_target = video_target.view(-1, c, h, w)
            video_embedding_target = self.image_model(video_target)
            video_embedding_target = video_embedding_target.view(b, t, -1)
            video_embedding_target = self.fusion_model(video_embedding_target)
            video_embedding_target = self.target_adapter(video_embedding_target)
            output_target = self.classifier(video_embedding_target)
            loss = F.cross_entropy(output_target, y_target)
        else:
            raise ValueError("no loss weight")
        return loss

    def training_step_normal(self, train_batch):

        if self.hparams.domain_shift:
            (
                video_source,
                y_source,
                index_source,
                video_target,
                y_target,
                index_target,
            ) = train_batch
        else:
            video_target = y_target = index_target = None
            (
                video_source,
                y_source,
            ) = train_batch

        # ============================ SOURCE ============================ #

        if self.hparams.loss.source.weight:

            # compute embeddings
            video_source2 = video_source.clone()
            video_source = video_source.view(
                (-1, self.extra_args["num_segments"], 3) + video_source.size()[-2:]
            )
            if self.extra_args["im2vid"]:
                video_source = video_source[
                               :, int(self.extra_args["num_segments"] / 2), :, :, :
                               ]
                video_source = video_source.unsqueeze(1)
            b, t, c, h, w = video_source.size()
            video_source = video_source.view(-1, c, h, w)
            video_embedding_source = self.image_model(video_source)
            video_embedding_source = video_embedding_source.view(b, t, -1)
            video_embedding_source = self.fusion_model(video_embedding_source)


            text_id = np.random.randint(self.num_text_aug, size=len(y_source))
            texts_source = torch.stack(
                [self.text_dict[j][i, :] for i, j in zip(y_source, text_id)]
            )
            texts_source = texts_source.to(self.device)
            if self.adapter is not None:
                video_embedding_source = self.source_adapter(video_embedding_source)
            text_embedding_source = self.text_model(texts_source)

            if self.hparams.network.fix_text:
                text_embedding_source.detach_()

            # compute logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_video_source, logits_per_text_source = create_logits(
                video_embedding_source, text_embedding_source, logit_scale
            )

            # generate ground-truth label
            ground_truth_source = (
                torch.tensor(gen_label(y_source), dtype=video_embedding_source.dtype)
                .float()
                .to(self.device)
            )

            # compute accuracy
            with torch.no_grad():
                text_inputs = self.classes.to(self.device)
                label_embedding = self.text_model(text_inputs)
                similarity = 100.0 * video_embedding_source @ label_embedding.t()
                similarity = similarity.view(b, self.num_text_aug, -1).softmax(dim=-1)
                logits = similarity.mean(dim=1, keepdim=False)
            self.metrics["train/source/acc"](logits, y_source)
            self.log_dict(
                self.metrics["train/source/acc"],
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
            )

            # compute loss
            loss_video_source = self.loss_video(
                logits_per_video_source, ground_truth_source
            )
            loss_text_source = self.loss_text(
                logits_per_text_source, ground_truth_source
            )
            source_loss = (loss_video_source + loss_text_source) / 2
            self.log("source_loss", source_loss, sync_dist=True)
            loss = source_loss * self.hparams.loss.source.weight

            labels_manager = LabelsManager(
                data=self.extra_args,
            )

        else:
            loss = torch.tensor(0.0).to(self.device)

        # ============================ TARGET ============================ #

        if self.hparams.domain_shift:

            if self.hparams.loss.target.weight:

                # fetch target label
                if (
                    self.hparams.loss.target.use_gt
                    or self.hparams.loss.target.filtering
                    != "top_k_confident_samples_v2"
                ):
                    target_label = y_target
                    b = video_target.size(0)
                    accept_mask = torch.ones(b, dtype=torch.bool)
                    accepted = True
                else:
                    # compute pseudo-labels with an inference step
                    with torch.no_grad():
                        text_inputs_target = self.classes.to(self.device)
                        text_embedding_target = self.text_model(text_inputs_target)
                        video_target = video_target.view(
                            (-1, self.extra_args["num_segments"], 3)
                            + video_target.size()[-2:]
                        )
                        b, t, c, h, w = video_target.size()
                        video_target = video_target.view(-1, c, h, w)
                        video_embedding_target = self.clip_model.encode_image(
                            video_target
                        ).view(b, t, -1)
                        video_embedding_target = self.fusion_model(
                            video_embedding_target
                        )

                        video_embedding_target /= video_embedding_target.norm(
                            dim=-1, keepdim=True
                        )
                        text_embedding_target /= text_embedding_target.norm(
                            dim=-1, keepdim=True
                        )
                        similarity = (
                            100.0 * video_embedding_target @ text_embedding_target.T
                        )
                        similarity = similarity.view(b, self.num_text_aug, -1).softmax(
                            dim=-1
                        )
                        similarity = similarity.mean(dim=1, keepdim=False)
                        values, pseudo_y_target = similarity.topk(1, dim=-1)

                        # define acceptance mask
                        accept_mask = torch.zeros(b, dtype=torch.bool)

                        # gather target samples and their confidence
                        for i in range(b):
                            label = pseudo_y_target[i]
                            entry = (index_target[i].item(), values[i].item())
                            self.new_samples_per_pseudo_labels[label.item()].append(
                                entry
                            )

                        if (
                            self.hparams.loss.target.filtering
                            == "top_k_confident_samples_v2"
                        ):
                            accept_mask = torch.ones(b, dtype=torch.bool)
                        else:
                            raise ValueError(
                                "Filtering method {} not recognized!".format(
                                    self.hparams.loss.target.filtering
                                )
                            )
                        accepted = torch.sum(accept_mask)

                    target_label = pseudo_y_target.view(b)

                if accepted:

                    # compute embeddings
                    video_target = video_target.view(
                        (-1, self.extra_args["num_segments"], 3)
                        + video_target.size()[-2:]
                    )
                    b, t, c, h, w = video_target.size()

                    text_id = np.random.randint(
                        self.num_text_aug, size=len(target_label)
                    )
                    target_label = target_label.to(self.device)
                    texts_target = torch.stack(
                        [
                            self.text_dict[j].to(self.device)[i, :]
                            for i, j in zip(target_label, text_id)
                        ]
                    )

                    video_target = video_target.view(-1, c, h, w)
                    video_embedding_target = self.image_model(video_target)

                    video_embedding_target = video_embedding_target.view(b, t, -1)
                    video_embedding_target = self.fusion_model(
                        video_embedding_target
                    )

                    if self.adapter is not None:
                        video_embedding_target = self.target_adapter(
                            video_embedding_target
                        )

                    video_embedding_target = video_embedding_target[accept_mask]
                    text_embedding_target = self.text_model(texts_target)
                    text_embedding_target = text_embedding_target[accept_mask]

                    if self.hparams.network.fix_text:
                        text_embedding_target.detach_()

                    # compute logits
                    logit_scale = self.clip_model.logit_scale.exp()
                    logits_per_video_target, logits_per_text_target = create_logits(
                        video_embedding_target, text_embedding_target, logit_scale
                    )

                    # generate ground truth label
                    if not self.hparams.loss.target.use_gt:
                        target_label = target_label[accept_mask]
                    ground_truth_target = (
                        torch.tensor(
                            gen_label(target_label), dtype=video_embedding_target.dtype
                        )
                        .float()
                        .to(self.device)
                    )

                    # compute loss
                    loss_video_target = self.loss_video(
                        logits_per_video_target, ground_truth_target
                    )

                    loss_text_target = self.loss_text(
                        logits_per_text_target, ground_truth_target
                    )

                    # compute accuracy
                    with torch.no_grad():
                        text_inputs = self.classes.to(self.device)
                        label_embedding = self.text_model(text_inputs)
                        similarity = (
                            100.0 * video_embedding_target @ label_embedding.t()
                        )
                        similarity = similarity.view(b, self.num_text_aug, -1).softmax(
                            dim=-1
                        )
                        logits = similarity.mean(dim=1, keepdim=False)
                    self.metrics["train/target/acc"](logits, y_target)
                    self.log_dict(
                        self.metrics["train/target/acc"],
                        prog_bar=True,
                        sync_dist=True,
                        on_epoch=True,
                    )

                    target_loss = (loss_video_target + loss_text_target) / 2
                    self.log("target_loss", target_loss, sync_dist=True)
                    loss += target_loss * self.hparams.loss.target.weight

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def distill_step(self, train_batch):
        assert (
            self.hparams.domain_shift
        ), "Distillation is only supported for domain shift"
        # assert self.image_model_student is not None, "Student model not found"
        (
            _,
            _,
            _,
            video_target,
            y_target,
            _,
        ) = train_batch
        video_target = video_target.view(
            (-1, self.extra_args["num_segments"], 3) + video_target.size()[-2:]
        )
        b, t, c, h, w = video_target.size()
        video_target = video_target.view(-1, c, h, w)

        # text
        text_inputs = self.classes.to(self.device)
        with torch.no_grad():
            text_embedding = self.text_model(text_inputs)
        middle_dim = self.num_text_aug

        # ======================== teacher forward ======================== #

        # zs clip
        self.image_model.eval()
        self.source_adapter.eval()
        self.target_adapter.eval()
        with torch.no_grad():
            video_embedding_teacher = self.image_model(video_target)
            video_embedding_teacher /= video_embedding_teacher.norm(
                dim=-1, keepdim=True
            )
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            similarity_teacher = video_embedding_teacher @ text_embedding.T
            similarity_teacher = similarity_teacher.view(b, middle_dim, -1).mean(dim=1)
            similarity_teacher = similarity_teacher.view(b, t, -1).mean(dim=1)
            _, pseudo_y_teacher = similarity_teacher.topk(1, dim=-1)
            pseudo_y_teacher = pseudo_y_teacher.view(-1)

            # source model
            self.source_adapter = self.source_adapter.to(self.device)
            video_embedding_teacher_source = self.source_adapter(
                video_embedding_teacher
            )
            video_embedding_teacher_source /= video_embedding_teacher_source.norm(
                dim=-1, keepdim=True
            )
            similarity_teacher_source = (
                video_embedding_teacher_source @ text_embedding.T
            )
            similarity_teacher_source = similarity_teacher_source.view(
                b, middle_dim, -1
            ).mean(dim=1)
            similarity_teacher_source = similarity_teacher_source.view(
                b, t, -1
            ).mean(dim=1)

            _, pseudo_y_teacher_source = similarity_teacher_source.topk(1, dim=-1)
            pseudo_y_teacher_source = pseudo_y_teacher_source.view(-1)

            # target model
            self.target_adapter = self.target_adapter.to(self.device)
            video_embedding_teacher_target = self.target_adapter(
                video_embedding_teacher
            )
            video_embedding_teacher_target /= video_embedding_teacher_target.norm(
                dim=-1, keepdim=True
            )
            similarity_teacher_target = (
                video_embedding_teacher_target @ text_embedding.T
            )
            similarity_teacher_target = similarity_teacher_target.view(
                b, middle_dim, -1
            ).mean(dim=1)
            similarity_teacher_target = similarity_teacher_target.view(
                b, t, -1
            ).mean(dim=1)

            _, pseudo_y_teacher_target = similarity_teacher_target.topk(1, dim=-1)
            pseudo_y_teacher_target = pseudo_y_teacher_target.view(-1)

            # weighted average
            weighted_average = (
                similarity_teacher * 0.5
                + similarity_teacher_source * 0.3
                + similarity_teacher_target * 0.2
            )
            _, weighted_average_prediction = weighted_average.topk(1, dim=-1)
            weighted_average_prediction = weighted_average_prediction.view(-1)

            # compute agreement
            cond1 = pseudo_y_teacher == pseudo_y_teacher_source
            cond2 = pseudo_y_teacher_source == pseudo_y_teacher_target
            cond3 = pseudo_y_teacher == pseudo_y_teacher_target

            # compute pseudo labels
            pseudo_y = torch.zeros_like(pseudo_y_teacher).long()
            pseudo_y[cond1] = pseudo_y_teacher[cond1]
            pseudo_y[cond2] = pseudo_y_teacher_source[cond2]
            pseudo_y[cond3] = pseudo_y_teacher_target[cond3]
            pseudo_y[pseudo_y == 0] = weighted_average_prediction[pseudo_y == 0]

        # ======================== student forward ======================== #

        # text_embedding = self.text_model_student(text_inputs)
        video_embedding_student = video_embedding_teacher
        video_embedding_student = video_embedding_student.view(b, t, -1)
        video_embedding_student = self.fusion_model(video_embedding_student)
        if not self.hparams.distill_on_clip:
            video_embedding_student = self.final_adapter(video_embedding_student)
        similarity_student = video_embedding_student @ text_embedding.T
        similarity_student = similarity_student.view(b, middle_dim, -1).mean(dim=1)

        # ======================== compute loss ======================== #

        # soft loss
        soft_outputs = torch.nn.functional.log_softmax(
            similarity_student / self.hparams.loss.temperature, dim=1
        )
        soft_targets = (
            (similarity_teacher / self.hparams.loss.temperature).softmax(dim=-1)
            + (similarity_teacher_source / self.hparams.loss.temperature).softmax(
                dim=-1
            )
            + (similarity_teacher_target / self.hparams.loss.temperature).softmax(
                dim=-1
            )
        ) / 3
        loss = soft_loss = (
            torch.nn.functional.kl_div(
                soft_outputs, soft_targets.detach(), reduction="batchmean"
            )
            * self.hparams.loss.soft.weight
        )

        # hard loss
        hard_loss = torch.nn.functional.cross_entropy(
            similarity_student,
            pseudo_y,
            label_smoothing=self.hparams.loss.label_smoothing_value,
        )
        loss += hard_loss * (1 - self.hparams.loss.soft.weight)

        return loss

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def validation_step(self, val_batch, batch_idx):
        video, y = val_batch
        # hierarchical text prompts
        text_inputs = self.classes.to(self.device)
        middle_dim = self.num_text_aug
        text_embedding = self.text_model(text_inputs)
        if self.hparams.distillation:
            text_embedding_student = text_embedding


        # encode video
        video = video.view(
            (-1, self.extra_args["num_segments"], 3) + video.size()[-2:]
        )
        b, t, c, h, w = video.size()
        video = video.view(-1, c, h, w)

        video_embedding = self.clip_model.encode_image(video).view(b, t, -1)
        if self.hparams.distillation:
            video_embedding_student = video_embedding

        video_embedding_no_adapter = None

        # model ensemble
        if self.hparams.network.ensemble.pred_ensemble in [
            "zs_clip+source_model",
            "zs_clip+target_model",
            "zs_clip+source_model+target_model",
        ]:
            assert self.adapter is not None, "Adapter must be defined!"

        # check fusion input
        assert self.hparams.network.fusion_input in [
            "logits",
        ], "Fusion input must be logits"

        # collect data
        similarities = {}
        video_embeddings = {}
        scores = {}

        similarity_no_adapter, scores_no_adapter = compute_similarity(
            self,
            video_embedding,
            text_embedding,
            middle_dim,
        )
        similarities["no_adapter"] = similarity_no_adapter

        if self.hparams.distillation and self.hparams.distill_on_clip:
            similarity_student, scores_student = compute_similarity(
                self,
                video_embedding_student,
                text_embedding_student,
                middle_dim,
            )
            similarities["student"] = similarity_student

        # feed embedding to adapter
        if self.adapter is not None:
            if self.hparams.distillation:
                (
                    similarity_final_adapter,
                    scores_final_adapter,
                    video_embedding_final_adapter,
                ) = compute_adapter_outputs(
                    video_embedding=video_embedding_student,
                    text_embedding=text_embedding_student,
                    middle_dim=middle_dim,
                    model=self,
                    adapter=self.final_adapter,
                    device=self.device,
                )
                similarities["final"] = similarity_final_adapter
                video_embeddings["final"] = video_embedding_final_adapter
                scores["final"] = scores_final_adapter
            else:
                if "zs_clip" in self.hparams.network.ensemble.pred_ensemble:
                    video_embedding_no_adapter = video_embedding.clone()
                    video_embedding_no_adapter /= (
                        video_embedding_no_adapter.norm(dim=-1, keepdim=True)
                    )
                    video_embeddings["no_adapter"] = video_embedding_no_adapter
                (
                    similarity_source_adapter,
                    scores_source_adapter,
                    video_embedding_source_adapter,
                ) = compute_adapter_outputs(
                    video_embedding=video_embedding,
                    text_embedding=text_embedding,
                    middle_dim=middle_dim,
                    model=self,
                    adapter=self.source_adapter,
                    device=self.device,
                )
                similarities["source"] = similarity_source_adapter
                video_embeddings["source"] = video_embedding_source_adapter
                scores["source"] = scores_source_adapter

                if (
                        "target_model"
                        in self.hparams.network.ensemble.pred_ensemble
                ) or self.hparams.validation_adapter == "target":
                    (
                        similarity_target_adapter,
                        scores_target_adapter,
                        video_embedding_target_adapter,
                    ) = compute_adapter_outputs(
                        video_embedding=video_embedding,
                        text_embedding=text_embedding,
                        middle_dim=middle_dim,
                        model=self,
                        adapter=self.target_adapter,
                        device=self.device,
                    )
                    similarities["target"] = similarity_target_adapter
                    video_embeddings["target"] = video_embedding_target_adapter
                    scores["target"] = scores_target_adapter

        # apply ensemble
        if "zs_clip" in self.hparams.network.ensemble.pred_ensemble:
            assert (
                    video_embedding_no_adapter is not None
            ), "Video embedding with no adapter is None"

            similarity_no_adapter, scores_no_adapter = compute_similarity(
                self,
                video_embedding_no_adapter,
                text_embedding,
                middle_dim,
            )
            similarities["no_adapter"] = similarity_no_adapter

            # normalize scores
            scores_no_adapter /= scores_no_adapter.norm(dim=-1, keepdim=True)
            if "source_model" in self.hparams.network.ensemble.pred_ensemble:
                scores_source_adapter = scores["source"]
                scores_source_adapter /= scores_source_adapter.norm(
                    dim=-1, keepdim=True
                )
                scores_source_adapter -= scores_no_adapter.mean(dim=1, keepdim=True)
                scores_source_adapter /= scores_no_adapter.std(dim=1, keepdim=True)
                scores["source"] = scores_source_adapter
            if "target_model" in self.hparams.network.ensemble.pred_ensemble:
                scores_target_adapter = scores["target"]
                scores_target_adapter /= scores_target_adapter.norm(
                    dim=-1, keepdim=True
                )
                scores_target_adapter -= scores_no_adapter.mean(dim=1, keepdim=True)
                scores_target_adapter /= scores_no_adapter.std(dim=1, keepdim=True)
                scores["target"] = scores_target_adapter

            similarity = compute_ensemble(
                model=self,
                similarities=similarities,
                video_embeddings=video_embeddings,
                scores=scores,
                scores_no_adapter=scores_no_adapter,
            )
            similarities["ensemble"] = similarity

        if self.hparams.distillation:
            if self.hparams.distill_on_clip:
                similarity = similarities["student"]
            else:
                similarity = similarities["final"]
        elif self.hparams.network.ensemble.pred_ensemble != "none":
            similarity = similarities["ensemble"]
        else:
            if self.hparams.validation_adapter == "none":
                assert (
                    not self.hparams.network.use_adapter
                ), "use_adapter is True but validation_adapter is none"
                similarity = similarities["no_adapter"]
            else:
                similarity = similarities[self.hparams.validation_adapter]

        # fetch predictions
        values_1, indices_1 = similarity.topk(1, dim=-1)
        values_5, indices_5 = similarity.topk(5, dim=-1)

        self.num += b

        for i in range(b):
            predicted_label = indices_1[i]
            label = y[i]
            if indices_1[i].item() == label.item():
                self.corr_1 += 1
            if y[i] in indices_5[i]:
                self.corr_5 += 1
            self.instances_per_class[label] += 1
            if label.item() == predicted_label.item():
                self.correct_per_class[label] += 1
        self.pred.extend([i[0] for i in indices_1.cpu().tolist()])
        self.gt.extend(list(y.cpu().tolist()))



    def on_validation_epoch_start(self):
        if self.extra_args["prototypes_extraction"]:
            self.statistics_matrices = torch.tensor([]).to(self.device)
            self.labels = torch.tensor([]).to(self.device)
        elif self.hparams.umap:
            self.umap_features = []
            self.umap_labels = []
        else:
            self.num = 0
            self.corr_1 = 0
            self.corr_5 = 0
            self.correct_per_class = [0 for _ in range(self.num_classes + 1)]
            self.instances_per_class = [0 for _ in range(self.num_classes + 1)]
            self.pred = []
            self.gt = []

    def on_fit_start(self):
        super().on_fit_start()

        if self.hparams.loss.target.weight:
            if not self.hparams.loss.target.use_gt:
                if self.hparams.loss.target.filtering == "top_k_confident_samples_v2":
                    if self.trainer.global_rank == 0:
                        predict_dataloader = (
                            self.trainer.datamodule.predict_dataloader()
                        )
                        outputs = []
                        confidences = []
                        with torch.no_grad():
                            for batch in track(
                                predict_dataloader,
                                description="Computing pseudo-labels...",
                            ):
                                video, y = batch
                                text_inputs = self.classes.to(self.device)
                                middle_dim = self.num_text_aug
                                text_embedding = self.text_model(text_inputs)
                                video = video.view(
                                    (-1, self.extra_args["num_segments"], 3)
                                    + video.size()[-2:]
                                )
                                b, t, c, h, w = video.size()
                                video = video.view(-1, c, h, w)

                                video = video.to(self.device)
                                video_embedding = self.clip_model.encode_image(
                                    video
                                ).view(b, t, -1)
                                video_embedding /= video_embedding.norm(
                                    dim=-1, keepdim=True
                                )
                                similarity, _ = compute_similarity(
                                    self, video_embedding, text_embedding, middle_dim
                                )
                                values_1, indices_1 = similarity.topk(1, dim=-1)
                                outputs.extend([i[0] for i in indices_1.cpu().tolist()])
                                confidences.extend(
                                    [i[0] for i in values_1.cpu().tolist()]
                                )
                        indices = select_samples_top_k(
                            outputs=outputs,
                            confidences=confidences,
                            top_k=self.hparams.loss.target.top_k,
                        )

                        for idx, video_record in enumerate(
                            self.trainer.datamodule.data_train_target.video_list
                        ):
                            video_record.label = outputs[idx]

                        train_subset = Subset(
                            self.trainer.datamodule.data_train_target, indices
                        )

                        self.trainer.datamodule.data_train_target = train_subset
                        self.trainer.datamodule.data_train.target_dataset = train_subset

                elif self.hparams.loss.target.filtering == "full_ensemble":
                    assert (
                        self.hparams.network.source_adapter_checkpoint != "none"
                    ), "Source adapter checkpoint not provided!"
                    assert (
                        self.hparams.network.target_adapter_checkpoint != "none"
                    ), "Target adapter checkpoint not provided!"
                    self.load_adapter_checkpoints()
                    predict_dataloader = self.trainer.datamodule.predict_dataloader()
                    predictions_clip_all = []
                    predictions_source_all = []
                    predictions_target_all = []
                    self.source_adapter = self.source_adapter.to(self.device)
                    self.target_adapter = self.target_adapter.to(self.device)
                    with torch.no_grad():
                        for batch in track(
                            predict_dataloader, description="Computing pseudo-labels..."
                        ):
                            video, y = batch
                            text_inputs = self.classes.to(self.device)
                            middle_dim = self.num_text_aug
                            text_embedding = self.text_model(text_inputs)
                            video = video.view(
                                (-1, self.extra_args["num_segments"], 3)
                                + video.size()[-2:]
                            )
                            b, t, c, h, w = video.size()
                            video = video.view(-1, c, h, w)

                            video = video.to(self.device)
                            video_embedding = (
                                self.clip_model.encode_image(video)
                                .view(b, t, -1)
                                .to(self.device)
                            )
                            video_embedding /= video_embedding.norm(
                                dim=-1, keepdim=True
                            )

                            # clip model
                            similarity_clip, _ = compute_similarity(
                                self, video_embedding, text_embedding, middle_dim
                            )
                            _, predictions_clip = similarity_clip.topk(1, dim=-1)
                            predictions_clip_all.extend(
                                [i[0] for i in predictions_clip.cpu().tolist()]
                            )

                            # source model
                            video_embedding_source = self.source_adapter(
                                video_embedding
                            )
                            similarity_source, _ = compute_similarity(
                                self, video_embedding_source, text_embedding, middle_dim
                            )
                            _, predictions_source = similarity_source.topk(1, dim=-1)
                            predictions_source_all.extend(
                                [i[0] for i in predictions_source.cpu().tolist()]
                            )

                            # target model
                            video_embedding_target = self.target_adapter(
                                video_embedding
                            )
                            similarity_target, _ = compute_similarity(
                                self, video_embedding_target, text_embedding, middle_dim
                            )
                            _, predictions_target = similarity_target.topk(1, dim=-1)
                            predictions_target_all.extend(
                                [i[0] for i in predictions_target.cpu().tolist()]
                            )

                        # compute agreement
                        predictions_clip_all = torch.tensor(predictions_clip_all)
                        predictions_source_all = torch.tensor(predictions_source_all)
                        predictions_target_all = torch.tensor(predictions_target_all)
                        cond1 = predictions_clip_all == predictions_source_all
                        cond2 = predictions_source_all == predictions_target_all
                        cond3 = predictions_clip_all == predictions_target_all

                        # compute mask
                        mask = (
                            (predictions_clip_all == predictions_source_all)
                            | (predictions_source_all == predictions_target_all)
                            | (predictions_clip_all == predictions_target_all)
                        )

                        # compute indices
                        indices = mask.nonzero().view(-1)

                        # compute pseudo labels
                        outputs = torch.zeros_like(mask).long()
                        outputs[cond1] = predictions_clip_all[cond1]
                        outputs[cond2] = predictions_source_all[cond2]
                        outputs[cond3] = predictions_target_all[cond3]

                        # update datamodule
                        train_subset = Subset(
                            self.trainer.datamodule.data_train_target, indices
                        )
                        self.trainer.datamodule.data_train_target = train_subset

                        for idx, video_record in enumerate(
                            self.trainer.datamodule.data_train.target_dataset.video_list
                        ):
                            video_record.label = outputs[idx]

    def on_test_epoch_start(self):
        self.load_adapter_checkpoints()

    def on_train_start(self):
        if self.hparams.distillation:
            self.load_adapter_checkpoints()

    def on_validation_epoch_end(self):

        val_accuracy = float(self.corr_1) / self.num
        self.log(
            "validation_accuracy",
            val_accuracy,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
        )
        # print("Validation_accuracy: {}\n".format(val_accuracy))

        is_best = False

        if val_accuracy > self.best:
            self.best = val_accuracy
            is_best = True

        with open(
            "{}/frame_activations.json".format(self.extra_args["working_dir"]), "w"
        ) as f:
            f.write(dumps(self.decomposition_dict, indent=2))

            # confusion matrix
            if not self.trainer.sanity_checking:
                labels = [i[1] for i in self.classes_names]
                labels_manager = LabelsManager(
                    data=self.extra_args,
                )
                cm = confusion_matrix(
                    labels_manager.convert(self.gt),
                    labels_manager.convert(self.pred),
                    labels=labels,
                )
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=labels
                )
                disp.plot(xticks_rotation="vertical")
                plt.savefig("{}/cm_last.png".format(self.extra_args["working_dir"]))

                if is_best:
                    copy(
                        "{}/cm_last.png".format(self.extra_args["working_dir"]),
                        "{}/cm_best.png".format(self.extra_args["working_dir"]),
                    )

                plt.close()
