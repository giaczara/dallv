from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.datamodules.components.video_dataset import (
    VideoDataset,
    VideoDatasetSourceAndTarget,
)
from src.datamodules.components.data_utils import get_augmentation
from src.utils.utils import get_classes


class VideoDataModule(LightningDataModule):
    def __init__(
        self,
        num_segments,
        random_shift,
        batch_size,
        input_size,
        randaug,
        workers,
        dataset,
        test_file,
        pin_memory,
        domain_shift,
        classes_limit,
        im2vid,
        source_train_file=None,
        target_train_file=None,
        source_dataset=None,
        target_dataset=None,
        balance_training_set=None,
        data_folder=None,
        train_file=None,
        split=None,
        prototype_extraction=False,
        subtasks_analysis=False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.classes_limit = classes_limit

        # data transformations
        self.transform_train = get_augmentation(True, self.hparams)
        self.transform_val = get_augmentation(False, self.hparams)

        self.data_train: Optional[Dataset] = None
        self.data_train_source: Optional[Dataset] = None
        self.data_train_target: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        self.extra_args = {
            "epic_kitchens": ("ek" in self.hparams.dataset),
            "daily_da": ("daily_da" in self.hparams.source_train_file),
            "sports_da": ("sports_da" in self.hparams.source_train_file),
            "hmdb_ucf": ("hmdb_ucf" in self.hparams.source_train_file)
            if self.hparams.source_train_file is not None
            else False,
            "num_classes": self.num_classes,
            "classes_limit": self.classes_limit
            if self.classes_limit > 0
            else self.num_classes,
        }

    @property
    def num_classes(self):
        res = len(get_classes(dict(self.hparams)))
        return res

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if (
            not self.data_train_source
            and not self.data_train_target
            and not self.data_val
        ):

            if self.domain_shift:

                self.data_train_source = VideoDataset(
                    self.hparams.source_train_file,
                    num_segments=self.hparams.num_segments,
                    random_shift=self.hparams.random_shift,
                    transform=self.transform_train,
                    extra_args=self.extra_args,
                )

                self.data_train_target = VideoDataset(
                    self.hparams.target_train_file,
                    num_segments=self.hparams.num_segments,
                    random_shift=self.hparams.random_shift,
                    transform=self.transform_train,
                    extra_args=self.extra_args,
                )

                self.data_train = VideoDatasetSourceAndTarget(
                    self.data_train_source, self.data_train_target
                )

            else:
                self.data_train = VideoDataset(
                    self.hparams.train_file,
                    num_segments=self.hparams.num_segments,
                    random_shift=self.hparams.random_shift,
                    transform=self.transform_train,
                    extra_args=self.extra_args,
                    data_folder=self.hparams.data_folder,
                )

            if self.hparams.prototype_extraction or self.hparams.subtasks_analysis:
                self.data_val = VideoDataset(
                    self.hparams.source_train_file,
                    num_segments=self.hparams.num_segments,
                    random_shift=self.hparams.random_shift,
                    transform=self.transform_train,
                    extra_args=self.extra_args,
                )
            else:
                self.data_val = VideoDataset(
                    self.hparams.test_file,
                    random_shift=False,
                    num_segments=self.hparams.num_segments,
                    transform=self.transform_val,
                    extra_args=self.extra_args,
                    data_folder=self.hparams.data_folder,
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        predict_dataset = VideoDataset(
            self.hparams.target_train_file,
            num_segments=self.hparams.num_segments,
            random_shift=self.hparams.random_shift,
            transform=self.transform_val,
            extra_args=self.extra_args,
        )
        return DataLoader(
            dataset=predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "video.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
