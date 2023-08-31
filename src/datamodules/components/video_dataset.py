import torch.utils.data as data
import numpy as np

from numpy.random import randint
from PIL import Image

from src.datamodules.components.data_utils import VideoRecord, natural_keys, find_frames


class VideoDataset(data.Dataset):
    def __init__(
        self,
        list_file,
        num_segments=1,
        new_length=1,
        transform=None,
        random_shift=True,
        test_mode=False,
        index_bias=1,
        data_folder=None,
        extra_args=None,
    ):

        self.extra_args = extra_args
        self.epic_kitchens = self.extra_args["epic_kitchens"]
        self.daily_da = self.extra_args["daily_da"]
        self.sports_da = self.extra_args["sports_da"]
        self.hmdb_ucf = self.extra_args["hmdb_ucf"]
        self.list_file = list_file
        if self.epic_kitchens:
            self.ek_videos = {}
            self.get_ek_videos()
        self.data_folder = data_folder

        self.num_segments = num_segments
        self.seg_length = new_length
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias

        self._parse_list()
        self.initialized = False

    def get_ek_videos(self):
        with open(self.list_file, "r") as file_list:
            i = 0
            for line in file_list:
                split_line = line.split()
                path = split_line[0]
                label = split_line[3]
                if int(label) < self.extra_args["num_classes"]:
                    kitchen = path.split("/")[-1]
                    if kitchen not in self.ek_videos:
                        kitchen_videos = find_frames(path)
                        kitchen_videos.sort(key=natural_keys)
                        self.ek_videos[kitchen] = kitchen_videos

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    def _parse_list(self):
        self.video_list = [
            VideoRecord(
                x.strip().split("\t" if (self.daily_da or self.sports_da or self.hmdb_ucf) else " "),
                epic_kitchens=self.epic_kitchens,
                daily_da=self.daily_da,
                sports_da=self.sports_da,
                hmdb_ucf=self.hmdb_ucf,
                data_folder=self.data_folder,
            )
            for x in open(self.list_file)
            if (
                int(x.split()[1 if (self.daily_da or self.sports_da or self.hmdb_ucf) else -1])
                < self.extra_args["num_classes"]
            )
            and (
                int(x.split()[1 if (self.daily_da or self.sports_da or self.hmdb_ucf) else -1])
                < int(self.extra_args["classes_limit"])
            )
        ]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                return (
                    np.mod(
                        np.arange(self.total_length) + randint(record.num_frames // 2),
                        record.num_frames,
                    )
                    + self.index_bias
                )
            offsets = np.concatenate(
                (
                    np.arange(record.num_frames),
                    randint(
                        record.num_frames, size=self.total_length - record.num_frames
                    ),
                )
            )
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [
            i * record.num_frames // self.num_segments
            for i in range(self.num_segments + 1)
        ]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames // 2], dtype=np.int32) + self.index_bias

        if record.num_frames <= self.total_length:
            if self.loop:
                return (
                    np.mod(np.arange(self.total_length), record.num_frames)
                    + self.index_bias
                )
            return (
                np.array(
                    [
                        i * record.num_frames // self.total_length
                        for i in range(self.total_length)
                    ],
                    dtype=np.int32,
                )
                + self.index_bias
            )
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return (
            np.array(
                [
                    i * record.num_frames / self.num_segments + offset + j
                    for i in range(self.num_segments)
                    for j in range(self.seg_length)
                ],
                dtype=np.int32,
            )
            + self.index_bias
        )

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = (
            self._sample_indices(record)
            if self.random_shift
            else self._get_val_indices(record)
        )
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def get(self, record, indices):
        images = list()
        if self.epic_kitchens:
            video_id = record.path.split("/")[-1]
            frames = self.ek_videos[video_id][record.start_frame : record.stop_frame]
        else:
            frames = find_frames(record.path)
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                seg_imgs = [Image.open(frames[p]).convert("RGB")]
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print("invalid indices: {}".format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, record.label#, indices, record.path

    def __len__(self):
        return len(self.video_list)


class VideoDatasetSourceAndTarget:
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return max([len(self.source_dataset), len(self.target_dataset)])

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        source_data = self.source_dataset[source_index]

        target_index = index % len(self.target_dataset)
        target_data = self.target_dataset[target_index]
        return *source_data, source_index, *target_data, target_index