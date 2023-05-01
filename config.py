#%%
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
from psutil import cpu_count
from pathlib import Path

from omegaconf import MISSING

dict_config =  {
  "task": "segmentation",
  "name": "msd" ,
  "root": "./media/datasets/MSD/"  ,
  "train_val_split": 0.9,
  "nifti_seg_options":
    {"test_split": 0.1,
    "resolution": 128,
    "msd_subtask": "liver",
    "cache": True,
    # slice_thickness: 1.0
    "normalization_type": "gaussian",
    "n_slices": 50,
    "data_stats":
      {"mean": -81.91,
      "std": 8.736},
    "ct_window": 
      {"low": -150,
      "high": 200}}}
# %%
from omegaconf import OmegaConf

class DatasetName(Enum):
    CIFAR10 = 1
    imagenet = 2
    tinyimagenet = 3
    fastmri = 4
    radimagenet = 5
    msd = 6
    ukbb_seg = 7
    ham10000 = 8


class MSDSubtask(Enum):
    braintumour = 1
    heart = 2
    liver = 3
    hippocampus = 4
    prostate = 5
    lung = 6
    pancreas = 7
    hepaticvessel = 8
    spleen = 9
    colon = 10


class DatasetTask(Enum):
    classification = 1
    reconstruction = 2
    segmentation = 3


class Normalization(Enum):
    raw = 0
    zeroone = 1
    gaussian = 2
    consecutive = 3


@dataclass
class DataStats:
    mean: float = MISSING
    std: float = MISSING


@dataclass
class CTWindow:
    low: int = MISSING
    high: int = MISSING


@dataclass
class FmriConfig:
    mask_type: str = "random"
    center_fractions: tuple[float] = (0.08,)
    accelerations: tuple[float] = (4,)
    challenge: str = "knee"
    resolution: int = 320
    new_data_root: Optional[str] = None


@dataclass
class RadimagenetConfig:
    datasplit_seed: Optional[int] = 0
    modality: str = "all"
    normalize_by_modality: bool = False
    allowed_body_regions: str = "all"
    allowed_labels: str = "all"
    split_folder: Optional[str] = None


@dataclass
class FilterOptionsNifti:
    resolution: Optional[tuple[int, int, int]] = None
    min_pixels_per_organ: Optional[tuple[int]] = None
    length_threshold: Optional[int] = None
    save_filtered_files: Optional[Path] = None
    reuse_filtered_files: Optional[Path] = None


@dataclass
class NiftiSegmentationConfig:
    slice_thickness: Optional[float] = None
    n_slices: Optional[int] = None
    cache: bool = False
    normalization_type: Normalization = MISSING
    data_stats: Optional[DataStats] = None
    ct_window: Optional[CTWindow] = None
    test_split: float = MISSING
    resolution: Optional[int] = None
    datasplit_seed: Optional[int] = 0
    assume_same_settings: bool = False
    msd_subtask: MSDSubtask = MISSING
    new_data_root: Optional[str] = None
    image_file_root: Optional[str] = None
    label_file_root: Optional[str] = None
    normalize_per_scan: bool = False
    filter_options: Optional[FilterOptionsNifti] = None
    limit_dataset: Optional[int] = None
    database: Optional[Path] = None


@dataclass
class HAM10000:
    merge_labels: bool = True  # only for HAM10000


@dataclass
class DatasetConfig:
    name: DatasetName = MISSING
    root: str = MISSING
    version: Optional[int] = None
    train_val_split: float = MISSING
    normalization: bool = False
    download: Optional[bool] = False
    fft: bool = False
    task: DatasetTask = MISSING
    test_split: float = 0.1
    radimagenet: Optional[RadimagenetConfig] = None
    fmri: Optional[FmriConfig] = None
    nifti_seg_options: Optional[NiftiSegmentationConfig] = None
    ham: Optional[HAM10000] = None


class Config:
    dataset: DatasetConfig

#%%
conf = OmegaConf.create(dict_config)
schema = OmegaConf.structured(DatasetConfig)
dataconf = OmegaConf.merge(schema, conf)


conf = Config()
conf.dataset = dataconf
print(conf)

# %%
from nifty.creator import NiftiSegCreator

train_set, val_set, test_set = NiftiSegCreator.make_datasets(conf, None)


print(len(train_set))