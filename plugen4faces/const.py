import dataclasses
import pathlib
import typing

import torch

MODELS = ["plugen4faces", "plugen", "styleflow"]
ModelValues = typing.Literal["plugen4faces", "plugen", "styleflow"]

ATTRS8 = ["gender", "glasses", "yaw", "pitch", "bald", "beard", "age", "smile"]
ATTRS9 = ["gender", "glasses", "bald", "beard", "smile", "age", "pitch", "roll", "yaw"]

NUM_ATTRS = 8
NUM_WS = 18
attr2idx = dict(zip(ATTRS8, range(8)))
attr2detidx = dict(zip(ATTRS9, range(9)))


@dataclasses.dataclass
class EnvironmentConfig:
    """Global constants. Encapsulates default paths to models, datasets, labels."""

    models_dir: pathlib.Path = pathlib.Path("pretrained_models")
    video_dir: pathlib.Path = pathlib.Path("evaluation_videos")

    plugen4faces_path: pathlib.Path = (
        models_dir / "realnvp_epochs1500_sigma0.7_decay0.9995_layers10_dropout0.0.pt"
    )
    plugen_path: pathlib.Path = models_dir / "model_e1000.pch"
    styleflow_path: pathlib.Path = models_dir / "styleflow_tuned.pch"
    stylegan_path: pathlib.Path = models_dir / "stylegan2-ffhq-config-f.pt"
    detector_path: pathlib.Path = models_dir / "resnet18_sd_11022023_shrnk_std.pt"
    arcface_path: pathlib.Path = models_dir / "arcface.tflite"
    e4e_path: pathlib.Path = models_dir / "e4e_ffhq_encoder.pt"

    data_root_dir: pathlib.Path = pathlib.Path("ffhq_256_e4e")
    det_out_cache_path: pathlib.Path = pathlib.Path(
        "ffhq_256_e4e_det_out_resnet18_sd_11022023_shrnk_std.pkl"
    )
    labels_path: pathlib.Path = data_root_dir / "ffhq_256_labels.json"

    detector_threshold_result_path: str = "detector_threshold_all_combined.pkl"
    detector_classes: int = 9

    landmark_predictor_path: pathlib.Path = (
        models_dir / "shape_predictor_68_face_landmarks.dat"
    )
    landmark_predictor_url: str = (
        # "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    )
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )


global_config = EnvironmentConfig()
