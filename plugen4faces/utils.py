import bz2
import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import face_recognition as fr
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .const import global_config
from .resnets import ResNet, ResNet18
from .StyleFlow.module.flow import CNF, cnf
from .StyleFlow.NICE import NiceFlow


def t2n(t: torch.Tensor) -> np.ndarray:
    """Convert image from torch to numpy. C,H,W -> W,H,C."""
    return t.permute(1, 2, 0).numpy()


def gen_img(
    ws: torch.Tensor, G: "Generator", target_size: Tuple[int, int] = (256, 256)
) -> torch.Tensor:
    img, _ = G.forward(
        [ws], input_is_latent=True, return_latents=False, randomize_noise=False
    )
    if img.shape[0] == 1:
        img = img[0]
    return tensor2im(T.resize(img, target_size))


def load_npz(path: str) -> torch.Tensor:
    return torch.tensor(np.load(path)["f"])


def load_stylegan2(
    path: str = str(global_config.stylegan_path),
    device: Union[str, torch.device] = "cuda",
) -> "Generator":
    from .stylegan2.model import Generator

    G = Generator(1024, 512, 8, channel_multiplier=2).eval().to(device)
    G.load_state_dict(torch.load(path, map_location=device)["g"], strict=True)
    logger.info("Loading StyleGAN from {}", path)
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def attrs2tensor(attrs: dict) -> torch.Tensor:
    """StyleFlow and PluGeN ordering."""
    t = torch.tensor
    result = [
        t(attrs["gender"] == "male"),
        t(attrs["glasses"] != "noGlasses"),
        t(attrs["head_pose"]["yaw"]),
        t(attrs["head_pose"]["pitch"]),
        t(attrs["hair"]["bald"]),
        t(attrs["facial_hair"]["beard"]),
        t(attrs["age"]),
        t(attrs["smile"]),
    ]
    return torch.stack(result, dim=0)


def attrs2detector(attrs: dict, num_classes: int = 9) -> torch.Tensor:
    return torch.tensor(
        [
            attrs["gender"] == "male",
            attrs["glasses"] != "noGlasses",
            attrs["hair"]["bald"],
            attrs["facial_hair"]["beard"],
            attrs["smile"],
            attrs["age"],
            attrs["head_pose"]["pitch"],
            attrs["head_pose"]["roll"],
            attrs["head_pose"]["yaw"],
        ][:num_classes]
    )


def det_postproc(
    det_out: torch.Tensor, _: str = str(global_config.detector_path)
) -> torch.Tensor:
    """Runs detector postprocessing on raw output."""
    return torch.cat(
        [
            torch.clamp(det_out[..., :2], 0, 1),
            det_out[..., 2:6],
            torch.clamp(det_out[..., 6:9], -180, 180),
        ],
        dim=-1,
    )


def normalize(
    x: torch.Tensor,
    values,
    min_: Optional[torch.Tensor] = None,
    max_: Optional[torch.Tensor] = None,
    keep=False,
):
    """Adapted from plugen"""
    if min_ is None:
        min_ = x.min()
    if max_ is None:
        max_ = x.max()

    x = (x - min_) / (max_ - min_)
    if keep or values == "continuous":
        return 2 * x - 1
    if values > 2:
        x[x == 1.0] = 0.9999
        x = ((values * x).int().float() / (values - 1)).float()
        x = 2 * x - 1
    else:
        x = 2 * x - 1
    return x


def tensor2im(img: torch.Tensor) -> torch.Tensor:
    """Rescale image-like tensor to [0,255] and cast to uint8.

    Args:
        img (torch.Tensor): Shape: [B, 3, H, W]. Image like tensor to
        convert to displayable image.
    """
    img = (img.cpu().detach() + 1) / 2
    img[img < 0] = 0
    img[img > 1] = 1
    return (img * 255).to(torch.uint8)


class NoFaceError(Exception):
    pass


def get_dists(img1: torch.Tensor, img2: torch.Tensor, arcface: "ArcFace") -> np.ndarray:
    emb1 = fr.api.face_encodings(t2n(img1), num_jitters=1)
    if len(emb1) == 0:
        raise NoFaceError
    emb1 = emb1[0]
    emb2 = fr.api.face_encodings(t2n(img2), num_jitters=1)
    if len(emb2) == 0:
        raise NoFaceError
    emb2 = emb2[0]
    d = fr.api.face_distance([emb1], emb2)
    cossim = F.cosine_similarity(torch.from_numpy(emb1), torch.from_numpy(emb2), dim=0)
    af_dist = arcface.get_distance_embeddings(
        arcface.calc_emb(t2n(img1)), arcface.calc_emb(t2n(img2))
    )
    return np.array([d[0], cossim, af_dist])


def roc_auc_ranking(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> List[float]:
    indices = torch.triu_indices(y_true.shape[0], y_true.shape[0], offset=1)
    real = (y_true.unsqueeze(0) < y_true.unsqueeze(1))[indices[0], indices[1], :]
    predicted = (y_pred.unsqueeze(0) < y_pred.unsqueeze(1))[indices[0], indices[1], :]
    auc = []
    for i in range(num_classes):
        auc.append(roc_auc_score(predicted[:, i], real[:, i]).round(4))
    return auc


def spearman_rankorder(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> List:
    r = []
    for i in range(num_classes):
        x = spearmanr(y_pred[:, i].numpy(), y_true[:, i].numpy())
        r.append(x)
    return r


def sha256sum(filename: str) -> str:
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def save_response_content(response: requests.Response, destination: str) -> None:
    CHUNK_SIZE = 32768
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    with open(destination, "wb") as f:
        pb = tqdm(total=total_size_in_bytes, unit="MiB", unit_scale=True)
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                pb.update(len(chunk))
                f.write(chunk)


def download_from_gdrive(
    file_id: str, target_path: str, *, force: bool = False
) -> None:
    """Downloads a file from Google Drive and saves at a target path."""
    if os.path.exists(target_path):
        if not force:
            logger.error("File {} already exists", target_path)
            return
        else:
            logger.error("File {} already exists. Overwriting...", target_path)
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    with session.get(
        url, params={"id": file_id, "confirm": 1}, stream=True
    ) as response:
        response.raise_for_status()
        logger.info("Downloading the checkpoint from {}", response.url)
        save_response_content(response, target_path)


def download_landmark_predictor(
    target_path: Union[str, Path],
    landmark_predictor_url: str = global_config.landmark_predictor_url,
) -> None:
    target_path = Path(target_path)
    bz2_path = Path(str(target_path) + ".bz2")
    if not target_path.exists():
        print("Downloading predictor file")
        if not bz2_path.exists():
            save_response_content(
                requests.get(landmark_predictor_url),
                str(bz2_path),
            )
        with open(target_path, "wb") as f:
            f.write(bz2.BZ2File(bz2_path).read())


def load_detector(
    detector_classes: int = 9,
    detector_path: Union[str, Path] = str(global_config.detector_path),
    device: Union[str, torch.device] = global_config.device,
) -> ResNet:
    detector_path = Path(detector_path)
    detector = ResNet18(num_classes=detector_classes).to(device).eval()
    detector.load_state_dict(torch.load(detector_path), strict=True)
    for p in detector.parameters():
        p.requires_grad = False
    return detector


def load_plugen(path: Union[str, Path] = str(global_config.plugen_path)) -> NiceFlow:
    path = Path(path)
    assert path.exists(), path
    plugen = (
        NiceFlow(input_dim=512, n_layers=4, n_couplings=4, hidden_dim=512).eval().cuda()
    )
    plugen.load_state_dict(torch.load(path)["model"], strict=True)
    return plugen


def load_styleflow(path: Union[str, Path] = str(global_config.styleflow_path)) -> CNF:
    path = Path(path)
    assert path.exists(), path
    styleflow = cnf(512, "512-512-512-512-512", 8, 1).cuda().eval()
    styleflow.load_state_dict(torch.load(path)["model"], strict=True)
    return styleflow
