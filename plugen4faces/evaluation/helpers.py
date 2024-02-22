"""Helper classes for evaluation."""
import dataclasses
import pathlib
import pickle
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple

import cv2
import face_recognition as fr
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics.functional as tmf
import torchvision.transforms.functional as T
from arcface import ArcFace
from torchvision.utils import make_grid

from ..utils import t2n, tensor2im
from .search import LinearThreshold, SearchAlgorithm

ResultType = DefaultDict[str, Dict[str, np.ndarray]]


@dataclasses.dataclass
class DetectorThresholdResult:
    """Object to store results of detector threshold search."""

    num_dists: int
    detector_classes: int
    dists: ResultType = dataclasses.field(default_factory=lambda: defaultdict(dict))
    detector_out: ResultType = dataclasses.field(
        default_factory=lambda: defaultdict(dict)
    )
    flow_values: ResultType = dataclasses.field(
        default_factory=lambda: defaultdict(dict)
    )
    skipped: DefaultDict[str, List[Tuple[str, str]]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )

    def set_zero(self, edit_name: str, config_name: str, num_samples: int) -> None:
        self.dists[edit_name][config_name] = np.zeros((num_samples, self.num_dists))
        self.detector_out[edit_name][config_name] = np.zeros(
            (num_samples, self.detector_classes)
        )
        self.flow_values[edit_name][config_name] = np.zeros((num_samples, 1))

    def save_det_out(
        self, edit_name: str, config_name: str, idx: int, value: np.ndarray
    ) -> None:
        self.detector_out[edit_name][config_name][idx, :] = value

    def save_flow_value(
        self, edit_name: str, config_name: str, idx: int, value: np.ndarray
    ) -> None:
        self.flow_values[edit_name][config_name][idx] = value

    def save_dists(
        self, edit_name: str, config_name: str, idx: int, value: np.ndarray
    ) -> None:
        self.dists[edit_name][config_name][idx, :] = value

    def add_skipped(
        self, config_name: str, path: str, kind: Literal["orig", "edited"]
    ) -> None:
        self.skipped[config_name].append((path, kind))

    def to_dict(
        self, paths: List[pathlib.Path], search_alg: SearchAlgorithm
    ) -> Dict[str, Any]:
        d = {
            "paths": paths,
            "det_out": self.detector_out,
            "dists": self.dists,
            "skipped": self.skipped,
            "flow_values": self.flow_values,
            "dists_labels": [
                "L2 face_recognition",
                "cossim face_recognition",
                "L2 ArcFace",
                "MSE",
                "PSNR",
                "SSIM",
            ],
        }
        if isinstance(search_alg, LinearThreshold):
            d["space_values"] = search_alg._space_values
            d["values_det_outs"] = search_alg._values_det_outs
        return d

    @staticmethod
    def from_dict(cls, d: Dict[str, Any]) -> Dict[str, Any]:
        result = cls(
            num_dists=len(d["dists_labels"]),
            detector_classes=9,
            dists=d["dists"],
            detector_out=d["det_out"],
            flow_values=d["flow_values"],
            skipped=d["skipped"],
        )
        return result

    def get_detector_out(self, edit_name: str, config_name: str) -> np.ndarray:
        return self.detector_out[edit_name][config_name]

    def get_dists(self, edit_name: str, config_name: str) -> np.ndarray:
        return self.dists[edit_name][config_name]

    def get_skipped(self, config_name: str) -> List[Tuple[str, str]]:
        return self.skipped[config_name]


@dataclasses.dataclass
class Mp4Recorder:
    """Object to record video of evaluation."""

    name_template: str = (
        "detector_threshold_{timestamp}_{attr}{change}_{config_name}.mp4"
    )

    def start_video(
        self,
        root_dir: pathlib.Path,
        name_kwargs: Dict,
        **writer_kwargs: Any,
    ) -> None:
        root_dir.mkdir(exist_ok=True, parents=True)
        self.video = imageio.get_writer(
            root_dir / self.name_template.format(**name_kwargs), **writer_kwargs
        )

    def _resize(self, img: torch.Tensor) -> torch.Tensor:
        return tensor2im(T.resize(img.squeeze(0), (512, 512)))

    def _process_frame(
        self,
        orig_img: torch.Tensor,
        result_img: torch.Tensor,
        det_out: float,
        flow_value: float,
        is_success: bool,
    ) -> np.ndarray:
        txtkwargs = {
            "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale": 0.7,
            "color": (255, 0, 0),
            "thickness": 2,
        }
        result_img = self._resize(result_img).cpu()
        orig_img = self._resize(orig_img).cpu()
        frame = make_grid(torch.stack((orig_img, result_img)), dim=0)
        # copy to actually have permuted dims, not just refs
        frame = t2n(frame).copy()
        frame = cv2.putText(frame, "input", (5, 20), **txtkwargs)
        frame = cv2.putText(frame, "result", (512 + 7, 20), **txtkwargs)
        frame = cv2.putText(
            frame,
            f"flow value: {round(flow_value, 4)}",
            (512 + 96, 20),
            **txtkwargs,
        )
        frame = cv2.putText(
            frame,
            f"det_out: {round(det_out, 4)}",
            (512 + 96 + 7 + 200, 20),
            **txtkwargs,
        )
        frame = cv2.putText(
            frame,
            f"success: {is_success}",
            (512 + 96 + 7 + 200, 60),
            **txtkwargs,
        )
        # align to nearest multiple of 16 (https://math.stackexchange.com/a/291494)
        frame = cv2.resize(
            frame,
            dsize=[((x - 1) | 15) + 1 for x in frame.shape[1::-1]],
        )  # cv2.resize transposes dimensions?
        return frame

    def append_frame(
        self,
        orig_img: torch.Tensor,
        result_img: torch.Tensor,
        det_out: float,
        flow_value: float,
        is_success: bool,
    ) -> None:
        frame = self._process_frame(
            orig_img, result_img, det_out, flow_value, is_success
        )
        self.video.append_data(frame)

    def end_video(self) -> None:
        self.video.close()


class OriginalFailedDetectionError(Exception):
    """FaceRecognition failed detection on original image."""


class EditedFailedDetectionError(Exception):
    """FaceRecognition failed detection on edited image."""


class MetricReporter:
    """Object to calculate metrics."""

    num_dists: int = 6

    def __init__(self, face_rec: ArcFace, cache_path: Optional[str]) -> None:
        """Initialize MetricReporter.

        Args:
            face_rec: ArcFace model.
            cache_path: Path to cache file.
        """
        self.face_rec = face_rec
        self.cache_path = cache_path

        if cache_path is not None:
            with open(cache_path, "rb") as f:
                self.orig_emb_cache = pickle.load(f)
        else:
            self.orig_emb_cache = {"embs_orig": {}, "af_orig": {}}

    def calculate(
        self, orig_img: torch.Tensor, result_img: torch.Tensor, name: str
    ) -> np.ndarray:
        assert orig_img.dim() == 3, orig_img.shape
        assert orig_img.shape == result_img.shape, (orig_img.shape, result_img.shape)

        # image-image metrics
        mse = F.mse_loss(result_img, orig_img).cpu().numpy()
        psnr = tmf.peak_signal_noise_ratio(result_img, orig_img).cpu().numpy()
        ssim = (
            tmf.structural_similarity_index_measure(
                result_img.unsqueeze(0), orig_img.unsqueeze(0)
            )
            .cpu()
            .numpy()
        )

        # embeddings
        # not sure why this is even different, is the difference only in cpu()?
        # img1024 = (
        #     tensor2im(result_img)[0]
        #     if linear
        #     else tensor2im(result_img[0]).cpu()
        # )
        img1024 = tensor2im(result_img.cpu())
        emb = fr.api.face_encodings(t2n(img1024), num_jitters=1)
        af = self.face_rec.calc_emb(t2n(img1024))
        if self.cache_path is None:
            orig = fr.api.face_encodings(t2n(orig_img), num_jitters=1)
            self.orig_emb_cache["embs_orig"][name] = orig
        orig = self.orig_emb_cache["embs_orig"][name]
        if orig is None:
            raise OriginalFailedDetectionError
        if len(emb) == 0:
            raise EditedFailedDetectionError

        emb = emb[0]
        euclid_dist = fr.api.face_distance([emb], orig)
        cossim = F.cosine_similarity(
            torch.from_numpy(orig), torch.from_numpy(emb), dim=0
        )
        if self.cache_path is None:
            self.orig_emb_cache["af_orig"][name] = self.face_rec.calc_emb(t2n(orig_img))
        af_dist = self.face_rec.get_distance_embeddings(
            self.orig_emb_cache["af_orig"][name], af
        )
        return np.stack(
            [euclid_dist[0], cossim.numpy(), af_dist, mse, psnr, ssim],
            axis=0,
        )
