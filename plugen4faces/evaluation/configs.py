"""Defines model wrappers that unify their APIs."""
import abc
import json
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
from arcface import ArcFace

from ..const import NUM_WS, attr2detidx, attr2idx, global_config
from ..StyleFlow.utils import normalize
from ..utils import attrs2tensor, load_detector


class ExperimentContext:
    def __init__(
        self,
        stylegan_path: Union[Path, str] = global_config.stylegan_path,
        detector_classes: int = global_config.detector_classes,
        detector_path: Union[Path, str] = global_config.detector_path,
        root_dir: Union[Path, str] = global_config.data_root_dir,
        labels_path: Union[Path, str] = global_config.labels_path,
        device: torch.device = global_config.device,
        load_labels: bool = True,  # only works for FaceAPI labels
    ) -> None:
        from ..utils import load_stylegan2

        stylegan_path = Path(stylegan_path)
        detector_path = Path(detector_path)
        root_dir = Path(root_dir)
        labels_path = Path(labels_path)

        self.onehots = (
            F.one_hot(torch.arange(NUM_WS), num_classes=NUM_WS)
            .float()
            .to(global_config.device)
        )
        self.zero_padding = torch.zeros(1, NUM_WS, 1).to(global_config.device)

        self.face_rec = ArcFace.ArcFace(str(global_config.arcface_path))
        # min & max used for normalization
        self.min = torch.tensor(
            [0.0000, 0.0000, -52.3000, -32.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        )
        self.max = torch.tensor(
            [1.0000, 1.0000, 58.1000, 28.5000, 0.9900, 0.9000, 72.0000, 1.0000]
        )
        self.attr2idx = attr2idx
        self.attr2detidx = attr2detidx

        self.G = load_stylegan2(str(stylegan_path), device=device)

        self.detector = load_detector(detector_classes, str(detector_path))
        self.det_filename = Path(detector_path).stem
        self.detector_classes = detector_classes

        if load_labels:
            with open(labels_path, "r", encoding="utf-8") as f:
                self.all_attrs = json.load(f)

            # dict2tensor and normalization
            # Note: this is FFHQ and FaceAPI specific
            attrs = []
            NUM_IMAGES = 70000
            for k in range(NUM_IMAGES):
                key = str(k).zfill(5)
                zz = (
                    attrs2tensor(self.all_attrs[key])
                    if self.all_attrs[key]["num_faces"] > 0
                    else torch.zeros(8)
                )
                attrs.append(zz)
            self.attrs = torch.stack(attrs, dim=0).to(global_config.device)
            del attrs
            for i in range(8):
                self.attrs[:, i] = normalize(
                    self.attrs[:, i], "continuous", False, keep=False
                )


class ModelConfig(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        ctx: ExperimentContext,
        ws: torch.Tensor,
        labels: torch.Tensor,
        attr: str,
        direction: int,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward(
        self, ctx: ExperimentContext, ws: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Convert stylegan latents to flow latents."""

    @abc.abstractmethod
    def inverse(
        self, ctx: ExperimentContext, z: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Convert latents back to stylegan latents."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name to save results under."""


class Plugen4FacesConfig(ModelConfig):
    """Plugen4Faces wrapper. Scale adjusts (-1,1) to (-x,x)."""

    def __init__(
        self, model: "SimpleRealNVP", scale: int = 1, translate: bool = False
    ) -> None:
        self.model = model
        self.attr2index = {
            x: i
            for i, x in enumerate(
                ["gender", "glasses", "yaw", "pitch", "bald", "beard", "age", "smile"]
            )
        }
        self.scale = scale
        self.translate = translate

    def __call__(
        self,
        ctx: ExperimentContext,
        ws: torch.Tensor,
        labels: torch.Tensor,
        attr: str,
        direction: int,
    ) -> torch.Tensor:
        z = self.forward(ctx, ws, labels)
        self.set_attr(z, attr, direction)
        return self.inverse(ctx, z, labels)

    def forward(
        self, ctx: ExperimentContext, ws: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.model.transform_to_noise(ws[0], ctx.onehots)

    def inverse(
        self, ctx: ExperimentContext, z: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.model.inverse(z, ctx.onehots)

    def set_attr(self, z: torch.Tensor, attr: str, direction: int) -> None:
        if self.translate:
            z[..., self.attr2index[attr]] += direction * self.scale
        else:
            z[..., self.attr2index[attr]] = direction * self.scale

    @property
    def name(self) -> str:
        return f"plugen4faces_-{self.scale}_{self.scale}"


class PlugenConfig(ModelConfig):
    """Standard Plugen (-1,1)"""

    def __init__(self, model: "NiceFlow", translate: bool = False) -> None:
        self.model = model
        self.attr2index = {
            x: i
            for i, x in enumerate(
                ["gender", "glasses", "yaw", "pitch", "bald", "beard", "age", "smile"]
            )
        }
        self.translate = translate

    def __call__(
        self,
        ctx: ExperimentContext,
        ws: torch.Tensor,
        labels: torch.Tensor,
        attr: str,
        direction: int,
    ) -> torch.Tensor:
        z = self.forward(ctx, ws, labels)
        self.set_attr(z, attr, direction)
        return self.inverse(ctx, z, labels)

    def forward(
        self, ctx: ExperimentContext, ws: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.model(ws.clone().transpose(0, 1))[0]

    def inverse(
        self, ctx: ExperimentContext, z: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.model.inv_flow(z).transpose(0, 1)

    def set_attr(self, z: torch.Tensor, attr: str, direction: int) -> None:
        if self.translate:
            z[..., self.attr2index[attr]] += direction
        else:
            z[..., self.attr2index[attr]] = direction

    @property
    def name(self) -> str:
        return "plugen1"


class Plugen2Config(PlugenConfig):
    """Plugen (-2, 2) except for +beard"""

    def set_attr(self, z: torch.Tensor, attr: str, direction: int) -> None:
        if attr == "beard" and direction == 1:
            super().set_attr(z, attr, direction)
        else:
            super().set_attr(z, attr, direction * 2)

    @property
    def name(self) -> str:
        return "plugen2"


class StyleFlowConfig(ModelConfig):
    def __init__(self, model: "CNF", scale: int, translate: bool = False) -> None:
        self.model = model
        self.attr2index = {
            x: i
            for i, x in enumerate(
                ["gender", "glasses", "yaw", "pitch", "bald", "beard", "age", "smile"]
            )
        }
        self.translate = translate
        self.scale = scale
        self.attr2subset = {
            "smile": [4, 5],
            "yaw": [0, 1, 2],
            "pitch": [0, 1, 2],
            "age": [4, 5, 6, 7],
            "gender": list(range(8)),
            "bald": [0, 1, 2, 3, 4],
            "beard": [5, 6, 7, 10],
            ("glasses", True): [0, 1, 2, 3, 4, 5],
            ("glasses", False): [0, 1, 2],
        }

    def __call__(
        self,
        ctx: ExperimentContext,
        ws: torch.Tensor,
        labels: torch.Tensor,
        attr: str,
        direction: int,
    ) -> torch.Tensor:
        z = self.forward(ctx, ws, labels)
        self.set_attr(z, labels, attr, direction)
        return self.inverse(ctx, z, labels)

    def forward(
        self, ctx: ExperimentContext, ws: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.model(ws, labels, ctx.zero_padding)[0]

    def inverse(
        self, ctx: ExperimentContext, z: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.model(z, labels, ctx.zero_padding, True)[0]

    def set_attr(
        self, z: torch.Tensor, labels: torch.Tensor, attr: str, direction: int
    ) -> None:
        if self.translate:
            labels[self.attr2index[attr]] += direction * self.scale
        else:
            labels[self.attr2index[attr]] = direction * self.scale

    def get_subset(
        self, attr: str, change: float, *, use_custom: bool = False
    ) -> List[int]:
        if use_custom:
            return self.attr2subset[(attr, change > 0) if attr == "glasses" else attr]
        # not sure why it's 16 not 18, should be NUM_WS anyway
        return list(torch.arange(16))

    @property
    def name(self) -> str:
        return f"styleflow{self.scale}"
