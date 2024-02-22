"""
This module defines Plugen4Faces model on SimpleRealNVP
the dataset class and training code.
"""
import json
import os
import re
import time
from datetime import datetime as dt
from pathlib import Path, PurePath
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from loguru import logger
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm
from nflows.utils import torchutils
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid

from .const import global_config
from .utils import attrs2tensor, load_stylegan2, normalize, tensor2im

torch.manual_seed(1337)


class ConditionalDiagonalNormal2(ConditionalDiagonalNormal):
    def _sample(self, num_samples, context):
        # Compute parameters.
        means, log_stds = self._compute_params(context)  # [16,]
        stds = torch.exp(log_stds)
        # means,std are [num_samples*context.shape[0], 512]
        means = torchutils.repeat_rows(
            means, num_samples
        )  # repeat_rows = repeat_interleave
        stds = torchutils.repeat_rows(stds, num_samples)

        # Generate samples.
        context_size = context.shape[0]
        noise = torch.randn(
            num_samples * context.shape[0], *self._shape, device=means.device
        )
        # noise = noise.repeat_interleave(context.shape[0], dim=0)
        assert noise.shape == means.shape == stds.shape, (
            noise.shape,
            means.shape,
            stds.shape,
        )
        samples = means + stds * noise
        assert samples.shape[0] == context.shape[0] * num_samples, (samples.shape,)
        return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def sample(self, num_samples, context, **kwargs):
        return self._sample(num_samples, context, **kwargs)

    def _log_prob(self, inputs, context, start_idx=0):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        # Compute parameters.
        means, log_stds = self._compute_params(context)
        assert means.shape == inputs.shape and log_stds.shape == inputs.shape

        # Compute log prob.
        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * torchutils.sum_except_batch(
            norm_inputs[..., start_idx:] ** 2, num_batch_dims=1
        )
        log_prob -= torchutils.sum_except_batch(
            log_stds[..., start_idx:], num_batch_dims=1
        )
        log_prob -= self._log_z
        return log_prob


class StandardNormal2(StandardNormal):
    def _log_prob(self, inputs, context, start_idx=0):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -0.5 * torchutils.sum_except_batch(
            inputs[..., start_idx:] ** 2, num_batch_dims=1
        )
        return neg_energy - self._log_z


class SimpleRealNVP(Flow):
    """An simplified version of Real NVP for 1-dim inputs.
    This implementation uses 1-dim checkerboard masking but doesn't use multi-scaling.
    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(
        self,
        features=512,
        hidden_features=512,
        context_features=18,
        num_layers=5,
        num_blocks_per_layer=2,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=True,
        conditional=True,  # only used in training
        context_encoder=nn.Linear(18, 1024),
    ):
        self.features = features
        self.context_features = context_features
        self.conditional = conditional

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        if self.conditional:
            dist = (
                ConditionalDiagonalNormal2([features], context_encoder=context_encoder),
            )
        else:
            dist = StandardNormal2([features])
        super().__init__(
            transform=CompositeTransform(layers),
            distribution=dist,
        )

    def _log_prob(
        self, inputs: torch.Tensor, context: torch.Tensor, start_idx: int = 0
    ) -> torch.Tensor:
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution._log_prob(
            noise, context=embedded_context, start_idx=start_idx
        )
        return log_prob + logabsdet

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        return self._distribution.sample(num_samples, context)

    def transform(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self._transform(inputs, context=self._embedding_net(context))

    def inverse(self, flow_noise: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        embedded_context = self._embedding_net(context)
        ws, _ = self._transform.inverse(
            flow_noise.reshape(-1, self.features), context=embedded_context
        )
        return ws.reshape(-1, self.context_features, self.features)


class ProjectionDataset(torch.utils.data.Dataset):
    YT_REGEX = re.compile(r"yt_(\d+)")
    CELEB_REAL_REGEX = re.compile(r"person_(\d+)/background_(\d+)")
    CELEB_FAKE_REGEX = re.compile(r"body_(\d+)/face_(\d+)/background_(\d+)")

    def __init__(
        self,
        path: str,
        attributes2tensor: Callable[[dict], torch.tensor],
        filename: str = "projected_w_frame_{idx}.npz",
        train: bool = True,
        attributes_path: str = "attributes.json",
        char_idx_start: int = 0,
        load_all: bool = False,
        get_attributes: Optional[Callable[[torch.Tensor, PurePath], dict]] = None,
        file_cap: Optional[int] = None,
    ):
        super().__init__()
        self.path = Path(path)
        self.filename = filename
        self.train = train
        self.ids = []
        video_id = None
        self.filename = filename
        self.id_set = set()
        self.FRAME_IDX_REGEX = re.compile(filename.format(idx=r"(\d+)"))

        with open(attributes_path, "r", encoding="utf-8") as file:
            self.attributes = json.load(file)
        self.attributes2tensor = attributes2tensor

        def _get_attributes(a: torch.Tensor, v: PurePath) -> dict:
            return self.attributes[str(v.relative_to(self.path))]

        self.get_attributes = (
            get_attributes if get_attributes is not None else _get_attributes
        )

        if train:
            file_counter = 0
            for v in self.path.glob(f"**/{filename}".format(idx="*")):
                file_counter += 1
                if file_cap is not None and file_counter > file_cap:
                    break
                frame_idx = self.FRAME_IDX_REGEX.search(str(v)).group(1)
                if "YouTube-real" in str(v):
                    match = self.YT_REGEX.search(str(v))
                    background_id = match.group(1)
                    video_id = ("yt", background_id, frame_idx)
                elif "Celeb-real" in str(v):
                    match = self.CELEB_REAL_REGEX.search(str(v))
                    person_id = match.group(1)
                    background_id = match.group(2)
                    video_id = ("real", person_id, background_id, frame_idx)
                elif "Celeb-synthesis" in str(v):
                    raise NotImplementedError("Celeb-fake")
                else:
                    # no sub-directories, single frame per 'video' and character,
                    # frame_idx is 'video' id
                    video_id = ("single", frame_idx, 0)

                if (
                    video_id not in self.id_set
                    and self.get_attributes(self.attributes, v) is not None
                ):
                    self.ids.append(video_id)
                    self.id_set.add(video_id[:-1])  # do not add frameidx
            logger.debug(
                "Added %d training ids ({} characters)", len(self.ids), len(self.id_set)
            )
            self.id_set = {
                x: i for i, x in enumerate(self.id_set, start=char_idx_start)
            }
        else:
            raise NotImplementedError("train=False")

        # load everything into memory
        self.data: list = []
        if load_all:
            from tqdm.auto import tqdm

            self.data = []
            _data = [None] * len(self)
            for idx in tqdm(range(len(self)), desc="Loading dataset"):
                _data[idx] = self[idx]
            self.data = _data

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple:
        if len(self.data) > idx:
            return self.data[idx]

        ids = self.ids[idx]
        frame_idx = ids[-1]

        # get path
        if ids[0] == "yt":
            dir_path = PurePath("YouTube-real") / f"yt_{ids[1]}"
        elif ids[0] == "real":
            dir_path = (
                PurePath("Celeb-real") / f"person_{ids[1]}" / f"background_{ids[2]}"
            )
        elif ids[0] == "single":
            dir_path = PurePath(".")
            frame_idx = str(ids[1]).zfill(4)
        else:
            raise NotImplementedError(ids[0])

        frame_path = dir_path / f"{self.filename}".format(idx=frame_idx)
        attrs = self.attributes2tensor(self.get_attributes(self.attributes, frame_path))
        frame_path = self.path / frame_path
        if str(frame_path).endswith(".npz"):
            ws = torch.tensor(np.load(frame_path)["f"])
        elif str(frame_path).endswith(".pt"):
            ws = torch.load(frame_path, map_location="cpu")
        return ws.squeeze(0), torch.tensor(self.id_set[ids[:-1]]), attrs


def load_plugen4faces(
    path: Union[str, Path] = str(global_config.plugen4faces_path)
) -> SimpleRealNVP:
    path = Path(path)
    assert path.exists(), path
    d = torch.load(path)
    model = SimpleRealNVP(**d["config"]).cuda().eval()
    model.load_state_dict(d["model"], strict=False)
    return model


def train(
    sigma: float,
    decay: float,
    epochs: int,
    num_layers: int,
    flow_divisor: float,
    attr_divisor: float,
    batch_size: int,
    dropout: float,
    batch_norm_between_layers: bool,
    snapshot_interval: int,
    remove_partials: bool,
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    data1 = ProjectionDataset(
        "/shared/sets/datasets/Celeb-DF_e4e_projections",
        filename="frame_{idx}.npz",
        attributes_path="/shared/sets/datasets/Celeb-DF_e4e_projections/attributes.json",
        attributes2tensor=attrs2tensor,
        load_all=True,
    )
    # "/shared/sets/datasets/stylegan2-ffhq-1024x1024_trunc05_e4e"
    data2 = ProjectionDataset(
        "/shared/sets/datasets/stylagan/stylegan2-ffhq-1024x1024_trunc05",
        filename="seed{idx}_w.pt",
        attributes_path="/shared/sets/datasets/stylegan2-ffhq-1024-trunc05.json",
        attributes2tensor=attrs2tensor,
        char_idx_start=len(data1.id_set) + 1,
        load_all=True,
    )
    data = ConcatDataset([data1, data2])
    dl = DataLoader(data, shuffle=True, batch_size=batch_size)

    num_ws = 18
    num_samples = 5
    K, N = 8, 512
    values = "continuous"

    # caching (min,max) for each attribute over entire dataset
    attrs = []
    for d in [data1, data2]:
        for i, a in enumerate(d.attributes.values()):
            if a is None:
                logger.debug("No attributes for {}", i)
            else:
                attrs.append(attrs2tensor(a))
    attrs = torch.stack(attrs, dim=0)
    logger.debug("attrs.shape: {}", attrs.shape)

    # var_ratio = torch.sum((attrs[..., :K] > 0.0).float(), dim=0) / attrs.shape[0]
    # logger.debug('var_ratio.shape: {}', var_ratio.shape)

    minmax = [[], []]
    for i in range(K):
        x = torch.aminmax(attrs[..., i])
        minmax[0].append(x.min)
        minmax[1].append(x.max)
    minmax = torch.tensor(minmax).to(DEVICE).view(2, 1, 1, K)
    logger.debug("minmax: {}", minmax)
    del attrs

    config = {
        "features": 512,
        "hidden_features": 512,
        "context_features": num_ws,
        "num_layers": num_layers,
        "num_blocks_per_layer": 2,
        "use_volume_preserving": False,
        "activation": F.relu,
        "dropout_probability": dropout,
        "batch_norm_within_layers": False,
        "batch_norm_between_layers": batch_norm_between_layers,
    }
    name = "realnvp_epochs{epochs}_sigma{sigma}_decay{decay}_layers{num_layers}_dropout{dropout}.pt"
    name = name.format(
        epochs="{epochs}",
        sigma=sigma,
        decay=decay,
        num_layers=config["num_layers"],
        dropout=config["dropout_probability"],
    )
    onehots = (
        torch.nn.functional.one_hot(torch.arange(num_ws), num_ws).float().to(DEVICE)
    )
    model = SimpleRealNVP(**config).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, cooldown=400, min_lr=1e-6
    )

    logger.info("dataset1: {} frames, {} characters", len(data1), len(data1.id_set))
    logger.info("dataset2: {} frames, {} characters", len(data2), len(data2.id_set))
    os.makedirs("samples", exist_ok=True)
    timestamp = dt.now().strftime("%Y-%m-%d_%H_%M_%S")
    logger.info("Model timestamp: {}", timestamp)

    G = load_stylegan2()

    sw = SummaryWriter()
    for i in range(epochs):
        mean_loss, mean_flow_loss, mean_ident_loss, mean_attr_loss = 0, 0, 0, 0
        model.train()
        epoch_start = time.time()
        sigma *= decay
        sw.add_scalar("sigma", sigma, i)
        for ws, char_idx, attrs in dl:
            ws, char_idx, attrs = ws.to(DEVICE), char_idx.to(DEVICE), attrs.to(DEVICE)
            optim.zero_grad()
            assert attrs.shape[-1] == K, (attrs.shape[-1], K)
            bs = ws.shape[0]
            idx = onehots.repeat(bs, 1).view(-1, num_ws).to(DEVICE)
            ws = ws.reshape(-1, N)

            # normalize adds 1 dim for some reason
            attrs = normalize(attrs, values, min_=minmax[0], max_=minmax[1]).squeeze(0)

            # flow loss from nflows
            z = model.transform_to_noise(inputs=ws, context=idx).view(bs, num_ws, -1)
            attributes, identity = z[..., :K], z[..., K:]
            flow_loss = -model._log_prob(inputs=ws, context=idx, start_idx=K).mean()
            flow_loss /= flow_divisor

            # identity loss
            mean = scatter(identity, char_idx, dim=0, reduce="mean")
            ident_loss = ((identity - mean[char_idx]) ** 2).mean()
            ident_loss *= i + 1

            # attribute loss
            sigma_ = torch.full_like(
                attrs.unsqueeze(1), sigma, device=DEVICE
            )  # [B,1,K]

            # rebalancing, not used
            # for a in [4,5]: # baldness, beard
            #    sigma_pos = sigma_[...,a] * torch.sqrt(2 * var_ratio[a])
            #    sigma_neg = sigma_[...,a] * torch.sqrt(2 * (1 - var_ratio[a]))
            #    sigma_[...,a] = torch.where(attrs[...,a] > 0, sigma_pos.squeeze(-1), sigma_neg.squeeze(-1)).unsqueeze(1)

            if i < 30:
                attr_loss = torch.tensor(0.0).float()
            else:
                dist = D.Normal(attrs.unsqueeze(1), sigma_)  # [B,1,K]
                attr_loss = -dist.log_prob(
                    attributes
                ).mean()  # .sum(dim=-1, keepdim=True) # [num_ws, B, 1]
                attr_loss /= attr_divisor

            loss = flow_loss + ident_loss + attr_loss

            mean_flow_loss += flow_loss / len(dl)
            mean_ident_loss += ident_loss / len(dl)
            mean_attr_loss += attr_loss / len(dl)
            mean_loss += loss.detach() / len(dl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        epoch_end = time.time()
        free, total = torch.cuda.mem_get_info(0)
        logger.info(
            f"epoch: {i}/{epochs}, mean_loss: {mean_loss:.4f}, elapsed: {epoch_end-epoch_start:.2f},"
            + f'lr: {optim.param_groups[0]["lr"]}, free_memory: {free/(2<<29):.2f}/{total/(2<<29):.2f}'
        )
        sw.add_scalar("mean_loss", mean_loss, i)
        sw.add_scalar("mean_flow_loss", mean_flow_loss, i)
        sw.add_scalar("mean_ident_loss", mean_ident_loss, i)
        sw.add_scalar("mean_attr_loss", mean_attr_loss, i)
        sw.add_scalar("lr", optim.param_groups[0]["lr"], i)
        scheduler.step(mean_loss)

        if (i + 1) % snapshot_interval == 0:
            path = f"results/{timestamp}/"
            os.makedirs(path, exist_ok=True)
            d = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "config": config,
                "minmax": minmax,
            }
            torch.save(d, path + name.format(epochs=i + 1))
            if remove_partials and i + 1 - snapshot_interval > 0:
                os.remove(path + name.format(epochs=i + 1 - snapshot_interval))

        if i > 0 and i % 10 == 0:
            torch.cuda.empty_cache()
            model.eval()
            ws = model.sample(num_samples, onehots).transpose(0, 1)
            assert list(ws.shape) == [num_samples, num_ws, N]
            synth_image, _ = G(
                [ws], input_is_latent=True, return_latents=False, randomize_noise=False
            )
            assert list(synth_image.shape) == [
                num_samples,
                3,
                1024,
                1024,
            ], synth_image.shape
            synth_image = make_grid(tensor2im(resize(synth_image, (256, 256))), nrow=5)
            sw.add_image("sample", synth_image, i)
    sw.close()
    path = f"results/{timestamp}/"
    os.makedirs(path, exist_ok=True)
    d = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "config": config,
        "minmax": minmax,
    }
    torch.save(d, path + name.format(epochs=epochs))
    logger.info("Finished at: {}", dt.now().strftime("%Y-%m-%d_%H_%M_%S"))
    if remove_partials and os.path.exists(
        path + name.format(epochs=epochs - snapshot_interval)
    ):
        os.remove(path + name.format(epochs=epochs - snapshot_interval))
