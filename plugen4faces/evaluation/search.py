"""Module with search algorithms."""
import abc
import dataclasses
import typing
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.functional as T

from ..utils import det_postproc
from .configs import ExperimentContext, ModelConfig, StyleFlowConfig


class SearchAlgorithm(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        config: ModelConfig,
        ws: torch.Tensor,
        labels: torch.Tensor,
        attr: str,
        change: float,
        threshold: float,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


def get_edit_name(attr: str, change: float, threshold: float) -> str:
    return f"{attr}_{change}_{threshold}"


@dataclasses.dataclass
class LinearThreshold(SearchAlgorithm):
    ctx: ExperimentContext
    num_steps: int
    search_batch_size: int
    edit_name_fn: typing.Callable[[str, float, float], typing.Hashable] = get_edit_name
    sf_subset: bool = False
    epsilon: float = 5e-3

    # used for saving linear specific data
    # incremented after each __call__
    idx: int = 0

    _space_values: typing.DefaultDict[
        str, typing.Dict[str, np.ndarray]
    ] = dataclasses.field(default_factory=lambda: defaultdict(dict))
    _values_det_outs: typing.DefaultDict[
        str, typing.Dict[str, np.ndarray]
    ] = dataclasses.field(default_factory=lambda: defaultdict(dict))

    def set_zero(self, edit_name: str, config_name: str, num_samples: int) -> None:
        self._space_values[edit_name][config_name] = np.zeros(
            (num_samples, self.num_steps)
        )
        self._values_det_outs[edit_name][config_name] = np.zeros(
            (num_samples, self.num_steps, self.ctx.detector_classes)
        )

    def __call__(
        self,
        config: ModelConfig,
        ws: torch.Tensor,
        labels: torch.Tensor,
        attr: str,
        change: float,
        threshold: float,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attr_idx = self.ctx.attr2idx[attr]
        det_attr_idx = self.ctx.attr2detidx[attr]
        edit_name = typing.cast(str, self.edit_name_fn(attr, change, threshold))

        z = config.forward(self.ctx, ws, labels)
        if "styleflow" in config.name:
            left, right = labels[attr_idx], torch.tensor(change)
        else:
            left, right = z[..., attr_idx].mean(), torch.tensor(change)
        left, right = torch.min(left, right), torch.max(left, right)

        space = torch.linspace(left, right, self.num_steps)
        num_ws = ws.shape[1]
        bs = self.search_batch_size  # points at a time on gpu
        space_imgs = []
        attr_det_outs = torch.zeros(space.shape)

        for i in range(0, self.num_steps, bs):
            current_ws = ws.repeat(bs, 1, 1).cuda()
            current_labels = torch.broadcast_to(labels, (bs, 18, 8)).clone().cuda()
            z = config.forward(
                self.ctx,
                current_ws.view(bs * num_ws, 1, -1),
                current_labels.view(-1, 8),
            )
            if isinstance(config, StyleFlowConfig):
                subset = config.get_subset(attr, change, use_custom=self.sf_subset)
                current_labels[:, :, attr_idx] = space[i : i + bs].view(bs, 1)

                # inverse
                ws2 = config.inverse(self.ctx, z, current_labels.view(-1, 8))
                ws2 = ws2.reshape(bs, num_ws, -1)
                current_ws[:, subset] = ws2[:, subset]
            else:  # this path is untested
                # inverse
                ws2 = config.inverse(self.ctx, z, current_labels.view(-1, 8))
                current_ws = ws2.reshape(bs, num_ws, -1)

            # get imgs and detector outputs
            imgs, _ = self.ctx.G.forward(
                [current_ws],
                input_is_latent=True,
                return_latents=False,
                randomize_noise=False,
            )
            space_imgs += list(imgs.cpu())
            det_outs = det_postproc(self.ctx.detector(T.resize(imgs, (256, 256)))).cpu()
            self._values_det_outs[edit_name][config.name][
                self.idx, i : i + bs
            ] = det_outs
            attr_det_outs[i : i + bs] = det_outs[:, det_attr_idx]
            torch.cuda.empty_cache()

        print("attrdetout", attr_det_outs)
        self._space_values[edit_name][config.name][self.idx] = space

        result_index = self.num_steps + 1
        if change > 0:
            is_success = attr_det_outs >= threshold - self.epsilon
            nz = is_success.nonzero()
            if len(nz) > 0:
                # get index of a smallest detector out above threshold
                result_index = nz[attr_det_outs[is_success].argmin()]
        else:
            is_success = attr_det_outs <= threshold + self.epsilon
            nz = is_success.nonzero()
            if len(nz) > 0:
                result_index = nz[attr_det_outs[is_success].argmax()]
        success = result_index < self.num_steps
        if not success:
            result_index = (
                attr_det_outs.argmax() if change > 0 else attr_det_outs.argmin()
            )
        result = space[result_index]
        result_det_out = self._values_det_outs[edit_name][config.name][
            self.idx, result_index
        ]
        print("is_success", is_success)
        print(
            "threshold",
            threshold,
            "change",
            change,
            "result_index",
            result_index,
            "result_det_out",
            result_det_out,
            "success",
            success,
        )
        self.idx += 1
        return result.cpu(), result_det_out, imgs[result_index].cpu(), is_success


@dataclasses.dataclass
class BinsearchThreshold(SearchAlgorithm):
    ctx: ExperimentContext
    max_steps: int
    epsilon: float = 5e-3

    def __call__(
        self,
        config: ModelConfig,
        ws: torch.Tensor,
        labels: torch.Tensor,
        attr: str,
        change: float,
        threshold: float,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attr_idx = self.ctx.attr2idx[attr]
        det_attr_idx = self.ctx.attr2detidx[attr]

        z = config.forward(self.ctx, ws, labels)
        if "styleflow" in config.name:
            left, right = labels[attr_idx], torch.tensor(change)
        else:
            left, right = z[..., attr_idx].mean(), torch.tensor(change)
        left, right = torch.min(left, right), torch.max(left, right)

        # first iteration with rightmost value
        mid = right if change > 0 else left
        if "styleflow" in config.name:
            labels[attr_idx] = mid
        else:
            z[..., attr_idx] = mid
        img, det_out = _get_det_out(config, self.ctx, z, labels)

        # epsilon only works left-side, > threshold has to look for lower,
        # similarily for negative change
        value = det_out[0, det_attr_idx]
        if change > 0:
            not_close = value < threshold - self.epsilon or value > threshold
        else:
            not_close = value > threshold + self.epsilon or value < threshold

        i = 0
        while not_close and i < self.max_steps:
            mid = (left + right) / 2
            if "styleflow" in config.name:
                labels[attr_idx] = mid
            else:
                z[..., attr_idx] = mid
            img, det_out = _get_det_out(config, self.ctx, z, labels)
            value = det_out[0, det_attr_idx]
            if value > threshold:
                right = mid
            elif value < threshold:
                left = mid
            i += 1
            if change > 0:
                not_close = value < threshold - self.epsilon or value > threshold
            else:
                not_close = value > threshold + self.epsilon or value < threshold
        # TODO: not sure if this is correct
        is_success = (
            change > 0 and value < threshold - self.epsilon or value > threshold
        ) or (change <= 0 and value > threshold + self.epsilon or value < threshold)
        return mid.cpu(), det_out.cpu(), img.cpu(), is_success


def _get_det_out(
    config, ctx: ExperimentContext, z: torch.Tensor, attrs: torch.Tensor
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    ws = config.inverse(ctx, z, attrs)
    img, _ = ctx.G.forward(
        [ws], input_is_latent=True, return_latents=False, randomize_noise=False
    )
    det_out = det_postproc(ctx.detector(T.resize(img, (256, 256))), ctx.det_filename)
    return img, det_out
