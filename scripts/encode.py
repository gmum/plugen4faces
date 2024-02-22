"""CLI script for mass encoding images with encoder4editing."""
import argparse
import os
import pathlib
from typing import Tuple, Union

import click
import numpy as np
import torch
import torchvision.transforms as transforms
from const import global_config
from PIL import Image
from tqdm.auto import tqdm

import dlib

from plugen4faces.encoder4editing.models.psp import pSp
from plugen4faces.encoder4editing.utils.alignment import align_face
from plugen4faces.utils import download_landmark_predictor, t2n, tensor2im


def try_align(image_path: str, face_predictor) -> Image.Image:
    try:
        img = align_face(filepath=image_path, predictor=face_predictor)
    except UnboundLocalError:
        print("Empty detection:", image_path)
        img = Image.open(image_path).convert("RGB")
    return img


def load_e4e(
    model_path: str, device: Union[str, torch.device] = global_config.device
) -> pSp:
    """Load encoder4editing from a checkpoint file."""
    ckpt = torch.load(model_path, map_location="cpu")
    opts = ckpt["opts"]
    opts["checkpoint_path"] = model_path
    opts["device"] = device
    opts = argparse.Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net = net.to(device)
    print("Model successfully loaded!")
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def encode(
    img: Union[np.ndarray, Image.Image],
    net: pSp,
    device: Union[str, torch.device] = global_config.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(img, np.ndarray) or isinstance(img, Image.Image):
        img = transforms.functional.to_tensor(img).float()
    if img.ndim == 3:
        img = img.unsqueeze(0)
    t = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = t(img).to(device)
    result_image, latent = net(
        img,
        resize=False,
        randomize_noise=False,
        return_latents=True,
    )
    return tensor2im(result_image[0]), latent


@click.command()
@click.option("--src", "source_dir", required=True, help="Source image directory.")
@click.option("--target", "target_dir", required=True, help="Path to store results in.")
@click.option(
    "--model",
    "model_path",
    show_default=True,
    default=str(global_config.e4e_path),
    help="Path to pretrained model file",
)
@click.option(
    "--align",
    "use_align",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to run face detector on images before encoding",
)
@click.option(
    "--predictor",
    "predictor_path",
    show_default=True,
    default=str(global_config.landmark_predictor_path),
    help="Path to predictor file used to align images. If None, default will be downloaded.",
)
@click.option(
    "--save-images",
    is_flag=True,
    show_default=True,
    default=False,
    help="Save images too otherwise just latents.",
)
@click.option(
    "--skip-existing/--overwrite-existing",
    is_flag=True,
    show_default=True,
    default=True,
    help="Whether to run overwrite or skip already existing files",
)
@torch.no_grad()
def main(
    source_dir,
    target_dir,
    model_path,
    use_align,
    predictor_path,
    save_images,
    skip_existing,
):
    net = load_e4e(model_path)

    if use_align:
        if not os.path.exists(global_config.landmark_predictor_path):
            download_landmark_predictor(global_config.landmark_predictor_path)
    predictor = dlib.shape_predictor(predictor_path) if use_align else None
    dataset = list(pathlib.Path(source_dir).glob("**/*.png"))
    os.makedirs(target_dir, exist_ok=True)

    s = dataset[0].relative_to(source_dir)
    path_start_idx = str(dataset[0]).find(str(s))
    existing = 0
    print("Found", len(dataset), "images")
    for path in tqdm(dataset):
        path = str(path)

        # mirroring subdirs in target_dir
        ss = path[path_start_idx:].split(".")[0].split("/")
        d = target_dir + "/".join(ss[:-1])
        os.makedirs(d, exist_ok=True)

        f = os.path.join(d, ss[-1] + ".npz")
        if os.path.exists(f):
            existing += 1
            if skip_existing:
                continue

        img = try_align(path, predictor) if use_align else Image.open(path, "RGB")
        result_image, latent = encode(img, net)

        np.savez(f, f=latent.cpu().numpy())
        if save_images:
            Image.fromarray(t2n(result_image.cpu()), "RGB").save(
                os.path.join(d, ss[-1] + ".png")
            )
    if skip_existing:
        print("Skipped files:", existing)
    else:
        print("Overwritten files:", existing)


if __name__ == "__main__":
    main()
