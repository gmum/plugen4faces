"""Experiment code for model evaluation."""
import os
from datetime import datetime as dt
from typing import Dict, List, Tuple

import click
import numpy as np
import torch
from plugen4faces.const import global_config
from plugen4faces.evaluation.configs import ExperimentContext
from plugen4faces.evaluation.evaluate import cache_embs as _cache_embs
from plugen4faces.evaluation.evaluate import detector_threshold as _detector_threshold
from plugen4faces.evaluation.evaluate import result_dump

torch.manual_seed(1337)


@click.group()
@click.pass_context
@click.option("--save_results", type=bool, default=True)
def main(ctx, save_results: bool) -> None:
    from plugen4faces.stylegan2.model import Generator

    global Generator

    ctx.ensure_object(dict)

    timestamp = dt.now().strftime("%Y-%m-%d_%H_%M_%S")
    ctx.obj["timestamp"] = timestamp
    ctx.obj["save_results"] = save_results
    print("Starting at:", timestamp)


@main.result_callback()
def save_results(results: Dict, save_results: bool) -> None:
    ctx = click.get_current_context()
    timestamp = ctx.obj["timestamp"]
    os.makedirs("evaluation_results", exist_ok=True)
    print("Finished at:", dt.now().strftime("%Y-%m-%d_%H_%M_%S"))
    if save_results and results is not None:
        result_dump(
            timestamp, results, ctx.invoked_subcommand.replace("-", "_"), overwrite=True
        )


@main.command()
@click.pass_context
@click.option(
    "--cache_path", default=None, type=str, help="Path to cache which to extend"
)
@click.option(
    "--root_dir",
    default=global_config.data_root_dir,
    show_default=True,
    type=str,
    help="Path to the root directory of the dataset for which to calculate embeddings",
)
@click.option(
    "--stylegan_path",
    default=global_config.stylegan_path,
    type=str,
    show_default=True,
    help="Path to stylegan2",
)
@torch.inference_mode()
def cache_embs(
    ctx: ExperimentContext, cache_path: str, root_dir: str, stylegan_path: str
) -> Dict[str, np.ndarray]:
    return _cache_embs(cache_path, root_dir, stylegan_path)


##
## Experiment subcommands
##


@main.command()
@click.pass_context
@click.option(
    "--edits",
    "-e",
    multiple=True,
    required=True,
    type=(str, int, float),
    help="(attr,change value,threshold)",
)
@click.option(
    "--linear", default=False, type=bool, help="linear search or binary search?"
)
@click.option(
    "--num_steps",
    default=40,
    type=int,
    help="how many points to consider in linear search",
)
@click.option(
    "--search_batch_size",
    default=10,
    type=int,
    help="how many points in a single batch",
)
@click.option("--max_steps", default=30, type=int, help="max number of binsearch steps")
@click.option(
    "--epsilon",
    default=5e-3,
    type=float,
    help="largest diff from threshold considered a success",
)
@click.option("--cache_path", required=True, type=str, help="embedding cache path")
@click.option(
    "--stylegan_path",
    default=global_config.stylegan_path,
    type=str,
    help="stylegan2 path",
)
@click.option("--detector_path", required=True, type=str, help="detector path")
@click.option(
    "--detector_classes", required=True, type=int, help="detector num_classes"
)
@click.option(
    "--root_dir",
    default=global_config.data_root_dir,
    type=str,
    help="dataset root dir",
)
@click.option(
    "--plugen_path",
    default=global_config.plugen_path,
    type=str,
    help="plugen path",
)
@click.option(
    "--plugen4faces_path",
    default=global_config.plugen4faces_path,
    type=str,
    help="plugen4faces path",
)
@click.option(
    "--styleflow_path",
    default=global_config.styleflow_path,
    type=str,
    help="styleflow path",
)
@click.option("--save_video", default=True, type=bool, help="save video?")
@click.option(
    "--sf_subset",
    default=True,
    type=bool,
    help="Change subset of Ws for styleflow or all?",
)
@torch.no_grad()
def detector_threshold(
    click_ctx,
    edits: List[Tuple[str, int, float]],
    linear: bool,
    num_steps: int,
    max_steps: int,
    epsilon: float,
    cache_path: str,
    stylegan_path: str,
    detector_path: str,
    detector_classes: int,
    root_dir: str,
    plugen_path: str,
    plugen4faces_path: str,
    styleflow_path: str,
    save_video: bool,
    search_batch_size: int,
    sf_subset: bool,
) -> Dict:
    return _detector_threshold(
        edits,
        linear,
        num_steps,
        max_steps,
        epsilon,
        cache_path,
        stylegan_path,
        detector_path,
        detector_classes,
        root_dir,
        plugen_path,
        plugen4faces_path,
        styleflow_path,
        save_video,
        search_batch_size,
        sf_subset,
    )


if __name__ == "__main__":
    main()
