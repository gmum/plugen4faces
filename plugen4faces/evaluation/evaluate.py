"""This module defines evaluation functions."""
import dataclasses
import os
import pathlib
import pickle
import typing
from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import face_recognition as fr
import numpy as np
import torch
from arcface import ArcFace
from tqdm.auto import tqdm

from ..const import global_config
from ..flow import load_plugen4faces
from ..utils import (
    load_npz,
    load_plugen,
    load_styleflow,
    load_stylegan2,
    t2n,
    tensor2im,
)
from .configs import (
    ExperimentContext,
    ModelConfig,
    Plugen4FacesConfig,
    PlugenConfig,
    StyleFlowConfig,
)
from .helpers import (
    DetectorThresholdResult,
    EditedFailedDetectionError,
    MetricReporter,
    Mp4Recorder,
    OriginalFailedDetectionError,
)
from .search import BinsearchThreshold, LinearThreshold, SearchAlgorithm, get_edit_name


def get_files(root_dir: pathlib.Path) -> List[pathlib.Path]:
    root_dir = pathlib.Path(root_dir)
    all_paths = list(root_dir.glob("**/*.npz"))
    # take everything, but deduplicate filenames
    paths = {p.name: p for p in all_paths}
    return list(paths.values())


@dataclasses.dataclass
class EmbeddingCache:
    face_rec_embs: Dict[str, np.ndarray]
    arcface_embs: Dict[str, np.ndarray]

    def __getitem__(
        self, key: typing.Literal["embs_orig", "af_orig"]
    ) -> Dict[str, np.ndarray]:
        """Return the embeddings for the given key.

        Only for compatibility with the old code, for new code use as attributes.
        Key can be either `embs_orig` or `af_orig`.
        """
        if key == "embs_orig":
            return self.face_rec_embs
        elif key == "af_orig":
            return self.arcface_embs
        else:
            raise KeyError(key)


@torch.inference_mode()
def cache_embs(
    ctx: ExperimentContext, cache_path: str, root_dir: str, stylegan_path: str
) -> Dict[str, np.ndarray]:
    """Cache face_recognition and ArcFace embeddings on original images."""
    if cache_path is not None:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    G = load_stylegan2(stylegan_path)

    root_dir_path = pathlib.Path(root_dir)
    paths = get_files(root_dir_path)

    face_rec = ArcFace.ArcFace(global_config.arcface_path)

    orig_imgs, embs_orig, af_orig = (
        {},
        cache.get("embs_orig", {}),
        cache.get("af_orig", {}),
    )
    failed = 0
    for path in tqdm(paths, desc="generating images"):
        name = path.name
        if name not in embs_orig and name not in af_orig:
            ws = load_npz(str(path))
            img_orig, _ = G.forward(
                [ws.cuda()],
                input_is_latent=True,
                return_latents=False,
                randomize_noise=False,
            )
            orig_imgs[name] = tensor2im(img_orig[0]).cpu()
            fr_emb, af_emb = calc_emb(orig_imgs[name], face_rec)
            if len(fr_emb) == 0:
                failed += 1
                print(path, "embedding failed: face_recognition returned an empty list")
                continue
            embs_orig[name] = fr_emb[0]
            af_orig[name] = af_emb
        else:
            print(name, "already in cache, skipping")
    print("failed:", failed)
    return dataclasses.asdict(
        EmbeddingCache(face_rec_embs=embs_orig, arcface_embs=af_orig)
    )


def result_dump(
    timestamp: str, results: Any, command: str, *, overwrite: bool = False
) -> None:
    out = f"evaluation_results/evaluate_{command}_{timestamp}.pkl"
    if os.path.exists(out) and not overwrite:
        print("File", out, "already exists!")
        return
    with open(out, "wb") as f:
        pickle.dump(results, f)
        print("Results saved in", out)


def calc_emb(
    img: torch.Tensor, face_rec: ArcFace.ArcFace
) -> Tuple[np.ndarray, ArcFace.ArcFace]:
    emb = fr.api.face_encodings(t2n(img), num_jitters=1)
    return (emb, face_rec.calc_emb(t2n(img)))


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
    timestamp = click_ctx.obj["timestamp"]
    save_results_ = click_ctx.obj["save_results"]
    ctx = ExperimentContext(
        stylegan_path=stylegan_path,
        detector_classes=detector_classes,
        detector_path=detector_path,
        root_dir=root_dir,
    )
    root_dir_path = pathlib.Path(root_dir)
    paths = get_files(root_dir_path)
    print("files:", len(paths))

    configs: List[ModelConfig] = []
    print(plugen4faces_path, plugen_path, styleflow_path)
    if plugen4faces_path != "":
        configs.append(
            Plugen4FacesConfig(load_plugen4faces(pathlib.Path(plugen4faces_path)), 1)
        )
    if plugen_path != "":
        configs.append(PlugenConfig(load_plugen(pathlib.Path(plugen_path))))
    if styleflow_path != "":
        configs.append(StyleFlowConfig(load_styleflow(pathlib.Path(styleflow_path)), 1))
    print("configs:", configs)

    reporter = MetricReporter(ctx.face_rec, cache_path)
    dt_result = DetectorThresholdResult(
        num_dists=reporter.num_dists, detector_classes=ctx.detector_classes
    )
    search_alg: SearchAlgorithm = (
        LinearThreshold(
            ctx=ctx,
            num_steps=num_steps,
            search_batch_size=search_batch_size,
            sf_subset=sf_subset,
            epsilon=epsilon,
        )
        if linear
        else BinsearchThreshold(
            ctx=ctx,
            max_steps=max_steps,
            epsilon=epsilon,
        )
    )
    if save_video:
        video_dir = pathlib.Path("evaluation_videos")
        video_dir.mkdir(exist_ok=True, parents=True)
        recorder = Mp4Recorder()
    else:
        video_dir = None
        recorder = None

    all_ws_ = []
    for idx, path in enumerate(paths):
        all_ws_.append(load_npz(str(path)))
    all_ws = torch.stack(all_ws_, dim=0)
    labels = torch.stack([ctx.attrs[int(path.stem)] for path in paths], dim=0)
    print("Loaded all ws")

    for attr, change, threshold in edits:
        det_attr_idx = ctx.attr2detidx[attr]
        edit_name = get_edit_name(attr, change, threshold)
        for config in configs:
            print("===", edit_name, config.name, "===")
            dt_result.set_zero(edit_name, config.name, len(paths))
            if isinstance(search_alg, LinearThreshold):
                search_alg.set_zero(edit_name, config.name, len(paths))

            dt_result = detector_threshold_single(
                ctx=ctx,
                config=config,
                paths=paths,
                all_ws=all_ws,
                all_labels=labels,
                attr=attr,
                change=change,
                threshold=threshold,
                edit_name=edit_name,
                dt_result=dt_result,
                save_video=save_video,
                video_dir=video_dir,
                timestamp=timestamp,
                search_alg=search_alg,
                reporter=reporter,
                recorder=recorder,
            )

        if save_results_:
            d = dt_result.to_dict(paths, search_alg)
            result_dump(timestamp, d, "detector_threshold", overwrite=True)

    # print summary -- this should be extracted to separate function too
    for attr, change, threshold in edits:
        det_attr_idx = ctx.attr2detidx[attr]
        edit_name = get_edit_name(attr, change, threshold)
        for config in configs:
            x = dt_result.get_detector_out(edit_name, config.name)[:, det_attr_idx]
            successes = (
                (x >= threshold - epsilon).sum()
                if change > 0
                else (x <= threshold + epsilon).sum()
            )
            print(
                config.name,
                attr,
                change,
                "mean success dists:",
                [
                    round(d.item(), 4)
                    for d in dt_result.get_dists(edit_name, config.name)[
                        x >= threshold
                    ].mean(axis=0)
                ],
                "successes:",
                successes,
                "images:",
                len(paths),
                "skipped:",
                len(dt_result.get_skipped(config.name)),
            )
    return dt_result.to_dict(paths, search_alg)


@torch.no_grad()
def detector_threshold_single(
    ctx: ExperimentContext,
    config: ModelConfig,
    search_alg: SearchAlgorithm,
    paths: List[pathlib.Path],
    all_ws: torch.Tensor,
    all_labels: torch.Tensor,
    attr: str,
    change: float,
    threshold: float,
    reporter: Optional[MetricReporter] = None,
    recorder: Optional[Mp4Recorder] = None,
    save_video: bool = True,
    video_dir: Optional[pathlib.Path] = None,
    edit_name: Optional[str] = None,
    dt_result: Optional[DetectorThresholdResult] = None,
    timestamp: Optional[str] = None,
) -> DetectorThresholdResult:
    if timestamp is None:
        timestamp = dt.now().strftime("%Y-%m-%d_%H_%M_%S")
    if edit_name is None:
        edit_name = get_edit_name(attr, change, threshold)
    if reporter is None:
        reporter = MetricReporter(ctx.face_rec, None)
    if dt_result is None:
        dt_result = DetectorThresholdResult(
            num_dists=reporter.num_dists, detector_classes=ctx.detector_classes
        )
        dt_result.set_zero(edit_name, config.name, len(all_ws))
    if video_dir is None:
        video_dir = global_config.video_dir
    if recorder is None:
        recorder = Mp4Recorder()

    recorder.start_video(
        video_dir,
        {
            "timestamp": timestamp,
            "attr": attr,
            "change": change,
            "config_name": config.name,
        },
        fps=2,
    )
    pb = tqdm(enumerate(paths), total=len(paths))
    for idx, path in pb:
        name = str(path.name)
        ws = all_ws[idx].cuda()
        orig_img, _ = ctx.G.forward(
            [ws],
            input_is_latent=True,
            return_latents=False,
            randomize_noise=False,
        )
        orig_img = orig_img.cpu()

        labels = all_labels[idx].cuda().unsqueeze(0)
        result_value, result_det_out, result_img, is_success = search_alg(
            config=config,
            ws=ws,
            labels=labels,
            attr=attr,
            change=change,
            threshold=threshold,
        )
        ident = edit_name, config.name, idx
        dt_result.save_det_out(*ident, result_det_out.cpu().numpy())
        dt_result.save_flow_value(*ident, result_value.cpu().numpy())

        try:
            dists = reporter.calculate(orig_img.squeeze(0), result_img.squeeze(0), name)
            dt_result.save_dists(edit_name, config.name, idx, dists)
        except OriginalFailedDetectionError:
            dt_result.add_skipped(config.name, str(path), "orig")
            print(path, "failed detection on original image")
            continue
        except EditedFailedDetectionError:
            dt_result.add_skipped(config.name, str(path), "edited")
            print(path, "failed detection on edited image")
            continue

        if save_video:
            recorder.append_frame(
                orig_img=orig_img.squeeze(0),
                result_img=result_img.squeeze(0),
                det_out=result_det_out.squeeze(0)[ctx.attr2idx[attr]].item(),
                flow_value=result_value.item(),
                is_success=is_success.item() > 0,
            )
    if save_video:
        recorder.end_video()
    return dt_result
