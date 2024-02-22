"""
Postprocessing code to generate the result table from possibly multiple run files from 
different models. This is how the workflow looks like:
```
# all the files from a single model, in this case styleflow
files = [
    'evaluation_results/evaluate_detector_threshold_2023-05-16_01_45_12.pkl',
    'evaluation_results/evaluate_detector_threshold_2023-05-16_01_45_34.pkl',
    'evaluation_results/evaluate_detector_threshold_2023-05-13_02_13_23.pkl'
]
# merge all these files into a single one
merge_runs(files, 'evaluation_results/evaluate_detector_threshold_styleflow_nosubset')
# then merge styleflow back with other models
merge_styleflow('evaluation_results/evaluate_detector_threshold_styleflow_nosubset_2023-05-20_12_28_00.pkl',
                'evaluation_results/evaluate_detector_threshold_all_sf_nosubset')
# finally, generate the table with results
get_metrics('evaluation_results/evaluate_detector_threshold_all_sf_nosubset_2023-05-20_12_55_15.pkl')
```
"""
import json
import pickle
from datetime import datetime as dt
from pathlib import Path
from typing import Union

import click
import numpy as np
import pandas as pd
import torch
from plugen4faces.const import global_config
from plugen4faces.utils import attrs2detector, spearman_rankorder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@click.group()
def main():
    pass


def merge_styleflow(
    styleflow_path: str,
    out_name: str,
    other: str = "detector_threshold_new_plugen_combined.pkl",
):
    """Takes a styleflow run and runs from different models and merges them together.
    Difference to merge_runs is that it merges dicts on model level, not single run.
    Honestly, no idea if this trash code generalizes to anything useful.
    """
    timestamp = dt.now().strftime("%Y-%m-%d_%H_%M_%S")
    with open(other, "rb") as f:
        combined2 = pickle.load(f)

    with open(styleflow_path, "rb") as f:
        styleflow = pickle.load(f)

    for edit in styleflow["dists"].keys():
        if edit == "glasses_2_0.7":
            for k in ["det_out", "dists", "flow_values"]:
                combined2[k]["glasses_2_0.6"]["styleflow"] = styleflow[k][edit][
                    "styleflow1"
                ]
        elif edit == "glasses_-2_0.2":
            # here we're changing thresholds, because styleflow has problems with that extreme threshold
            for k in ["det_out", "dists", "flow_values"]:
                combined2[k][edit]["new"] = combined2[k]["glasses_-2_0.09"]["new"]
                combined2[k][edit]["plugen"] = combined2[k]["glasses_-2_0.09"]["plugen"]
                combined2[k][edit]["styleflow"] = styleflow[k][edit]["styleflow1"]
                del combined2[k]["glasses_-2_0.09"]
        elif edit == "smile_-2_0.3":
            for k in ["det_out", "dists", "flow_values"]:
                combined2[k][edit]["new"] = combined2[k]["smile_-2_0.2"]["new"]
                combined2[k][edit]["plugen"] = combined2[k]["smile_-2_0.2"]["plugen"]
                combined2[k][edit]["styleflow"] = styleflow[k][edit]["styleflow1"]
                del combined2[k]["smile_-2_0.2"]
        elif edit == "beard_2_0.3":
            for k in ["det_out", "dists", "flow_values"]:
                combined2[k][edit]["new"] = combined2[k]["beard_1_0.3"]["new"]
                combined2[k][edit]["plugen"] = combined2[k]["beard_1_0.3"]["plugen"]
                combined2[k][edit]["styleflow"] = styleflow[k][edit]["styleflow1"]
                del combined2[k]["beard_1_0.3"]

        x = float(edit.split("_")[-1])
        if x == int(x):
            # TODO: TEST THIS
            for k in ["det_out", "dists", "flow_values"]:
                combined2[k][edit[: edit.rfind("_") + 1] + str(int(x))][
                    "styleflow"
                ] = styleflow[k][edit]["styleflow1"]
        else:
            for k in ["det_out", "dists", "flow_values"]:
                combined2[k][edit]["styleflow"] = styleflow[k][edit]["styleflow1"]

    with open(f"{out_name}_{timestamp}.pkl", "wb") as f:
        pickle.dump(combined2, f)


def overwrite(main_file: str, new_file: str, out_name: str) -> None:
    timestamp = dt.now().strftime("%Y-%m-%d_%H_%M_%S")

    with open(main_file, "rb") as f:
        main_data = pickle.load(f)
    with open(new_file, "rb") as f:
        new_data = pickle.load(f)
    main_runs = main_data["flow_values"].keys()
    new_runs = new_data["flow_values"].keys()

    for run in main_runs:
        if run in new_runs:
            for model in new_data["flow_values"][run].keys():
                name = model.split("_")[0]
                main_data["flow_values"][run][name] = new_data["flow_values"][run][
                    model
                ]
                main_data["det_out"][run][name] = new_data["det_out"][run][model]
                main_data["dists"][run][name] = new_data["dists"][run][model]
                if "space_values" in new_data.keys():
                    if "space_values" not in main_data.keys():
                        main_data["space_values"] = {}
                    if run not in main_data["space_values"]:
                        main_data["space_values"][run] = {}
                    main_data["space_values"][run][name] = new_data["space_values"][
                        run
                    ][model]

    with open(f"{out_name}_{timestamp}.pkl", "wb") as f:
        pickle.dump(main_data, f)


@main.command()
@click.option(
    "--main_file",
    type=str,
    help="Base file. Will not be modified, a copy is made instead.",
)
@click.option("--new_file", type=str, help="File to merge")
@click.option(
    "--output_name",
    type=str,
    required=True,
    help="Final name is {output_name}_{timestamp}.pkl",
)
def add_run(main_file: str, new_file: str, output_name: str) -> None:
    overwrite(main_file, new_file, output_name)


def merge_runs(files: list, out_name: str) -> None:
    """Takes list of files produced by evaluate.py (used with detector_threshold)
    and combines them to a single file.
    TODO: handle duplicated keys
    """
    timestamp = dt.now().strftime("%Y-%m-%d_%H_%M_%S")

    with open(files[0], "rb") as f:
        combined = pickle.load(f)

    for file in files[1:]:
        with open(file, "rb") as f:
            data = pickle.load(f)
        print(file, data["det_out"].keys())
        for run in data["det_out"]:
            for model in data["det_out"][run]:
                print(model)
                name = model.split("_")[0]
                combined["det_out"][run][name] = data["det_out"][run][model]
                combined["dists"][run][name] = data["dists"][run][model]
                combined["flow_values"][run][name] = data["flow_values"][run][model]
                if "space_values" in data:
                    combined["space_values"][run][name] = data["space_values"][run][
                        model
                    ]
                    combined["values_det_outs"][run][name] = data["values_det_outs"][
                        run
                    ][model]
        # combined['det_out'].update(data['det_out'])
        # combined['dists'].update(data['dists'])
        # combined['flow_values'].update(data['flow_values'])
        # if 'space_values' in data:
        #     combined['space_values'].update(data['space_values'])
        #     combined['values_det_outs'].update(data['values_det_outs'])

    with open(f"{out_name}_{timestamp}.pkl", "wb") as f:
        pickle.dump(combined, f)


def get_metrics(
    fname: str,
    det_out_cache_path: Union[Path, str] = global_config.det_out_cache_path,
    labels_path: Union[Path, str] = global_config.labels_path,
    epsilon: float = 5e-3,
    thresholds: dict = {},
):
    """Generates the results in the spreadsheet.
    All three models are expected to be in a single file.
    Thresholds overwrites edit_name threshold
    """
    with open(det_out_cache_path, "rb") as f:
        det_out_orig = pickle.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        all_attrs = json.load(f)

    attr_idx = {
        a: idx
        for a, idx in zip(
            [
                "gender",
                "glasses",
                "bald",
                "beard",
                "smile",
                "age",
                "pitch",
                "roll",
                "yaw",
            ],
            range(9),
        )
    }
    with open(fname, "rb") as f:
        result = pickle.load(f)

    # was needed for some version of result files, leaving to not mess anything up
    if isinstance(result["paths"], list):
        # paths = [x[1] for x in result["paths"]]
        paths = [x for x in result["paths"]]
    else:
        paths = result["paths"].values()

    attrs = torch.stack([attrs2detector(all_attrs[p.stem], 9) for p in paths], dim=0)
    orig = torch.stack([det_out_orig[p.name] for p in paths], dim=0)

    acc = []
    roc = []
    f1 = []
    for i in range(5):
        y_true = attrs[:, i] > 0.5
        y_pred = orig[:, i] > 0.5
        acc.append(accuracy_score(y_true, y_pred).round(4))
        f1.append(f1_score(y_true, y_pred).round(4))
        roc.append(roc_auc_score(y_true, y_pred).round(4))
    arr = []
    spearman = [round(x.statistic, 4) for x in spearman_rankorder(orig, orig, 9)]

    row = [None, None, None]
    arr.append(
        row
        + [
            "original",
            orig.shape[0],
            None,
            *([None] * 6),
            *spearman,
            *([None] * 9),
            *acc,
        ]
    )
    for edit_name in result["dists"].keys():
        a, c, t = edit_name.split("_")
        c, t = float(c), float(t)
        t = thresholds.get((a, c), t)
        print("==", a, c, t, "==")

        row = [a, c, t]
        succ = {}
        succ_intersection = None
        b = orig[:, attr_idx[a]]
        print(
            "successes from labels",
            (b >= t - epsilon if c > 0 else b <= t + epsilon).sum(),
        )
        for m in result["dists"][edit_name].keys():
            dists, det_out = (
                result["dists"][edit_name][m],
                result["det_out"][edit_name][m],
            )
            nzero = (dists[..., :3] == 0).sum(axis=1) != 3
            success = (
                det_out[:, attr_idx[a]] >= t - epsilon
                if c > 0
                else det_out[:, attr_idx[a]] <= t + epsilon
            )
            succ[m] = success & nzero
            succ_intersection = (
                succ[m] if succ_intersection is None else (succ_intersection & succ[m])
            )

        for m in result["dists"][edit_name].keys():
            edited = torch.from_numpy(result["det_out"][edit_name][m])
            assert orig.shape == edited.shape, (orig.shape, edited.shape)

            model = m.split("_")[0].rstrip("12")
            successes = succ[m].sum()
            dist = (
                result["dists"][edit_name][m][succ_intersection].mean(axis=0).round(4)
            )
            assert list(dist.shape) == [6]

            det_out_diff = [
                round(x.item(), 4) for x in ((edited - orig) ** 2).mean(dim=0)
            ]

            spearman = [
                round(x.statistic, 4) for x in spearman_rankorder(orig, edited, 9)
            ]
            acc = []
            roc = [None] * 5
            f1 = []
            det_out = result["det_out"][edit_name][m]
            for i in range(5):
                if i == attr_idx[a]:
                    y_true = (
                        np.zeros(det_out.shape[0])
                        if c < 0
                        else np.ones(det_out.shape[0])
                    )
                else:
                    y_true = attrs[:, i] > 0.5
                y_pred = det_out[:, i] > 0.5
                acc.append(accuracy_score(y_true, y_pred).round(4))
                f1.append(f1_score(y_true, y_pred).round(4))
            arr.append(
                row
                + [
                    model,
                    successes,
                    succ_intersection.sum(),
                    *dist,
                    *spearman,
                    *det_out_diff,
                    *acc,
                ]
            )

    cc = [
        "attr",
        "change",
        "threshold",
        "model",
        "successes",
        "success_intersection",
        "intersection L2 face_recognition",
        "intersection cossim face_recognition",
        "intersection L2 ArcFace",
        "intersection MSE",
        "intersection PSNR",
        "intersection SSIM",
    ]
    for x in ["spearman_rankorder", "det_out_diff"]:
        cc += [
            f"{x}/{y}"
            for y in [
                "gender",
                "glasses",
                "bald",
                "beard",
                "smile",
                "age",
                "pitch",
                "roll",
                "yaw",
            ]
        ]
    for x in ["acc"]:
        cc += [f"{x}/{y}" for y in ["gender", "glasses", "bald", "beard", "smile"]]
    assert all([len(x) == len(cc) for x in arr]), ([len(x) for x in arr], len(cc))
    df = pd.DataFrame(arr, columns=cc)
    for x in df.to_csv(index=False).split("\n"):
        print(x)
    return df


@main.command()
@click.option("--fname", required=True, type=str, help="file to get metrics from")
@click.option(
    "--epsilon",
    default=5e-3,
    type=float,
    help="Epsilon for detector_threshold, determines number of successes",
)
@click.option(
    "--det_out_cache_path",
    type=str,
    default=global_config.det_out_cache_path,
    help="Path to the detector output cache for original evaluation images",
)
@click.option(
    "--labels_path",
    type=str,
    default=global_config.labels_path,
    help="Path to the labels for original evaluation images",
)
def metrics(
    fname: str, det_out_cache_path: str, labels_path: str, epsilon: float
) -> None:
    get_metrics(
        fname,
        det_out_cache_path=det_out_cache_path,
        labels_path=labels_path,
        epsilon=epsilon,
        thresholds={},
    )


if __name__ == "__main__":
    main()
