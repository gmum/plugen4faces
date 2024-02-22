# Face Identity-Aware Disentanglement in StyleGAN

This repository contains the official code for PluGeN4Faces,
a method for explicitly disentangling attributes from person's identity.
This work was presented at WACV 2024.

[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Suwala_Face_Identity-Aware_Disentanglement_in_StyleGAN_WACV_2024_paper.pdf)

## Abstract 
Conditional GANs are frequently used for manipulating
the attributes of face images, such as expression, hairstyle,
pose, or age. Even though the state-of-the-art models successfully modify the requested attributes, they simultaneously modify other important characteristics of the image,
such as a person’s identity. In this paper, we focus on solving this problem by introducing PluGeN4Faces, a plugin to
StyleGAN, which explicitly disentangles face attributes from
a person’s identity. Our key idea is to perform training on
images retrieved from movie frames, where a given person
appears in various poses and with different attributes. By
applying a type of contrastive loss, we encourage the model
to group images of the same person in similar regions of
latent space. Our experiments demonstrate that the modifications of face attributes performed by PluGeN4Faces are
significantly less invasive on the remaining characteristics
of the image than in the existing state-of-the-art models.

## Citation

If you found this code useful, please cite:
```{bibtex}
@InProceedings{Suwala_2024_WACV,
    author    = {Suwa{\l}a, Adrian and W\'ojcik, Bartosz and Proszewska, Magdalena and Tabor, Jacek and Spurek, Przemys{\l}aw and \'Smieja, Marek},
    title     = {Face Identity-Aware Disentanglement in StyleGAN},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {5222-5231}
}
```


# Setup

## Creating the environment
```{bash}
conda env create -f environment.yml
```

## Training the model

All needed models are expected to be in `pretrained_models`.

To train the model you need to first encode the training data into latents, using e4e.
Once the checkpoint is saved, execute:
```{bash}
python scripts/encode.py --src DIR_WITH_IMAGES --target NEW_DIR_WITH_LATENTS
```

Training the model requires also a StyleGAN checkpoint compatible with [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).

```{bash}
chmod +x scripts/train_flow.sbatch
./scripts/train_flow.sbatch
```
The options are documented in `scripts/train.py` or `--help` on the called python script.

## Evaluation

For evaluation you'll also need ArcFace checkpoint and an attribute classifier checkpoint.

### Caching latents for original images

Before you run the evaluation itself you have to cache
`face_recognition`, `ArcFace` and StyleGAN latent space
on your evaluation data.

```{bash}
./scripts/cache_embs.sbatch
```

### Main evaluation

The main evaluation script searches for flow values to cross the given thresholds for edits in the given
range of values. It can use either binary search based on the outputs from the attribute classifier, or
a linear search, which uses equally spaced points across the given range. Linear search can be faster since it
knows all the points upfront, and uses vectorization to process several at the same time.

```{bash}
chmod +x scripts/detector_threshold.sbatch
./scripts/detector_threshold.sbatch
```

The meaning of the options is documented in the CLI code in `scripts/`, or just `--help` on the 
(underlying python script, not the bash script).


# Using the trained model

## Loading
```{python}
from plugen4faces.flow import load_plugen4faces

model = load_plugen4faces("model.pt")
```

## Converting to flow latent

```{python}
from plugen4faces.const import NUM_WS
onehots = (
    F.one_hot(torch.arange(NUM_WS), num_classes=NUM_WS)
    .float()
    #.repeat(B, 1, 1)
    .to(model.device)
)
# ws should have shape [B, NUM_WS, 512], and onehots [B, NUM_WS, NUM_WS]
z = model.transform_to_noise(ws, onehots)
```

## Converting to stylegan latent
```{python}
# z should have shape [B, NUM_WS, 512], and onehots [B, NUM_WS, NUM_WS]
ws = model.inverse(z, onehots)
```

## Editing attributes
```{python}
from plugen4faces.const import attr2idx

z[..., attridx["beard"]] = 2
z[..., attridx["gender"]] = -2
```
There is also `attr2detidx` that converts to indices of the attribute classifier.

## Loading other models
```{python}
from plugen4faces.utils import load_detector, load_stylegan, load_plugen, load_styleflow
```
All need a path, but have a default from `global_config` defined in `plugen4faces/const.py`

# Structure of the code

```
plugen4faces/
├── environment.yml
├── plugen4faces
│   ├── const.py         # globally used defaults, paths, values
│   ├── encoder4editing  # Image to StyleGAN latent inverter
│   ├── evaluation
│   │   ├── configs.py   # wrappers around evaluated models
│   │   ├── evaluate.py  # main evaluation code
│   │   ├── helpers.py
│   │   ├── __init__.py
│   │   └── search.py
│   ├── flow.py          # implements the method
│   ├── __init__.py
│   ├── resnets.py       # for attribute classifier
│   ├── StyleFlow        # implementation of PluGeN and StyleFlow
│   ├── stylegan2
│   └── utils.py
├── README.md
└── scripts
    ├── cache_embs.sbatch           # for caching orignal images, required to run evaluation
    ├── detector_threshold.sbatch   # main evaluation script, detector is the attribute classifier
    ├── encode.py                   # for encoding directories of images with e4e
    ├── evaluate.py                 # CLI interface for evaluation
    ├── postprocess.py              # for merging runs and calculating metrics from detector_threshold results
    ├── train_flow.sbatch           # training script
    └── train.py                    # CLI interface for training
```
