# Object Counting with DISCount

Estimating how many objects (people, birds, sheep, balloons) are in a large image by hand-counting only a few tiles, using **detector-based importance sampling for imperfect detectors.**

Counting every object in a high-resolution image takes a long time. Instead, the image is split into a grid of tiles, a few tiles are sampled and counted, and the total is estimated from those samples. DISCount picks tiles using a counting model's predicted density map, so tiles likely to hold more objects are sampled more often. Each sample is then reweighted so the estimate stays unbiased. This repository compares that proposal against uniform and triangle sampling, using hand-annotated ground truth.

## About this project

This project was done through the **Early Research Scholars Program (ERSP)** at the [UMass Amherst Manning College of Information and Computer Sciences](https://www.cics.umass.edu/). ERSP is a year-long, team-based research program for early undergraduates. Each team has two mentors: one from the program and one from a research lab.

- **ERSP mentor:** Prof. Neena Thota
- **Research mentor:** [Prof. Subhransu Maji](https://people.cs.umass.edu/~smaji/), Computer Vision Lab, UMass Amherst
- **PhD student mentors:** Max Hamilton and Rangel Daroya

The work was carried out in the UMass Amherst Computer Vision Lab and builds on DISCount, which came out of Prof. Maji's group.

## Repository layout

```
.
├── [ERSP]_Processing_Images.ipynb   # Main analysis notebook
├── Dataset/                         # Images + VIA point/box annotations
│   ├── balloons/
│   ├── birds_1/  birds_2/  birds_3/
│   ├── crowd_1/  crowd_2/  crowd_3/  crowd_4/
│   ├── crowd_use/                   # Image the notebook is configured for
│   └── sheep_1/
└── Object Counted by DISCount/      # Precomputed density maps (.npy)
    ├── AdobeStock_469974989.npy     # → Dataset/crowd_use
    ├── balloon.npy                  # → Dataset/balloons
    └── intermediate_birds.npy       # → Dataset/birds_1
```

### Datasets

Annotations were made with the [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) and exported as VIA JSON. Each object is either a **point** (`cx`, `cy`) or a **bounding box** (`x`, `y`, `width`, `height`).

| Folder | Image | Size (px) | Annotation | Objects |
|---|---|---|---|---|
| `crowd_use` | `AdobeStock_469974989.jpeg` | 5417 × 3611 | boxes, whole image | 429 |
| `crowd_1` | `basic_crowd.jpg` | 6016 × 4016 | points, whole image | 42 |
| `crowd_2` | `basic_crowd2.jpg` | 5467 × 3075 | points, whole image | 91 |
| `crowd_3` | `intermediate_crowd.jpg` | 4859 × 3218 | points, whole image | 488 |
| `crowd_4` | `complex_crowd.jpg` | 7952 × 5304 | points, per 600 px tile (126 tiles) | 3346 |
| `birds_1` | `intermediate_birds.jpg` | 5932 × 3956 | points, per 800 px tile (40 tiles) | 152 |
| `birds_2` | `complex_birds.jpg` | 5184 × 3888 | points, per 600 px tile (63 tiles) | 307 |
| `birds_3` | `basic_birds.jpg` | 6000 × 4000 | points, whole image | 111 |
| `sheep_1` | `basic_sheep.jpg` | 4896 × 3672 | points, whole image | 59 |
| `balloons` | `balloons.jpg` | 4896 × 3264 | points, whole image | 269 |

The `*_annotated.jpg` files are downscaled previews with the annotations drawn on. In the per-tile datasets, each `tile_<i>.jpg` is a crop of the full image and is annotated separately in the same JSON file. The tiles are numbered row by row from the top-left.

### Density maps

Each file in `Object Counted by DISCount/` is a `float32` array of shape `(1, 1, H, W)` (about 1050 × 1584). It holds a counting model's predicted density for one image, and its sum is the model's estimated total (for example, about 443 for the crowd image, whose ground truth is 429). The notebook sums the map over each tile and normalizes the result to get the DISCount proposal distribution.

## How the notebook works

`[ERSP]_Processing_Images.ipynb` runs these steps for a single image:

1. **Load** the image, its VIA JSON and its density map. Paths and the object name are set in the first code cells.
2. **Tile** the image into a `tile_size` × `tile_size` grid (300 px by default, which gives 19 × 13 = 247 tiles for the crowd image).
3. **Ground truth.** Each annotation's center is assigned to a tile, giving a per-tile count array and the true total.
4. **Build proposal distributions** over tiles:
   - **Uniform**: every tile is equally likely.
   - **Triangle**: the probability rises linearly to the middle tile index, then falls.
   - **DISCount**: the density map summed per tile, then normalized.
5. **Importance sampling.** `run_importance_sampling(data, q_probs, n)` draws `n` tiles with replacement from `q` and weights each count by `1 / (N · q_i)`. It returns the running estimate of the total count, and its variance, after 1…n samples.
6. **Compare.** The run is repeated many times (100 for the crowd image) and averaged. The plot shows each method's estimate against the number of samples, with a 90% band (±1.645 σ) and the ground truth as a dashed line.

The estimator for N tiles and sampled tiles i₁…iₙ is:

$$\hat{C} = \frac{1}{n}\sum_{k=1}^{n} \frac{c_{i_k}}{q_{i_k}}$$

This estimate is unbiased for any proposal that is nonzero wherever the count is nonzero. Its variance shrinks as `q` gets closer to the true count distribution, so a good density map should make the DISCount band narrower than the uniform one.

## Getting started

Requirements: Python 3.10+ (the notebook was last run on 3.12) and Jupyter.

```bash
pip install numpy pandas matplotlib scipy jupyter
jupyter notebook "[ERSP]_Processing_Images.ipynb"
```

Start Jupyter from the repository root. The notebook builds its paths from `Path.cwd()`.

### Running on a different image

Change the paths in the configuration cell:

```python
img_path    = str(repo_root / "Dataset" / "balloons" / "balloons.jpg")
json_path   = str(repo_root / "Dataset" / "balloons" / "balloons_annotated.json")
q_probs_path = str(repo_root / "Object Counted by DISCount" / "balloon.npy")
obj_name    = "balloons"
```

You will also need to adjust a few values that are hard-coded for the crowd image:

- **Annotation shape.** The coordinate-extraction cell reads `x`, `y`, `width` and `height`, which only exist for box annotations. Every dataset except `crowd_use` uses points, so read `shape_attributes['cx']` and `['cy']` for those.
- **Density-map block size.** The cell that sums the density map uses blocks of 87 px, chosen so that its grid lines up with the 19 × 13 image grid. Set it to about `density_map_width / tiles_per_row` for other images or tile sizes.
- **`NUM_COUNT = 247`** is the tile count for the crowd image. Set it to `tiles_count`.
- **Per-tile datasets** (`birds_1`, `birds_2`, `crowd_4`) store coordinates relative to each tile. Sum the regions per JSON entry to get the tile counts directly instead of re-tiling the full image.
