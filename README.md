# TFG---Reconstruction-of-wind-turbines


## Description

This project provides various functions to align an input image (`image`) to a template (`template`) using different methods: ORB, SIFT, SuperGlue/LightGlue, exhaustive searches, randomized optimization, differential evolution, affine transformations, and homographic transformations. The main script is `implementacio1.py`, which selects the method via a command-line argument and saves two output files: a stacked image (side by side) and an overlay image (50% transparency).

## Folder Structure


/Project/
├── alingn.py       # Alignment functions
├── implementation.py   # Main CLI script
├── superglue_wrapper.py
├── images/             # Example folder with images
│   ├── img1.jpg
│   └── tmp1.jpg
└── README.txt          # (This file)


## Dependencies

* Python ≥ 3.7
* numpy
* opencv-python
* torch (PyTorch)
* scipy
* imutils
* lightglue and superpoint (for “superglue” or “lightglue” methods)

## Functions in `align.py`

* **`align_images_fast(image, template, maxFeatures=500, keepPercent=0.2, debug=False)`**
  ORB + BFMatcher (Hamming) + RANSAC → homography
  Returns: `(aligned_image, aligned_template, H)`
  If `debug=True`, saves an inliers visualization (e.g., in `metode_sift_fast/matches_inliers_fast.png`).

* **`align_images_sift(image, template, maxFeatures=500, keepPercent=0.2, debug=False)`**
  SIFT + BFMatcher (L2 + crossCheck) + RANSAC → homography
  Returns: `(aligned_image, aligned_template, H)`
  If `debug=True`, saves an inliers visualization (e.g., in `metode_sift_fast/matches_inliers_fast.png`).

* **`align_images_superglue(image, template, prefix="", debug=True)`**
  SuperPoint + SuperGlue → homography
  Returns: `(aligned_image, aligned_template, H)`
  If `debug=True`, saves an inliers visualization in the `vis_superglue/` folder.

* **`align_images_lightglue(image, template, prefix="", debug=True)`**
  SuperPoint + LightGlue → homography
  Returns: `(aligned_image, aligned_template, H)`
  If `debug=True`, saves an inliers visualization in the `vis_lightblue/` folder.

* **`exhaustive_search(image, template)`**
  Exhaustive search on CPU (fixed X, Y range) computing MSE
  Returns: `(aligned_image, aligned_template)`.

* **`exhaustive_search_torch(image, template)`**
  Exhaustive search on GPU (or MPS) with PyTorch
  Returns: `(aligned_image, aligned_template)`.

* **`randomized_search(image, template)`**
  Iterative stochastic search (alternating variations on Y and X)
  Returns: `(aligned_image, aligned_template, shiftx, shifty)`.

* **`differential_ev(image, template)`**
  Global optimization with Differential Evolution (SciPy), TX, TY
  Returns: `(aligned_image, aligned_template, x_min, y_min)`.

* **`align_affine(image, template)`**
  Full 6-DOF affine optimization via Powell
  Returns: `(aligned_image, aligned_template, M)` where `M` is a 2×3 matrix.

* **`align_homography(image, template)`**
  8-parameter homographic optimization via Powell
  Returns: `(aligned_image, aligned_template, H)` where `H` is a 3×3 matrix.


## Main Script (`implementation.py`)

* Loads the input image and template.
* Resizes both to half their original size.
* Parses `-m` (method) and `-n` (name) to decide which function from `alineacio1.py` to call.
* After alignment, resizes the results (width = 500) and saves two images:

  * `a<name>_alineacion_s_<method>.png` → stacked image (side by side).
  * `a<name>_alineacion_o_<method>.png` → overlay image (50% transparency).

## Usage

From the project folder, in a terminal:


python implementacio1.py \
    -i path to image \
    -t path to template \
    -m <method> \
    -n <name>


* `-i`: path to image .
* `-t`: path to template.
* `-m`: one of these methods:

  * `fast`
  * `sift`
  * `superglue`
  * `lightglue`
  * `exhaustive`
  * `exhaustive_torch`
  * `randomized`
  * `diff_ev`
  * `affine`
  * `homografia`
* `-n`: suffix for the output filenames (e.g., `test`).

Example:


python implementation.py \
    -i images/img1.jpg \
    -t images/tmp1.jpg \
    -m fast \
    -n test


Expected output:


atest_alineacion_s_fast.png
aptest_alineacion_o_fast.png
