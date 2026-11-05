# Reproduction levels

## 1. Fixed scores, categories and saved-prediction metrics

`python code/reproduce_saved_results.py --verify-images`

This standard-library check recomputes frozen F2/inbred scores and categories, verifies the 230-to-222 inclusion rule, summaries of the saved OOF/historical predictions and the common 275-record contingency table, and checks the 275 shared F2 image checksums when the F2 archive is extracted. This was actually tested on the assembled package. It does not refit models, rerun selection or recompute bootstrap intervals.

## 2. Same-image feature replay

Install the optional image dependencies in a separate environment, then run:

```bash
python code/replay_core_image_features.py --limit 3
```

The supplied wrapper can replay up to the 30 archived QA images. It uses the exact byte-linked image filenames and original v8 code. Its scope is computational reproducibility of gap count, gap ratio and Color_S, not anatomical gap accuracy or repeat photography. The new wrapper was syntax-checked only in this packaging environment, where OpenCV was unavailable. Archived prior 30-image computational results remain in `data/validation/`.

## 3. Historical full model/selection workflows

Original Stage2B, Stage3B, Stage3C, Stage4AB, Stage4CD and Stage6 scripts and their saved input/configuration/checkpoint files are supplied for inspection. They have not been rerun during packaging. They are not called by the default reproduction. Some historical CLI defaults, labels and header statements require the implementation notes; a full re-estimation also requires rebuilding the historical working directory and archived environment. Do not run their unqualified defaults expecting a new release to match the frozen result or to remove the historical test-use limitation.

For example, the original Stage2B script defaults to `min_mse`; the frozen primary workflow uses `one_se`. The final 1,000-run checkpoint is already included. Avoid rerunning it just to regenerate outputs already saved. Any deliberate reanalysis belongs to a new output directory and version, never overwriting the frozen package.

## 4. Figures

The figure source data and original rendering code are provided in `figure_reproduction/`. Numeric plots use frozen tables and do not retrain the OCI. Exact typeface reproduction requires an independently installed licensed Times New Roman. Fig.1 and Supplementary Fig.1 additionally depend on two historical layout/photo PDF assets outside this candidate upload; see the figure README. No figure was redrawn during data packaging.

### Image-sharing update

The 198 original advanced-inbred photographs are not included in this deposit at the author’s instruction. Retained numerical and image-linkage metadata do not imply access to those originals. `--verify-images` requires only the 275 shared F2 images. No alternative access route for the withheld originals has been specified.
