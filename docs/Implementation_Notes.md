# Implementation and provenance notes

## Frozen analysis

The primary model is `ENet_one_se_thr70`; Stage3C is a separate retrospective policy-choice sensitivity, not a replacement coefficient set. The 70% policy was chosen during historical exploration. Final selection used 1,000 stratified 80% subsamples without replacement; repeated outer evaluation used 100 selection subsamples per outer-training fold. The final checkpoint, configurations, models and all saved prediction pairs are included so these runs need not be repeated to inspect or apply the results.

Original method scripts and original configs are kept byte-identical for source provenance. Some historical header/config statements describe inner-fold screening more broadly than the actual implementation. **Inner tuning inherits the parent training-stage candidate pool; selection does not restart from all 20 variables in every inner fold.** No pairwise |r| > 0.95 pruning was performed by the final code. Both Text_Entropy and Text_ASM remain in the eight-variable training candidate pool. Do not treat an archived `inner_cv_fold_training_only_screen: true` flag or a “leakage-safe” header as a stronger claim than the implementation supports.

The archived freeze registry predates the mechanical-file audit and still has older unresolved export-field metadata. The numerical freeze remains unchanged. Read `config/measurement_semantics_latest.json` for the newer GR field/unit evidence and the remaining interval/zero/readout gaps. An update of interpretation metadata does not alter a frozen measurement or model input.

Stage4CD archive outputs can retain historical FH/GC-style labels. Those are provenance labels, not current biological classifications. The current `source_data/derived/Fig6_*neutral_labels.csv` tables and current figure legends use neutral mechanical/OCI disagreement directions. Non-significant differences are not statistical equivalence.

## Image extraction

The v8 routine uses grayscale, a 5x5 Gaussian filter, per-image global Otsu thresholding and a filled largest external contour. Gap counts include 8-connected dark components with area strictly greater than five pixels. Gap ratio includes **all** dark pixels inside the mask, including smaller components. Color_S is mean OpenCV HSV saturation over that mask. No verified physical-scale or colour correction is added.

The original function named `get_glcm_features` does not construct a GLCM: it calculates a histogram entropy, histogram squared-probability sum and Laplacian response SD. Its historical name/comments are retained as source code; they are not the current method definition. The signed box-counting output is not a validated anatomical fractal dimension. The unused legacy `Size_Diameter` source field is not needed for OCI and lacks an independently established extraction/scale definition.

`code/replay_core_image_features.py` is a portable entry point for the three core inputs. It calls the original v8 gap function and duplicates only the original image-loading/threshold/mask/HSV steps. It defaults to three saved QA images. The archived 30-image replay and threshold-perturbation tables are supplied; this packaging run did not recreate segmentation ground truth or run new acquisition tests. The new wrapper was syntax-checked, but not executed because OpenCV was absent from the packaging runtime.

## Numerical precision and identity

Original numeric tables are copied byte-identically except specifically described metadata-only linkage transformations. No rounding, exclusions, model coefficient, cutoff or sample split was changed. The portable verifier tolerates the original Stage4AB %.12g serialization, not an arbitrary model discrepancy. Source spreadsheet cell values and extreme GR readings remain untouched.

198 available images are author-confirmed independent plants, with 197 exact filename links to retained records. One unmatched file is retained without an invented inbred ID. Twenty-five of the 222 retained rows lack supplied matching images. Eight provenance-ambiguous records stay in the 230 archive and the exclusion file. A filename match or equal image hash is not a new genotype confirmation.
