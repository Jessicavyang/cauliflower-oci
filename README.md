# OCI cauliflower data and code

Supporting data and code for **A Digital Phenotype for Cauliflower Curd Compactness Integrating Sectional Gap and Color Signals**, prepared for Smart Agricultural Technology.

This collection contains the frozen image-feature matrices, mechanical exports and linkage, model parameters, saved predictions, figure source data, original analysis scripts, and available-image manifests. It supports inspection and reproducible application of an explicit sectional image phenotype. This is a local deposit candidate; no repository record, DOI or open licence has been assigned.

## Data collections

|Collection|Records/files|Use|
|---|---:|---|
|F2 features and frozen scores|275|220 training records and 55 historical test records; historical test data had informed prior exploration.|
|F2 original photographs|275 JPEGs|Exact frozen filename matches, supplied in the separate F2 image archive.|
|Advanced-inbred archive|230|Complete immutable record-level output retained for provenance.|
|Current advanced-inbred application|222|Excludes eight records with unresolved duplicated source-image identities; current Fig.7 uses this subset.|
|Local inbred photographs, not deposited|198 JPEGs|Excluded from public upload at the author’s instruction. Metadata retain 197 exact filename links and one unlinked image; the 222-record table has 25 records without a matching image in the local collection.|
|Same-image computational QA|30|Saved replay and threshold-perturbation diagnostics, with byte-level image links.|
|Available manual outer contours|2|Existing 23AU_004/23AU_013 masks and saved overlap/boundary metrics.|
|Independent internal-gap evaluation|45 ROIs/15 parent images planned|Not a completed independent-gap validation dataset.|

The current 222-record counts are **102 Compact, 92 Medium, 28 Loose**, with **56** records outside at least one training input range. The 230-record archive has **107/94/29** and **58** out-of-range records. These are record counts, not confirmation of 222 or 230 independent genotypes.

## Get started

Extract `OCI_Data_Code_v1.zip` and `OCI_F2_Images_275_v1.zip` into the **same parent directory**. These two archives share the top folder `oci_data_code_v1`. The 198 original inbred photographs are excluded from this deposit; their record-level data and linkage metadata remain included.

The default reproduction uses only the Python standard library (tested with Python 3.12.14):

```bash
cd oci_data_code_v1
python code/reproduce_saved_results.py
```

To additionally require and verify the 275 shared F2 original JPEGs:

```bash
python code/reproduce_saved_results.py --verify-images
```

Results are written to `outputs/reproduction/reproduction_result.json`. No feature selection, model fitting, resampling, threshold optimization or figure rendering is invoked. Scores use the full frozen parameters, not rounded manuscript displays. The comparison allows only the declared 12-significant-digit serialization error for Stage4AB outputs.

## Contents and authoritative fields

- `data/raw/mechanical/`: original XLS/XLSX and open CSV mirrors. These are saved result exports, not original force–time curves.
- `data/processed/`: full F2 features, historical partition input, current frozen scores, 230 archive, 222 main application and eight exclusions.
- `data/linkage/` and `data/image_manifests/`: mechanical reading traces, exact filename links, missing/unmatched records and image SHA-256 values.
- `config/OCI_frozen_scaling_and_grades.json`: final model feature order, coefficients, sample-SD scaling, score anchors and exact category boundaries.
- `data/predictions/`: saved Stage3B and retrospective Stage3C predictions, separated by model, repeat and fold.
- `data/statistics/`: saved summaries and conditional interval supplements. First-repeat, repeated-CV, historical-test and policy-sensitivity estimates are distinct.
- `source_data/derived/`: current figure and supplementary-table source CSVs. File/column definitions are in `data_dictionary.csv`.
- `code/analysis/`: byte-identical original method scripts. See implementation notes before interpreting historical headers or rerunning them.
- `figure_reproduction/`: the final figure-rendering code with its expected directory geometry and frozen source data; see the figure README for photographic-asset and font requirements.
- `manifests/`: file checksums and package provenance. Repository metadata under `metadata/` is a draft for author confirmation.

`OCI_Raw_recomputed` / `OCI_Score_recomputed` / `OCI_Grade_recomputed` are authoritative for F2. `OCI_Raw_frozen` / `OCI_Score_frozen` / `OCI_Grade_frozen` are authoritative for inbred records. Older `OCI`, `Grade`, `Mech_Composite` and cluster fields in partition or original-supplement inputs remain historical provenance only. Use `Set`/`image_id` from the historical partition file, not its old target or score fields, to reconstruct the current split.

## Measurement and interpretation

GR is force–time gradient, exported as `Gradient F-T 1:2` in `g/sec` and interpreted as `gf s^-1`. DD is displacement in mm from the author-confirmed 5 gf trigger reference to the 30,000 gf target load. The mechanical reference is `(-Ztrain(GR) + Ztrain(DD))/2`. The precise GR marker interval and DD software zero/readout rule remain unresolved. The newer measurement sidecar documents the evidence update without changing the original computational freeze.

OCI is an operational phenotype derived from one two-dimensional longitudinal section and calibrated to mechanical responses. It is not independently established anatomical compactness, three-dimensional porosity, an official DUS grade, external prediction accuracy or a non-destructive breeding method. Computational replay and two outer-contour masks do not complete independent internal-gap or repeated-acquisition validation.

## Rights and citation

Publication permissions, data/image and code licences, dataset creators and the eventual publication date require author confirmation. `RIGHTS_AND_LICENSING.md` describes the pending decisions. No MIT, CC-BY or CC0 grant is silently applied. Cite the actual record and its version-specific DOI only after they exist. The proposed creator order follows the supplied manuscript and is explicitly marked as a dataset-metadata draft.

## Author-confirmed image-sharing scope

Only the 275 F2 original images are designated for this deposit. The 198 local advanced-inbred originals are not uploaded. Their feature/score/category and linkage tables remain available, but image-based re-extraction for those originals cannot be performed from this deposit alone. No access-on-request promise or restriction justification has been supplied.
