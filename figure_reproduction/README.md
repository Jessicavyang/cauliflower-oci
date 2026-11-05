# Figure source code

The directory geometry matches the final Times New Roman figure package. Install the optional figure requirements and a licensed local Times New Roman font. From this directory:

```bash
python code/render_figures.py Fig2 Fig3 Fig4 Fig5 Fig6 Fig7 SupplementFig2 SupplementFig3 SupplementFig4 SupplementFig5
```

These plots use frozen source data, not new modelling. Current Fig.7 is the 222-record application; Supplementary Fig.4 has three cases. No current Supplementary Fig.6 is released. The three matched case JPEGs are present without editing; the full originals are also in the F2 image archive.

Fig.1 and Supplementary Fig.1 require `originals/Fig1_V2.pdf` and `originals/Supplement_Figure_Summary.pdf`. Those historical layout/photo containers are held outside the upload candidates pending confirmation of their image/equipment-illustration redistribution scope. Do not run the script with no figure arguments while these two assets are absent, because its historical default attempts all figures. No replacement or simulated photographic asset is supplied.

The release was not redrawn in this packaging run; data and source code preserve the already reviewed design. These optional scripts need installed scientific/plotting dependencies and may create large high-resolution files under figures, supplementary and qa.
