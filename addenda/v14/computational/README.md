# v14 computational addendum

Research article under preparation: **Action sufficiency and update risk in rolling storage optimization**. Sole author: **Junjie Zhang**. Original software and documentation retain the repository's MIT licence; third-party market data retain their own source terms. See LICENSE and DATA_TERMS.md.

The recorded exploratory decision is **STOP_NEW_COMPRESSION_ALGORITHM**. The retention gate keeps the complete inverse-lead streaming moments and adds optimization calls, so this pilot does not establish an online-storage advantage over exact streaming IL. The useful result is a tested boundary for rolling action sufficiency and update risk.

## Supplied evidence

- Six research/controller modules and a reproduction wrapper, with only two recorded ROOT-path substitutions in scientific code.
- Two processed-input parquet subsets at the original relative paths. Evaluation is NSW1 January 2024, cutoff 60: 1,488 half-hour origins for each of raw, current_smooth_a050, sparse_equal, full_equal, and inverse_lead. HISTORY retains the complete 2023 validation rows for raw, the four current processors, and inverse_lead, so the 24 training indices and targets are unchanged.
- Eighteen saved closed-loop trajectories and reports: nine policies or ablations at initial inventories 0.2/1.8 MWh. There are also a locked design, training masks, benchmark, shared-state diagnostics, and an executed synthetic-control receipt.
- A previously executed fresh check ran IL and static9 for all 1,488 January origins at initial 0.2 MWh. Seven trajectory fields and marked values were bitwise/exactly equal to the frozen references; static9 was exact to IL; all 18 synthetic checks passed. The compact package was verified before this public copy. Public preparation verifies copied bytes; it does not claim a new public checkout replay.

The static9 control is an exact structural nine-coefficient map for this implemented controller, not a reproduction of a learned static-compression algorithm. Masks were developed on 24 chronological 2023 origins; they are preserved rather than refitted by the minimum reproduction.

## Verify and reproduce

Use Python 3.12 and the versions in ENVIRONMENT.json/requirements.txt. HiGHS is accessed through SciPy's private interface; the recorded execution used Windows with these versions, and portability to another environment is not established.

From this directory:

```text
python -B scripts/verify_manifest.py .
python -B scripts/reproduce_v14_minimum.py --output D:/MLWork/rdia-v14-fresh
```

The output path must be new and outside the supplied code/data directory. Use an explicit fresh path suitable for the host; numerical work on Windows follows the author's D-drive convention. The wrapper runs two complete 1,488-origin replays and the synthetic controls, and records a fresh receipt. It verifies 24 training indices/targets, seven trajectory fields, marked values, and synthetic scientific outputs. It leaves the supplied evidence unchanged. Python paths resolve from the scripts, so execution also works when invoked from the enclosing repository.

## Scope of the gate and provenance

The gate uses floating-point feasible incumbents and reported solver objective bounds plus a numerical cushion. It checks same-state, one-window first-action objective regret. It is not interval/rational arithmetic or a rigorous floating-point certificate, and does not certify cumulative realized profits, noninferiority, or safety under future forecast updates. January 2024 and the older data are exploratory/reanalysed; no independent holdout or universal profitability is claimed.

Original raw archives, unsubmitted manuscripts, and internal assessments are excluded. MANIFEST.json checks this standalone addendum; it is separate from the root v1.0 manifests. Historical source locations are normalized to relative names in public JSON metadata. The unified diff retains a former machine ROOT solely as an already-applied transformation record. Source checksum metadata refer to the original full processed files, while this package supplies the losslessly reduced subsets described in SOURCE_TRANSFORMATIONS.json.


## Optional Figure 7 rendering

Figure rendering is separate from the minimum replay and is not run by default. The portable entry preserves the original numeric, plotting, font, canvas, 1200 dpi raster, vector-export and collision-check parameters. Its four input files are included byte for byte. Install the optional recorded dependencies and use a new output directory outside this package:

```text
python -m pip install -r requirements-figures.txt
python -B scripts/portable_figure7.py --output D:/MLWork/rdia-v14-figure7-fresh
```

From the repository root, invoke `addenda/v14/computational/scripts/portable_figure7.py` with the same explicit `--output`. Existing or overlapping directories are rejected. Matplotlib's configuration/cache files stay inside that fresh output. Arial is required to reproduce the recorded font metrics; other environments remain unverified and the retained layout checks must pass. Native PDF/SVG exports are vectors, while PNG/TIFF are freshly rendered at 1200 dpi. Redrawing uses the saved evidence and does not rerun optimization. See FIGURE_CODE_ADAPTATIONS.json, FIGURE_ENVIRONMENT.json and the unified diff for the original source SHA and the path-only transformation. No new 1200 dpi rendering is claimed during public packaging.
