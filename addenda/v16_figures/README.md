# RDIA v16 frozen figure redrawing package

This small package redraws Figures 1–6 from frozen processed values. It performs no model training, optimization, statistical re-estimation, raw-data download or manuscript generation. Junjie Zhang is the sole study author. Repository: https://github.com/williamjay1/rdia-rolling-storage.

## Reproduce

Place the package in a writable research directory (the validated execution used D: on Windows). Install the dependencies in `requirements.txt` in your Python environment, then run:

```sh
python scripts/reproduce_v16_figures.py
```

All six figures are generated under `figures/revision_v16/` as native 174 mm PDF/SVG and original 1200 dpi PNG/LZW TIFF, with 300 dpi previews, source arrays and layout receipts. Matplotlib caches remain under the package's `cache/`. The typography requires an installed, properly licensed Arial font; no proprietary font file is redistributed. The fresh execution receipt records the tested host and package versions. PDF metadata timestamps may change without changing geometry or values.

The command verifies exact scientific source arrays against `reference/`, real PDF font size/width, raster metadata and final PDF text/graphic collision checks. Display-only `layout` and deterministic horizontal `display_x` records are excluded from the scientific-value comparison. Economic y values, all interval endpoints, forecast coefficients, branch cash paths, acceptance counts and stored timing observations are preserved. The recorded replay times are frozen observations, not new benchmarks of your host.

## Included material

Figure 1 is an information-clock schematic. Figures 2–4 use original study-generated aggregate or diagnostic outputs. Figure 5 includes only two selected processed Australian forecast rows and 96 selected processed trajectory rows. Figure 6 includes a synthetic control, all six 384-case numerical-gate groups and eight recorded exploratory replay summaries. The 24-hour adverse example and NSW January 2024 pilot are exploratory, not independent confirmation. The method-decision and resource scope stay in the source arrays.

No AEMO raw archive, Ontario/IESO source input, manuscript, correspondence, internal reviewer report, DOI or credential is included. See `DATA_LICENSE.md`, `THIRD_PARTY_NOTICES.md`, `provenance/processed_data_manifest.json` and `manifest.json` for terms and provenance.

The copied `nature-skills` QA code is an independent third-party project, not an official Nature product. Nature-style design does not imply endorsement or acceptance by any journal.
