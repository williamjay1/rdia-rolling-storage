# Additive v15 computational checks

Junjie Zhang; accompanying computational addendum to the rolling-storage information evaluation study. The frozen public v1.0 release remains unchanged. This package supplies a reproducible historical reanalysis, not independent confirmation or a new compression algorithm. No unpublished manuscript or internal review is included.

## Included scope

* Exact derived daily operating-cash fixtures and realised boundary totals: strict purged 2023 current-program selection (364 dates); full frozen evaluation (973 interior dates, 974 nominal days); and its 2024–2025 execution-day subpath (731 dates). All models/regions share calendar indices. The shared 5 September 2024 source gap has 47 saved actions; all other interior dates have 48, with no imputation.
* Executed consistent-recentered Hansen SPA, one-sided studentised Romano–Wolf max-t stepdown and Tmax MCS at 7/28 day blocks, 10,000 draws per block, plus current-program reselection and signed cost-budget sensitivity. The primary family is S/F/IL relative to fixed strict-2023 C*, and MCS includes R/C*/S/F/IL. Retained membership is not equivalence or noninferiority.
* Five-asset daily scores are Y=(973X+B)/974, B=full marked value−interior operating cash. Realised boundary cash and SOC price marks are fixed in resampling; this is bookkeeping rather than observed daily marked cash. Cost is unobserved. Annual per-MW rates are portfolio daily rates×365/5 for five 1 MW stylised assets, not commercial profit or region-specific fees.
* Current-program selection sensitivity only resamples saved program value paths, without forecasting-model refit or regenerated SOC paths. Combined train/evaluation ranges condition on the saved candidate set and realised endpoint marks. They are not unconditional selection-adjusted confidence intervals.
* Eleven groups of finite synthetic theory checks. They verify specified algebra on small instances, not an empirical/runtime certificate for cumulative cash.
* All 18 saved January2024 NSW parameter paths (six settings×C*/S/IL) and exact three-policy January curve fixture. The fresh default check replays only the locked hourly-update IL setting: 1,488 actions from 744 plans, with original initial1 MWh, power/SOC feasibility, cutoff and physical accounting. S−C* is negative and IL−C* positive in all six saved settings; these are descriptive same-month results.
* `legacy_diagnostics/` preserves 24 earlier v13 scientific CSVs byte for byte, with relative-source PROVENANCE.json. These retain compressed-out regional Holm, 100 branch starts, delayed-sale opportunities/frequency, action-tolerance and conditional precision/MDE diagnostics. They are archival outputs, not newly rerun v15 results or a substitute for the defined new family. Earlier cost/equivalence scenarios remain illustrative rather than formal evidence of noninferiority.

## Reproduce without modifying this package

Python 3.12.10 with the supplied exact versions was used. The controller uses SciPy's private bundled HiGHS API; other versions require verification rather than assumed numerical identity. Run from an extracted copy:

```console
python reproduce_v15.py --check-only
python reproduce_v15.py --output /absolute/path/to/a/new/output-directory
```

The output directory must be new and outside the package. On Windows the research default is a new D-drive directory. The second command recomputes all statistical and theory checks plus the one full January hourly IL replay, compares numerical CSV/JSON and cash/stock/clock fields against frozen outputs, and emits execution_receipt.json there. It does not ingest original raw archives or repeat the full five-region optimisation. It never modifies an earlier output directory.

Individual entries, again with new output directories, are:

```console
python scripts/revision_v15_statistics.py --input-fixture fixtures/statistics --output /absolute/new/statistics
python scripts/revision_v15_parameter_pilot.py --mode single --setting update60min_execute_two_planned_steps --policy inverse_lead --output /absolute/new/parameters
```

The theory script writes beside its script file; the default wrapper copies it to the new theory output folder before executing it. MANIFEST.json covers every payload. provenance.json records original source SHA values, path adaptations, omitted manuscript renderers and exact scientific-function source identity. FRESH_VERIFICATION.json records the actually executed package checks. Respect DATA_LICENSE.md and the root release's authorship and source permissions.
