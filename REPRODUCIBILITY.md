# Reproducing the released RDIA materials

This guide covers the scientific files actually selected for version 1.0.0 of
[rdia-rolling-storage](https://github.com/williamjay1/rdia-rolling-storage).
It distinguishes checksum verification, fresh numerical execution, statistical
reconstruction, model refitting and raw-source reconstruction. Unsubmitted
manuscripts and internal editorial files are not part of the public package.

## Package and environment

The Git checkout contains code, the authentic 128-origin NSW1 fixture, aggregate
CSV summaries, researcher-generated Ontario model parameters, seven figure
exports and provenance. Large scientific inputs retain the original `results/`
tree and are supplied in the complete GitHub Release attachment. GitHub's
automatically generated source ZIP does not include these gitignored files.

`data_assets/selection.json` declares 303 selected scientific files totaling
1,149,951,158 bytes. They cover the five-region NEM input/core trajectory
comparisons, selected branch/common-state diagnostics, statistics and the NSW1
matched learning experiment. They are a documented subset of the original
research workspace; this package does not claim to contain every earlier
experiment. `package_manifest.json` checks the scientific code, fixture and
selected data; `release_manifest.json` covers the complete assembled release.
See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for clocks, units and splits, and
[DATA_LICENSE.md](DATA_LICENSE.md) for source-dependent reuse terms.

Use Python 3.12 and the numerical versions in `requirements.txt`. The recorded
public validation used Python 3.12.10, NumPy 2.5.1, pandas 3.0.5, SciPy 1.18.0,
PyArrow 25.0.0, scikit-learn 1.9.0 and threadpoolctl 3.6.0 on Windows. HiGHS is
accessed through SciPy; a separate `highspy` installation is not required.
Plotting/figure audit dependencies are optional and listed separately.

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements.txt
python scripts/public_reproduce.py --help
```

Commands below assume the current directory is the package root. Replace every
example output path with a new, absent directory **outside** the code and data
roots. The public numerical and download adapters currently enforce D-drive
computation output on Windows, following the author's host convention. Linux
and macOS use explicit writable paths without drive letters; Linux execution
has not been validated in this release. Frozen modules can retain a historical
project `ROOT` literal for byte-identical provenance. Use the public adapters,
which redirect input/output locations, rather than invoking historical modules
directly.

## Verify and run the small fixture

```bash
python scripts/public_reproduce.py --package-root . --check-only
python scripts/public_reproduce.py --package-root . --smoke --output-root D:/MLWork/rdia-smoke-fresh
python scripts/public_reproduce.py --package-root . --oracle-only --output-root D:/MLWork/rdia-oracle-fresh
```

`--check-only` verifies byte counts and SHA-256 for declared files that are
installed and requires the code/fixture. A Git-only checkout can report
`PASS_AVAILABLE_DATA_ASSET_NOT_INSTALLED` together with missing data paths.
This is not complete-data verification. With the full data installed, use
`--verify-all`; any missing declared scientific file or checksum mismatch fails.

The smoke command performs fresh controller solves for
`tests/fixtures/NSW1_raw_c60_inputs_128.parquet`. It compares seven state/action/
cash fields against `NSW1_raw_c60_reference_128.parquet` **after** solving,
with an absolute tolerance of `1e-6`, and checks bounds and operating modes.
The fixture is a real evaluation prefix, not synthetic data or the full sample.
The oracle command tests 30 flat/random objective cases, primary-value
agreement with an all-binary controller, SPO+ subgradients/upper bounds and
the fixed-wear reflection identity. It does not fit a forecast model.

Executed modes write `public_execution_receipt.json` with runtime, input
checksums, adapter checksum, elapsed time and the precise task scope. Read the
status and scope together; successful fixture execution does not establish
external replication of all results.

## Install the complete release data

```bash
python scripts/download_release.py --output-root D:/MLWork/rdia-release-download-fresh
```

The helper retrieves `rdia-rolling-storage-v1.0.0.zip` and `SHA256SUMS` from
the versioned GitHub Release. It verifies the archive before extraction,
rejects unsafe paths/symlinks/duplicates and unexpected expansion, checks ZIP
integrity and verifies all declared extracted files. It retains a
`download_verification.json` receipt. This operation downloads and verifies
files; it does not solve, fit or reconstruct raw data. Release availability is
separate from the locally validated package: a download can only succeed once
that attachment has actually been published.

The complete package is extracted under the download directory's `package/`.
You may run from that root. Alternatively, keep the Git checkout as code and
point `--data-root` to an extracted root containing `results/`:

```bash
python scripts/public_reproduce.py --package-root . --data-root D:/MLWork/rdia-release-download-fresh/package --verify-all
```

Do not pass the `results/` directory itself as `--data-root`. Both roots are
read-only inputs to execution, and the output directory must overlap neither.

## Supported task scope

All modes below use `--task NAME` with an explicit fresh `--output-root`.
Data-dependent modes currently require the entire declared scientific data
selection; they fail explicitly when it is incomplete. The default region list
is `NSW1 QLD1 SA1 TAS1 VIC1`; use `--regions` to restrict applicable tasks.

| Task | What it computes | Packaging-time execution |
|---|---|---|
| `il-mixed` | New mixed trading/continuation input replays for inverse-lead attribution. | Implemented; not rerun through the public adapter. |
| `current-eval` | Full evaluation replay of current-curve candidates. `--policy` can select one candidate. | Implemented; not rerun through the public adapter. |
| `sensitivity` | Efficiency/wear parameter switches for raw, sparse and inverse-lead policies. | Implemented; not rerun through the public adapter. |
| `walkforward` | Rolling current-processor reselection and controller execution; saved execution prefixes are not reused. | Implemented; not rerun through the public adapter. |
| `statistics` | Block-bootstrap summaries on saved continuous trajectories; no controller solves or refit. | Implemented; not rerun through the public adapter. |
| `direct-tests` | Direct paired tests, absorption uncertainty and annual context on saved paths. | Executed with 10,000 draws; five CSV outputs equal frozen references byte for byte. |
| `theory` | Pure finite/reachable/storage/quadratic/projection control checks; observed-surrogate extension omitted. | Implemented; not rerun through the public adapter. |
| `synthetic` | Explicit synthetic controls, including a recorded plan; not observed market data. | Implemented; not rerun through the public adapter. |
| `branches` | Fresh solves on frozen selected event windows with subsequent updates. | Implemented; not rerun through the public adapter. |
| `common-state` | Fresh action-range diagnostics on saved selected common states. | Implemented; not rerun through the public adapter. |
| `spo-fit` | Matched local SPO+/Ridge fit, validation selection, two evaluation replays and audit for NSW1. | Implemented; no new public-package refit performed. |

Representative commands after full-data verification:

```bash
python scripts/public_reproduce.py --task direct-tests --draws 10000 --output-root D:/MLWork/rdia-direct-fresh
python scripts/public_reproduce.py --task il-mixed --regions NSW1 --output-root D:/MLWork/rdia-il-fresh
python scripts/public_reproduce.py --task current-eval --regions NSW1 --policy current_smooth_a025 --output-root D:/MLWork/rdia-current-fresh
python scripts/public_reproduce.py --task sensitivity --regions NSW1 --setting eta0.85 --policy sparse_equal --output-root D:/MLWork/rdia-sensitivity-fresh
python scripts/public_reproduce.py --task common-state --regions NSW1 --history inverse_lead --stride 32 --output-root D:/MLWork/rdia-state-fresh
```

Supported sensitivity settings are `eta0.85`, `eta0.95`, `kappa2`, `kappa10`;
the setting changes each charge/discharge efficiency, not a directly specified
round-trip efficiency. Sensitivity policies are `raw`, `sparse_equal`,
`inverse_lead`. Current candidates are `raw`, `current_shrink_a025`,
`current_shrink_a050`, `current_smooth_a025`, `current_smooth_a050`.
`--history` accepts `sparse_equal` or `inverse_lead` for diagnostics.
Bootstrap draws default to 10,000 and must be at least 2,000. The task
implementation determines its regional scope: `spo-fit` is NSW1 only.

Some tasks reuse frozen comparators, selected events or saved trajectories by
design. An `il-mixed` run recomputes the two mixed-input paths; it does not by
itself reconstruct every term in every final table. A branch or common-state
run is conditional on the documented event/state selection. Statistical
resampling does not rerun forecast selection or MPC within each bootstrap draw.
The package does not promise one command that rebuilds the entire unpublished
manuscript, all historical experiments or every table.

## Figures and Ontario reconstruction

Install `requirements-figures.txt` to redraw the supplied seven figure exports.
The data root must contain the complete package, including `summaries/ontario/`.

```bash
python -m pip install -r requirements-figures.txt
python scripts/public_figures.py --data-root . --output-root D:/MLWork/rdia-figures-fresh
```

These exports use the original numerical summaries, editable PDF/SVG and native
1200 dpi raster output. Redrawing is distinct from rerunning the underlying
experiments. Fixed upstream QA script versions and licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

IESO source CSVs, price-containing Ontario inputs, per-origin generated
forecasts and complete Ontario trajectories are excluded from both Git and the
Release data selection. Aggregates and researcher-generated fitted parameters
are included. Obtain source data directly from IESO under its applicable terms;
the reconstruction adapter checks the recorded exact source versions in
`provenance/ieso_source_versions.json`. It requires both paths to be new:

```bash
python scripts/public_ontario.py --raw-root F:/AcademicData/RDIA_Ontario/raw-fresh --output-root D:/MLWork/rdia-ontario-fresh --stage prepare
```

`download` acquires and checks source files; `prepare` also rebuilds inputs;
`all` additionally runs the two-arm solver grid and analysis. A changed source
checksum fails before replay and preserves downloaded files for inspection.
The operator three-hour forecast arm and researcher-generated twelve-hour arm
remain distinct; CAD outcomes are not pooled with AUD results. This public
preparation workflow was executed afresh at packaging time: both reconstructed
input files were byte-identical to their preserved references, as recorded in
`provenance/public_figures_ontario_validation.json`. The complete 24-trajectory
Ontario policy grid was not rerun in this packaging check. Preparation is not
execution of the full Australian RDIA layer A/B/C diagnostics in Ontario.

## What was actually verified

`provenance/public_validation.json` records fresh internal execution from the
staged public code/data on Windows: 330 declared scientific files passed
size/SHA-256 checks; the 128-origin smoke replay had zero differences in all
seven compared fields; the oracle passed 30 cases with 361 oracle calls; and
10,000-draw direct tests reproduced five frozen CSVs byte for byte.

`provenance/public_figures_ontario_validation.json` separately records seven
fresh figure exports with identical source data and the two Ontario preparation
comparisons. These checks do not rerun the underlying figure experiments or
the full Ontario policy grid.

The full public task grid, new model refit, Australian raw-source reconstruction
and Linux execution were not performed. Preserved frozen results document earlier
research computations; their inclusion is not a claim that every computation
has been independently rerun. `provenance/script_changes.json` and
`provenance/figure_ontario_publication_changes.json` record published-code
changes without disguising path adapters as scientific algorithm changes.

## Supplementary v13 diagnostics

The v13 revision appends cost/conditional-precision and exploratory-equivalence
calculations on saved paths, operational classification of the 100 fixed
nonrandom branch windows, material-action thresholds on saved state ranges,
objective-tolerance sensitivity on a fixed subsample, and a worked SA
objective-oscillation certificate. These research-revision computations have
their own retained execution scope and are distinct from the public adapter's
packaging-time smoke and oracle checks. The finalized release manifest records
the diagnostic files included. The completed extra 30-minute availability
delay consists of 20 research-project replays, each with 46,741 origins on
the unchanged common chronology. LASTCHANGED+30 minutes must be no later
than the original T-60 cutoff; nominal issue eligibility stays unchanged.
The original 2023 C* weights are retained without reselection. Five-asset
S-R and IL-R contrasts are +A$86,079.35 and +A$147,162.83, while S-C*
is -A$3,944.46 and IL-C* is +A$57,139.02 over 974 nominal days.
`results/revision_v13/action/final_execution_audit.json` records physical
constraints, chronology and marked-value reconstruction checks. These are
point sensitivities without new confidence intervals, not observed receipt
times or a fresh public-adapter rerun of the complete 20-policy grid.

The 7,305 saved common states per history comparison are not 7,305 newly solved
states in the objective-tolerance sensitivity: that fresh subsample contains
1,125 all-binary action-range calculations. The 11-of-100 delayed-sale count
uses the 0.1 MW discharge-gap threshold and the selected-window classifier; it
is not a population failure probability or causal loss attribution. Maintenance
cost scenarios are illustrative, and conditional MDE is not observed power or
prospective sample-size planning. Exploratory equivalence does not replace the
primary confidence intervals or resolve outcome-informed study development.

## Additive action and stricter-clock commands

The 330-file core scientific manifest is retained unchanged. Additional action
code, inputs and outputs are listed in `provenance/v13_action_files.json`
and supplied with the complete Release attachment. The original scientific
modules retain their bytes; `scripts/public_action.py` locates declared inputs,
verifies checksums and redirects outputs. It supports `material`, `ranges`,
`case`, `delay-build` and `delay-replay` modes.

```bash
python scripts/public_action.py --package-root . --mode material --output-root /path/to/fresh-material
python scripts/public_action.py --package-root . --mode ranges --output-root /path/to/fresh-ranges
python scripts/public_action.py --package-root . --mode case --output-root /path/to/fresh-case
python scripts/public_action.py --package-root . --mode delay-build --raw-root /path/to/read-only-official-monthly-zips --output-root /path/to/fresh-delayed-inputs
python scripts/public_action.py --package-root . --mode delay-replay --region NSW1 --policy sparse_equal --output-root /path/to/fresh-delayed-replay
```

Use fresh output paths outside the code, data and raw-input roots; apply the
Windows D-drive convention described above where required. `delay-build`
requires the recorded 33 official AEMO monthly ZIPs in a read-only raw root.
Those raw ZIPs, monthly caches, partial files, logs and unpublished TeX writing
fragments are excluded from the public data selection. `delay-replay` runs
only the declared region/policy, using supplied delayed inputs. The complete
20-replay grid can be requested as separate fresh runs; it has not been rerun
as a public-package grid validation. Fresh public-adapter range execution passed on 125 fixed common states per
history across five regions and three allowances: 750 comparison rows and
1,125 all-binary range calculations. Threshold recounts and the worked SA
oscillation case also passed against preserved references.
`provenance/v13_action_validation.json` records the executed scope, receipt
checksums and exact comparisons. These checks are distinct from the complete
20-replay grid and do not refit models or establish independent external replication.
