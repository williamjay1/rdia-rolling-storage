# Data dictionary

This guide distinguishes observed market records, researcher-generated forecasts,
controller trajectories and statistical summaries. Large Australian inputs are
provided in the separately checksummed release data asset, not in a normal Git
checkout. `data_assets/selection.json` lists the selected large core files;
the separate v13 provenance manifests identify added dependencies and outputs.
The small fixture under `tests/fixtures/` is a real-data prefix for
engineering checks; it is not the full evaluation sample.

Unsubmitted manuscripts, internal review responses, editorial assessments,
cover letters, personal machine configuration and temporary progress snapshots
are outside the public data selection. Original monthly archives are not bundled.
Data retain the provider terms described in `DATA_LICENSE.md` and
`THIRD_PARTY_NOTICES.md`; the MIT code licence does not relicense them.

## Appended v13 diagnostics

Additional Australian results are under `results/revision_v13/action/` and
`results/revision_v13/economic/`. CSV headers identify currencies, units,
contrasts, block lengths, thresholds and conditioning scope. Source identities
and public comparisons are recorded in `provenance/v13_action_files.json`,
`provenance/v13_action_validation.json`, `provenance/v13_economic_inputs_manifest.json`
and `provenance/v13_economic_validation.json`. These separate manifests do not
replace the immutable 330-entry core scientific manifest. The complete Release
manifest covers the core and additions.

Delayed inputs require LASTCHANGED+30 minutes no later than T-60, with nominal
issue still no later than T-60. Their replays retain all 46,741 origins, original
2023 processing weights and initial inventory. Values are AUD point clock
sensitivities; no new confidence intervals are supplied. The 100 branch starts
are a nonrandom forecast-defined event sample; the delayed-sale classifier
describes those windows. Cost columns are illustrative incremental annual costs
per MW, not measured prices. Conditional MDE and positive-width equivalence
diagnostics retain the observed block-error distribution; they do not provide
prospective power or confirmation.

## Australian price and forecast inputs

The five region identifiers are `NSW1`, `QLD1`, `SA1`, `TAS1` and `VIC1`.
Prices are Australian dollars per MWh (AUD/MWh). NEM timestamps use fixed AEST,
UTC+10, without daylight-saving adjustment; a timezone-naive timestamp in these
files is not a region's daylight-saving civil time.

`target = T` labels the end of a half-hour execution interval `[T−30 min, T]`.
The main archive cutoff is `T−60 min`; the freshness sensitivity uses `T−30 min`.
`LASTCHANGED` is an archive modification timestamp used as an availability proxy,
not an observed participant receipt timestamp. Historical weights must not be
interpreted as verified real-time publication latencies.

The input families retain the original relative paths:

| Path pattern | Role |
|---|---|
| `results/revision_v5/history/{region}_baseline_inputs.parquet` | Current curves and current-only processing candidates, with common labels. |
| `results/revision_v5/history/{region}_inputs_aligned.parquet` | Target-aligned archive representations used in the rolling comparisons. |
| `results/revision_v5/history/{region}_incremental_inputs_aligned.parquet` | Restricted learned/incremental representation comparisons. |
| `results/revision_v5/history/{region}_history_features_aligned.parquet` | Current/history features and observed labels for learning and diagnostics. |

A release can contain only a selected subset of these families; consult the
selection manifest before running a task. File rows can repeat one origin across
policies or cutoffs. Do not count all rows as independent decision origins or
join on `target` alone.

This release selects exactly 303 large scientific files (1,149,951,158 bytes).
Its thirteen `history/` files comprise baseline and aligned inputs for each of
the five regions, the NSW1 history-feature file used by the matched learning
experiment, and two verification CSVs. The incremental-input family and
history-feature files for the other four regions are not included. The path
table describes source conventions; it does not claim that every historical
experiment can be rerun from this selected public package.

The first three families have the following actual column conventions:

| Column | Meaning and unit |
|---|---|
| `region` | One of the five region identifiers. |
| `target` | First half-hour interval-end label, in NEM time. |
| `phase` | `validation` or `evaluation`, defining chronological use. |
| `cutoff_minutes` | Minutes from the first target's end back to the archive cutoff. |
| `policy` | Representation/processor identifier; it is not an observed market action. |
| `query_time` | Coherent current-reference curve archive time. It is not a single issue timestamp of a target-by-target history composite. |
| `p_01` … `p_12` | Twelve objective-input prices, AUD/MWh. `p_01` corresponds to `target`; `p_j` corresponds to `target + (j−1) × 30 min`. |
| `actual_00` … `actual_11` | Ex-post observed price labels at the same twelve target positions, AUD/MWh. `actual_00` pairs with `p_01`. Labels are used for evaluation/training eligibility, not supplied to the controller as forecasts. |

The main controller trades over the first eight forecast steps and uses the
remaining four prices in its continuation proxy. It executes only the first
action and then updates the inputs. This is a six-hour forecast input with a
four-hour trading horizon, not twelve hours of Australian operation.

`validation` is the purged 2023 selection episode. Complete target horizons must
precede the next evaluation cutoff; a row is not eligible merely because its
first target is in 2023. Evaluation covers January 2024 through August 2026.
Earlier years in the learning feature family support the documented training
split; they are not additional evaluation assets. The baseline marked-value
normalization uses 974 nominal exposure days, while the realized retained
interval span is reported separately. Do not infer the denominator from a row
count without checking the episode definition.

## Archive features and representation identifiers

In `*_history_features_aligned.parquet`, horizon suffixes `_00` … `_11` are
zero-based and each increment is 30 minutes. `REGIONID` is the region field and
`curve_asof` is the coherent current-reference archive timestamp. Important
field families are:

| Field family | Meaning and unit |
|---|---|
| `fcst_XX`, `raw_XX` | Current-reference forecast prices, AUD/MWh; retained source/reference conventions are documented by the input builder. |
| `actual_XX` | Observed target price labels, AUD/MWh. |
| `sparse_equal_XX` | Equal mean of current plus target-aligned offsets 1, 2, 4 and 6; AUD/MWh. Actual archive ages can vary. |
| `full_equal_XX` | Equal mean of available target-specific archive versions, AUD/MWh. |
| `inverse_lead_XX` | Available versions weighted by inverse delivery-minus-nominal-run-start lead; AUD/MWh. |
| `inverse_archive_lead_XX` | Sensitivity using delivery-minus-`LASTCHANGED` in the weighting denominator; AUD/MWh. |
| `count_XX` | Number of available versions, dimensionless integer. |
| `sd_XX` | Population standard deviation across those available forecast values, AUD/MWh. |
| `lead_min_XX`, `nominal_lead_min_XX` | Delivery minus the matched archive/nominal issue time, in minutes. |
| `oldest_age_min_XX`, `latest_age_min_XX` | Archive cutoff minus oldest/latest eligible modification time, in minutes. These are ages relative to the cutoff, unlike delivery lead. |
| `lag1_p_XX`, `lag2_p_XX`, `lag4_p_XX`, `lag6_p_XX` | Sparse constituent forecast values, AUD/MWh. |
| `lag1_lead_min_XX` and analogous lag fields | Delivery minus the corresponding constituent archive time, in minutes. |
| `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` | Dimensionless calendar encodings. |
| `weekday` | Integer calendar weekday used by the implementation. |
| `decision_lead_min`, `cutoff_minutes` | Minute-valued clock/eligibility features. |

Nominal AEMO run start is reconstructed from the sequence-date and period code:
04:00 plus `(PP−1) × 30 min`. The PP01 run label ends at 04:30; its inferred start
is not a measured file receipt. The raw representation `R` is `raw`; sparse `S`
is `sparse_equal`; inverse delivery-lead `IL` is `inverse_lead`. `C*` is a
region-specific processor selected from current-only candidates on the prior
validation episode, not a common extra forecast column or the hindsight winner
on evaluation. Current candidates include `current_shrink_a025/a050` and
`current_smooth_a025/a050`.

## Executed Australian controller trajectories

Trajectory paths follow
`results/revision_v5/control/{scenario}/{region}_{phase}_c{cutoff}_{policy}.parquet`;
the matching JSON records controller parameters, input hash, episode totals and
audits. Read the scenario and JSON before pooling sensitivity runs.

| Column | Meaning and unit |
|---|---|
| `region`, `target` | Region and half-hour end label. |
| `actual_price` | Realized price of the executed interval, AUD/MWh. |
| `execution_start` | `target − 30 min`. |
| `execution_day` | Calendar day obtained by flooring `execution_start`, in NEM time. |
| `soc_start`, `soc_end` | Stored energy before/after execution, MWh; these are not percentages. |
| `charge_mw`, `discharge_mw` | Nonnegative grid-side power over the executed interval, MW. |
| `cashflow_aud` | Gross executed cash, AUD. |
| `degradation_aud` | Linear throughput wear charge, AUD; not a measured lifetime degradation cost. |
| `net_aud` | Gross cash less the throughput charge, AUD. |
| `primary_optimum_aud`, `primary_loss_aud`, `primary_loss_upper_aud`, `primary_gap_aud` | Solver/implemented-policy numerical diagnostic values in AUD; they are not realized operating cash. |

For time step `dt = 0.5 h`, executed gross cash is
`actual_price × (discharge_mw − charge_mw) × dt`; wear is
`kappa × (charge_mw + discharge_mw) × dt`. Energy balance is
`soc_end = soc_start + eta_c × charge_mw × dt − discharge_mw × dt / eta_d`.
The base asset is 1 MW / 2 MWh with 0.2–1.8 MWh operating bounds, initial energy
1 MWh, `eta_c = eta_d = 0.91`, and `kappa = 5 AUD per grid-MWh`. Boundary files
change the explicitly recorded parameter; they must not be treated as base runs.

`net_operating_aud` is the sum of executed net cash. `endpoint_mark_aud` is
`final_energy × final_actual_price − initial_energy × initial_actual_price`.
`net_value_aud` is operating cash plus this common endpoint accounting mark.
The endpoint mark is distinct from the forecast continuation coefficient used
inside each optimization problem.

## Statistical and displayed results

`results/revision_v7/statistics/daily_contrast_panel.parquet` has
`execution_day`, `region`, `contrast`, and `net_cash_difference_aud`. Each value
is a paired day-level cash difference on retained continuous policy trajectories,
in AUD; it excludes the separate endpoint contrast. Regions share the same
calendar resampling indices. A five-asset sum is a sum of the five model assets,
not an estimate of the total Australian market.

The layer-A decomposition summaries use `history_rule`, `region`, `component`,
`full_value_aud`, `aud_per_nominal_day`, `block7_ci_low`, `block7_ci_high` and
`definition`. In the four mixed-input runs, `J_RR`, `J_HR`, `J_RH` and `J_HH`
denote realized marked values; the first position selects trading inputs and the
second selects continuation inputs (`R` is raw/current, `H` is history).
The order-averaged allocations are
`T = [(J_HR−J_RR)+(J_HH−J_RH)]/2` and
`C = [(J_RH−J_RR)+(J_HH−J_HR)]/2`; their sum is
`D = J_HH−J_RR`. The interaction is
`I = J_HH−J_HR−J_RH+J_RR`. These are realized input-channel allocations under
the common controller, not additive physical revenue streams. Use each row's
`definition` to identify the normalization and interval represented.

Layer-B common-state action ranges are grid net injection `discharge−charge`
in MW, conditional on the same inventory in MWh; they are not energy intervals.
Layer-C `marked_aud` contrasts are finite-window, inventory-marked branch values
in AUD. Branch `positive`/`negative` counts refer to selected diagnostic events,
not independently sampled market experiments. Selection of events and the
subsequent update schedule remain part of the interpretation.

CSV result files keep units in their field names: `_aud` denotes AUD totals;
`_aud_per_day` or daily means denote AUD/day; `_aud_per_mwh` denotes AUD/MWh;
`_cad` and `_cad_per_day` denote Canadian dollar quantities. `block_days` is a
circular moving-block bootstrap block length in days. `bootstrap_draws` is a
count; `seed` is a pseudorandom generator seed. CI bounds, p-values and Monte
Carlo standard errors are different fields. Reported uncertainty conditions on
the saved policies, continuous state paths and disclosed selection design; it
does not silently rerun selection or the controller inside every draw.

CSV summaries are derived outputs, not new participant or market observations.
The theory/synthetic controls are explicitly constructed examples; they must not
be relabelled as observed market records. The source/result mapping and public
reproduction entry point identify which displayed results are recalculated and
which are supplied frozen outputs.

## Ontario download-and-rebuild boundary

IESO raw annual CSVs and Ontario inputs/trajectories that contain IESO source
price/forecast values are excluded from this public release. The following
schema describes the reconstruction code, not data claimed to be present in a
fresh public checkout. Official versioned URLs/checksums and IESO attribution
must accompany reconstruction. CAD/MWh values are not converted to AUD/MWh.

Rebuilt `operator_short_inputs.parquet` uses three actual operator lead columns;
`researcher_12h_inputs.parquet` uses researcher-generated twelve-hour curves.
Their common fields are `interval_start_est`, `interval_end_est`,
`interval_start_utc`, `actual_cad_mwh`, `origin_index`, `nominal_issue_est`,
`phase`, and `common_current_complete`. EST is fixed UTC−5; the UTC column is
timezone-aware. `nominal_issue_est = interval_start_est − 1 h`. Historical issue
times are reconstructed from delivery-interval **start** minus lead, not end
minus lead. Actual publication/receipt/run identities remain unavailable.

Each policy has `{policy}_00...` price columns in CAD/MWh. Operator indices
00–02 cover three hours, with two trading and one continuation step. Researcher
indices 00–11 cover twelve hours, with eight trading and four continuation steps.
`latest_feature_price_end_est` belongs to the researcher arm and records the
feature-availability cutoff check under the assumed 24-hour price delay. An
incomplete current operator curve triggers the common idle guard: retain the
hour, use zero flows and carry inventory; do not impute actual prices or drop
the hour independently by policy.

The public `summaries/ontario/ridge_models.npz` contains researcher-generated
fitted parameters: `coefficient`, `scaler_mean` and `scaler_scale` have shape
`(18, 55)`, `intercept` has shape `(18,)`, and `train_origins` is an integer
origin-index array. These eighteen fitted horizons support the twelve-step curve
and earlier target-aligned vintages; they are not an operator eighteen-hour
forecast. Ridge fitting uses 2022; prior current-policy selection uses
October–December 2023; evaluation uses February–July 2024. The accompanying
aggregate CSVs report CAD outcomes and CAD/MWh forecast errors. Original study
output permissions are described separately in `DATA_LICENSE.md`.

Reconstruction can also produce `generated_vintages.npz`, with integer `origins`
and `predictions` containing eighteen direct horizon columns. This per-origin
prediction array, the price-containing inputs and the complete Ontario
trajectories are excluded from the public package. The parameter file alone
does not reconstruct a forecast without the separately obtained source features.

Rebuilt hourly trajectories use `soc_start_mwh`, `soc_end_mwh`, `charge_mw`,
`discharge_mw`, `cashflow_cad`, `degradation_cad`, `net_cash_cad`,
`primary_loss_upper_cad`, `primary_gap_cad` and `solver_executed` in addition to
the clock/price fields. The two arms have different forecast processes and
temporal support; their results are not pooled as one replication effect.
