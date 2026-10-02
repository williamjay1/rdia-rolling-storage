# RDIA for rolling storage optimization

Research software and reproducibility materials for the current v14 manuscript, **Action sufficiency and update risk in rolling storage optimization**. The frozen v1.0 release accompanies the earlier title, **A protocol for evaluating information representations in rolling storage optimization**. Sole author: **Junjie Zhang**. The manuscript targets *Annals of Operations Research*; this repository does not claim journal acceptance or a DOI.

Affiliations:

- Shanghai Academy of Global Governance and Area Studies, Shanghai International Studies University, Shanghai 201620, China.
- School of Economics and Finance, Shanghai International Studies University, Shanghai 201620, China.

Contact: <junjiezhang2024@shisu.edu.cn>. ORCID: [0009-0004-8821-4018](https://orcid.org/0009-0004-8821-4018).

The frozen v1.0 study uses rolling-decision information attribution (RDIA) to compare forecast representations against attainable alternatives using three complementary layers: objective-input channel allocation along rolling paths, tolerance-aware action ranges at a common inventory state, and one-time versus sustained branches exposed to subsequent forecast updates. The Australian application uses five regional electricity markets and one implemented storage controller. Local decision-focused learning and Ontario comparisons provide partial boundary checks. Ontario's two forecast arms do not replicate the full Australian three-layer diagnostic design. Results, confidence intervals and resampling tests are conditional on the retained comparator, information clock, controller, selected states/events and observed history; negative or imprecise results are retained.

## What is released

The repository contains scientific code, a small real-data test fixture, machine-readable result summaries, seven original figure exports, attribution notices, data descriptions and reproduction instructions. Large, derived AEMO inputs and reference trajectories are supplied as a checksummed GitHub Release attachment rather than Git LFS pointers. The complete attachment, `rdia-rolling-storage-v1.0.0.zip`, combines code and data so it can be uploaded manually to Zenodo with `SHA256SUMS`.

The core Australian numerical inputs and outcomes preserve the v10 scientific freeze; v11 supplies the figure design. The v13 revision adds separately identified supplementary diagnostics on numerical/material-action thresholds, branch-start selection and a worked objective-oscillation certificate. These additions do not represent a new prospective evaluation or a rerun of the whole frozen study. The finalized release manifest identifies the diagnostic code and outputs actually included in this version. The additive action and delay selection is declared separately in `provenance/v13_action_files.json`, preserving the core scientific manifest.

The appended diagnostics distinguish these scopes:

- Break-even archive-cost thresholds, illustrative cost scenarios, conditional bootstrap precision/MDE and exploratory equivalence checks on saved paths. Actual maintenance costs were not measured; exploratory equivalence is not the core result or a prospective power calculation.
- Material-action thresholds evaluated on 1,461 saved common states per region (7,305 states across five regions for each history comparison), with 1,125 fresh all-binary action-range calculations on a fixed subsample for objective-tolerance sensitivity.
- An operational delayed-sale classifier flags failure in 11 of 100 selected nonrandom branch windows at the 0.1 MW discharge-gap threshold. This is a selected-window diagnostic, not a population failure rate or a causal estimate of a single trade's loss.
- A worked SA objective-oscillation certificate, distinct from the action ranges and cumulative branch results.
- Twenty complete stricter-clock replays, each with 46,741 origins, apply LASTCHANGED+30 minutes before the original T-60 cutoff. The original 2023 current-processing weights are retained. Five-asset S-C* changes sign to -A$3,944.46, while IL-C* remains +A$57,139.02 over 974 nominal days. These are point sensitivity results without new confidence intervals or an observed participant-receipt claim. The completed research runs are distinct from fresh execution through the public adapter. Fresh public-adapter action-range and SA-case checks passed within the fixed-subsample scope recorded in `provenance/v13_action_validation.json`; the complete 20-replay grid was not rerun through that public entry.

Unsubmitted manuscript files, internal review documents, credentials, local machine configuration and raw monthly source archives are excluded. Ontario inputs and trajectories containing IESO source price or forecast columns are also excluded; source links, version checksums and reconstruction code are provided. Source terms apply to data separately from the MIT license for original code.

## v14 computational addendum

The current revision examines action sufficiency and update risk in rolling storage optimization. [addenda/v14](addenda/v14/) supplies a separately manifested compact package with 18 exploratory NSW1 January 2024 closed-loop trajectories, nine policies/ablations at initial 0.2/1.8 MWh, shared-state diagnostics, a locked design and synthetic controls. The pilot records **STOP_NEW_COMPRESSION_ALGORITHM**: the candidate gate retains full inverse-lead streaming moments and adds optimization calls, so it has no established online-storage advantage over exact streaming IL.

The compact package was checked before this public copy: fresh IL and static9 replays each covered 1,488 origins at initial 0.2 MWh; seven trajectory fields and marked values matched exactly, and all 18 synthetic checks passed. The public preparation verifies copied scientific bytes; it does not claim a new checkout-wide replay. The floating-point gate concerns same-state, one-window objective regret and is not a rigorous cumulative-profit certificate. Native Ontario materials contain schema-probe code and version/source manifests without source-price data or an executed external-market replay.

The root v1.0 citation/Zenodo metadata, scientific inputs, and manifests are retained. Its versioned Release attachment remains frozen under the earlier manuscript title. Root explanatory documents are updated to identify the separately scoped v14 work; the old full-release manifest still describes the original v1.0 attachment rather than those subsequently updated documentation bytes. The addendum has its own manifests and is excluded from the v1.0 verification scope.

## v15 computational addendum

[addenda/v15](addenda/v15/) adds conditional Hansen SPA, Romano–Wolf stepdown and model-confidence-set calculations, current-program reselection sensitivity, signed cost-budget intervals, eleven finite theory checks and eighteen saved parameter paths. Seven- and twenty-eight-day blocks resample dates synchronously across all regions. Model-confidence-set membership is not equivalence; combined selection/evaluation ranges still include zero for inverse lead and do not remove all prior research-search uncertainty. Costs were not observed.

The compact package was freshly extracted and executed before publication: both block lengths used 10,000 statistical draws, six reconstructed CSVs and formal JSON matched exactly, eleven synthetic groups passed, and one full 1,488-action hourly-update IL replay matched cash, inventory and clocks exactly. The wrapper does not reconstruct raw MMS archives, refit forecasting models, rerun the full five-region study or provide independent confirmation. Twenty-four earlier diagnostic CSVs are retained byte for byte with provenance. The original v1.0 release and the v14 addendum remain unchanged; v15 has its own manifest. Unsubmitted manuscript and review files and IESO source prices are excluded.

## Start here

Use Python 3.12 in a virtual environment. The numerical dependency versions in `requirements.txt` were used in the original numerical work. Figure and metadata tools have separate optional dependencies. All new output must go to a fresh directory outside the supplied read-only inputs.

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements.txt
python scripts/public_reproduce.py --help
python scripts/public_reproduce.py --check-only
python scripts/public_reproduce.py --smoke --output-root /path/to/fresh-rdia-smoke
python scripts/public_reproduce.py --oracle-only --output-root /path/to/fresh-rdia-oracle
```

Replace example output paths with new directories appropriate for your machine. For the full data package, download the Release attachment and verify it using `SHA256SUMS`, or run `python scripts/download_release.py --output-root /path/to/fresh-download`. Then execute `public_reproduce.py` from the extracted package. The download helper verifies the archive and every supplied file before reporting success; it does not solve the controller or refit a model. A release download can succeed only after the attachment is published.

Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact supported commands and verification scope. [DATA_DICTIONARY.md](DATA_DICTIONARY.md) defines the information clock, columns, units and sample splits. Verification of checksums, fresh solver execution, statistical reconstruction, model refitting and raw-source reconstruction are different operations; a successful small test does not establish external replication of the entire study.

## Data, code and citation

Original software is licensed under [MIT](LICENSE). Third-party data retain their source terms; see [DATA_LICENSE.md](DATA_LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). AEMO is the source of the redistributed Australian forecast and settlement price material. IESO source data are download-and-rebuild only in this public release.

Use [CITATION.cff](CITATION.cff) to cite the versioned research software. A DOI is deliberately absent until the author deposits the release in Zenodo. The [Chinese Zenodo guide](ZENODO_GUIDE_zh.md) explains manual deposit, version selection and DOI backfilling. Large Release attachments are not assumed to be included in GitHub's automatically generated source ZIP or Zenodo's GitHub integration.

## AI use and research boundaries

OpenAI Codex assisted research design, literature and source review, programming, numerical analysis, figure preparation, manuscript drafting and revision. Reported results are checked against identified market records or labelled synthetic controls, executed code and mathematical arguments. Implemented checks and executed results are distinguished from proposals. The author remains responsible for source interpretation, claim accuracy, final human review and submission. This disclosure is not restricted to language editing.

The study is exploratory and uses public market records rather than human participants. It does not establish universal gains from forecast history, a new forecast estimator, end-to-end rolling-policy training, economic causal identification or independent external replication. Provider clocks, selection rules, numerical tolerance ordering and unmeasured archive costs are documented with the results.
