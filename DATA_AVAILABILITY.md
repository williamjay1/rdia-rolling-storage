# Data and code availability

Repository: <https://github.com/williamjay1/rdia-rolling-storage>.
Sole author: **Junjie Zhang**, Shanghai Academy of Global Governance and Area
Studies and School of Economics and Finance, Shanghai International Studies
University, Shanghai 201620, China. Contact: <junjiezhang2024@shisu.edu.cn>;
ORCID: <https://orcid.org/0009-0004-8821-4018>.

The complete version 1.0.0 code-plus-data package is distributed through the
matching GitHub Release as **rdia-rolling-storage-v1.0.0.zip**, with **SHA256SUMS**.
The download requires that version's attachment to have been published.
GitHub's automatically generated source-code archive is smaller and omits the
large scientific data under `results/`.

The core scientific selection contains 303 Australian derived-data and
reference-output files, about 1.15 GB before ZIP compression. The authoritative
selection and scientific file identities are in `data_assets/selection.json`,
`data_assets/manifest.json` and `package_manifest.json`. `release_manifest.json`
also covers the documentation, figure, additional diagnostic and other public files. Source-dependent
reuse terms are described in `DATA_LICENSE.md` and `THIRD_PARTY_NOTICES.md`.

To obtain and verify the complete archive from a normal checkout:

```bash
python scripts/download_release.py --output-root /path/to/fresh-download
python /path/to/fresh-download/package/scripts/public_reproduce.py --verify-all
```

Use a fresh output directory appropriate for your machine.
Follow `REPRODUCIBILITY.md` for actual solver, statistical and fitting commands.
Download/hash verification alone does not execute these scientific tasks.

Raw AEMO monthly archives are not mirrored here. The supplied derived inputs
allow frozen-input replay; they are not a claim that the full historical raw
archive-to-input pipeline has been publicly rerun. The [official NEMWEB
archives](https://www.nemweb.com.au/) remain the source of market records.

Core outputs preserve the v10 scientific freeze; v11 supplies figure design.
The v13 supplementary diagnostic calculations are separately identified and do
not overwrite those original results. They include illustrative cost and
conditional precision/MDE checks, exploratory equivalence, fixed selected-window
failure classification, material-action thresholds, subsample objective-tolerance
sensitivity and a worked SA objective-oscillation certificate. Twenty extra
30-minute availability-delay replays have been completed in the research
project, with 46,741 origins per replay and the same chronology in all five
regions. They retain the original 2023 C* weights. The five-asset marked
S-C* contrast is -A$3,944.46 and IL-C* is +A$57,139.02 over 974 nominal
days. No new confidence intervals are calculated for this clock sensitivity.
The source-project execution audit is `results/revision_v13/action/final_execution_audit.json`;
`provenance/v13_action_files.json` declares the additive public selection.
Completion of those source-project runs is not a fresh public-adapter grid rerun.
Fresh public-adapter action-range and SA-case checks passed: 750 comparison
rows, 1,125 all-binary range calculations and exact reference comparisons
are recorded in `provenance/v13_action_validation.json`. This does not
recompute all 7,305 saved common states per history comparison.
Confidence intervals and block tests are
conditional on retained policies, selection rules and observed history; they
do not repair the study's exploratory design. The manifest, rather than a
version label alone, identifies the scientific files supplied.

IESO annual source CSVs, Ontario price-bearing aligned inputs and recoverable
period-level trajectories are excluded. Exact official source URLs, versions
and hashes are in `provenance/ieso_source_versions.json`. Reconstruct locally:

```bash
python scripts/public_ontario.py --raw-root /path/to/new-raw-directory --output-root /path/to/new-computation-directory --stage prepare
```

Use separate fresh raw-download and computation directories. The preparation command downloads the three official
versions, verifies their frozen hashes and prepares both disclosed forecast
arms. It does not execute the complete Ontario policy grid. `--stage all`
additionally requests that grid and analysis, which was not rerun for this
public-package preparation check. Rebuilt IESO material must retain IESO's terms
and notice and must not be added to this public release.

The retained packaging receipt records fresh internal preparation of both
Ontario arms with byte-identical inputs, and fresh export of seven figures from
the supplied summaries. These checks are separate from the 128-origin NSW
solver replay, 30 SPO+ oracle validation cases, and reconstruction of paired
statistics on saved paths. They do not establish a fresh full policy grid,
full Australian raw-source reconstruction, or independent external replication.

The author will deposit the complete attachment personally in Zenodo.
No Zenodo DOI is claimed in this release. See `ZENODO_GUIDE_zh.md` for accurate
version citation and metadata updates after the record is published.
