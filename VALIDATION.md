# Executed public-package checks

The following are actual execution checks of the curated public copies, not
claims of independent external replication or execution of every experiment.

| Check | Executed result | Scientific scope |
|---|---|---|
| Fresh NSW rolling replay | 128 newly solved origins; maximum difference from the frozen reference in all seven state/action/cash fields: 0 | A short authentic prefix; not the full evaluation grid |
| SPO+ oracle validation | 30 validation cases, 361 oracle calls; maximum primary-objective difference approximately 1.14 × 10⁻¹³ AUD | Oracle, loss and subgradient checks; not a new full training run |
| Direct statistical reconstruction | 10,000 bootstrap draws; five output CSVs are byte-identical to frozen results | Computation on saved paths; no controller re-solve or selection refit |
| Figure reconstruction | All seven figures regenerated; their numerical source arrays equal the frozen figure arrays | Figure data and rendering workflow; not new market evidence |
| IESO source reconstruction | All three exact official version hashes matched; both reconstructed Ontario input files are byte-identical to the frozen inputs | Download and input preparation; full Ontario policy replay was not repeated in this public-release check |

The checks use the stated Python 3.12 numerical environment. The public entry
point verifies supplied scientific file identities and requires a separate fresh
output directory. A normal Git checkout intentionally lacks the large data;
`--check-only` reports that distinction rather than pretending it verified absent
files. Use the complete Release archive for `--verify-all` and the full tasks.

Ontario source and reconstructed price-bearing files from the local check are
not redistributed. Original records and intermediate execution receipts are
retained locally; concise public receipts contain version identities and scope.
The GitHub workflow separately attempts the short replay and oracle checks on
Linux. Its actual run status should be read in Actions; workflow configuration
alone is not a passing execution result.

## Executed appended diagnostics

- Public economic script: 354 numerical checks passed; 23 generated outputs
  were byte-identical to the source computation. One source-evidence JSON differs
  only in publication metadata, with scientific values identical.
- Public action adapter: 750 fixed-sample comparisons, 1,125 fresh all-binary
  range calculations and the SA oscillation example passed exact comparisons.
- Research-source stricter clock: 33 monthly archives were reconstructed and
  20 full replays completed, each preserving 46,741 origins. Independent chronology,
  physics and marked-cash reconstruction passed. The public adapter has not
  rerun the complete 20-policy grid.

Exact receipts are `provenance/v13_economic_validation.json` and
`provenance/v13_action_validation.json`. These additions preserve the core
scientific manifest and distinguish saved-data calculations, newly solved
ranges and completed research-source replays.
