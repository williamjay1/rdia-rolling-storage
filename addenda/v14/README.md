# v14: action sufficiency and update risk

Current manuscript title: **Action sufficiency and update risk in rolling storage optimization**. Sole author: **Junjie Zhang**. This addendum is separately scoped from the frozen v1.0 release and does not change its scientific inputs, manifests, citation metadata, or Zenodo metadata.

[computational/](computational/) is a standalone small numerical package. It preserves 18 exploratory January 2024 closed-loop trajectories and the locked design, shared-state diagnostics, and synthetic controls. The recorded decision is **STOP_NEW_COMPRESSION_ALGORITHM**: the gate requires full streaming IL moments and additional solves, so it does not provide a storage advantage over the exact streaming alternative. It tests conditions and failure boundaries for rolling action sufficiency.

Before this public copy, the compact package passed complete manifests and fresh IL/static9 replays of 1,488 origins each at initial 0.2 MWh; seven trajectory fields and marked values were exact, and 18 synthetic checks passed. [COMPACT_PACKAGE_FRESH_VALIDATION.json](COMPACT_PACKAGE_FRESH_VALIDATION.json) preserves that executed scope. Copy verification is distinct from a new public checkout rerun. The numerical gate is a floating-point local-window objective check, not a rigorous cumulative-profit certificate or an independent noninferiority result.

[native_ontario/](native_ontario/) contains only schema-probe code and official source/version manifests with price-free scope metadata. It contains no Ontario price data or source files and does not claim an executed native-market replication. Reuse follows source terms independently of the repository's software licence.

The frozen v1.0 Release attachment and its original manifests remain the record for the prior RDIA study. This addendum has its own manifests. Unsubmitted manuscripts, internal assessments, raw market archives and locally generated fresh outputs are excluded. No journal acceptance or new DOI is claimed.
