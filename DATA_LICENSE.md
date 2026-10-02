# Data and derived-material reuse terms

This is a **mixed-content release**. The software, original study outputs and provider-derived data do not all have the same license. The repository's [MIT license](LICENSE) covers original research software; it does not grant rights that belong to AEMO, IESO or another third party.

Sole author: **Junjie Zhang**, Shanghai Academy of Global Governance and Area Studies and School of Economics and Finance, Shanghai International Studies University, Shanghai 201620, China.  
Contact: <junjiezhang2024@shisu.edu.cn>; ORCID: <https://orcid.org/0009-0004-8821-4018>.  
Repository: <https://github.com/williamjay1/rdia-rolling-storage>

## Permissions by material class

| Material | Distribution and reuse terms |
|---|---|
| Original research code and original documentation | MIT, subject to the root `LICENSE`. Identified third-party files retain their own licenses. |
| Copied Nature Figure QA scripts | Apache-2.0; retain the upstream license and notices in `scripts/nature_qa_v11/`. |
| Original study-generated aggregate results, model coefficients and original figure artwork | The author permits copying, use, redistribution and adaptation for any purpose with accurate attribution to this version of the study and preservation of provenance. This grant covers the author's original contribution only; it does not relicense incorporated third-party material. |
| Supplied aligned Australian forecast and price inputs, source-derived fixture values, and Australian trajectories | Reuse is governed by [AEMO copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions). Preserve accurate attribution to AEMO and the relevant source material, with the supplied source/version information and processing description. Do not represent these data as uniformly MIT or Creative Commons licensed. |
| IESO source data, Ontario aligned price/forecast inputs, and full price-recoverable Ontario trajectories | **Not redistributed** in the repository or Release attachments. Download from the official sources and reconstruct locally, following [IESO Terms of Use](https://www.ieso.ca/terms-of-use) and any applicable supplemental terms. |
| Ontario aggregate outcomes and researcher-generated coefficients | Only the original, non-source-level study outputs are supplied. The original-contribution permission above applies; it does not confer IESO source-data redistribution or sublicensing rights. |
| Material with unclear or unverified redistribution rights | Not supplied as data. Official links and reconstruction instructions are used instead. |

The additive v13 common-state and stricter-clock Australian inputs/outputs
listed in `provenance/v13_action_files.json` retain the same material-class
permissions and AEMO attribution boundary. Raw monthly ZIPs and unpublished
TeX writing fragments are excluded.

The large Australian-data ZIP attached to a GitHub Release has the **same source-dependent terms** as the material it contains. Compression, alignment, optimization or relocation to Zenodo does not remove the underlying source rights. Source archives are not assigned a new project license.

## Attribution and provenance

When using the released material:

1. Cite the author, software title and exact released version using [CITATION.cff](CITATION.cff). If the author later deposits that version in Zenodo, cite its actual version DOI; do not invent a DOI or reuse the manuscript's DOI as a dataset/software identifier.
2. Attribute Australian market source material to the **Australian Energy Market Operator (AEMO)** and identify the relevant public report/archive and periods. Distinguish source prices and forecasts from this project's alignment, forecast transformations, controller decisions and computed outcomes.
3. Retain supplied provenance and checksums when redistributing processed inputs. Describe further changes and preserve the applicable source notices.
4. Obtain Ontario sources directly from IESO using the recorded versions in [provenance/ieso_source_versions.json](provenance/ieso_source_versions.json). Follow the full required IESO reproduction notice and all applicable terms. This package does not grant a license on IESO's behalf.

For publisher-related expectations, [Annals of Operations Research's data policy](https://link.springer.com/journal/10479/submission-guidelines) requires an original-research data availability statement, encourages public repositories and requires authors to possess necessary sharing rights. It does not force a single license for all third-party material. The download-and-rebuild route documents a rights-conscious availability boundary.

## Zenodo metadata

Do not apply Zenodo's default **CC BY 4.0** indiscriminately to the entire mixed package. Declare the licenses and source terms by material class. Where offered, **Other (Open)** can describe the mixed-content record together with this notice; Zenodo also supports **Add custom** and multiple license declarations. Name a custom notice “RDIA mixed licenses and source terms” and link it to this version of `DATA_LICENSE.md`. List MIT for original software and Apache-2.0 for the identified third-party scripts, while retaining AEMO's source conditions. Consult the [official license-field instructions](https://help.zenodo.org/docs/deposit/describe-records/licenses/).

These terms describe the released materials and the author's permissions, not permission from a data provider beyond its published terms. No IESO raw or price-recoverable inputs are included. The same material-class terms apply to the v14–v16 addenda and the complete v1.1.0 and v1.1.1 attachments. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source links and software provenance. Provider terms were rechecked on **2026-10-03**.
