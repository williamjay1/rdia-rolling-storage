# Third-party notices

Repository: <https://github.com/williamjay1/rdia-rolling-storage>  
Original research software sole author: **Junjie Zhang**, Shanghai Academy of Global Governance and Area Studies and School of Economics and Finance, Shanghai International Studies University, Shanghai 201620, China.  
Contact: <junjiezhang2024@shisu.edu.cn>; ORCID: <https://orcid.org/0009-0004-8821-4018>.

The root [MIT license](LICENSE) applies to the author's original software. It does not replace data-provider terms or the licenses of third-party software. See [DATA_LICENSE.md](DATA_LICENSE.md) for the scope of the data permissions.

## Australian market material: AEMO

The Australian electricity forecast and settlement-price material originates from the **Australian Energy Market Operator (AEMO)**, including the NEMWEB archive's public price reports. Publicly available material created by or on behalf of AEMO is subject to AEMO's stated copyright permissions: accurate and appropriate attribution to the relevant material and AEMO is required. AEMO excludes confidential documents and reports commissioned by another copyright owner from that general permission. The project does not assert a Creative Commons license over AEMO material.

- [AEMO copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions)
- [NEMWEB public reports and disclaimer](https://web.nemweb.com.au/)

For supplied aligned Australian inputs and trajectories, retain the source report, region, period, version or archive identifier, source checksum where recorded, and transformation description in the accompanying provenance. Attribute the source as AEMO; identify alignment, forecast representation construction and optimization outputs as this project's processing. Public access is not an assertion of AEMO endorsement of this work. The additive v13 action and stricter-clock selection in `provenance/v13_action_files.json` retains these attribution requirements; a new processing step does not remove provider rights. Raw monthly ZIP archives are not mirrored in the public release.

## Ontario market material: IESO

The Ontario source is the **Independent Electricity System Operator (IESO)**. Its [Terms of Use](https://www.ieso.ca/terms-of-use), under **Intellectual Property Rights**, describe a limited, non-exclusive, non-sublicensable and non-transferable permission for use and reproduction, with a prescribed notice; other uses can require prior written permission. Separate or supplemental terms may take precedence.

This release adopts a conservative distribution boundary. It does **not** include IESO source CSV files, aligned price/forecast inputs, or full Ontario period-by-period trajectories that expose or permit recovery of source prices. This boundary is a project packaging choice, not a claim that IESO forbids every form of reproduction. The package supplies source URLs, versions and checksums, reconstruction code, aggregate study outcomes and researcher-generated model coefficients. Users rebuilding Ontario inputs must obtain them from IESO and follow all applicable terms, including the full prescribed notice on reproductions. Retain [provenance/ieso_source_versions.json](provenance/ieso_source_versions.json).

## Figure workflow and quality-assurance sources

The following fixed versions were actually consulted or used. They are software provenance, not evidence for the study's scientific findings.

| Upstream project | Fixed commit and license | Actual use in this release |
|---|---|---|
| K-Dense-AI/scientific-agent-skills | [`91497e335489dcb544ec8ddc8f6b7ce5fd6d1121`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/91497e335489dcb544ec8ddc8f6b7ce5fd6d1121/skills/scientific-visualization); [MIT](https://github.com/K-Dense-AI/scientific-agent-skills/blob/91497e335489dcb544ec8ddc8f6b7ce5fd6d1121/LICENSE.md) | Scientific-visualization guidance was read and applied to the plotting workflow. The full skill library is not required to execute this research code. |
| garrettj403/SciencePlots | [`b9b16959570bd2fbc9ff5118bacc423c3bddd592`](https://github.com/garrettj403/SciencePlots/tree/b9b16959570bd2fbc9ff5118bacc423c3bddd592); [MIT](https://github.com/garrettj403/SciencePlots/blob/b9b16959570bd2fbc9ff5118bacc423c3bddd592/LICENSE) | `science.mplstyle` and `nature.mplstyle` were read as parameter references. The project uses explicit Matplotlib parameters and does not require installation of SciencePlots. |
| Yuan1z0825/nature-skills | [`84880815fb37317b3766bff2c2abba395b8993c3`](https://github.com/Yuan1z0825/nature-skills/tree/84880815fb37317b3766bff2c2abba395b8993c3/skills/nature-figure/scripts); [Apache-2.0](https://github.com/Yuan1z0825/nature-skills/blob/84880815fb37317b3766bff2c2abba395b8993c3/LICENSE) | The four upstream scripts `validate_figure.py`, `audit_pdf_text.py`, `audit_figure_collisions.py` and `audit_panel_alignment.py` are included under `scripts/nature_qa_v11/` with their upstream license. They were run for source, text, collision and panel-alignment checks. The project-specific audit wrapper is original software. |

For the legacy v11 exports, the upstream Nature-style font preset was adapted to a 160 mm canvas, with regular text at 9 pt, panel labels at 10 pt, axis lines at 0.55 pt and data lines normally at 1.1 pt. These seven earlier exports include vector PDF/SVG and native 1200 dpi PNG/TIFF. Their automatic checks do not establish scientific validity or journal acceptance. The six current 174 mm v16 figures are identified separately below.

Optional software citation: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://arxiv.org/abs/2609.00065), [DOI](https://doi.org/10.48550/arXiv.2609.00065). This is a **preprint** and is cited here only as software provenance.

Python dependencies retain their own licenses. Installing a dependency does not place that dependency under this project's MIT license. Official [Nature artwork guidance](https://www.nature.com/nature/for-authors/final-submission) and [Annals of Operations Research author instructions](https://link.springer.com/journal/10479/submission-guidelines) informed figure preparation; neither publisher endorses this repository.

Terms and documentation were checked on **2026-10-02**. Provider terms may change; retain the recorded versions and consult the linked terms before a new redistribution.

## Current v16 figure addendum

Earlier references above to seven figures and a 160 mm canvas describe the
legacy v11 exports. The current six v16 figures use a native 174 mm canvas,
embedded Arial lettering of at least 9 pt, native PDF/SVG and original 1200 dpi
PNG/TIFF when redrawn. Current vector exports are under `figures/v16/`.
The minimal redrawing package, source terms, pinned Nature QA Apache-2.0 notice
and source-array classifications are in `addenda/v16_figures/`. No proprietary
Arial font is redistributed. Frozen redraws are not new market experiments.
The package's author-created aggregate/synthetic figure material and any
AEMO-derived processed points retain their distinct source permissions.
