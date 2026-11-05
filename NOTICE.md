# Third-party code and molecular inputs

| Component | Attribution | Included notice |
|---|---|---|
| FEAT networks and losses | [FEAT](https://github.com/jiajunhe98/FEAT/tree/0fb77e1acee9c3806e6c9aa3fce6ae3e1745ba84), commit `0fb77e1acee9c3806e6c9aa3fce6ae3e1745ba84` | [MIT](src/varipath/baselines/FEAT-LICENSE) |
| Constrained velocity Verlet and ALDP starting structure | [openmmtools](https://github.com/choderalab/openmmtools/blob/a42cbdc862a0a4fe996a579dd1b04611ce9916a4/openmmtools/integrators.py), commit `a42cbdc862a0a4fe996a579dd1b04611ce9916a4` | [MIT](src/varipath/LICENSE.openmmtools) |
| Ac-A4-NMe topology and starting structures | [Wales PATHSAMPLE tetra-alanine catalogue](https://github.com/wales-group/examples/tree/5e07a80b9761a032cf8d97768e0e0bc1fd6cea12/tetra_ALA/05_Expanding_PATHSAMPLE_database/expected_output), minima 0, 14, 28 and 42 | Molecular input data |

Force-field parameters are serialized from OpenMM's ff14SB/GBn2+ACE and AMBER99SB-ILDN/OBC1 definitions. Molecular input files are in [assets](src/varipath/assets).

MNF-IC implements periodic peptide maps and the BD/BAR/HD objectives described in the [MNF author archive](https://doi.org/10.34770/nrpq-qk33). MBAR estimation uses [PyMBAR](https://pymbar.readthedocs.io/en/stable/).
