# VariPath

Research code for **VariPath: Variational Transport Paths with Saddle Anchors for Molecular Free Energy Estimation**.

**The same correction can help on one route and hurt on another.**
VariPath jointly learns spline transport routes and saddle-anchored local fields, using exact finite-map weights to estimate molecular free energies.

## Installation

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The direct dependencies are pinned in `pyproject.toml` and `requirements.txt`. Use `requirements-lock.txt` for the complete pinned environment. Computation uses float64.

## Molecular inputs

| System | Endpoint transformation | Force field | Atoms / shape dimensions |
|---|---|---|---|
| ALDP | 550 → 300 K | ff14SB / GBn2 + ACE | 22 / 48 |
| Ac-A4-NMe | GB polar coupling s = λ², from 0 to 1 at 300 K | AMBER99SB-ILDN / OBC1 | 52 / 123 |

Both systems use the all-L conditional ensemble, hydrogen-bond-length constraints and the full constrained mass-volume measure. Overall rotation is retained. Topology, force-field definitions and initial structures are bundled in [assets](src/varipath/assets).

## Quick start

Run the analytical example and controls:

```bash
varipath smoke --output outputs/example
varipath analytical --output outputs/analytical
```

Run a molecular example:

```bash
varipath doctor configs/aldp-example.json
varipath prepare configs/aldp-example.json --output outputs/ALDP/prepared
varipath run configs/aldp-example.json --preparation outputs/ALDP/prepared \
  --method E-MBAR --replica 0 --output outputs/ALDP/E-MBAR/0
```

Add `--resume` to continue the same run. Checkpoints bind the configuration, physical system, input pools and random state.


```bash
pip install -e '.[cuda]'
varipath doctor configs/ala4-example.json
varipath prepare configs/ala4-example.json --output outputs/ALA4/prepared
varipath endpoint-md configs/ala4-example.json --preparation outputs/ALA4/prepared \
  --replica 0 --output outputs/ALA4/endpoint-MD/0
```

The CUDA extra installs OpenMM's CUDA 12 backend and requires an NVIDIA GPU with a compatible driver. This example uses double-precision GPU proposals with CPU preparation and Reference-platform energy correction.

## Methods and experiments

| Method | Estimator |
|---|---|
| V | VariPath with saddle-anchored fields |
| V0 | Spline transport without local fields |
| G | Fixed generic local fields |
| MG | Local fields propagated by a pretrained base flow |
| FEAT | Stochastic forward/reverse paths and NEBAR |
| MNF-IC | Periodic peptide maps, BD/BAR/HD training and bidirectional BAR |
| E-MBAR | Energy-window sampling and MBAR |
| EB-MBAR | Energy and conformational bias windows with MBAR reweighting |

## Tests

```bash
python -m pytest -q
```

The tests cover molecular energies and measures, finite-map Jacobians, training gradients, route interventions, native baseline estimators, independent random streams, recovery and failure handling.

## Structure

```text
configs/          two minimal molecular examples
src/varipath/     implementation, bundled molecular inputs and baseline adapters
tests/            mathematical, physical and workflow tests
```

## Acknowledgements

VariPath uses [PyTorch](https://pytorch.org/), [OpenMM](https://openmm.org/), [ASE](https://ase-lib.org/) and [PyMBAR](https://pymbar.readthedocs.io/en/stable/). Adapted code and molecular inputs are credited in [NOTICE.md](NOTICE.md).

## License

[MIT License](LICENSE) — Use this however you want. Make it your own.
