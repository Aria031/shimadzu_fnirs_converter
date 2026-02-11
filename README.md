[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18607367.svg)](https://doi.org/10.5281/zenodo.18607367)

### shimadzu-fnirs-converter

**Shimadzu fNIRS TXT → SNIRF (.snirf) & MNE FIF (.fif)**

Convert Shimadzu proprietary TXT exports into open neuroimaging formats for modern, reproducible pipelines.

Shimadzu raw files are not directly compatible with most analysis toolkits.  
This package converts them into **SNIRF (standard)** and **MNE FIF (native)** formats for seamless use in Python/MATLAB ecosystems.

Compatible with:

- MNE-Python
- Homer3
- NIRS-KIT
- Brainstorm
- any SNIRF workflow

---

### Features

- TXT → SNIRF
- TXT → MNE FIF (native Raw object)
- Optode/channel geometry parsing
- Origin / others coordinate support
- Event & trigger integration
- Python API + CLI
- Batch processing
- NumPy 2 compatible
- pytest tested
- Zenodo DOI archived releases

---

### Installation

### From PyPI (recommended)

```bash
pip install shimadzu-fnirs-converter

### Development install
```bash
git clone https://github.com/Aria031/shimadzu-fnirs-converter
cd shimadzu-fnirs-converter
pip install -e .
```

### Quick Start
### Python API
### Save both FIF + SNIRF
```python
from shimadzu_fnirs_converter import convert

convert(
    txt_path="raw_data.TXT",
    origin_path="origin.csv",
    others_path="others.csv",
    out_fif="sub01_raw.fif",
    out_snirf="sub01_raw.snirf",
    subject="sub-01",
)

```
### Save only FIF (MNE)
```python
from shimadzu_fnirs_converter import convert_fif

convert_fif(
    txt_path="raw_data.TXT",
    origin_path="origin.csv",
    others_path="others.csv",
    out_fif="sub01_raw.fif",
    subject="sub-01",
)
```
### Save only SNIRF
```python
from shimadzu_fnirs_converter import convert_snirf

convert_snirf(
    txt_path="raw_data.TXT",
    origin_path="origin.csv",
    others_path="others.csv",
    out_snirf="sub01_raw.snirf",
    subject="sub-01",
)
```
### Batch conversion
```python
from shimadzu_fnirs_converter import convert_batch

jobs = [
    dict(
        txt_path="sub01.TXT",
        origin_path="origin.csv",
        others_path="others.csv",
        out_fif="sub01.fif",
        subject="sub-01",
    ),
    dict(
        txt_path="sub02.TXT",
        origin_path="origin.csv",
        others_path="others.csv",
        out_fif="sub02.fif",
        subject="sub-02",
    ),
]

convert_batch(jobs)
```

### Command Line
```bash
shimadzu-fnirs-converter --help
```
Example:

```bash
shimadzu-fnirs-converter \
  --txt sub01_run1.TXT \
  --origin optodes_origin.csv \
  --others optodes_others.csv \
  --out sub01_raw.fif \
  --snirf sub01_raw.snirf \
  --subject sub-01
```

### Requirements
- Python ≥ 3.9
- numpy ≥ 2.0
- pandas ≥ 2.0
- h5py ≥ 3.10
- mne ≥ 1.6

### Citation
```bibtex
@software{dong2026shimadzu,
  author  = {Dong, Jiaran and Zhang, Jingyan and Feng, Chen},
  title   = {shimadzu-fnirs-converter: Shimadzu fNIRS TXT to SNIRF/FIF converter},
  year    = {2026},
  doi     = {10.5281/zenodo.18605987},
  url     = {https://doi.org/10.5281/zenodo.18605987}
}

```
### License
MIT License

### Authors
Jiaran Dong, Jingyan Zhang, Chen Feng
