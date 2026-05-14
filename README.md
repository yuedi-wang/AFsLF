# AFsLF

Official implementation of **AFsLF (Autofocus for Scanning Light Field Microscopy)**, a real-time robust autofocus method for sustained intravital scanning light field imaging.

AFsLF estimates the defocus distance from the disparity between the central view and an edge angular view in scanning light field microscopy (sLFM), and supports focal plane correction using a motorized translation stage.

---

## Code Availability

All relevant source code for AFsLF is available at:

```text
https://github.com/yuedi-wang/AFsLF
```

This software is released under the **GNU General Public License v2.0 (GPL-2.0-only)**. See the `LICENSE` file for details.

---

## Data Availability


Source Data associated with the manuscript are provided in `source-data.xlsx` and are additionally deposited at Zenodo:

```text
https://doi.org/10.5281/zenodo.19660029
```

---

## System Requirements

### Operating System

The released Python code can be run on a standard desktop or workstation environment.

Tested operating system:

```text
 Windows 11
```

### Python Version

```text
Python = 3.8
```

### Dependencies

Install the required Python packages with:

```bash
pip install numpy tifffile opencv-python
```

The released script also uses the Python standard library module `pickle`, which does not require separate installation.


### Hardware

The offline Python demo does not require a GPU.

The AFsLF evaluation reported in the manuscript was performed on a desktop computer with:

```text
CPU: Intel i9-10940X
RAM: 256 GB
GPU: NVIDIA GeForce RTX 3090
```

For real-time autofocus correction on newly acquired data, an sLFM/csLFM/RUSH3D imaging system and a motorized translation stage are required.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yuedi-wang/AFsLF.git
cd AFsLF
```

Create a Python environment:

```bash
conda create -n afslf python=3.8
conda activate afslf
```

Install dependencies:

```bash
pip install numpy tifffile opencv-python
```

Typical installation time on a normal desktop computer:

```text
approximately 2 hours
```

---

## Demo

### Required Files

Before running the demo, make sure the following files are available:

```text
autofocus_valid.py
data/
undistort_params_dict_points_240620.pkl
```


### Run the Demo

Run:

```bash
python autofocus_valid.py
```

The current script contains a dataset-specific input path. If needed, update the following line in `autofocus_valid.py`:

```python
raw_path = f"Y:/C2/B18_{i}.tiff"
```

to the path of the provided demo data or your own TIFF data.

### Expected Output

The script estimates frame-wise defocus values and saves them to:

```text
output_fft.txt
```

During execution, the script prints the estimated image shift and defocus value for each frame.

Expected runtime for the demo:

```text
approximately 0.1 s
```

---

## Instructions for Use on New Data

To run AFsLF on new sLFM data, prepare:

```text
1. TIFF image sequence or TIFF stack from an sLFM/csLFM/RUSH3D system
2. Pre-calibrated undistortion parameter file
3. System-specific parameters
```

In `autofocus_valid.py`, update the system-specific parameters, including:

```python
H, W = 10748, 14304
centerX, centerY = 7151, 5373
crop_H, crop_W = 1995, 1995
kk = 1
```

Also update the input data path:

```python
raw_path = "path/to/your/tiff/data"
```

The released code performs the following main steps:

```text
1. Load pre-calibrated geometric correction parameters.
2. Extract the selected light field region.
3. Generate two angular-view images.
4. Estimate the translational shift using ECC-based image registration.
5. Convert the x-direction shift into a defocus value.
6. Save the estimated defocus sequence to output_fft.txt.
```

---

## Reproduction Instructions

To reproduce the quantitative results reported in the manuscript:

```text
1. Clone this repository.
2. Install the required Python dependencies.
3. Place the required calibration parameter file in the repository root directory.
4. Prepare the TIFF image sequence and update the input path in autofocus_valid.py.
5. Run python autofocus_valid.py.
6. Use output_fft.txt together with source-data.xlsx to reproduce the corresponding quantitative analyses and plots.
```



---

## License

This software is licensed under the **GNU General Public License v2.0 (GPL-2.0-only)**.

Recommended SPDX identifier:

```text
SPDX-License-Identifier: GPL-2.0-only
```


