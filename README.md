# Antenna-Array

Python code accompanying the paper

**V. Rabinovich, "Adaptive Interference Nulling with a Single-Receiver Phased Array"** (2026).

The paper describes an adaptive phased array with passive weighting units and a single receiver. The array correlation coefficients are not measured directly. Instead, they are estimated from output-power readings taken at the different states of 2-bit phase shifters (0°, 90°, 180°, 270°). From these estimates the processor computes the weights that place nulls on the interferers.

## Contents

| File | Description |
|---|---|
| `single_receiver_nulling_examples.py` | Numerical examples of Section 5 of the paper (cases A–E, Table 1, Figs. 4 and 5) |
| `adaptive_array_patterns.py` | Adaptive pattern for a linear, planar (M × M) or circular (concentric rings) array with any number of interferers |

## What the program does

For each case (linear array N = 4 and planar 3 × 3 array, one to four interferers) the program:

1. simulates the K = 1 + 3N + 9N(N−1)/2 output-power readings, Eq. (4.3);
2. estimates the correlation coefficients R̂ᵢₖ by Eqs. (4.6)–(4.9);
3. solves the linear system (4.11) and scales the weights by Eq. (4.12);
4. computes the null depths (4.17), the SINR before (4.18) and after (4.16) adaptation, the loss relative to SINR₀ (4.19) and the adaptation gain.

It prints the results of each case together with the adaptive weights wₙ (magnitude and phase) and saves two figures:

- `fig4_case_B.png`: pattern of the linear array, case B (Fig. 4 of the paper);
- `fig5_case_E.png`: pattern cuts of the planar array, case E (Fig. 5 of the paper).

The figures are saved in the same folder as the program.

## Pattern program for other arrays

`adaptive_array_patterns.py` applies the same processing to an array chosen in the SETTINGS block at the top of the file:

- `GEOMETRY = "linear"`: `N_LINEAR` elements with spacing `SPACING`;
- `GEOMETRY = "planar"`: `M_PLANAR` × `M_PLANAR` elements;
- `GEOMETRY = "circular"`: concentric rings `RINGS = [(radius, number of elements), ...]`, optionally with a center element.

The number of interferers `J` is set there too; their directions are random (seed `SEED`) or given explicitly in `INTERFERERS`. The program prints the null depths, the SINR before and after adaptation and the weights, and saves `array_geometry.png`, `pattern_linear.png` (linear array) or `pattern_map.png` and `pattern_cuts.png` (planar and circular arrays). The dominant mode rejection (DMR) correction of the estimated matrix can be switched on or off with `USE_DMR`.

## Requirements

- Python 3.8 or newer
- numpy
- matplotlib

Install the libraries with

```
pip install numpy matplotlib
```

## Usage

```
python single_receiver_nulling_examples.py
python adaptive_array_patterns.py
```

## Assumptions

The examples assume ideal conditions: identical uncoupled isotropic elements at λ/2 spacing, ideal 2-bit phase shifters, passive lossless channels, noise only in the receiver, a broadside desired signal with SINR₀ = 10 dB, and uncorrelated interferers.

## Citation

If you use this code, please cite the paper above.

## Author

Victor Rabinovich, Independent Researcher, Toronto, ON, Canada
