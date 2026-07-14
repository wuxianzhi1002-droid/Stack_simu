# Single spectrum comparison: TMM vs StackRT

## Selection

- Source NPZ: `dynamic_spectra_20260708_112955.npz`
- Time index: `0`
- Time: `0 s`
- Air cavity: `1000 um`
- Spectrum: p polarization (`Rp`), normal incidence
- Stack: `RefReflector / Air / HSQ / PSS / SOC / TiO2 / Cu`
- Thicknesses: Air `1000 um`, HSQ `40 nm`, PSS `5 nm`, SOC `50 nm`, TiO2 `20 nm`
- First and last layers are treated as semi-infinite media.

## Material model

The TMM uses exactly the refractive-index arrays defined in `main_dynamic_v2.py`:

- RefReflector: `5.8284`
- Air: `1.0`
- HSQ: `1.41`
- PSS: `1.50 + 0.05j`
- SOC: `1.55 + 0.005 / wavelength_um^2`
- TiO2: `2.4 + 0.02 / wavelength_um^2`
- Cu: `1.1 + 2.5j`

## Frequency and sign convention

`main_dynamic_v2.py` constructs the StackRT frequency vector as
`f = 3e8 / nominal_wavelength`. The matched TMM therefore uses the same
frequency and computes propagation phase with
`phase_wavelength = 299792458 / f`. Material arrays remain evaluated at the
nominal wavelength, matching the original construction of `n_matrix`.

For complex index `n + i*k`, the characteristic matrices use `-i` in the
off-diagonal terms. This convention reproduces StackRT attenuation and phase.
Using the opposite sign or directly using nominal wavelength for propagation
produces a large artificial mismatch for the 1 mm cavity.

## Metrics

- MAE: `7.05560645021e-13`
- RMSE: `1.43549384711e-12`
- Maximum absolute error: `1.29316835018e-11` at nominal wavelength `225.081254 nm`
- Pearson correlation: `1`

## Zoom plot

- Center wavelength: `500 nm`
- Requested span: `40 nm`
- Actual plotted range: `480-520 nm`
- The zoom figure also includes a central 1 nm detail panel so individual fringes remain visible for the 1 mm cavity.

## Files

- `single_spectrum_compare_tmm_stackrt.png`: full-range overlay and residual plot
- `single_spectrum_compare_tmm_stackrt_zoom.png`: configurable zoom, central 1 nm detail, and zoom residual
- `single_spectrum_compare_tmm_stackrt.csv`: per-wavelength comparison table
- `single_spectrum_compare_tmm_stackrt.npz`: compact numerical arrays and metadata
- `comparison_metrics.json`: parameters and scalar metrics
- `compare_single_spectrum.py`: reproducible comparison script
