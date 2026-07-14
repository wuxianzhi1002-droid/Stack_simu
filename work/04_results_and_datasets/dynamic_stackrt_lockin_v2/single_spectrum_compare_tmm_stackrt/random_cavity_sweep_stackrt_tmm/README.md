# Random cavity sweep: StackRT vs TMM

- Generator: `lumapi`
- Source NPZ files: `fresh lumapi StackRT calls`
- Samples: `30` random cavity lengths
- Random seed: `20260714`
- Requested cavity range for lumapi mode: `999.0-1001.0 um`
- Actual sampled range: `999.079740984-1000.959024227 um`
- Wavelength range: `0.2-0.6 um`
- Spectral resolution: `0.02 nm`
- Wavelength points: `20001`
- Mean MAE: `8.24161206552e-13`
- Maximum MAE: `1.17970311016e-12`
- Maximum absolute error over all spectra: `2.28666530155e-11`
- Minimum Pearson correlation: `1`

## Files

- `random_cavity_sweep_metrics.csv`: one row of matching metrics per random cavity length
- `random_cavity_sweep_spectra.npz`: cavity lengths, StackRT/TMM spectra, and residual matrix
- `random_cavity_sweep_matching_metrics.png`: MAE, RMSE, maximum error, and correlation comparison
- `random_cavity_sweep_residual_heatmap.png`: residual over wavelength and cavity length
- `random_cavity_sweep_worst_case.png`: worst-MAE spectrum and residual
- `random_cavity_sweep_summary.json`: configuration and aggregate metrics
