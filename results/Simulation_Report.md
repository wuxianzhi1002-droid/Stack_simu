# Lumerical STACK Simulation Report

## 1. Model Overview
This report summarizes the 1D planar film stack optical modeling for focus and leveling research.

### Layer Stack Schematic (Top to Bottom):
1. **HSQ (Imaging Layer)**: 30 nm
2. **Conductive Layer**: PSS (5 nm) or Cr (5 nm)
3. **SOC (Planarization Layer)**: 45 nm
4. **Hardmask**: TiO2 (20 nm) or HfO2 (20 nm)
5. **Copper (Reflector)**: 100 nm (Assumed)
6. **SiN (Substrate)**: Semi-infinite

## 2. Summary Statistics
| Stack Name | Avg Reflection | Max Reflection | Min Reflection | Oscillation Amp | Peak Count |
|------------|----------------|----------------|----------------|-----------------|------------|
| PSS_TiO2 | 0.5650 | 0.8948 | 0.1948 | 0.7000 | 0 |
| Cr_TiO2 | 0.0425 | 0.1722 | 0.0114 | 0.1609 | 1 |
| PSS_HfO2 | 0.6135 | 0.9061 | 0.2500 | 0.6561 | 0 |
| Cr_HfO2 | 0.0350 | 0.0748 | 0.0107 | 0.0641 | 1 |

## 3. Key Analysis Insights
- Conductive layer (PSS vs Cr) significantly shifts average reflectivity by 52.2%.
- Hardmask material has minor impact on average reflectivity.

## 4. Visualizations
![All Stacks](all_stacks_reflection.png)
![PSS vs Cr](PSS_vs_Cr_comparison.png)
![TiO2 vs HfO2](TiO2_vs_HfO2_comparison.png)

## 5. Limitations & Future Work
- Model is limited to 1D planar infinite layers.
- Materials HSQ, SOC, PSS use approximate dispersive models.
- Ultra-thin Cr (5nm) may show sensitive behavior depending on material model accuracy.
- Future work: FDTD for 3D structures, angle scans, and sensitivity analysis.
