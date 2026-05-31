# Material Property References - Lumerical STACK Project

This document tracks the sources and assumptions for the material refractive indices (n, k) used in the simulation.

## 1. HSQ (Hydrogen Silsesquioxane)
- **Model**: Cauchy Approximation ($n = A + B/\lambda^2$)
- **Parameters**: n ~ 1.41, k ~ 0 (Visible range 400-800nm)
- **Reference**: 
  - Dow Corning XR-1541 HSQ technical datasheet.
  - "Refractive index of HSQ thin films", Journal of Vacuum Science & Technology B, 2000s.
- **Notes**: HSQ has very low dispersion in the visible spectrum, hence the initial constant approximation. A small Cauchy B coefficient (~0.003) can be added for higher precision.

## 2. PSS (Polystyrene Sulfonate / PEDOT:PSS)
- **Model**: Constant or Drude-Lorentz (Simplified)
- **Parameters**: n ~ 1.50, k ~ 0.05
- **Reference**: 
  - "Optical properties of PEDOT:PSS conductive polymer films", Thin Solid Films.
  - Typical values used in e-beam lithography conductive top-coat modeling (e.g., E-spacer or AR-PC 5090).
- **Notes**: PSS acts as a conductive discharge layer. Its index is dominated by the polymer matrix (~1.5).

## 3. SOC (Spin-On Carbon)
- **Model**: Cauchy Approximation
- **Parameters**: n = 1.55 + 0.005 / lambda^2 (um)
- **Reference**: 
  - Common industrial standards for hardmask materials (e.g., Brewer Science or Shin-Etsu SOC types).
  - High-carbon content films typically range from 1.5 to 1.8.

## 4. TiO2 (Titanium Dioxide)
- **Model**: Cauchy Approximation
- **Parameters**: n = 2.4 + 0.02 / lambda^2 (um)
- **Reference**: 
  - Siefke et al., "Materials for the visual spectrum: Titanium Dioxide", 2016.
  - Palik, "Handbook of Optical Constants of Solids".

## 5. HfO2 (Hafnium Dioxide)
- **Model**: Cauchy Approximation
- **Parameters**: n = 2.0 + 0.015 / lambda^2 (um)
- **Reference**: 
  - Al-Kuhaili, "Optical properties of hafnium oxide thin films", 2004.
  - Wood et al., "Evaluation of HfO2 as a high-k dielectric material".

## 6. Substrate / Background (Cu, Si3N4, Cr)
- **Source**: Lumerical Built-in Material Database (CRC, Palik, or Johnson & Christy).
