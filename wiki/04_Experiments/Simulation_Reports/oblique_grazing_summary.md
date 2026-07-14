# Grazing Validation Summary

StackRT simulation was not executed.

Reason: 'Failed to put variable'

The new workflow code has been generated. Run the commands below after lumapi is available.

```powershell
python "work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/run_grazing_validation.py"
python "work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/main_grazing_stackrt.py"
python "work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/solve_grazing_asd.py" --input "<grazing_stackrt_npz>"
python "work/01_simulation_models/01_Lumerical_Workflow/oblique incidence/simulate_grazing_triangulation.py" --input "<grazing_asd_npz>"
```
