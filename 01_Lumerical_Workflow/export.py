import pandas as pd
import matplotlib.pyplot as plt
import os

class DataExporter:
    """
    Handles data export, plotting, and report generation.
    """
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def export_csv(self, results):
        """
        Exports wavelengths and reflectivity to CSV.
        """
        for name, data in results.items():
            df = pd.DataFrame({
                "Wavelength_nm": data["wavelengths"].flatten(),
                "Reflection": data["R"].flatten()
            })
            file_path = os.path.join(self.output_dir, f"{name}_reflection.csv")
            df.to_csv(file_path, index=False)
            print(f"Exported {file_path}")

    def generate_plots(self, results):
        """
        Generates comparison plots and zoomed-in fringe plots.
        """
        # A. Reflection vs Wavelength (All)
        plt.figure(figsize=(10, 6))
        for name, data in results.items():
            plt.plot(data["wavelengths"].flatten(), data["R"].flatten(), label=name)
        plt.title("Reflection vs Wavelength (All Stacks)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflection")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "all_stacks_reflection.png"))
        plt.close()

        # B. Zoomed-in plot for fringes (500-500.5nm)
        plt.figure(figsize=(10, 6))
        for name, data in results.items():
            w = data["wavelengths"].flatten()
            r = data["R"].flatten()
            mask = (w >= 500) & (w <= 500.5)
            plt.plot(w[mask], r[mask], label=name)
        plt.title("Fringe Detail (Zoom 500-500.5nm) - 2mm Cavity")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflection")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "fringe_zoom_500nm.png"))
        plt.close()

        # C. PSS vs Cr comparison (fixed TiO2) - using zoomed window for clarity
        if "PSS_TiO2" in results and "Cr_TiO2" in results:
            plt.figure(figsize=(10, 6))
            w = results["PSS_TiO2"]["wavelengths"].flatten()
            mask = (w >= 590) & (w <= 610)
            plt.plot(w[mask], results["PSS_TiO2"]["R"].flatten()[mask], label="PSS_TiO2")
            plt.plot(w[mask], results["Cr_TiO2"]["R"].flatten()[mask], label="Cr_TiO2")
            plt.title("Fringe Comparison: PSS vs Cr (with TiO2)")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflection")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "PSS_vs_Cr_comparison.png"))
            plt.close()

    def generate_report(self, summary, insights):
        """
        Generates a Markdown report.
        """
        report_path = os.path.join(self.output_dir, "Simulation_Report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Lumerical STACK Simulation Report\n\n")
            f.write("## 1. Model Overview\n")
            f.write("This report summarizes the 1D planar film stack optical modeling for focus and leveling research.\n")
            f.write("**New Feature**: 3-Surface Zemax Equivalent Model with 2mm & 1um cavities.\n\n")
            
            f.write("## 2. Summary Statistics & Fringe Analysis\n")
            f.write("| Stack Name | Avg R | Max R | Visibility | Fringe Period (nm) | Aliasing Risk | Peak Count |\n")
            f.write("|------------|-------|-------|------------|--------------------|---------------|------------|\n")
            for name, stats in summary.items():
                f.write(f"| {name} | {stats['avg_R']:.4f} | {stats['max_R']:.4f} | {stats['visibility']:.4f} | {stats['avg_period_nm']:.4f} | {stats['aliasing_risk']} | {stats['peak_count']} |\n")
            
            f.write("\n## 3. Key Analysis Insights\n")
            for insight in insights:
                f.write(f"- {insight}\n")
            
            f.write("\n## 4. Visualizations\n")
            f.write("### Broadband Spectrum\n")
            f.write("![All Stacks](all_stacks_reflection.png)\n\n")
            f.write("### High-Frequency Fringe Detail (500nm Zoom)\n")
            f.write("![Fringe Zoom](fringe_zoom_500nm.png)\n\n")
            
            f.write("\n## 5. Limitations & Future Work\n")
            f.write("- Model includes 2mm virtual cavity, requiring very high resolution (0.01nm).\n")
            f.write("- FDTD visualization of 2mm gap is scaled down to 2um to allow mesh generation.\n")
            
        print(f"Generated report: {report_path}")
