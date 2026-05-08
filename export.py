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
        Generates comparison plots.
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

        # C. PSS vs Cr comparison (fixed TiO2)
        if "PSS_TiO2" in results and "Cr_TiO2" in results:
            plt.figure(figsize=(10, 6))
            plt.plot(results["PSS_TiO2"]["wavelengths"].flatten(), results["PSS_TiO2"]["R"].flatten(), label="PSS_TiO2")
            plt.plot(results["Cr_TiO2"]["wavelengths"].flatten(), results["Cr_TiO2"]["R"].flatten(), label="Cr_TiO2")
            plt.title("Conductive Layer Comparison: PSS vs Cr (with TiO2)")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflection")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "PSS_vs_Cr_comparison.png"))
            plt.close()

        # D. TiO2 vs HfO2 comparison (fixed PSS)
        if "PSS_TiO2" in results and "PSS_HfO2" in results:
            plt.figure(figsize=(10, 6))
            plt.plot(results["PSS_TiO2"]["wavelengths"].flatten(), results["PSS_TiO2"]["R"].flatten(), label="PSS_TiO2")
            plt.plot(results["PSS_HfO2"]["wavelengths"].flatten(), results["PSS_HfO2"]["R"].flatten(), label="PSS_HfO2")
            plt.title("Hardmask Comparison: TiO2 vs HfO2 (with PSS)")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflection")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "TiO2_vs_HfO2_comparison.png"))
            plt.close()

    def generate_report(self, summary, insights):
        """
        Generates a Markdown report.
        """
        report_path = os.path.join(self.output_dir, "Simulation_Report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Lumerical STACK Simulation Report\n\n")
            f.write("## 1. Model Overview\n")
            f.write("This report summarizes the 1D planar film stack optical modeling for focus and leveling research.\n\n")
            
            f.write("## 2. Summary Statistics\n")
            f.write("| Stack Name | Avg Reflection | Max Reflection | Min Reflection | Oscillation Amp | Peak Count |\n")
            f.write("|------------|----------------|----------------|----------------|-----------------|------------|\n")
            for name, stats in summary.items():
                f.write(f"| {name} | {stats['avg_R']:.4f} | {stats['max_R']:.4f} | {stats['min_R']:.4f} | {stats['oscillation_amplitude']:.4f} | {stats['peak_count']} |\n")
            
            f.write("\n## 3. Key Analysis Insights\n")
            for insight in insights:
                f.write(f"- {insight}\n")
            
            f.write("\n## 4. Visualizations\n")
            f.write("![All Stacks](all_stacks_reflection.png)\n")
            f.write("![PSS vs Cr](PSS_vs_Cr_comparison.png)\n")
            f.write("![TiO2 vs HfO2](TiO2_vs_HfO2_comparison.png)\n")
            
            f.write("\n## 5. Limitations & Future Work\n")
            f.write("- Model is limited to 1D planar infinite layers.\n")
            f.write("- Materials HSQ, SOC, PSS use approximate dispersive models.\n")
            f.write("- Ultra-thin Cr (5nm) may show sensitive behavior depending on material model accuracy.\n")
            f.write("- Future work: FDTD for 3D structures, angle scans, and sensitivity analysis.\n")
            
        print(f"Generated report: {report_path}")
