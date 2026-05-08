import numpy as np

class ResultAnalyzer:
    """
    Analyzes simulation results and performs validity checks.
    """
    def __init__(self):
        pass

    def check_validity(self, results):
        """
        Checks if R and T are physically reasonable.
        """
        for name, data in results.items():
            R = data["R"]
            if np.any(R > 1.0) or np.any(R < 0):
                print(f"Warning: Physical anomaly in {name}: R out of [0, 1] range.")
            if np.any(np.isnan(R)):
                print(f"Warning: NaN detected in {name} results.")

    def analyze_oscillation(self, wavelengths, R):
        """
        Calculates oscillation amplitude and basic periodicity.
        """
        R = R.flatten()
        amplitude = np.max(R) - np.min(R)
        # Simple periodicity check: number of peaks
        peaks = 0
        for i in range(1, len(R) - 1):
            if R[i] > R[i-1] and R[i] > R[i+1]:
                peaks += 1
        return amplitude, peaks

    def compare_stacks(self, results):
        """
        Compares different stacks to identify major contributors to reflection change.
        """
        summary = {}
        for name, data in results.items():
            amp, peaks = self.analyze_oscillation(data["wavelengths"], data["R"])
            summary[name] = {
                "avg_R": np.mean(data["R"]),
                "max_R": np.max(data["R"]),
                "min_R": np.min(data["R"]),
                "oscillation_amplitude": amp,
                "peak_count": peaks
            }
        return summary

    def sensitivity_insight(self, summary):
        """
        Provides qualitative insights based on summary statistics.
        """
        # Compare PSS vs Cr
        # Compare TiO2 vs HfO2
        insights = []
        
        # This is a bit hard-coded based on stack names defined in stack_builder
        if "PSS_TiO2" in summary and "Cr_TiO2" in summary:
            diff = summary["PSS_TiO2"]["avg_R"] - summary["Cr_TiO2"]["avg_R"]
            if abs(diff) > 0.05:
                insights.append(f"Conductive layer (PSS vs Cr) significantly shifts average reflectivity by {diff*100:.1f}%.")
            else:
                insights.append("Conductive layer type has minor impact on average reflectivity.")

        if "PSS_TiO2" in summary and "PSS_HfO2" in summary:
            diff = summary["PSS_TiO2"]["avg_R"] - summary["PSS_HfO2"]["avg_R"]
            if abs(diff) > 0.05:
                insights.append(f"Hardmask material (TiO2 vs HfO2) significantly shifts average reflectivity by {diff*100:.1f}%.")
            else:
                insights.append("Hardmask material has minor impact on average reflectivity.")
                
        return insights
