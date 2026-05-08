import os
import sys
from materials import MaterialManager
from stack_builder import StackBuilder
from simulation import SimulationEngine
from analysis import ResultAnalyzer
from export import DataExporter

def main():
    print("Initializing Lumerical STACK Simulation...")
    
    # Initialize components
    mm = MaterialManager()
    sb = StackBuilder(mm)
    sim = SimulationEngine(mm)
    analyzer = ResultAnalyzer()
    exporter = DataExporter()

    # Get stack configurations
    stacks = sb.get_stacks()
    results = {}

    try:
        # Start Lumerical session
        sim.start_session()
        
        # Run simulations for each stack
        for stack_config in stacks:
            name = stack_config["name"]
            print(f"Running simulation for stack: {name}...")
            res = sim.run_stack(stack_config)
            results[name] = res
            
        # Analysis phase
        print("Analyzing results...")
        analyzer.check_validity(results)
        summary = analyzer.compare_stacks(results)
        insights = analyzer.sensitivity_insight(summary)
        
        # Export phase
        print("Exporting data and plots...")
        exporter.export_csv(results)
        exporter.generate_plots(results)
        exporter.generate_report(summary, insights)
        
        print("Simulation task completed successfully.")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close Lumerical session
        sim.close_session()

if __name__ == "__main__":
    main()
