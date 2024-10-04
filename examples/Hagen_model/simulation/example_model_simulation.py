import os
import sys

# ncpi toolbox
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ncpi

if __name__ == "__main__":
    # Create a Simulation object
    sim = ncpi.Simulation(param_folder='params', python_folder='python', output_folder='output')

    # Run the network, simulation and analysis scripts
    sim.network('network.py', 'network_params.py')
    sim.simulate('simulation.py', 'simulation_params.py')
    sim.analysis('analysis.py', 'analysis_params.py')
