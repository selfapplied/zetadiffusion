"""
energy.py

Implements the Topological Energy Extraction Hamiltonian.
Models the transformation of geometric stress (Ricci curvature) 
into computational work via topological surgery (genus change).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ManifoldState:
    """Represents the state of the Informational Manifold."""
    curvature_grid: np.ndarray
    viscosity_gamma: float
    current_genus: int = 0
    
    @property
    def potential_energy(self) -> float:
        """
        V ~ alpha * integral(|R|^2)
        """
        # Simplified summation over the grid
        alpha = 1.0 # Elastic modulus
        return 0.5 * alpha * np.sum(self.curvature_grid**2)

    def reset_curvature(self):
        """Relaxes the manifold to flatness after a shock."""
        self.curvature_grid = np.zeros_like(self.curvature_grid)

class TopologicalHarvester:
    """
    The Engine: Transforms Shock Waves into Work.
    H_extract = -eta * d(chi)/dt * Phi_harvest
    """
    def __init__(self, efficiency: float = 0.8):
        self.efficiency = efficiency
        self.accumulated_work = 0.0
        self.shock_history: List[Dict] = []
        
    def process_event(self, state: ManifoldState, mach_number: float) -> Dict:
        """
        Evaluates the manifold state for shock conditions and harvests energy.
        
        Args:
            state: Current ManifoldState.
            mach_number: Geometric Mach number (ratio of flow velocity to sound speed).
            
        Returns:
            Event report dictionary.
        """
        potential_V = state.potential_energy
        
        # Shock Condition: Mach > 1.0 implies supersonic flow -> shock formation
        if mach_number > 1.0:
            # --- THE SNAP ---
            # 1. Calculate Energy Release
            # In this model, the shock releases the stored potential energy
            release_E = potential_V
            
            # 2. Harvest Work
            captured_work = self.efficiency * release_E
            entropy_loss = release_E - captured_work
            
            # 3. Update Engine State
            self.accumulated_work += captured_work
            
            # 4. Topological Surgery (Genus Jump)
            # Each shock adds a handle to the manifold
            old_genus = state.current_genus
            state.current_genus += 1
            
            # 5. Relaxation (Cooling)
            state.reset_curvature()
            
            event = {
                "status": "SHOCK_HARVESTED",
                "mach": mach_number,
                "energy_released": release_E,
                "work_captured": captured_work,
                "entropy_loss": entropy_loss,
                "topology_change": f"g{old_genus}->g{state.current_genus}"
            }
            self.shock_history.append(event)
            return event
            
        else:
            # --- LOADING PHASE ---
            return {
                "status": "LOADING",
                "mach": mach_number,
                "stored_potential": potential_V
            }

def run_cycle(n_steps=50):
    """
    Simulates a loading -> shock -> harvest cycle.
    """
    # Initialize flat manifold
    grid_size = 100
    state = ManifoldState(
        curvature_grid=np.zeros(grid_size),
        viscosity_gamma=0.5
    )
    harvester = TopologicalHarvester(efficiency=0.85)
    
    print(f"{'Step':<5} | {'Mach':<6} | {'Potential':<10} | {'Status':<15} | {'Accumulated Work'}")
    print("-" * 70)
    
    # Simulation loop
    for t in range(n_steps):
        # 1. Driver: Chaos injects curvature
        # Stress builds up linearly or exponentially
        injection = np.random.normal(0.1, 0.05, grid_size) * (t % 15) # Periodic loading
        state.curvature_grid += injection
        
        # 2. Calculate Mach number
        # Mach number proportional to total curvature stress?
        # Let's assume M ~ sqrt(Potential) / c_s
        mach = np.sqrt(state.potential_energy) / 5.0 
        
        # 3. Process
        report = harvester.process_event(state, mach)
        
        # Output trace
        status = report['status']
        work = harvester.accumulated_work
        pot = state.potential_energy
        
        if status == "SHOCK_HARVESTED":
            print(f"{t:<5} | {mach:<6.2f} | {report['energy_released']:<10.2f} | {status:<15} | {work:.2f} (+{report['work_captured']:.2f})")
        elif t % 5 == 0:
             print(f"{t:<5} | {mach:<6.2f} | {pot:<10.2f} | {status:<15} | {work:.2f}")

if __name__ == "__main__":
    run_cycle()

