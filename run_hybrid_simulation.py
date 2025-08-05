#!/usr/bin/env python3
"""
Run Hybrid Cell Evolution Simulation
Demonstrates cells that evolve both biological traits and problem-solving abilities.
"""

from hybrid_population import run_hybrid_evolution_experiment
from environment import (create_benign_environment, create_competitive_environment, 
                        create_harsh_environment, Environment, SelectionPressure, SelectionType)
from genome import ComputationalGenome


def main():
    print("=" * 70)
    print("HYBRID CELL EVOLUTION SIMULATION")
    print("Cells evolve both biological traits and AI problem-solving abilities")
    print("=" * 70)
    
    # Create environment that rewards both efficiency and intelligence
    env = Environment("Intelligence-Favoring Environment")
    env.conditions.carrying_capacity = 500
    env.conditions.nutrient_density = 0.8
    
    # Add selection pressure for energy efficiency
    energy_pressure = SelectionPressure(
        name="energy_efficiency",
        target_parameter="energy_efficiency",
        selection_type=SelectionType.DIRECTIONAL,
        strength=0.5
    )
    env.add_selection_pressure(energy_pressure)
    
    # Note: Problem-solving ability is already selected for via energy rewards
    
    # Run the experiment
    print("\nStarting simulation...")
    print("- Cells solve addition problems every 10 ticks")
    print("- Correct answers give energy rewards")
    print("- Better problem solvers survive and reproduce more")
    print("- Watch as the population evolves higher accuracy!\n")
    
    population = run_hybrid_evolution_experiment(
        environment=env,
        max_time=1000.0,
        max_generations=100,
        snapshot_interval=100.0,
        problem_frequency=10
    )
    
    # Analyze evolution of intelligence
    print("\n" + "=" * 70)
    print("ANALYSIS: Evolution of Problem-Solving Intelligence")
    print("=" * 70)
    
    if population.population_accuracy_history:
        # Get initial and final accuracy
        initial_acc = population.population_accuracy_history[0]['accuracy']
        final_acc = population.population_accuracy_history[-1]['accuracy']
        
        print(f"\nInitial average accuracy: {initial_acc:.1f}%")
        print(f"Final average accuracy: {final_acc:.1f}%")
        print(f"Improvement: {final_acc - initial_acc:.1f} percentage points")
        
        # Show top performers
        print("\nTop 5 Problem Solvers:")
        top_solvers = sorted(population.cells, key=lambda c: c.problem_accuracy, reverse=True)[:5]
        for i, cell in enumerate(top_solvers, 1):
            print(f"{i}. Cell {cell.cell_id} (Gen {cell.generation}): "
                  f"{cell.problem_accuracy:.1f}% accuracy, "
                  f"{cell.problems_solved}/{cell.total_problems} solved")
    
    print("\nSimulation complete!")
    
    # Optional: Save results
    save_choice = input("\nSave results? (y/n): ")
    if save_choice.lower() == 'y':
        import json
        filename = input("Filename (default: hybrid_results.json): ") or "hybrid_results.json"
        
        results = {
            'summary': population.get_population_summary(),
            'accuracy_history': population.population_accuracy_history,
            'best_solver_history': population.best_solver_history,
            'statistics_history': population.statistics_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
