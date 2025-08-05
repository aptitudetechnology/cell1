"""
Example Scenarios and Demonstrations
Showcases different evolutionary scenarios and educational examples.
"""
import random
import time
from typing import Dict, List, Optional, Tuple

from genome import ComputationalGenome
from cell import EvolutionaryCell
from environment import Environment, SelectionPressure, SelectionType, create_benign_environment, create_competitive_environment, create_harsh_environment
from population import Population, run_evolution_experiment


def founder_effect_experiment() -> Population:
    """
    Demonstrate founder effect - evolution from single cell.
    
    Returns:
        Population after founder effect experiment
    """
    print("=== FOUNDER EFFECT EXPERIMENT ===")
    print("Starting with a single founder cell and observing initial population expansion")
    print("This demonstrates how genetic bottlenecks affect evolution")
    
    # Create founder with specific genome
    founder_genome = ComputationalGenome()
    print(f"Founder genome: {founder_genome}")
    print(f"Founder parameters: {founder_genome.express_genes()}")
    
    # Use benign environment for initial expansion
    env = create_benign_environment()
    
    # Run evolution
    population = run_evolution_experiment(
        initial_genome=founder_genome,
        environment=env,
        max_time=150.0,
        max_generations=15,
        snapshot_interval=30.0
    )
    
    print(f"\nFounder Effect Results:")
    print(f"- Started with 1 cell, ended with {len([c for c in population.cells if c.is_alive])} cells")
    print(f"- Reached generation {population.generation}")
    print(f"- Genetic diversity: {population.get_population_summary()['genetic_diversity']:.3f}")
    
    return population


def selection_pressure_experiment() -> Population:
    """
    Demonstrate evolution under strong selection pressure.
    
    Returns:
        Population after selection experiment
    """
    print("\n=== SELECTION PRESSURE EXPERIMENT ===")
    print("Introducing nutrient limitation to observe adaptive evolution")
    print("Selection should favor energy-efficient cells")
    
    # Create environment with nutrient limitation
    env = Environment("Nutrient Limited Environment")
    env.conditions.nutrient_density = 0.4
    env.conditions.carrying_capacity = 300
    
    # Add strong selection for energy efficiency
    energy_selection = SelectionPressure(
        name="energy_conservation",
        target_parameter="energy_efficiency",
        selection_type=SelectionType.DIRECTIONAL,
        strength=1.2  # Strong selection
    )
    env.add_selection_pressure(energy_selection)
    
    # Start with random founder
    founder = ComputationalGenome()
    initial_efficiency = founder.express_genes()['energy_efficiency']
    print(f"Initial energy efficiency: {initial_efficiency:.3f}")
    
    # Run evolution
    population = run_evolution_experiment(
        initial_genome=founder,
        environment=env,
        max_time=200.0,
        max_generations=25,
        snapshot_interval=40.0
    )
    
    # Analyze results
    alive_cells = [c for c in population.cells if c.is_alive]
    if alive_cells:
        final_efficiency = sum(c.parameters['energy_efficiency'] for c in alive_cells) / len(alive_cells)
        improvement = final_efficiency - initial_efficiency
        
        print(f"\nSelection Pressure Results:")
        print(f"- Initial energy efficiency: {initial_efficiency:.3f}")
        print(f"- Final energy efficiency: {final_efficiency:.3f}")
        print(f"- Improvement: {improvement:.3f} ({improvement/initial_efficiency*100:.1f}%)")
        print(f"- Population adapted to nutrient limitation!")
    
    return population


def mutation_rate_evolution_experiment() -> Population:
    """
    Demonstrate evolution of optimal mutation rates.
    
    Returns:
        Population after mutation rate evolution
    """
    print("\n=== MUTATION RATE EVOLUTION EXPERIMENT ===")
    print("Observing evolution of optimal mutation rates in changing environment")
    
    # Create environment with moderate variability
    env = Environment("Variable Environment")
    env.change_frequency = 0.02  # Changes every 50 time units on average
    env.change_magnitude = 0.3
    
    # Add cyclic changes to create ongoing selection pressure
    env.add_cyclic_change('nutrient_density', 0.4, 80.0)  # Cycle every 80 time units
    
    # Selection for moderate mutation rates (optimal for adaptation)
    mutation_selection = SelectionPressure(
        name="evolvability",
        target_parameter="mutation_rate",
        selection_type=SelectionType.STABILIZING,
        strength=0.6,
        optimal_value=0.003  # Optimal mutation rate
    )
    env.add_selection_pressure(mutation_selection)
    
    # Start with high mutation rate
    founder = ComputationalGenome()
    founder.set_gene_value('mutation_rate', 200)  # High mutation rate initially
    initial_mutation_rate = founder.express_genes()['mutation_rate']
    print(f"Initial mutation rate: {initial_mutation_rate:.6f}")
    
    # Run evolution
    population = run_evolution_experiment(
        initial_genome=founder,
        environment=env,
        max_time=300.0,
        max_generations=30,
        snapshot_interval=60.0
    )
    
    # Analyze results
    alive_cells = [c for c in population.cells if c.is_alive]
    if alive_cells:
        final_mutation_rate = sum(c.parameters['mutation_rate'] for c in alive_cells) / len(alive_cells)
        
        print(f"\nMutation Rate Evolution Results:")
        print(f"- Initial mutation rate: {initial_mutation_rate:.6f}")
        print(f"- Final mutation rate: {final_mutation_rate:.6f}")
        print(f"- Optimal mutation rate: 0.003000")
        print(f"- Convergence to optimal: {abs(final_mutation_rate - 0.003):.6f} from optimal")
    
    return population


def genetic_drift_experiment() -> Tuple[Population, Population]:
    """
    Compare evolution in small vs large populations to show genetic drift.
    
    Returns:
        Tuple of (small_population, large_population)
    """
    print("\n=== GENETIC DRIFT EXPERIMENT ===")
    print("Comparing evolution in small vs large populations")
    print("Small populations should show more genetic drift")
    
    # Use same founder for both populations
    founder = ComputationalGenome()
    print(f"Common founder genome: {founder}")
    
    # Small population environment
    small_env = create_benign_environment()
    small_env.name = "Small Population"
    small_env.conditions.carrying_capacity = 50
    
    # Large population environment  
    large_env = create_benign_environment()
    large_env.name = "Large Population"
    large_env.conditions.carrying_capacity = 500
    
    print("\nRunning small population evolution...")
    small_pop = run_evolution_experiment(
        initial_genome=ComputationalGenome(founder.genome),  # Copy founder
        environment=small_env,
        max_time=200.0,
        max_generations=20,
        snapshot_interval=50.0
    )
    
    print("\nRunning large population evolution...")
    large_pop = run_evolution_experiment(
        initial_genome=ComputationalGenome(founder.genome),  # Copy founder
        environment=large_env,
        max_time=200.0,
        max_generations=20,
        snapshot_interval=50.0
    )
    
    # Compare genetic diversity
    small_diversity = small_pop.get_population_summary()['genetic_diversity']
    large_diversity = large_pop.get_population_summary()['genetic_diversity']
    
    print(f"\nGenetic Drift Results:")
    print(f"- Small population diversity: {small_diversity:.3f}")
    print(f"- Large population diversity: {large_diversity:.3f}")
    print(f"- Difference: {large_diversity - small_diversity:.3f}")
    
    if small_diversity < large_diversity:
        print("- Small population shows more genetic drift (lower diversity)")
    else:
        print("- Results may vary due to stochastic effects")
    
    return small_pop, large_pop


def adaptive_radiation_experiment() -> List[Population]:
    """
    Demonstrate adaptive radiation in multiple environmental niches.
    
    Returns:
        List of populations from different environments
    """
    print("\n=== ADAPTIVE RADIATION EXPERIMENT ===")
    print("Evolving populations in different environmental niches")
    print("Each environment should select for different optimal traits")
    
    # Common ancestor
    ancestor = ComputationalGenome()
    print(f"Common ancestor: {ancestor}")
    
    populations = []
    
    # Environment 1: High energy requirements (select for efficiency)
    env1 = Environment("High Energy Demand")
    env1.conditions.nutrient_density = 0.3
    energy_pressure = SelectionPressure(
        name="energy_efficiency_selection",
        target_parameter="energy_efficiency", 
        selection_type=SelectionType.DIRECTIONAL,
        strength=0.8
    )
    env1.add_selection_pressure(energy_pressure)
    
    # Environment 2: Rapid growth advantage (select for growth rate)
    env2 = Environment("Rapid Growth Environment")
    env2.conditions.carrying_capacity = 200
    growth_pressure = SelectionPressure(
        name="growth_rate_selection",
        target_parameter="growth_rate",
        selection_type=SelectionType.DIRECTIONAL, 
        strength=0.8
    )
    env2.add_selection_pressure(growth_pressure)
    
    # Environment 3: Stable conditions (select for larger size)
    env3 = Environment("Stable Large Size")
    size_pressure = SelectionPressure(
        name="size_selection",
        target_parameter="division_threshold",
        selection_type=SelectionType.DIRECTIONAL,
        strength=0.6
    )
    env3.add_selection_pressure(size_pressure)
    
    environments = [env1, env2, env3]
    env_names = ["Energy Efficient", "Fast Growth", "Large Size"]
    
    for i, (env, name) in enumerate(zip(environments, env_names)):
        print(f"\nEvolving in {name} environment...")
        
        pop = run_evolution_experiment(
            initial_genome=ComputationalGenome(ancestor.genome),  # Copy ancestor
            environment=env,
            max_time=180.0,
            max_generations=18,
            snapshot_interval=45.0
        )
        
        populations.append(pop)
    
    # Compare final populations
    print(f"\nAdaptive Radiation Results:")
    for i, (pop, name) in enumerate(zip(populations, env_names)):
        alive_cells = [c for c in pop.cells if c.is_alive]
        if alive_cells:
            avg_params = {
                'energy_efficiency': sum(c.parameters['energy_efficiency'] for c in alive_cells) / len(alive_cells),
                'growth_rate': sum(c.parameters['growth_rate'] for c in alive_cells) / len(alive_cells),
                'division_threshold': sum(c.parameters['division_threshold'] for c in alive_cells) / len(alive_cells)
            }
            
            print(f"  {name}:")
            print(f"    Energy Efficiency: {avg_params['energy_efficiency']:.3f}")
            print(f"    Growth Rate: {avg_params['growth_rate']:.3f}")
            print(f"    Division Threshold: {avg_params['division_threshold']:.1f}")
    
    return populations


def demonstrate_all_scenarios():
    """
    Run all demonstration scenarios in sequence.
    """
    print("COMPUTATIONAL GENOME CELL EVOLUTION SYSTEM")
    print("=" * 60)
    print("Running all demonstration scenarios...")
    
    start_time = time.time()
    
    # Run all experiments
    founder_pop = founder_effect_experiment()
    time.sleep(1)  # Brief pause between experiments
    
    selection_pop = selection_pressure_experiment()
    time.sleep(1)
    
    mutation_pop = mutation_rate_evolution_experiment()
    time.sleep(1)
    
    small_pop, large_pop = genetic_drift_experiment()
    time.sleep(1)
    
    radiation_pops = adaptive_radiation_experiment()
    
    total_time = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Total execution time: {total_time:.1f} seconds")
    
    # Summary statistics
    all_populations = [founder_pop, selection_pop, mutation_pop, small_pop, large_pop] + radiation_pops
    
    print(f"\nFinal Summary:")
    print(f"- Experiments run: {len(all_populations)}")
    print(f"- Total cells evolved: {sum(p.total_births for p in all_populations)}")
    print(f"- Highest generation reached: {max(p.generation for p in all_populations)}")
    print(f"- Most diverse population: {max(p.get_population_summary()['genetic_diversity'] for p in all_populations):.3f}")
    
    return all_populations


def interactive_experiment():
    """
    Allow user to configure and run custom experiments.
    """
    print("\n=== INTERACTIVE EXPERIMENT ===")
    print("Configure your own evolution experiment!")
    
    # Get user preferences
    try:
        max_time = float(input("Maximum simulation time (default 200): ") or "200")
        max_generations = int(input("Maximum generations (default 20): ") or "20")
        
        print("\nEnvironment options:")
        print("1. Benign (no selection pressure)")
        print("2. Competitive (resource competition)")
        print("3. Harsh (multiple stressors)")
        print("4. Custom")
        
        env_choice = input("Choose environment (1-4, default 2): ") or "2"
        
        if env_choice == "1":
            env = create_benign_environment()
        elif env_choice == "3":
            env = create_harsh_environment()
        elif env_choice == "4":
            env = Environment("Custom Environment")
            # Could add more customization options here
        else:
            env = create_competitive_environment()

        # Add a larger cyclic food drop: every 30 time units, nutrient_density increases by 1.5 (bigger pulse)
        env.add_cyclic_change('nutrient_density', amplitude=1.5, period=30.0, phase=0.0)

        print(f"\nRunning experiment with {env.name}...")

        population = run_evolution_experiment(
            environment=env,
            max_time=max_time,
            max_generations=max_generations,
            snapshot_interval=max_time/5
        )
        
        print(f"\nExperiment completed!")
        summary = population.get_population_summary()
        print(f"Final population summary: {summary}")
        
        # Offer to save results
        save_choice = input("\nSave results to file? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Enter filename (default: experiment_results.json): ") or "experiment_results.json"
            population.save_snapshot(filename)
            print(f"Results saved to {filename}")
        
        return population
        
    except (ValueError, KeyboardInterrupt):
        print("Experiment cancelled or invalid input")
        return None


if __name__ == "__main__":
    # Run demonstration
    print("Choose experiment type:")
    print("1. Run all demonstration scenarios")
    print("2. Run single scenario")
    print("3. Interactive custom experiment")
    
    try:
        choice = input("Enter choice (1-3, default 1): ") or "1"
        
        if choice == "1":
            demonstrate_all_scenarios()
        elif choice == "3":
            interactive_experiment()
        else:
            print("\nSingle scenarios:")
            print("1. Founder Effect")
            print("2. Selection Pressure")
            print("3. Mutation Rate Evolution")
            print("4. Genetic Drift")
            print("5. Adaptive Radiation")
            
            scenario_choice = input("Choose scenario (1-5): ")
            
            if scenario_choice == "1":
                founder_effect_experiment()
            elif scenario_choice == "2":
                selection_pressure_experiment()
            elif scenario_choice == "3":
                mutation_rate_evolution_experiment()
            elif scenario_choice == "4":
                genetic_drift_experiment()
            elif scenario_choice == "5":
                adaptive_radiation_experiment()
            else:
                print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
