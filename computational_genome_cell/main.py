"""
Main Simulation Runner
Entry point for the Computational Genome Cell Evolution System.
"""
import sys
import argparse
import json
import time
from pathlib import Path

from genome import ComputationalGenome, create_random_population, analyze_genetic_diversity
from cell import EvolutionaryCell
from environment import (Environment, create_benign_environment, create_competitive_environment, 
                        create_harsh_environment, SelectionPressure, SelectionType)
from population import Population, run_evolution_experiment
from examples import (founder_effect_experiment, selection_pressure_experiment, 
                     mutation_rate_evolution_experiment, genetic_drift_experiment,
                     adaptive_radiation_experiment, demonstrate_all_scenarios,
                     interactive_experiment)

# Try to import visualization, but make it optional
try:
    from visualization import EvolutionVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Note: Visualization not available. Install matplotlib and seaborn for plotting.")


def create_custom_environment(config: dict) -> Environment:
    """
    Create custom environment from configuration dictionary.
    
    Args:
        config: Environment configuration
        
    Returns:
        Environment instance
    """
    env = Environment(config.get('name', 'Custom Environment'))
    
    # Set environmental conditions
    conditions = config.get('conditions', {})
    env.conditions.temperature = conditions.get('temperature', 37.0)
    env.conditions.ph = conditions.get('ph', 7.0)
    env.conditions.oxygen_level = conditions.get('oxygen_level', 1.0)
    env.conditions.nutrient_density = conditions.get('nutrient_density', 1.0)
    env.conditions.toxin_level = conditions.get('toxin_level', 0.0)
    env.conditions.carrying_capacity = conditions.get('carrying_capacity', 1000)
    
    # Add selection pressures
    pressures = config.get('selection_pressures', [])
    for pressure_config in pressures:
        pressure = SelectionPressure(
            name=pressure_config['name'],
            target_parameter=pressure_config['target_parameter'],
            selection_type=SelectionType(pressure_config.get('selection_type', 'directional')),
            strength=pressure_config.get('strength', 1.0),
            optimal_value=pressure_config.get('optimal_value')
        )
        env.add_selection_pressure(pressure)
    
    # Set environmental dynamics
    dynamics = config.get('dynamics', {})
    env.change_frequency = dynamics.get('change_frequency', 0.0)
    env.change_magnitude = dynamics.get('change_magnitude', 0.1)
    
    # Add cyclic changes
    cyclic_changes = dynamics.get('cyclic_changes', [])
    for change in cyclic_changes:
        env.add_cyclic_change(
            change['parameter'],
            change['amplitude'],
            change['period'],
            change.get('phase', 0.0)
        )
    
    return env


def run_simulation(args):
    """
    Run main simulation based on command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    print("Computational Genome Cell Evolution System")
    print("=" * 50)
    
    # Create or load initial genome
    if args.genome_file:
        with open(args.genome_file, 'r') as f:
            genome_data = json.load(f)
        initial_genome = ComputationalGenome.from_dict(genome_data)
        print(f"Loaded initial genome from {args.genome_file}")
    elif args.genome_value:
        initial_genome = ComputationalGenome(args.genome_value)
        print(f"Using specified genome value: 0x{args.genome_value:08X}")
    else:
        initial_genome = ComputationalGenome()
        print(f"Generated random initial genome: {initial_genome}")
    
    # Create environment
    if args.environment_file:
        with open(args.environment_file, 'r') as f:
            env_config = json.load(f)
        environment = create_custom_environment(env_config)
        print(f"Loaded environment from {args.environment_file}")
    elif args.environment == 'benign':
        environment = create_benign_environment()
    elif args.environment == 'competitive':
        environment = create_competitive_environment()
    elif args.environment == 'harsh':
        environment = create_harsh_environment()
    else:
        environment = create_benign_environment()
    
    print(f"Environment: {environment.name}")
    
    # Run evolution experiment
    start_time = time.time()
    
    population = run_evolution_experiment(
        initial_genome=initial_genome,
        environment=environment,
        max_time=args.max_time,
        max_generations=args.max_generations,
        snapshot_interval=args.snapshot_interval
    )
    
    runtime = time.time() - start_time
    
    # Display results
    print(f"\nSimulation completed in {runtime:.2f} seconds")
    summary = population.get_population_summary()
    
    print(f"\nFinal Results:")
    print(f"- Generation: {summary['generation']}")
    print(f"- Population size: {summary['alive_cells']}")
    print(f"- Average fitness: {summary['average_fitness']:.3f}")
    print(f"- Genetic diversity: {summary['genetic_diversity']:.3f}")
    print(f"- Total births: {summary['total_births']}")
    print(f"- Total deaths: {summary['total_deaths']}")
    
    # Show fittest individuals
    fittest = population.get_fittest_cells(5)
    print(f"\nTop 5 fittest cells:")
    for i, cell in enumerate(fittest, 1):
        print(f"  {i}. Cell {cell.cell_id}: Fitness={cell.get_fitness():.3f}, "
              f"Gen={cell.generation}, Mass={cell.mass:.1f}")
    
    # Save results if requested
    if args.output_file:
        population.save_snapshot(args.output_file)
        print(f"\nResults saved to {args.output_file}")
    
    # Generate visualizations if requested and available
    if args.visualize and VISUALIZATION_AVAILABLE:
        print(f"\nGenerating visualizations...")
        visualizer = EvolutionVisualizer(population)
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            visualizer.plot_population_dynamics(str(output_dir / "population_dynamics.png"))
            visualizer.plot_parameter_evolution(str(output_dir / "parameter_evolution.png"))
            visualizer.plot_genetic_diversity(str(output_dir / "genetic_diversity.png"))
            visualizer.plot_genome_heatmap(save_path=str(output_dir / "genome_heatmap.png"))
            visualizer.create_evolution_dashboard(str(output_dir / "evolution_dashboard.png"))
            
            print(f"Visualizations saved to {args.output_dir}")
        else:
            visualizer.create_evolution_dashboard()


def run_example_scenarios(args):
    """
    Run predefined example scenarios.
    
    Args:
        args: Parsed command line arguments
    """
    scenario_functions = {
        'all': demonstrate_all_scenarios,
        'founder': founder_effect_experiment,
        'selection': selection_pressure_experiment,
        'mutation': mutation_rate_evolution_experiment,
        'drift': genetic_drift_experiment,
        'radiation': adaptive_radiation_experiment,
        'interactive': interactive_experiment
    }
    
    if args.scenario in scenario_functions:
        print(f"Running {args.scenario} scenario...")
        result = scenario_functions[args.scenario]()
        
        if args.output_file and hasattr(result, 'save_snapshot'):
            result.save_snapshot(args.output_file)
            print(f"Results saved to {args.output_file}")
    else:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available scenarios: {list(scenario_functions.keys())}")


def test_system():
    """
    Run system tests to verify all components work correctly.
    """
    print("Running system tests...")
    
    # Test genome system
    print("1. Testing genome system...")
    genome = ComputationalGenome()
    mutated = genome.mutate()
    assert genome.hamming_distance(mutated) >= 0
    print("   ✓ Genome system working")
    
    # Test cell system
    print("2. Testing cell system...")
    cell = EvolutionaryCell(genome)
    stats = cell.update(1.0)
    assert 'status' in stats
    print("   ✓ Cell system working")
    
    # Test environment system
    print("3. Testing environment system...")
    env = create_benign_environment()
    env.update(1.0)
    assert hasattr(env, 'time')
    print("   ✓ Environment system working")
    
    # Test population system
    print("4. Testing population system...")
    population = Population(genome, env)
    population.update(1.0)
    assert len(population.statistics_history) > 0
    print("   ✓ Population system working")
    
    # Test short evolution
    print("5. Testing evolution...")
    test_pop = run_evolution_experiment(
        max_time=10.0,
        max_generations=3,
        snapshot_interval=5.0
    )
    assert test_pop.time > 0
    print("   ✓ Evolution system working")
    
    print("\nAll systems tests passed! ✓")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Computational Genome Cell Evolution System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic simulation
  python main.py simulate --max-time 500 --max-generations 50
  
  # Run with harsh environment
  python main.py simulate --environment harsh --output results.json
  
  # Run with custom genome
  python main.py simulate --genome-value 0x12345678 --visualize
  
  # Run example scenarios
  python main.py examples --scenario all
  python main.py examples --scenario founder
  
  # Run tests
  python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run evolution simulation')
    sim_parser.add_argument('--max-time', type=float, default=200.0,
                           help='Maximum simulation time (default: 200)')
    sim_parser.add_argument('--max-generations', type=int, default=20,
                           help='Maximum generations (default: 20)')
    sim_parser.add_argument('--snapshot-interval', type=float, default=40.0,
                           help='Progress reporting interval (default: 40)')
    sim_parser.add_argument('--environment', choices=['benign', 'competitive', 'harsh'],
                           default='benign', help='Predefined environment type')
    sim_parser.add_argument('--environment-file', type=str,
                           help='JSON file with custom environment configuration')
    sim_parser.add_argument('--genome-value', type=lambda x: int(x, 0),
                           help='Initial genome value (hex or decimal)')
    sim_parser.add_argument('--genome-file', type=str,
                           help='JSON file with initial genome')
    sim_parser.add_argument('--output-file', type=str,
                           help='Save results to JSON file')
    sim_parser.add_argument('--visualize', action='store_true',
                           help='Generate visualizations')
    sim_parser.add_argument('--output-dir', type=str,
                           help='Directory to save visualization files')
    
    # Examples command
    examples_parser = subparsers.add_parser('examples', help='Run example scenarios')
    examples_parser.add_argument('--scenario', 
                                choices=['all', 'founder', 'selection', 'mutation', 
                                        'drift', 'radiation', 'interactive'],
                                default='all',
                                help='Example scenario to run')
    examples_parser.add_argument('--output-file', type=str,
                                help='Save results to JSON file')
    
    # Test command
    subparsers.add_parser('test', help='Run system tests')
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    # Execute command
    try:
        if args.command == 'simulate':
            run_simulation(args)
        elif args.command == 'examples':
            run_example_scenarios(args)
        elif args.command == 'test':
            test_system()
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        if __name__ == "__main__":
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
