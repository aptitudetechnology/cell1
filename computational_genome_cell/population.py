"""
Population Management System
Manages cell populations, tracks evolution, and implements selection dynamics.
"""
import random
import statistics
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import json
import time
from collections import defaultdict, Counter

from genome import ComputationalGenome, analyze_genetic_diversity
from cell import EvolutionaryCell
from environment import Environment


@dataclass
class PopulationStatistics:
    """Statistics about the current population."""
    generation: int
    time: float
    population_size: int
    alive_cells: int
    average_fitness: float
    max_fitness: float
    min_fitness: float
    average_age: float
    average_mass: float
    average_atp: float
    genetic_diversity: float
    mutation_rate_avg: float
    energy_efficiency_avg: float
    growth_rate_avg: float
    division_threshold_avg: float
    births_this_generation: int
    deaths_this_generation: int


@dataclass
class LineageNode:
    """Node in the evolutionary lineage tree."""
    cell_id: int
    genome_value: int
    parent_id: Optional[int]
    birth_time: float
    death_time: Optional[float]
    generation: int
    fitness: float
    children: List[int]


class Population:
    """
    Manages a population of evolutionary cells and tracks their evolution.
    """
    
    def __init__(self, initial_genome: ComputationalGenome = None, 
                 environment: Environment = None):
        """
        Initialize population with a founder cell.
        
        Args:
            initial_genome: Genome for the founder cell (random if None)
            environment: Environment for the population
        """
        # Initialize founder cell
        if initial_genome is None:
            initial_genome = ComputationalGenome()
        
        self.founder_genome = initial_genome
        self.cells = [EvolutionaryCell(initial_genome, cell_id=1)]
        self.environment = environment
        
        # Population tracking
        self.generation = 0
        self.time = 0.0
        self.next_cell_id = 2
        self.total_births = 1  # Founder counts as birth
        self.total_deaths = 0
        
        # Lineage tracking
        self.lineage_tree: Dict[int, LineageNode] = {}
        self.lineage_tree[1] = LineageNode(
            cell_id=1,
            genome_value=initial_genome.genome,
            parent_id=None,
            birth_time=0.0,
            death_time=None,
            generation=0,
            fitness=self.cells[0].get_fitness(),
            children=[]
        )
        
        # Statistics tracking
        self.statistics_history: List[PopulationStatistics] = []
        self.generation_times = []
        
        # Evolution tracking
        self.beneficial_mutations = []
        self.fixed_mutations = []
        self.selection_coefficients = {}
    
    def update(self, time_step: float = 1.0) -> PopulationStatistics:
        """
        Update the entire population for one time step.
        
        Args:
            time_step: Time elapsed
            
        Returns:
            PopulationStatistics for this time step
        """
        self.time += time_step
        
        if self.environment:
            self.environment.update(time_step)
        
        # Track births and deaths for this update
        births_this_step = 0
        deaths_this_step = 0
        new_cells = []
        living_cells = []
        
        # Get current population size for nutrient calculation
        population_size = len([c for c in self.cells if c.is_alive])
        nutrient_availability = 1.0
        
        if self.environment:
            nutrient_availability = self.environment.get_nutrient_availability(population_size)
        
        # Update each cell
        for cell in self.cells:
            if not cell.is_alive:
                continue
            
            # Update cell state
            update_stats = cell.update(time_step, nutrient_availability)
            
            # Check for death
            if not cell.is_alive:
                deaths_this_step += 1
                self.total_deaths += 1
                # Record death time in lineage
                if cell.cell_id in self.lineage_tree:
                    self.lineage_tree[cell.cell_id].death_time = self.time
            else:
                living_cells.append(cell)
                
                # Check for division
                if cell.can_divide:
                    daughter = cell.divide()
                    if daughter is not None:
                        daughter.cell_id = self.next_cell_id
                        self.next_cell_id += 1
                        new_cells.append(daughter)
                        births_this_step += 1
                        self.total_births += 1
                        
                        # Record in lineage tree
                        self.lineage_tree[daughter.cell_id] = LineageNode(
                            cell_id=daughter.cell_id,
                            genome_value=daughter.genome.genome,
                            parent_id=cell.cell_id,
                            birth_time=self.time,
                            death_time=None,
                            generation=daughter.generation,
                            fitness=daughter.get_fitness(),
                            children=[]
                        )
                        
                        # Update parent's children list
                        if cell.cell_id in self.lineage_tree:
                            self.lineage_tree[cell.cell_id].children.append(daughter.cell_id)
        
        # Update population
        self.cells = living_cells + new_cells
        
        # Apply environmental selection if environment exists
        if self.environment and len(self.cells) > 0:
            self._apply_environmental_selection()
        
        # Calculate statistics
        stats = self._calculate_statistics(births_this_step, deaths_this_step)
        self.statistics_history.append(stats)
        
        # Check for generation advancement
        if births_this_step > 0:
            avg_generation = statistics.mean([c.generation for c in self.cells])
            if avg_generation > self.generation:
                self.generation = int(avg_generation)
                self.generation_times.append(self.time)
        
        return stats
    
    def _apply_environmental_selection(self):
        """Apply environmental selection pressures to the population."""
        if not self.environment or len(self.cells) == 0:
            return
        
        # Calculate fitness for each cell including environmental effects
        cell_fitnesses = []
        for cell in self.cells:
            base_fitness = cell.get_fitness()
            
            # Apply selection pressures
            selection_modifier = self.environment.apply_selection(cell.parameters)
            
            # Apply environmental stress
            stress = self.environment.calculate_environmental_stress()
            stress_modifier = max(0.1, 1.0 - stress * 0.5)
            
            final_fitness = base_fitness * selection_modifier * stress_modifier
            cell_fitnesses.append((cell, final_fitness))
        
        # Sort by fitness
        cell_fitnesses.sort(key=lambda x: x[1], reverse=True)
        
        # Apply carrying capacity limitation
        carrying_capacity = self.environment.conditions.carrying_capacity
        if len(self.cells) > carrying_capacity:
            # Keep only the fittest cells
            survivors = [cell for cell, fitness in cell_fitnesses[:carrying_capacity]]
            
            # Mark eliminated cells as dead
            eliminated = [cell for cell, fitness in cell_fitnesses[carrying_capacity:]]
            for cell in eliminated:
                cell.is_alive = False
                if cell.cell_id in self.lineage_tree:
                    self.lineage_tree[cell.cell_id].death_time = self.time
            
            self.cells = [c for c in self.cells if c.is_alive]
    
    def _calculate_statistics(self, births: int, deaths: int) -> PopulationStatistics:
        """Calculate population statistics for current state."""
        alive_cells = [c for c in self.cells if c.is_alive]
        
        if not alive_cells:
            return PopulationStatistics(
                generation=self.generation,
                time=self.time,
                population_size=0,
                alive_cells=0,
                average_fitness=0,
                max_fitness=0,
                min_fitness=0,
                average_age=0,
                average_mass=0,
                average_atp=0,
                genetic_diversity=0,
                mutation_rate_avg=0,
                energy_efficiency_avg=0,
                growth_rate_avg=0,
                division_threshold_avg=0,
                births_this_generation=births,
                deaths_this_generation=deaths
            )
        
        # Basic statistics
        fitnesses = [c.get_fitness() for c in alive_cells]
        ages = [c.age for c in alive_cells]
        masses = [c.mass for c in alive_cells]
        atps = [c.atp for c in alive_cells]
        
        # Genetic statistics
        genomes = [c.genome for c in alive_cells]
        diversity_stats = analyze_genetic_diversity(genomes)
        
        # Parameter averages
        mutation_rates = [c.parameters['mutation_rate'] for c in alive_cells]
        energy_efficiencies = [c.parameters['energy_efficiency'] for c in alive_cells]
        growth_rates = [c.parameters['growth_rate'] for c in alive_cells]
        division_thresholds = [c.parameters['division_threshold'] for c in alive_cells]
        
        return PopulationStatistics(
            generation=self.generation,
            time=self.time,
            population_size=len(self.cells),
            alive_cells=len(alive_cells),
            average_fitness=statistics.mean(fitnesses),
            max_fitness=max(fitnesses),
            min_fitness=min(fitnesses),
            average_age=statistics.mean(ages),
            average_mass=statistics.mean(masses),
            average_atp=statistics.mean(atps),
            genetic_diversity=diversity_stats['genetic_diversity'],
            mutation_rate_avg=statistics.mean(mutation_rates),
            energy_efficiency_avg=statistics.mean(energy_efficiencies),
            growth_rate_avg=statistics.mean(growth_rates),
            division_threshold_avg=statistics.mean(division_thresholds),
            births_this_generation=births,
            deaths_this_generation=deaths
        )
    
    def get_genotype_frequencies(self) -> Dict[int, float]:
        """
        Calculate frequencies of different genotypes in the population.
        
        Returns:
            Dictionary mapping genome values to their frequencies
        """
        alive_cells = [c for c in self.cells if c.is_alive]
        if not alive_cells:
            return {}
        
        genome_counts = Counter(c.genome.genome for c in alive_cells)
        total = len(alive_cells)
        
        return {genome: count / total for genome, count in genome_counts.items()}
    
    def get_fittest_cells(self, n: int = 10) -> List[EvolutionaryCell]:
        """
        Get the n fittest cells in the population.
        
        Args:
            n: Number of cells to return
            
        Returns:
            List of fittest cells
        """
        alive_cells = [c for c in self.cells if c.is_alive]
        return sorted(alive_cells, key=lambda c: c.get_fitness(), reverse=True)[:n]
    
    def get_lineage_depth(self) -> int:
        """
        Get the maximum lineage depth in the population.
        
        Returns:
            Maximum generation number
        """
        if not self.cells:
            return 0
        return max(c.generation for c in self.cells if c.is_alive)
    
    def track_beneficial_mutations(self):
        """Identify and track beneficial mutations in the population."""
        # This is a simplified version - could be much more sophisticated
        genotype_frequencies = self.get_genotype_frequencies()
        
        # Look for genotypes that are increasing in frequency
        if len(self.statistics_history) > 10:
            # Compare current frequencies to frequencies 10 steps ago
            # This would require more sophisticated tracking
            pass
    
    def calculate_selection_coefficients(self) -> Dict[str, float]:
        """
        Calculate selection coefficients for different traits.
        
        Returns:
            Dictionary mapping traits to their selection coefficients
        """
        alive_cells = [c for c in self.cells if c.is_alive]
        if len(alive_cells) < 10:
            return {}
        
        coefficients = {}
        
        # Calculate selection coefficient for each parameter
        for param_name in ['energy_efficiency', 'growth_rate', 'division_threshold', 'mutation_rate']:
            param_values = [c.parameters[param_name] for c in alive_cells]
            fitnesses = [c.get_fitness() for c in alive_cells]
            
            # Simple correlation-based selection coefficient
            if len(set(param_values)) > 1:  # Need variation to calculate
                correlation = self._calculate_correlation(param_values, fitnesses)
                coefficients[param_name] = correlation
        
        return coefficients
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def save_snapshot(self, filename: str):
        """
        Save current population state to file.
        
        Args:
            filename: File to save to
        """
        snapshot = {
            'metadata': {
                'generation': self.generation,
                'time': self.time,
                'total_births': self.total_births,
                'total_deaths': self.total_deaths,
                'founder_genome': self.founder_genome.to_dict()
            },
            'cells': [cell.to_dict() for cell in self.cells if cell.is_alive],
            'environment': self.environment.to_dict() if self.environment else None,
            'statistics_history': [
                {
                    'generation': s.generation,
                    'time': s.time,
                    'population_size': s.population_size,
                    'alive_cells': s.alive_cells,
                    'average_fitness': s.average_fitness,
                    'genetic_diversity': s.genetic_diversity,
                    'mutation_rate_avg': s.mutation_rate_avg,
                    'energy_efficiency_avg': s.energy_efficiency_avg,
                    'growth_rate_avg': s.growth_rate_avg,
                    'division_threshold_avg': s.division_threshold_avg
                }
                for s in self.statistics_history[-100:]  # Save last 100 stats
            ],
            'lineage_tree': {
                str(cell_id): {
                    'cell_id': node.cell_id,
                    'genome_value': node.genome_value,
                    'parent_id': node.parent_id,
                    'birth_time': node.birth_time,
                    'death_time': node.death_time,
                    'generation': node.generation,
                    'fitness': node.fitness,
                    'children': node.children
                }
                for cell_id, node in self.lineage_tree.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(snapshot, f, indent=2)
    
    def get_population_summary(self) -> Dict:
        """
        Get a summary of the current population state.
        
        Returns:
            Dictionary with population summary
        """
        if not self.statistics_history:
            return {'status': 'no_data'}
        
        current_stats = self.statistics_history[-1]
        
        return {
            'generation': self.generation,
            'time': self.time,
            'population_size': current_stats.population_size,
            'alive_cells': current_stats.alive_cells,
            'average_fitness': round(current_stats.average_fitness, 3),
            'genetic_diversity': round(current_stats.genetic_diversity, 3),
            'lineage_depth': self.get_lineage_depth(),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'most_frequent_genotypes': dict(list(self.get_genotype_frequencies().items())[:5]),
            'selection_coefficients': self.calculate_selection_coefficients(),
            'environment': self.environment.name if self.environment else None
        }
    
    def is_extinct(self) -> bool:
        """Check if the population is extinct."""
        return len([c for c in self.cells if c.is_alive]) == 0
    
    def __str__(self) -> str:
        """String representation of population."""
        alive = len([c for c in self.cells if c.is_alive])
        return f"Population(gen={self.generation}, alive={alive}, time={self.time:.1f})"


def run_evolution_experiment(initial_genome: ComputationalGenome = None,
                           environment: Environment = None,
                           max_time: float = 1000.0,
                           max_generations: int = 100,
                           snapshot_interval: float = 100.0) -> Population:
    """
    Run a complete evolution experiment.
    
    Args:
        initial_genome: Starting genome (random if None)
        environment: Environment for evolution
        max_time: Maximum simulation time
        max_generations: Maximum generations to simulate
        snapshot_interval: How often to print progress
        
    Returns:
        Final Population object
    """
    population = Population(initial_genome, environment)
    
    print(f"Starting evolution experiment with {environment.name if environment else 'no environment'}")
    print(f"Founder genome: {population.founder_genome}")
    
    last_snapshot_time = 0.0
    
    while (population.time < max_time and 
           population.generation < max_generations and 
           not population.is_extinct()):
        
        # Update population
        stats = population.update(1.0)
        
        # Print progress
        if population.time - last_snapshot_time >= snapshot_interval:
            summary = population.get_population_summary()
            print(f"Time {population.time:.0f}: Gen {summary['generation']}, "
                  f"Pop {summary['alive_cells']}, "
                  f"Fitness {summary['average_fitness']:.2f}, "
                  f"Diversity {summary['genetic_diversity']:.3f}")
            last_snapshot_time = population.time
        
        # Early termination conditions
        if stats.alive_cells == 0:
            print("Population extinct!")
            break
        
        if stats.alive_cells > 2000:
            print("Population size limit reached, applying additional selection...")
            # Could implement additional population control here
    
    print(f"\nExperiment completed!")
    print(f"Final summary: {population.get_population_summary()}")
    
    return population


if __name__ == "__main__":
    # Test the population system
    print("Testing Population Management System")
    print("=" * 50)
    
    from environment import create_competitive_environment
    
    # Create test environment
    env = create_competitive_environment()
    
    # Run short evolution experiment
    population = run_evolution_experiment(
        environment=env,
        max_time=200.0,
        max_generations=20,
        snapshot_interval=50.0
    )
    
    # Show final results
    print(f"\nFinal population: {population}")
    fittest = population.get_fittest_cells(3)
    print(f"Top 3 fittest cells:")
    for i, cell in enumerate(fittest, 1):
        print(f"  {i}. {cell} - Fitness: {cell.get_fitness():.3f}")
        print(f"     Parameters: {cell.parameters}")
