"""
Hybrid Population Management
Manages populations of hybrid cells with problem-solving capabilities.
"""
import random
import time
from typing import List, Dict, Optional, Tuple

from hybrid_integrated_cell import HybridEvolutionaryCell, AIGenome
from genome import ComputationalGenome
from environment import Environment
from population import LineageNode


class HybridPopulation:
    """
    Manages a population of hybrid evolutionary cells with AI capabilities.
    """
    
    def __init__(self, initial_genome: ComputationalGenome = None, 
                 environment: Environment = None,
                 problem_frequency: int = 10):
        """
        Initialize population with a founder cell.
        
        Args:
            initial_genome: Starting genome (random if None)
            environment: Environmental conditions
            problem_frequency: How often cells solve problems (every N ticks)
        """
        if initial_genome is None:
            initial_genome = ComputationalGenome()
        
        if environment is None:
            from environment import create_benign_environment
            environment = create_benign_environment()
        
        self.environment = environment
        self.problem_frequency = problem_frequency
        
        # Create founder cell
        founder = HybridEvolutionaryCell(initial_genome, AIGenome())
        self.cells = [founder]
        
        # Population tracking
        self.generation = 0
        self.time = 0.0
        self.total_births = 1
        self.total_deaths = 0
        
        # Problem-solving tracking
        self.population_accuracy_history = []
        self.best_solver_history = []
        
        # Lineage tracking
        self.lineage_tree = {
            founder.cell_id: LineageNode(
                cell_id=founder.cell_id,
                genome_value=founder.genome.genome,
                parent_id=founder.parent_id,
                birth_time=0.0,
                death_time=None,
                generation=founder.generation,
                fitness=0.0,
                children=[]
            )
        }
        
        # Statistics history
        self.statistics_history = []
    
    def generate_problem(self) -> Tuple[float, float, float]:
        """Generate a random addition problem."""
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        return a, b, a + b
    
    def update(self, time_step: float):
        """
        Update population for one time step.
        
        Args:
            time_step: Time to advance
        """
        self.time += time_step
        
        # Check if it's a problem-solving tick
        problem_tick = (int(self.time) % self.problem_frequency) == 0
        
        # Get nutrient availability based on population
        nutrients = self.environment.get_nutrient_availability(len(self.cells))
        
        # Update environment
        self.environment.update(time_step)
        
        # Lists for population changes
        new_cells = []
        dead_cells = []
        
        # Update each cell
        total_accuracy = 0.0
        cells_with_problems = 0
        
        for cell in self.cells:
            if not cell.is_alive:
                continue
            
            # Update cell
            stats = cell.update(time_step, nutrients, problem_tick)
            
            # Track problem-solving performance
            if cell.total_problems > 0:
                total_accuracy += cell.problem_accuracy
                cells_with_problems += 1
            
            # Check for division
            if cell.can_divide:
                daughter = cell.divide()
                if daughter:
                    new_cells.append(daughter)
                    self.total_births += 1
                    
                    # Update lineage
                    parent_node = self.lineage_tree.get(cell.cell_id)
                    if parent_node:
                        daughter_node = LineageNode(
                            cell_id=daughter.cell_id,
                            genome_value=daughter.genome.genome,
                            parent_id=cell.cell_id,
                            birth_time=self.time,
                            death_time=None,
                            generation=daughter.generation,
                            fitness=0.0,
                            children=[]
                        )
                        parent_node.children.append(daughter_node.cell_id)
                        self.lineage_tree[daughter.cell_id] = daughter_node
            
            # Check for death
            if not cell.is_alive:
                dead_cells.append(cell)
                self.total_deaths += 1
                if cell.cell_id in self.lineage_tree:
                    self.lineage_tree[cell.cell_id].death_time = self.time
        
        # Add new cells
        self.cells.extend(new_cells)
        
        # Remove dead cells
        self.cells = [c for c in self.cells if c.is_alive]
        
        # Apply environmental selection
        self._apply_environmental_selection()
        
        # Update generation counter
        if self.cells:
            self.generation = max(c.generation for c in self.cells)
        
        # Record population accuracy
        if cells_with_problems > 0:
            avg_accuracy = total_accuracy / cells_with_problems
            self.population_accuracy_history.append({
                'time': self.time,
                'accuracy': avg_accuracy,
                'population_size': len(self.cells)
            })
        
        # Track best solver
        if self.cells:
            best_solver = max(self.cells, key=lambda c: c.problem_accuracy)
            self.best_solver_history.append({
                'time': self.time,
                'cell_id': best_solver.cell_id,
                'accuracy': best_solver.problem_accuracy,
                'genome': best_solver.genome.genome,
                'ai_weights': best_solver.ai_genome.weights.copy()
            })
        
        # Record statistics
        self._record_statistics()
    
    def _apply_environmental_selection(self):
        """Apply environmental selection pressures."""
        if not self.environment or len(self.cells) == 0:
            return
        
        # Calculate fitness for each cell
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
        
        # Apply carrying capacity
        carrying_capacity = self.environment.conditions.carrying_capacity
        if len(self.cells) > carrying_capacity:
            survivors = [cell for cell, _ in cell_fitnesses[:carrying_capacity]]
            eliminated = [cell for cell, _ in cell_fitnesses[carrying_capacity:]]
            
            for cell in eliminated:
                cell.is_alive = False
                self.total_deaths += 1
            
            self.cells = [c for c in self.cells if c.is_alive]
    
    def _record_statistics(self):
        """Record population statistics."""
        if not self.cells:
            return
        
        stats = {
            'time': self.time,
            'generation': self.generation,
            'population_size': len(self.cells),
            'average_fitness': sum(c.get_fitness() for c in self.cells) / len(self.cells),
            'average_atp': sum(c.atp for c in self.cells) / len(self.cells),
            'average_mass': sum(c.mass for c in self.cells) / len(self.cells),
            'average_accuracy': sum(c.problem_accuracy for c in self.cells) / len(self.cells),
            'best_accuracy': max(c.problem_accuracy for c in self.cells),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths
        }
        
        self.statistics_history.append(stats)
    
    def get_population_summary(self) -> Dict:
        """Get summary of current population state."""
        if not self.cells:
            return {
                'generation': self.generation,
                'time': self.time,
                'population_size': 0,
                'alive_cells': 0,
                'average_fitness': 0,
                'average_accuracy': 0,
                'best_accuracy': 0,
                'environment': str(self.environment)
            }
        
        alive_cells = [c for c in self.cells if c.is_alive]
        
        return {
            'generation': self.generation,
            'time': self.time,
            'population_size': len(self.cells),
            'alive_cells': len(alive_cells),
            'average_fitness': sum(c.get_fitness() for c in alive_cells) / len(alive_cells),
            'average_accuracy': sum(c.problem_accuracy for c in alive_cells) / len(alive_cells),
            'best_accuracy': max(c.problem_accuracy for c in alive_cells),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'environment': str(self.environment)
        }
    
    def get_best_problem_solver(self) -> Optional[HybridEvolutionaryCell]:
        """Get the cell with highest problem-solving accuracy."""
        if not self.cells:
            return None
        return max(self.cells, key=lambda c: c.problem_accuracy)
    
    def get_fittest_cells(self, n: int = 10) -> List[HybridEvolutionaryCell]:
        """Get top n fittest cells."""
        return sorted(self.cells, key=lambda c: c.get_fitness(), reverse=True)[:n]


def run_hybrid_evolution_experiment(initial_genome: ComputationalGenome = None,
                                  environment: Environment = None,
                                  max_time: float = 1000.0,
                                  max_generations: int = 100,
                                  snapshot_interval: float = 100.0,
                                  problem_frequency: int = 10) -> HybridPopulation:
    """
    Run a hybrid evolution experiment with problem-solving cells.
    
    Args:
        initial_genome: Starting genome
        environment: Environmental conditions
        max_time: Maximum simulation time
        max_generations: Maximum number of generations
        snapshot_interval: How often to print progress
        problem_frequency: How often cells solve problems
        
    Returns:
        Final population
    """
    print(f"Starting hybrid evolution experiment with problem-solving cells")
    
    if initial_genome is None:
        initial_genome = ComputationalGenome()
    
    print(f"Founder genome: {initial_genome}")
    
    # Create population
    population = HybridPopulation(initial_genome, environment, problem_frequency)
    
    last_snapshot = 0.0
    start_time = time.time()
    
    while population.time < max_time and population.generation < max_generations:
        # Update population
        population.update(1.0)
        
        # Progress snapshots
        if population.time - last_snapshot >= snapshot_interval:
            summary = population.get_population_summary()
            print(f"Time {population.time:.0f}: Gen {summary['generation']}, "
                  f"Pop {summary['population_size']}, "
                  f"Fitness {summary['average_fitness']:.2f}, "
                  f"Accuracy {summary['average_accuracy']:.1f}%, "
                  f"Best {summary['best_accuracy']:.1f}%")
            last_snapshot = population.time
        
        # Check for extinction
        if len(population.cells) == 0:
            print("Population extinct!")
            break
    
    runtime = time.time() - start_time
    print(f"\nExperiment completed in {runtime:.2f} seconds!")
    
    # Final summary
    summary = population.get_population_summary()
    print(f"Final summary: {summary}")
    
    # Show best problem solver
    best_solver = population.get_best_problem_solver()
    if best_solver:
        print(f"\nBest problem solver:")
        print(f"  Cell ID: {best_solver.cell_id}")
        print(f"  Generation: {best_solver.generation}")
        print(f"  Accuracy: {best_solver.problem_accuracy:.1f}%")
        print(f"  Problems solved: {best_solver.problems_solved}/{best_solver.total_problems}")
        print(f"  AI weights: {[f'{w:.3f}' for w in best_solver.ai_genome.weights]}")
        print(f"  Biological traits: Energy={best_solver.parameters['energy_efficiency']:.2f}, "
              f"Growth={best_solver.parameters['growth_rate']:.2f}")
    
    return population


# Example usage
if __name__ == "__main__":
    from environment import create_benign_environment, create_competitive_environment
    
    print("=" * 60)
    print("HYBRID CELL EVOLUTION WITH PROBLEM SOLVING")
    print("=" * 60)
    
    # Run experiment with competitive environment
    env = create_competitive_environment()
    population = run_hybrid_evolution_experiment(
        environment=env,
        max_time=500.0,
        max_generations=50,
        snapshot_interval=50.0,
        problem_frequency=10
    )
    
    print("\n" + "=" * 60)
    print("EVOLUTION OF PROBLEM-SOLVING ABILITY")
    print("=" * 60)
    
    # Show accuracy evolution
    if population.population_accuracy_history:
        print("\nAccuracy over time:")
        for i in range(0, len(population.population_accuracy_history), 10):
            record = population.population_accuracy_history[i]
            print(f"  Time {record['time']:.0f}: {record['accuracy']:.1f}% "
                  f"(Pop size: {record['population_size']})")
