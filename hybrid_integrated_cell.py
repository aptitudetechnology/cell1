"""
Hybrid Integrated Cell System
Combines the existing computational genome cell evolution with AI problem-solving capabilities.
"""
import random
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from genome import ComputationalGenome
from cell import Organelle, CellularComponents


class AIGenome:
    """Simple AI genome for problem-solving capability."""
    
    def __init__(self, weights: List[float] = None):
        """Initialize AI genome with neural network weights."""
        if weights is None:
            # 4 weights for simple network: 2 inputs -> 2 hidden -> 1 output
            self.weights = [random.uniform(-1.0, 1.0) for _ in range(4)]
        else:
            self.weights = weights.copy()
    
    def solve_problem(self, a: float, b: float) -> float:
        """Use simple neural network to solve addition problem."""
        # Hidden layer activations
        h1 = a * self.weights[0] + b * self.weights[1]
        h2 = a * self.weights[2] + b * self.weights[3]
        
        # Output is sum of hidden activations
        return h1 + h2
    
    def mutate(self, mutation_rate: float) -> 'AIGenome':
        """Create mutated copy of AI genome."""
        new_weights = self.weights.copy()
        for i in range(len(new_weights)):
            if random.random() < mutation_rate:
                new_weights[i] += random.uniform(-0.1, 0.1)
                new_weights[i] = max(-1.0, min(1.0, new_weights[i]))
        return AIGenome(new_weights)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {'weights': self.weights}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AIGenome':
        """Create from dictionary."""
        return cls(weights=data['weights'])


class HybridEvolutionaryCell:
    """
    Enhanced evolutionary cell with both biological and AI capabilities.
    Combines the existing cell system with problem-solving abilities.
    """
    
    # Class variable for unique cell IDs
    _next_cell_id = 1
    
    def __init__(self, genome: ComputationalGenome, ai_genome: AIGenome = None, 
                 parent_id: Optional[int] = None):
        """
        Initialize hybrid cell with both genomes.
        
        Args:
            genome: Computational genome for biological traits
            ai_genome: AI genome for problem-solving
            parent_id: ID of parent cell if any
        """
        # Existing cell properties
        self.cell_id = HybridEvolutionaryCell._next_cell_id
        HybridEvolutionaryCell._next_cell_id += 1
        
        self.genome = genome
        self.ai_genome = ai_genome if ai_genome else AIGenome()
        self.generation = 0
        self.parent_id = parent_id
        
        # Express genes to get parameters
        self.parameters = genome.express_genes()
        
        # Cell state
        self.age = 0.0
        self.mass = 50.0
        self.atp = 100.0
        self.max_atp = 500.0
        self.is_alive = True
        self.can_divide = False
        self.energy_debt = 0.0
        
        # Problem-solving stats
        self.problems_solved = 0
        self.total_problems = 0
        self.problem_accuracy = 0.0
        
        # Components (simplified for hybrid model)
        self.components = CellularComponents(
            mitochondria=10,
            ribosomes=15,
            proteins=100.0,
            lipids=50.0,
            nucleotides=30.0
        )
        
        # Initialize organelles based on components
        self._initialize_organelles()
    
    def _initialize_organelles(self):
        """Initialize cellular organelles."""
        # Import here to avoid circular imports
        from cell import Mitochondrion, Ribosome, Nucleus
        
        self.mitochondria = [Mitochondrion() for _ in range(self.components.mitochondria)]
        self.ribosomes = [Ribosome() for _ in range(self.components.ribosomes)]
        self.nucleus = Nucleus(self.genome, mutation_rate=self.parameters['mutation_rate'])
    
    def solve_math_problem(self, a: float, b: float) -> Tuple[float, float]:
        """
        Solve addition problem and return guess and reward.
        
        Args:
            a, b: Numbers to add
            
        Returns:
            (guess, energy_reward)
        """
        guess = self.ai_genome.solve_problem(a, b)
        correct = a + b
        error = abs(correct - guess)
        
        # Energy reward based on accuracy (max 10 energy)
        reward = max(0, 10 - error)
        
        # Track statistics
        self.total_problems += 1
        if error < 0.5:  # Count as correct if close enough
            self.problems_solved += 1
        self.problem_accuracy = (self.problems_solved / self.total_problems) * 100
        
        return guess, reward
    
    def update(self, time_step: float, nutrients_available: float = 1.0, 
               problem_tick: bool = False) -> Dict:
        """
        Update cell state for one time step.
        
        Args:
            time_step: Time elapsed
            nutrients_available: Environmental nutrient availability
            problem_tick: Whether to solve a problem this tick
            
        Returns:
            Dictionary with update statistics
        """
        if not self.is_alive:
            return {'status': 'dead'}
        
        self.age += time_step
        
        # 1. ATP Production (existing logic)
        atp_produced = 0.0
        for mitochondrion in self.mitochondria:
            atp_produced += mitochondrion.produce_atp(
                self.parameters['energy_efficiency'], 
                time_step
            )
        atp_produced *= nutrients_available
        self.atp += atp_produced
        
        # 2. Problem Solving Phase (new)
        problem_reward = 0.0
        if problem_tick:
            a = random.randint(0, 10)
            b = random.randint(0, 10)
            guess, reward = self.solve_math_problem(a, b)
            self.atp += reward
            problem_reward = reward
        
        # 3. Maintenance and Growth (existing logic)
        # Energy cost of living
        maintenance_cost = self.mass * 0.01 * time_step
        self.atp -= maintenance_cost
        
        # Protein synthesis and growth
        proteins_synthesized = 0.0
        if len(self.ribosomes) > 0 and self.atp > 0:
            atp_for_proteins = min(self.atp * 0.4, self.atp)
            atp_per_ribosome = atp_for_proteins / len(self.ribosomes)
            
            for ribosome in self.ribosomes:
                proteins_made = ribosome.synthesize_proteins(
                    self.parameters['growth_rate'],
                    atp_per_ribosome,
                    time_step
                )
                proteins_synthesized += proteins_made
                self.atp -= proteins_made * 0.5
        
        # Mass increase
        mass_gained = proteins_synthesized * 0.8
        self.mass += mass_gained
        
        # 4. Check division readiness
        if self.mass >= self.parameters['division_threshold'] and not self.nucleus.is_replicating:
            self.nucleus.start_dna_replication()
        
        # 5. DNA replication
        replication_complete = False
        if self.nucleus.is_replicating:
            atp_for_replication = min(self.atp * 0.3, 10.0 * time_step)
            replication_complete = self.nucleus.continue_dna_replication(
                atp_for_replication, time_step
            )
            self.atp -= atp_for_replication
        
        if replication_complete:
            self.can_divide = True
        
        # 6. Death conditions
        if self.atp < -50:
            self.energy_debt += abs(self.atp + 50)
            if self.energy_debt > 100:
                self.is_alive = False
        
        # Cap ATP
        self.atp = min(self.atp, self.max_atp)
        
        return {
            'status': 'alive' if self.is_alive else 'dead',
            'atp_produced': atp_produced,
            'problem_reward': problem_reward,
            'proteins_synthesized': proteins_synthesized,
            'mass_gained': mass_gained,
            'current_mass': self.mass,
            'current_atp': self.atp,
            'can_divide': self.can_divide,
            'problem_accuracy': self.problem_accuracy
        }
    
    def divide(self) -> Optional['HybridEvolutionaryCell']:
        """
        Perform binary fission to create daughter cell.
        
        Returns:
            Daughter HybridEvolutionaryCell or None
        """
        if not self.can_divide or not self.is_alive:
            return None
        
        # Get mutated biological genome
        daughter_genome = self.nucleus.get_replicated_genome()
        
        # Get mutated AI genome
        daughter_ai_genome = self.ai_genome.mutate(self.parameters['mutation_rate'])
        
        # Create daughter cell
        daughter = HybridEvolutionaryCell(
            daughter_genome, 
            daughter_ai_genome,
            parent_id=self.cell_id
        )
        daughter.generation = self.generation + 1
        
        # Divide resources
        self.mass *= 0.5
        daughter.mass = self.mass
        self.atp *= 0.5
        daughter.atp = self.atp
        
        # Divide organelles
        half_mito = len(self.mitochondria) // 2
        daughter.mitochondria = self.mitochondria[half_mito:]
        self.mitochondria = self.mitochondria[:half_mito]
        
        half_ribo = len(self.ribosomes) // 2
        daughter.ribosomes = self.ribosomes[half_ribo:]
        self.ribosomes = self.ribosomes[:half_ribo]
        
        # Update component counts
        self.components.mitochondria = len(self.mitochondria)
        self.components.ribosomes = len(self.ribosomes)
        daughter.components.mitochondria = len(daughter.mitochondria)
        daughter.components.ribosomes = len(daughter.ribosomes)
        
        # Reset division state
        self.can_divide = False
        self.nucleus.is_replicating = False
        self.nucleus.dna_replication_progress = 0.0
        
        return daughter
    
    def get_fitness(self) -> float:
        """
        Calculate overall fitness combining biological and AI performance.
        
        Returns:
            Fitness score
        """
        # Base biological fitness
        bio_fitness = (self.atp / 100.0) * (self.mass / 100.0)
        
        # AI fitness component
        ai_fitness = self.problem_accuracy / 100.0
        
        # Combined fitness (weighted average)
        # Higher weight on problem-solving to encourage AI evolution
        combined_fitness = bio_fitness * 0.4 + ai_fitness * 0.6
        
        return max(0.1, combined_fitness)
    
    def to_dict(self) -> Dict:
        """Convert cell to dictionary for serialization."""
        return {
            'cell_id': self.cell_id,
            'genome': self.genome.to_dict(),
            'ai_genome': self.ai_genome.to_dict(),
            'generation': self.generation,
            'parent_id': self.parent_id,
            'age': self.age,
            'mass': self.mass,
            'atp': self.atp,
            'is_alive': self.is_alive,
            'problems_solved': self.problems_solved,
            'total_problems': self.total_problems,
            'problem_accuracy': self.problem_accuracy,
            'parameters': self.parameters
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"HybridCell(id={self.cell_id}, gen={self.generation}, "
                f"bio={self.genome}, ai_accuracy={self.problem_accuracy:.1f}%)")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Hybrid Evolutionary Cell")
    print("=" * 50)
    
    # Create test genome and cell
    genome = ComputationalGenome()
    ai_genome = AIGenome()
    cell = HybridEvolutionaryCell(genome, ai_genome)
    
    print(f"Initial cell: {cell}")
    print(f"Biological parameters: {cell.parameters}")
    print(f"AI weights: {cell.ai_genome.weights}")
    
    # Test problem solving
    a, b = 5, 7
    guess, reward = cell.solve_math_problem(a, b)
    print(f"\nProblem: {a} + {b} = ?")
    print(f"Cell's guess: {guess:.2f}")
    print(f"Energy reward: {reward:.2f}")
    
    # Test update with problem solving
    stats = cell.update(1.0, nutrients_available=1.0, problem_tick=True)
    print(f"\nUpdate stats: {stats}")
    
    # Test division
    if cell.mass > 80:
        cell.can_divide = True
        daughter = cell.divide()
        if daughter:
            print(f"\nDaughter cell: {daughter}")
            print(f"Daughter AI weights: {daughter.ai_genome.weights}")
