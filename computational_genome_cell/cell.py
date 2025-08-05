"""
Evolutionary Cell Implementation
Cell class controlled by computational genome with organelles and binary fission.
"""
import random
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from genome import ComputationalGenome


@dataclass
class CellularComponents:
    """Data class to track cellular component quantities."""
    mitochondria: int = 10
    ribosomes: int = 30
    proteins: float = 100.0
    lipids: float = 50.0
    nucleotides: float = 30.0


class Organelle:
    """Base class for cellular organelles."""
    
    def __init__(self, efficiency: float = 1.0):
        self.efficiency = efficiency
        self.damage = 0.0
        self.age = 0
    
    def age_organelle(self, time_step: float):
        """Age the organelle and accumulate damage."""
        self.age += time_step
        # Organelles accumulate damage over time
        damage_rate = 0.001 * time_step
        self.damage += damage_rate * random.uniform(0.5, 1.5)
        self.damage = min(1.0, self.damage)  # Cap at 100% damage
    
    def get_current_efficiency(self) -> float:
        """Get current efficiency accounting for damage."""
        return self.efficiency * (1.0 - self.damage)


class Mitochondrion(Organelle):
    """Mitochondria - cellular powerhouses producing ATP."""
    
    def __init__(self, efficiency: float = 1.0):
        super().__init__(efficiency)
        self.atp_production_rate = 2.0  # Base ATP per time unit
    
    def produce_atp(self, energy_efficiency_gene: float, time_step: float) -> float:
        """
        Produce ATP based on genetic control and current state.
        
        Args:
            energy_efficiency_gene: Gene expression value for energy efficiency
            time_step: Time elapsed
            
        Returns:
            Amount of ATP produced
        """
        base_production = self.atp_production_rate * time_step
        genetic_modifier = energy_efficiency_gene
        efficiency_modifier = self.get_current_efficiency()
        
        atp_produced = base_production * genetic_modifier * efficiency_modifier
        
        # Age the mitochondrion
        self.age_organelle(time_step)
        
        return atp_produced


class Ribosome(Organelle):
    """Ribosomes - protein synthesis machinery."""
    
    def __init__(self, efficiency: float = 1.0):
        super().__init__(efficiency)
        self.protein_synthesis_rate = 1.5  # Base protein per time unit
    
    def synthesize_proteins(self, growth_rate_gene: float, atp_available: float, 
                          time_step: float) -> float:
        """
        Synthesize proteins based on genetic control and available energy.
        
        Args:
            growth_rate_gene: Gene expression value for growth rate
            atp_available: Available ATP for protein synthesis
            time_step: Time elapsed
            
        Returns:
            Amount of protein synthesized
        """
        # Protein synthesis requires ATP
        atp_required_per_protein = 0.5
        max_proteins_by_energy = atp_available / atp_required_per_protein
        
        base_production = self.protein_synthesis_rate * time_step
        genetic_modifier = growth_rate_gene
        efficiency_modifier = self.get_current_efficiency()
        
        potential_proteins = base_production * genetic_modifier * efficiency_modifier
        actual_proteins = min(potential_proteins, max_proteins_by_energy)
        
        # Age the ribosome
        self.age_organelle(time_step)
        
        return actual_proteins


class Nucleus(Organelle):
    """Nucleus - contains genome and controls DNA replication."""
    
    def __init__(self, genome: ComputationalGenome):
        super().__init__(efficiency=1.0)
        self.genome = genome
        self.dna_replication_progress = 0.0
        self.is_replicating = False
    
    def start_dna_replication(self) -> bool:
        """
        Start DNA replication process.
        
        Returns:
            True if replication started successfully
        """
        if not self.is_replicating:
            self.is_replicating = True
            self.dna_replication_progress = 0.0
            return True
        return False
    
    def continue_dna_replication(self, atp_available: float, time_step: float) -> bool:
        """
        Continue DNA replication process.
        
        Args:
            atp_available: Available ATP for replication
            time_step: Time elapsed
            
        Returns:
            True if replication is complete
        """
        if not self.is_replicating:
            return False
        
        # DNA replication requires significant ATP
        atp_required_per_progress = 10.0
        max_progress_by_energy = atp_available / atp_required_per_progress
        
        base_replication_rate = 0.1 * time_step  # 10 time units to complete
        efficiency_modifier = self.get_current_efficiency()
        
        potential_progress = base_replication_rate * efficiency_modifier
        actual_progress = min(potential_progress, max_progress_by_energy)
        
        self.dna_replication_progress += actual_progress
        
        # Age the nucleus
        self.age_organelle(time_step)
        
        if self.dna_replication_progress >= 1.0:
            self.is_replicating = False
            return True
        
        return False
    
    def get_replicated_genome(self) -> ComputationalGenome:
        """
        Get a mutated copy of the genome after replication.
        
        Returns:
            New ComputationalGenome with potential mutations
        """
        return self.genome.mutate()


class EvolutionaryCell:
    """
    Cell controlled by computational genome with realistic cellular processes.
    """
    
    def __init__(self, genome: ComputationalGenome, cell_id: int = None):
        """
        Initialize cell with genome and cellular components.
        
        Args:
            genome: ComputationalGenome controlling this cell
            cell_id: Unique identifier for this cell
        """
        self.genome = genome
        self.cell_id = cell_id or random.randint(100000, 999999)
        self.generation = genome.generation
        self.birth_time = time.time()
        self.age = 0.0
        
        # Get expressed parameters from genome
        self.parameters = genome.express_genes()
        
        # Initialize cellular components
        self.components = CellularComponents()
        self.mass = 100.0  # Starting mass
        self.atp = 50.0    # Starting ATP
        self.max_atp = 200.0
        
        # Initialize organelles
        self.mitochondria = [Mitochondrion() for _ in range(self.components.mitochondria)]
        self.ribosomes = [Ribosome() for _ in range(self.components.ribosomes)]
        self.nucleus = Nucleus(genome)
        
        # Cell state
        self.is_alive = True
        self.can_divide = False
        self.energy_debt = 0.0
    
    MIN_RIBOSOMES_FOR_DIVISION = 10

    def update(self, time_step: float, nutrients_available: float = 1.0) -> Dict:
        """
        Update cell state for one time step.
        
        Args:
            time_step: Time elapsed
            nutrients_available: Environmental nutrient availability (0-1)
            
        Returns:
            Dictionary with update statistics
        """
        if not self.is_alive:
            return {'status': 'dead'}
        
        self.age += time_step
        
        # 1. ATP Production (Mitochondria)
        atp_produced = 0.0
        for mitochondrion in self.mitochondria:
            atp_produced += mitochondrion.produce_atp(
                self.parameters['energy_efficiency'], 
                time_step
            )
        
        # Nutrient availability affects ATP production
        atp_produced *= nutrients_available
        self.atp += atp_produced
        self.atp = min(self.atp, self.max_atp)
        
        # 2. Protein Synthesis (Ribosomes)
        proteins_synthesized = 0.0
        if len(self.ribosomes) > 0:
            atp_for_proteins = min(self.atp * 0.6, self.atp)  # Use up to 60% of ATP
            atp_per_ribosome = atp_for_proteins / len(self.ribosomes)
            for ribosome in self.ribosomes:
                proteins_made = ribosome.synthesize_proteins(
                    self.parameters['growth_rate'],
                    atp_per_ribosome,
                    time_step
                )
                proteins_synthesized += proteins_made
                self.atp -= proteins_made * 0.5  # ATP cost
            self.components.proteins += proteins_synthesized
        else:
            # No ribosomes, no protein synthesis
            proteins_synthesized = 0.0
        
        # 3. Growth and Mass Increase
        growth_efficiency = 0.8  # 80% of proteins convert to mass
        mass_gained = proteins_synthesized * growth_efficiency
        self.mass += mass_gained

        # 4. Ribosome Synthesis (convert proteins to ribosomes)
        ribosome_protein_cost = 20.0  # Number of proteins needed for 1 ribosome
        new_ribosomes = 0
        while self.components.proteins >= ribosome_protein_cost:
            self.components.proteins -= ribosome_protein_cost
            self.ribosomes.append(Ribosome())
            new_ribosomes += 1
        self.components.ribosomes = len(self.ribosomes)

        # 5. Maintenance Costs
        maintenance_cost = self.mass * 0.01 * time_step  # 1% per time unit
        self.atp -= maintenance_cost

        # 6. Check for Division Readiness (require enough ribosomes)
        can_attempt_division = (
            self.mass >= self.parameters['division_threshold']
            and len(self.ribosomes) >= self.MIN_RIBOSOMES_FOR_DIVISION
            and not self.nucleus.is_replicating
        )
        if can_attempt_division:
            self.nucleus.start_dna_replication()

        # 7. DNA Replication
        replication_complete = False
        if self.nucleus.is_replicating:
            atp_for_replication = min(self.atp * 0.3, self.atp)  # Use up to 30% of ATP
            replication_complete = self.nucleus.continue_dna_replication(
                atp_for_replication, time_step
            )
            if atp_for_replication > 0:
                self.atp -= min(atp_for_replication, 10.0 * time_step)

        if replication_complete:
            self.can_divide = True
        
        # 7. Death Conditions
        if self.atp < -50:  # Severe energy debt
            self.energy_debt += abs(self.atp + 50)
            if self.energy_debt > 100:
                self.is_alive = False
        
        # Natural aging death
        if self.age > 1000:  # Maximum lifespan
            death_probability = (self.age - 1000) * 0.001
            if random.random() < death_probability:
                self.is_alive = False
        
        return {
            'status': 'alive' if self.is_alive else 'dead',
            'atp_produced': atp_produced,
            'proteins_synthesized': proteins_synthesized,
            'mass_gained': mass_gained,
            'current_mass': self.mass,
            'current_atp': self.atp,
            'can_divide': self.can_divide,
            'replication_progress': self.nucleus.dna_replication_progress,
            'new_ribosomes': new_ribosomes,
            'ribosome_count': len(self.ribosomes)
        }
    
    def divide(self) -> Optional['EvolutionaryCell']:
        """
        Perform binary fission to create daughter cell.
        
        Returns:
            Daughter EvolutionaryCell or None if division not possible
        """
        if not self.can_divide or not self.is_alive:
            return None
        
        # Get mutated genome for daughter
        daughter_genome = self.nucleus.get_replicated_genome()
        
        # Create daughter cell
        daughter = EvolutionaryCell(daughter_genome)
        
        # Divide cellular components
        self.mass *= 0.5
        daughter.mass = self.mass
        
        self.atp *= 0.5
        daughter.atp = self.atp
        
        # Divide organelles, ensure at least 1 of each in both cells
        total_mito = len(self.mitochondria)
        half_mito = max(1, total_mito // 2)
        if total_mito - half_mito == 0:
            half_mito = total_mito - 1  # Ensure at least 1 in daughter
        daughter.mitochondria = self.mitochondria[half_mito:]
        self.mitochondria = self.mitochondria[:half_mito]
        if len(self.mitochondria) == 0:
            self.mitochondria = [Mitochondrion()]
        if len(daughter.mitochondria) == 0:
            daughter.mitochondria = [Mitochondrion()]

        total_ribo = len(self.ribosomes)
        if total_ribo <= 1:
            # Only one ribosome, give one to each
            self.ribosomes = [Ribosome()]
            daughter.ribosomes = [Ribosome()]
        else:
            half_ribo = total_ribo // 2
            daughter.ribosomes = self.ribosomes[half_ribo:]
            self.ribosomes = self.ribosomes[:half_ribo]
            if len(self.ribosomes) == 0:
                self.ribosomes = [Ribosome()]
            if len(daughter.ribosomes) == 0:
                daughter.ribosomes = [Ribosome()]

        # Update components counts
        self.components.mitochondria = len(self.mitochondria)
        self.components.ribosomes = len(self.ribosomes)
        daughter.components.mitochondria = len(daughter.mitochondria)
        daughter.components.ribosomes = len(daughter.ribosomes)
        
        # Reset division state
        self.can_divide = False
        self.nucleus.dna_replication_progress = 0.0
        
        return daughter
    
    def get_fitness(self) -> float:
        """
        Calculate cell fitness based on survival and reproduction potential.
        
        Returns:
            Fitness value (higher is better)
        """
        if not self.is_alive:
            return 0.0
        
        # Base fitness from being alive
        fitness = 1.0
        
        # Bonus for energy efficiency
        if self.atp > 0:
            fitness += self.atp / 100.0
        
        # Bonus for growth potential
        fitness += (self.mass / 100.0) * 0.5
        
        # Bonus for division readiness
        if self.can_divide:
            fitness += 2.0
        
        # Penalty for age
        age_penalty = self.age / 1000.0
        fitness -= age_penalty
        
        return max(0.0, fitness)
    
    def to_dict(self) -> Dict:
        """
        Convert cell to dictionary for serialization.
        
        Returns:
            Dictionary representation of cell
        """
        return {
            'cell_id': self.cell_id,
            'generation': self.generation,
            'age': self.age,
            'mass': self.mass,
            'atp': self.atp,
            'is_alive': self.is_alive,
            'can_divide': self.can_divide,
            'fitness': self.get_fitness(),
            'genome': self.genome.to_dict(),
            'parameters': self.parameters,
            'components': {
                'mitochondria': self.components.mitochondria,
                'ribosomes': self.components.ribosomes,
                'proteins': self.components.proteins
            }
        }
    
    def __str__(self) -> str:
        """String representation of cell."""
        status = "alive" if self.is_alive else "dead"
        return f"Cell(id={self.cell_id}, gen={self.generation}, mass={self.mass:.1f}, {status})"
    
    def __repr__(self) -> str:
        """Detailed representation of cell."""
        return f"EvolutionaryCell(id={self.cell_id}, genome=0x{self.genome.genome:08X})"


if __name__ == "__main__":
    # Test the cell system
    print("Testing Evolutionary Cell System")
    print("=" * 40)
    
    # Create test cell
    genome = ComputationalGenome()
    cell = EvolutionaryCell(genome)
    
    print(f"Initial cell: {cell}")
    print(f"Parameters: {cell.parameters}")
    print(f"Initial fitness: {cell.get_fitness():.2f}")
    
    # Simulate cell growth
    print("\nSimulating cell growth...")
    for step in range(20):
        stats = cell.update(time_step=1.0, nutrients_available=1.0)
        if step % 5 == 0:
            print(f"Step {step}: Mass={cell.mass:.1f}, ATP={cell.atp:.1f}, "
                  f"Can divide={cell.can_divide}, Fitness={cell.get_fitness():.2f}")
        
        if cell.can_divide:
            daughter = cell.divide()
            if daughter:
                print(f"Division occurred! Daughter: {daughter}")
                break
    
    print(f"\nFinal cell state: {cell}")
