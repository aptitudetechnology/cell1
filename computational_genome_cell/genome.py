"""
Computational Genome System
Implements a 32-bit genome with 4 essential genes for cellular evolution simulation.
"""
import random
import struct
from typing import Dict, List, Tuple
import json


class ComputationalGenome:
    """
    32-bit genome divided into 4 essential genes (8 bits each).
    Controls cellular parameters through gene expression.
    """
    
    # Gene positions in the 32-bit genome
    GENE_POSITIONS = {
        'energy_efficiency': (0, 8),    # bits 0-7
        'growth_rate': (8, 16),         # bits 8-15
        'division_threshold': (16, 24),  # bits 16-23
        'mutation_rate': (24, 32)       # bits 24-31
    }
    
    def __init__(self, genome_value: int = None):
        """
        Initialize genome with either a specific value or random generation.
        
        Args:
            genome_value: 32-bit integer representing the genome (None for random)
        """
        if genome_value is None:
            self.genome = random.randint(0, 2**32 - 1)
        else:
            self.genome = genome_value & 0xFFFFFFFF  # Ensure 32-bit
        
        self.generation = 0
        self.mutation_history = []
        self.parent_genome = None
    
    def get_gene_value(self, gene_name: str) -> int:
        """
        Extract the 8-bit value for a specific gene.
        
        Args:
            gene_name: Name of the gene to extract
            
        Returns:
            8-bit integer value (0-255)
        """
        if gene_name not in self.GENE_POSITIONS:
            raise ValueError(f"Unknown gene: {gene_name}")
        
        start_bit, end_bit = self.GENE_POSITIONS[gene_name]
        # Extract bits by shifting and masking
        shifted = self.genome >> start_bit
        masked = shifted & ((1 << (end_bit - start_bit)) - 1)
        return masked
    
    def set_gene_value(self, gene_name: str, value: int):
        """
        Set the value for a specific gene.
        
        Args:
            gene_name: Name of the gene to modify
            value: New 8-bit value (0-255)
        """
        if gene_name not in self.GENE_POSITIONS:
            raise ValueError(f"Unknown gene: {gene_name}")
        
        value = max(0, min(255, value))  # Clamp to 8-bit range
        start_bit, end_bit = self.GENE_POSITIONS[gene_name]
        
        # Clear the existing gene bits
        mask = ~(((1 << (end_bit - start_bit)) - 1) << start_bit)
        self.genome &= mask
        
        # Set the new gene value
        self.genome |= (value << start_bit)
    
    def express_genes(self) -> Dict[str, float]:
        """
        Convert raw gene values to cellular parameters using scaling functions.
        
        Returns:
            Dictionary mapping parameter names to expressed values
        """
        parameters = {}
        
        # Energy efficiency: 0-255 -> 0.1-2.0 (ATP production multiplier)
        energy_raw = self.get_gene_value('energy_efficiency')
        parameters['energy_efficiency'] = 0.1 + (energy_raw / 255.0) * 1.9
        
        # Growth rate: 0-255 -> 0.1-3.0 (protein synthesis speed multiplier)
        growth_raw = self.get_gene_value('growth_rate')
        parameters['growth_rate'] = 0.1 + (growth_raw / 255.0) * 2.9
        
        # Division threshold: 0-255 -> 50-200 (mass units required for division)
        division_raw = self.get_gene_value('division_threshold')
        parameters['division_threshold'] = 50 + (division_raw / 255.0) * 150
        
        # Mutation rate: 0-255 -> 0.0001-0.01 (probability per replication)
        mutation_raw = self.get_gene_value('mutation_rate')
        parameters['mutation_rate'] = 0.0001 + (mutation_raw / 255.0) * 0.0099
        
        return parameters
    
    def mutate(self) -> 'ComputationalGenome':
        """
        Create a mutated copy of this genome through point mutations.
        
        Returns:
            New ComputationalGenome with potential mutations
        """
        # Get current mutation rate
        mutation_rate = self.express_genes()['mutation_rate']
        
        # Create copy of genome
        new_genome_value = self.genome
        mutations = []
        
        # Check each bit for potential mutation
        for bit_position in range(32):
            if random.random() < mutation_rate:
                # Flip this bit
                new_genome_value ^= (1 << bit_position)
                mutations.append(bit_position)
        
        # Create new genome instance
        new_genome = ComputationalGenome(new_genome_value)
        new_genome.generation = self.generation + 1
        new_genome.parent_genome = self.genome
        new_genome.mutation_history = self.mutation_history + [mutations] if mutations else self.mutation_history
        
        return new_genome
    
    def hamming_distance(self, other: 'ComputationalGenome') -> int:
        """
        Calculate genetic distance between two genomes.
        
        Args:
            other: Another ComputationalGenome to compare with
            
        Returns:
            Number of differing bits
        """
        xor_result = self.genome ^ other.genome
        return bin(xor_result).count('1')
    
    def to_binary_string(self) -> str:
        """
        Convert genome to binary string representation.
        
        Returns:
            32-character binary string
        """
        return format(self.genome, '032b')
    
    def to_dict(self) -> Dict:
        """
        Convert genome to dictionary for serialization.
        
        Returns:
            Dictionary representation of genome
        """
        return {
            'genome_value': self.genome,
            'generation': self.generation,
            'mutation_history': self.mutation_history,
            'parent_genome': self.parent_genome,
            'genes': {name: self.get_gene_value(name) for name in self.GENE_POSITIONS.keys()},
            'expressed_parameters': self.express_genes(),
            'binary_representation': self.to_binary_string()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComputationalGenome':
        """
        Create genome from dictionary representation.
        
        Args:
            data: Dictionary containing genome data
            
        Returns:
            ComputationalGenome instance
        """
        genome = cls(data['genome_value'])
        genome.generation = data.get('generation', 0)
        genome.mutation_history = data.get('mutation_history', [])
        genome.parent_genome = data.get('parent_genome')
        return genome
    
    def __str__(self) -> str:
        """String representation of genome."""
        genes = {name: self.get_gene_value(name) for name in self.GENE_POSITIONS.keys()}
        return f"Genome(gen={self.generation}, genes={genes})"
    
    def __repr__(self) -> str:
        """Detailed representation of genome."""
        return f"ComputationalGenome(0x{self.genome:08X}, generation={self.generation})"


def create_random_population(size: int) -> List[ComputationalGenome]:
    """
    Create a population of random genomes.
    
    Args:
        size: Number of genomes to create
        
    Returns:
        List of ComputationalGenome instances
    """
    return [ComputationalGenome() for _ in range(size)]


def analyze_genetic_diversity(population: List[ComputationalGenome]) -> Dict:
    """
    Analyze genetic diversity in a population.
    
    Args:
        population: List of ComputationalGenome instances
        
    Returns:
        Dictionary with diversity statistics
    """
    if not population:
        return {'diversity': 0, 'unique_genomes': 0, 'avg_hamming_distance': 0}
    
    unique_genomes = len(set(g.genome for g in population))
    
    # Calculate average pairwise Hamming distance
    total_distance = 0
    comparisons = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            total_distance += population[i].hamming_distance(population[j])
            comparisons += 1
    
    avg_hamming = total_distance / comparisons if comparisons > 0 else 0
    
    return {
        'population_size': len(population),
        'unique_genomes': unique_genomes,
        'genetic_diversity': unique_genomes / len(population),
        'avg_hamming_distance': avg_hamming,
        'max_possible_distance': 32
    }


if __name__ == "__main__":
    # Test the genome system
    print("Testing Computational Genome System")
    print("=" * 40)
    
    # Create test genome
    genome = ComputationalGenome()
    print(f"Random genome: {genome}")
    print(f"Binary: {genome.to_binary_string()}")
    print(f"Expressed parameters: {genome.express_genes()}")
    
    # Test mutation
    mutated = genome.mutate()
    print(f"\nMutated genome: {mutated}")
    print(f"Hamming distance: {genome.hamming_distance(mutated)}")
    
    # Test population diversity
    population = create_random_population(10)
    diversity_stats = analyze_genetic_diversity(population)
    print(f"\nPopulation diversity: {diversity_stats}")
