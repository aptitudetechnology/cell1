# Computational Genome Cell Evolution System

## Objective
Create a Python program that implements a simple cell with a computational genome that can evolve and reproduce through binary fission. Focus on creating a minimal but scientifically-grounded system that demonstrates evolutionary principles through computational biology.

## Core Requirements

### 1. Computational Genome Structure
- **32-bit genome** divided into functional genes (8 bits each)
- **4 essential genes**:
  - `energy_efficiency` (0-255): Controls mitochondrial ATP production rate
  - `growth_rate` (0-255): Controls ribosome protein synthesis speed
  - `division_threshold` (0-255): Mass required before cell can divide
  - `mutation_rate` (0-255): Probability of mutations during replication

### 2. Gene Expression System
- Each 8-bit gene value maps to cellular parameters using scaling functions
- Genes directly control organelle behavior and cellular properties
- Simple regulatory logic: gene values determine protein production rates
- Phenotypic effects should be clearly traceable to genotypic values

### 3. Cell Implementation
- Build upon the binary fission cell concept from our previous session
- Replace hardcoded values with genome-controlled parameters
- Organelles (mitochondria, ribosomes, nucleus) respond to genetic control
- Energy metabolism, growth, and division all governed by genome
- Cell lifecycle: Growth → DNA Replication → Division → Inheritance

### 4. Mutation System
- **Point mutations**: Random bit flips during DNA replication
- **Mutation rate** controlled by genome (self-modifying mutation rates)
- Mutations occur with small probability during each replication cycle
- Track mutation history and genomic lineages

### 5. Population Evolution
- Start with single "founder" cell with random genome
- Simulate population growth through binary fission
- Environmental selection pressures:
  - Limited nutrients (competition for resources)
  - Energy costs for cellular processes
  - Survival of fittest based on reproduction rate
- Track population genetics over generations

## Technical Implementation

### Program Structure
```
computational_genome_cell/
├── cell.py              # Cell class with genome integration
├── genome.py            # Genome class and gene expression
├── population.py        # Population management and evolution
├── environment.py       # Environmental conditions and selection
├── visualization.py     # Evolution tracking and plotting
├── examples.py          # Demo scenarios and experiments
└── main.py             # Main simulation runner
```

### Key Classes
- `ComputationalGenome`: 32-bit genome with gene expression methods
- `EvolutionaryCell`: Cell class controlled by computational genome
- `Population`: Manages multiple cells and tracks evolution
- `Environment`: Controls selection pressures and resource availability

### Data Persistence
- Save population snapshots at regular intervals
- Export genomic data in JSON format
- Track evolutionary statistics (fitness, diversity, mutations)
- Store complete lineage trees for phylogenetic analysis

## Scientific Accuracy Requirements

### Biological Realism
- Realistic energy costs for cellular processes
- Proper DNA replication timing and energy requirements
- Cellular aging and death mechanisms
- Resource competition and carrying capacity limits

### Evolutionary Principles
- Demonstrate natural selection through differential reproduction
- Show genetic drift in small populations
- Exhibit adaptive evolution under selection pressure
- Maintain genetic diversity through mutation

### Computational Biology
- Gene expression should follow biological logic
- Mutation rates should be biologically plausible
- Population dynamics should match ecological models
- Clear genotype-to-phenotype mapping

## Educational Features

### Interactive Experiments
- Compare evolution under different selection pressures
- Observe effects of varying mutation rates
- Track emergence of beneficial mutations
- Demonstrate evolutionary trade-offs

### Visualization Components
- Real-time population size and diversity plots
- Genome heatmaps showing gene frequency changes
- Phylogenetic trees of evolutionary relationships  
- Fitness landscapes and adaptive peaks

### Analysis Tools
- Calculate population genetic statistics
- Measure selection coefficients for different genotypes
- Track fixation of beneficial mutations
- Analyze evolutionary trajectories

## Success Criteria

### Core Functionality
- Cells successfully reproduce via binary fission
- Genomes are inherited with realistic mutation rates
- Population evolves under selection pressure
- Clear fitness differences between genotypes

### Educational Value
- Demonstrates key evolutionary concepts
- Shows how genotype determines phenotype
- Illustrates natural selection in action
- Provides hands-on evolution experimentation

### Scientific Validity
- Based on real biological and evolutionary principles
- Produces realistic population genetic patterns
- Mutations have appropriate phenotypic effects
- Selection acts on heritable variation

## Example Simulation Scenarios

1. **Founder Effect**: Start with single cell, observe initial population expansion
2. **Selection Pressure**: Introduce nutrient limitation, watch adaptation
3. **Mutation Rate Evolution**: Observe evolution of optimal mutation rates
4. **Genetic Drift**: Compare evolution in small vs. large populations
5. **Adaptive Radiation**: Multiple environmental niches favoring different traits

## Deliverables
- Complete Python implementation with all classes
- Working examples demonstrating evolutionary principles
- Visualization tools for tracking evolution
- Documentation explaining genetic code and system behavior
- Test scenarios showing system responds correctly to selection

Create a system that serves as both a computational biology research tool and an educational platform for understanding evolutionary principles through hands-on simulation.
