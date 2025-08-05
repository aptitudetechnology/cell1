# Computational Genome Cell Evolution System

A Python implementation of cellular evolution with computational genomes, demonstrating evolutionary principles through simulation of cells that reproduce via binary fission.

## Overview

This system implements cells with 32-bit computational genomes that control cellular parameters through gene expression. Cells evolve under selection pressure, showing natural selection, genetic drift, mutation, and adaptation in action.

## Features

### Core Components
- **32-bit Computational Genome**: 4 genes (8 bits each) controlling cellular functions
- **Evolutionary Cells**: Cells with organelles controlled by genome expression
- **Population Management**: Tracks evolution, lineages, and population genetics
- **Environmental Pressures**: Configurable selection pressures and conditions
- **Visualization Tools**: Real-time plotting of evolution dynamics

### Genes and Functions
1. **Energy Efficiency** (0-255): Controls mitochondrial ATP production rate
2. **Growth Rate** (0-255): Controls ribosome protein synthesis speed  
3. **Division Threshold** (0-255): Mass required before cell division
4. **Mutation Rate** (0-255): Probability of mutations during DNA replication

### Educational Scenarios
- **Founder Effect**: Evolution from single ancestor cell
- **Selection Pressure**: Adaptation under resource limitation
- **Mutation Rate Evolution**: Optimal mutation rate evolution
- **Genetic Drift**: Small vs large population dynamics
- **Adaptive Radiation**: Evolution in different environmental niches

## Installation

1. Clone or download the project
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run All Example Scenarios
```bash
python main.py examples --scenario all
```

### Run Basic Simulation
```bash
python main.py simulate --max-time 500 --max-generations 50
```

### Run with Visualization
```bash
python main.py simulate --environment competitive --visualize --output-dir ./results
```

### Run Interactive Experiment
```bash
python main.py examples --scenario interactive
```

## Usage Examples

### Command Line Interface

```bash
# Test system components
python main.py test

# Run founder effect experiment
python main.py examples --scenario founder

# Run harsh environment simulation
python main.py simulate --environment harsh --output results.json

# Custom genome simulation
python main.py simulate --genome-value 0x12345678 --max-generations 100
```

### Python API

```python
from genome import ComputationalGenome
from cell import EvolutionaryCell
from environment import create_competitive_environment
from population import run_evolution_experiment

# Create custom experiment
genome = ComputationalGenome()
environment = create_competitive_environment()

population = run_evolution_experiment(
    initial_genome=genome,
    environment=environment,
    max_time=200.0,
    max_generations=25
)

print(f"Final population: {population.get_population_summary()}")
```

## System Architecture

```
computational_genome_cell/
├── genome.py           # 32-bit genome with gene expression
├── cell.py            # Evolutionary cell with organelles
├── environment.py     # Environmental conditions and selection
├── population.py      # Population management and evolution tracking
├── visualization.py   # Evolution plotting and analysis
├── examples.py        # Demonstration scenarios
├── main.py           # Command-line interface
└── requirements.txt  # Python dependencies
```

## Scientific Accuracy

### Biological Realism
- Realistic energy costs for cellular processes
- Proper DNA replication timing and energy requirements
- Cellular aging and death mechanisms
- Resource competition and carrying capacity

### Evolutionary Principles
- Natural selection through differential reproduction
- Genetic drift in small populations
- Mutation-selection balance
- Adaptive evolution under selection pressure

## Visualization

The system includes comprehensive visualization tools:

- **Population Dynamics**: Size and fitness over time
- **Parameter Evolution**: Genetic parameter changes
- **Genome Heatmaps**: Bit pattern visualization
- **Fitness Landscapes**: 2D parameter fitness plots
- **Lineage Trees**: Evolutionary relationships
- **Selection Coefficients**: Trait selection strength

## Configuration

### Custom Environments

Create JSON configuration files for custom environments:

```json
{
  "name": "Custom Environment",
  "conditions": {
    "temperature": 40.0,
    "nutrient_density": 0.5,
    "carrying_capacity": 500
  },
  "selection_pressures": [
    {
      "name": "energy_selection",
      "target_parameter": "energy_efficiency",
      "selection_type": "directional",
      "strength": 0.8
    }
  ]
}
```

### Custom Genomes

Save and load specific genomes:

```python
# Save genome
genome = ComputationalGenome()
with open('genome.json', 'w') as f:
    json.dump(genome.to_dict(), f)

# Load genome
with open('genome.json', 'r') as f:
    genome_data = json.load(f)
genome = ComputationalGenome.from_dict(genome_data)
```

## Educational Use

This system serves as both a research tool and educational platform:

### Learning Objectives
- Understand how genotype determines phenotype
- Observe natural selection in action
- Explore mutation-selection balance
- Study population genetics principles
- Investigate evolutionary trade-offs

### Classroom Activities
1. Compare evolution under different selection pressures
2. Investigate optimal mutation rates
3. Study founder effects and genetic bottlenecks
4. Explore adaptive radiation scenarios
5. Analyze lineage trees and phylogenetic relationships

## Advanced Features

### Population Analysis
- Genetic diversity metrics
- Selection coefficient calculations
- Lineage tracking and phylogenies
- Fixation probability analysis
- Mutation accumulation studies

### Data Export
- JSON snapshots of entire populations
- CSV export of evolution statistics
- Genome lineage trees
- Environmental condition histories

## Performance Notes

- Optimized for populations up to ~2000 cells
- Visualization may require significant memory for large datasets
- Long simulations can be saved and resumed using snapshots

## Troubleshooting

### Common Issues

1. **Import Errors**: Install required packages with `pip install -r requirements.txt`
2. **Visualization Not Working**: Matplotlib and seaborn are optional but required for plots
3. **Memory Issues**: Reduce population size or simulation time for large experiments
4. **Slow Performance**: Large populations (>1000 cells) may run slowly

### System Requirements
- Python 3.7+
- ~100MB RAM for typical simulations
- Optional: GPU acceleration for large populations (future enhancement)

## Contributing

This is an educational and research tool. Contributions welcome for:
- Additional selection pressure types
- New visualization methods
- Performance optimizations
- Educational scenarios
- Documentation improvements

## License

Open source - suitable for educational and research use.

## Citation

If using this system for research or educational purposes, please cite:

"Computational Genome Cell Evolution System - A tool for simulating cellular evolution with computational genomes"

## Contact

For questions, suggestions, or issues, please refer to the project documentation or create an issue in the project repository.
