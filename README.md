# Computational Genome Cell Evolution System

A Python implementation of cellular evolution with computational genomes, demonstrating evolutionary principles through simulation of cells that reproduce via binary fission. Now includes hybrid bio-AI cells that can evolve problem-solving abilities!

## Overview

This system implements cells with 32-bit computational genomes that control cellular parameters through gene expression. Cells evolve under selection pressure, showing natural selection, genetic drift, mutation, and adaptation in action.

### New: Hybrid Bio-AI Cells
The system now includes hybrid cells that combine biological evolution with AI capabilities:
- Cells have both biological genomes and neural network weights
- They solve math problems to gain extra energy
- Natural selection favors better problem-solvers
- Watch intelligence evolve over generations!

## Features

### Core Components
- **32-bit Computational Genome**: 4 genes (8 bits each) controlling cellular functions
- **Evolutionary Cells**: Cells with organelles controlled by genome expression
- **Population Management**: Tracks evolution, lineages, and population genetics
- **Environmental Pressures**: Configurable selection pressures and conditions
- **Visualization Tools**: Real-time plotting of evolution dynamics
- **AI Problem Solving** (NEW): Cells with neural networks that evolve to solve problems

### Genes and Functions
1. **Energy Efficiency** (0-255): Controls mitochondrial ATP production rate
2. **Growth Rate** (0-255): Controls ribosome protein synthesis speed  
3. **Division Threshold** (0-255): Mass required before cell division
4. **Mutation Rate** (0-255): Probability of mutations during DNA replication

### Hybrid Cell AI Features (NEW)
- **Neural Network**: Simple 4-weight network for problem solving
- **Energy Rewards**: Correct answers provide ATP boost
- **Co-evolution**: Both biological and AI traits evolve together
- **Selection Pressure**: Intelligence becomes a survival advantage

### Educational Scenarios
- **Founder Effect**: Evolution from single ancestor cell
- **Selection Pressure**: Adaptation under resource limitation
- **Mutation Rate Evolution**: Optimal mutation rate evolution
- **Genetic Drift**: Small vs large population dynamics
- **Adaptive Radiation**: Evolution in different environmental niches
- **AI Evolution** (NEW): Watch cells evolve problem-solving intelligence

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

### Run Hybrid Bio-AI Evolution (NEW)
```bash
# Run the hybrid cell simulation with problem-solving
python run_hybrid_simulation.py

# Or run the standalone MVP version
python hybridcell.py
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

# Run hybrid evolution experiment (NEW)
python hybrid_population.py
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

### Hybrid Bio-AI API (NEW)

```python
from hybrid_integrated_cell import HybridEvolutionaryCell, AIGenome
from hybrid_population import run_hybrid_evolution_experiment
from environment import create_competitive_environment

# Run hybrid evolution with problem-solving cells
population = run_hybrid_evolution_experiment(
    environment=create_competitive_environment(),
    max_time=1000.0,
    problem_frequency=10  # Solve problems every 10 ticks
)

# Get the smartest cell
best_solver = population.get_best_problem_solver()
print(f"Best solver accuracy: {best_solver.problem_accuracy:.1f}%")
```

## System Architecture

```
computational_genome_cell/
├── genome.py                    # 32-bit genome with gene expression
├── cell.py                     # Evolutionary cell with organelles
├── environment.py              # Environmental conditions and selection
├── population.py               # Population management and evolution tracking
├── visualization.py            # Evolution plotting and analysis
├── examples.py                 # Demonstration scenarios
├── main.py                     # Command-line interface
├── requirements.txt            # Python dependencies
│
├── hybridcell.py              # Standalone MVP hybrid cell implementation (NEW)
├── hybrid_integrated_cell.py   # Integrated hybrid cell with AI (NEW)
├── hybrid_population.py        # Population manager for hybrid cells (NEW)
├── run_hybrid_simulation.py    # Script to run hybrid evolution (NEW)
└── new-prompt.md              # Design doc for hybrid cells (NEW)
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
- **NEW**: Watch intelligence evolve through natural selection
- **NEW**: See how problem-solving ability becomes a survival trait

### Classroom Activities
1. Compare evolution under different selection pressures
2. Investigate optimal mutation rates
3. Study founder effects and genetic bottlenecks
4. Explore adaptive radiation scenarios
5. Analyze lineage trees and phylogenetic relationships
6. **NEW**: Observe AI evolution - cells learning to solve problems
7. **NEW**: Compare biological vs computational trait evolution

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

## What's New: Hybrid Bio-AI Evolution

The latest addition introduces cells that evolve both biological traits and problem-solving intelligence:

- **Dual Evolution**: Cells have both biological genomes and neural network weights
- **Problem Solving**: Cells solve math problems to earn extra energy (ATP)
- **Natural Selection for Intelligence**: Smarter cells survive and reproduce more
- **Co-evolution**: Watch how biological and computational traits evolve together
- **Research Applications**: Study emergence of intelligence through evolution

### Running Hybrid Simulations

```bash
# Full integrated hybrid simulation
python run_hybrid_simulation.py

# Standalone MVP version (< 200 lines)
python hybridcell.py

# Direct API usage
python hybrid_population.py
```

## Contributing

This is an educational and research tool. Contributions welcome for:
- Additional selection pressure types
- New visualization methods
- Performance optimizations
- Educational scenarios
- Documentation improvements
- Enhanced AI capabilities for cells
- More complex problem types for cells to solve

## License

Open source - suitable for educational and research use.

## Citation

If using this system for research or educational purposes, please cite:

"Computational Genome Cell Evolution System - A tool for simulating cellular evolution with computational genomes"

## Contact

For questions, suggestions, or issues, please refer to the project documentation or create an issue in the project repository.
