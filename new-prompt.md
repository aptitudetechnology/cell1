# Minimum Viable Computational Genome Cell Prototype

## Objective
Create a simple Python prototype that demonstrates a cell with both biological and computational genomes that evolve together. Focus on core concepts with minimal complexity to prove the hybrid bio-AI evolution concept works.

## Core Requirements

### 1. Simple Hybrid Cell
```python
class HybridCell:
    bio_genome: int        # 16-bit biological traits
    ai_genome: list        # Simple neural network weights [4 weights]
    energy: float          # Current energy level
    mass: float           # Cell mass (grows over time)
```

**Biological Genome (16-bit total)**:
- `metabolism` (4-bit, 0-15): Energy production rate per tick
- `growth_rate` (4-bit, 0-15): How fast cell gains mass
- `division_size` (4-bit, 0-15): Mass needed to divide (scaled 10-25)
- `mutation_rate` (4-bit, 0-15): Chance of mutations (scaled 0-10%)

**AI Genome (4 weights)**:
- Simple neural network: 2 inputs → 2 hidden → 1 output
- Weights stored as list of 4 floats [-1.0 to +1.0]
- Solves basic math problems (addition of two numbers)

### 2. Problem-Solving System
- **Single Problem Type**: Add two random numbers (0-10)
- **Input**: Two numbers fed to neural network
- **Output**: Network's guess at the sum
- **Scoring**: Energy reward = max(0, 10 - |correct_answer - guess|)
- **Frequency**: Each cell gets 1 problem every 10 simulation ticks

### 3. Energy Economy
- **Energy Sources**:
  - Base metabolism: `metabolism_gene * 0.5` energy per tick
  - Problem solving: 0-10 energy based on accuracy
- **Energy Costs**:
  - Living: 1 energy per tick
  - Growing: 0.1 energy per mass gained
  - Division: 10 energy cost

### 4. Simple Evolution
- **Reproduction**: When mass > division_size, spend 10 energy to divide
- **Inheritance**: 
  - Bio genome copied with point mutations
  - AI weights copied with small random noise (+/- 0.1)
- **Death**: Cell dies when energy < 0

### 5. Basic Population
- Start with 5 random cells
- Run for 1000 ticks
- Track: population size, average problem-solving accuracy, genome diversity

## Minimal Implementation

### File Structure
```
mvp_hybrid_cells/
├── cell.py           # HybridCell class
├── simulation.py     # Main simulation loop
└── main.py          # Run simulation and show results
```

### Key Functions
```python
# cell.py
class HybridCell:
    def __init__(self, bio_genome=None, ai_genome=None)
    def update(self)                    # Live one tick
    def solve_problem(self, a, b)       # Use AI to solve a+b
    def can_divide(self)               # Check if ready to reproduce
    def divide(self)                   # Create offspring with mutations
    
# simulation.py  
def run_simulation(initial_cells, ticks=1000)
def generate_problem()              # Create random addition problem
def calculate_stats(population)     # Track evolution metrics
```

### Core Logic Flow
```
For each tick:
  For each cell:
    1. Gain energy from metabolism
    2. Lose 1 energy for living
    3. If tick % 10 == 0: solve math problem, gain energy reward
    4. Grow mass using energy
    5. If big enough and enough energy: divide with mutations
    6. If energy <= 0: die
  
  Every 100 ticks: print population stats
```

## Success Criteria
- Cells successfully reproduce via binary fission
- Both biological and AI traits are inherited with mutations  
- Population evolves better problem-solving over time
- Clear correlation between AI accuracy and reproductive success
- Simple visualization showing evolution progress

## Expected Outputs
- Population size over time
- Average problem-solving accuracy over time  
- Example genomes from generation 1 vs final generation
- Evidence that better problem-solvers reproduce more

## Constraints
- Total code < 200 lines
- No external libraries except random, math
- Single problem type (addition only)
- Fixed environment (no resource competition)
- Simple console output (no fancy graphics)

This MVP proves the core concept: cells with computational genomes can evolve to become better problem-solvers through natural selection, creating a foundation for more complex bio-AI evolution systems.