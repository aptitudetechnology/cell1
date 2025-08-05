# mvp_hybrid_cells/cell.py

import random

class HybridCell:
    """
    A simple cell with both a biological and a computational (AI) genome.
    """
    def __init__(self, bio_genome=None, ai_genome=None):
        if bio_genome is None:
            self.bio_genome = random.getrandbits(16)
        else:
            self.bio_genome = bio_genome
        
        if ai_genome is None:
            self.ai_genome = [random.uniform(-1.0, 1.0) for _ in range(4)]
        else:
            self.ai_genome = ai_genome
            
        self.energy = 50.0
        self.mass = 1.0

        # Decode biological genome
        self.metabolism = (self.bio_genome >> 12) & 0b1111
        self.growth_rate = (self.bio_genome >> 8) & 0b1111
        self.division_size = 10 + ((self.bio_genome >> 4) & 0b1111)
        self.mutation_rate = ((self.bio_genome & 0b1111) / 15.0) * 0.1

    def update(self):
        """Simulates one tick of the cell's life."""
        # Energy gain from metabolism
        self.energy += self.metabolism * 0.5
        
        # Energy cost of living
        self.energy -= 1.0
        
        # Grow mass
        mass_gain = self.growth_rate * 0.1 * (self.energy / 50.0)
        self.mass += max(0, mass_gain)
        self.energy -= mass_gain * 0.1 # Energy cost of growing

    def solve_problem(self, a, b):
        """Uses the AI genome (a simple neural network) to solve a math problem."""
        # Neural network with 2 inputs, 2 hidden neurons, 1 output
        # Weights: w1 (input1->h1), w2 (input2->h1), w3 (input1->h2), w4 (input2->h2)
        # Note: This is a simplified network for the MVP.
        h1 = a * self.ai_genome[0] + b * self.ai_genome[1]
        h2 = a * self.ai_genome[2] + b * self.ai_genome[3]
        
        # Simple summation of hidden layers for output
        output = h1 + h2
        return output

    def can_divide(self):
        """Checks if the cell is ready to reproduce."""
        return self.mass >= self.division_size and self.energy >= 10.0

    def divide(self):
        """Creates a new offspring with mutations."""
        # Inherit and mutate biological genome
        new_bio_genome = self.bio_genome
        if random.random() < self.mutation_rate:
            # Flip a random bit
            bit_to_flip = random.randint(0, 15)
            new_bio_genome ^= (1 << bit_to_flip)

        # Inherit and mutate AI genome
        new_ai_genome = list(self.ai_genome)
        for i in range(len(new_ai_genome)):
            if random.random() < self.mutation_rate:
                new_ai_genome[i] += random.uniform(-0.1, 0.1)
                new_ai_genome[i] = max(-1.0, min(1.0, new_ai_genome[i]))
        
        offspring = HybridCell(new_bio_genome, new_ai_genome)
        self.energy -= 10.0
        self.mass /= 2.0
        offspring.mass = self.mass
        
        return offspring

# mvp_hybrid_cells/simulation.py

import math
# from cell import HybridCell # assuming cell.py is in the same directory

def generate_problem():
    """Generates a simple addition problem."""
    a = random.randint(0, 10)
    b = random.randint(0, 10)
    return a, b, a + b

def calculate_stats(population):
    """Calculates and returns key statistics for the current population."""
    if not population:
        return 0, 0.0, 0.0, ""

    total_accuracy = 0.0
    total_cells_evaluated = 0
    
    # Simple metric for AI genome diversity
    avg_ai_genome = [0.0, 0.0, 0.0, 0.0]
    
    for cell in population:
        # Generate a sample problem for accuracy check
        a, b, correct_answer = generate_problem()
        guess = cell.solve_problem(a, b)
        
        # Use a small tolerance for floating point comparisons
        error = abs(correct_answer - guess)
        if error < 0.5: # a good enough guess
            total_accuracy += 1
        
        total_cells_evaluated += 1

        for i in range(4):
            avg_ai_genome[i] += cell.ai_genome[i]

    avg_accuracy = (total_accuracy / total_cells_evaluated) * 100 if total_cells_evaluated > 0 else 0
    avg_mass = sum(c.mass for c in population) / len(population)
    
    for i in range(4):
        avg_ai_genome[i] /= len(population)
        
    return len(population), avg_accuracy, avg_mass, f"[{', '.join(f'{w:.2f}' for w in avg_ai_genome)}]"

def run_simulation(initial_cells, ticks=1000):
    """The main simulation loop."""
    population = initial_cells
    
    for tick in range(1, ticks + 1):
        if not population:
            print("Population died out.")
            break
            
        next_generation = []
        
        for cell in population:
            cell.update()
            
            # Problem solving phase
            if tick % 10 == 0:
                a, b, correct = generate_problem()
                guess = cell.solve_problem(a, b)
                reward = max(0, 10 - abs(correct - guess))
                cell.energy += reward
            
            # Reproduction
            if cell.can_divide():
                offspring = cell.divide()
                next_generation.append(offspring)
            
            # Survival check
            if cell.energy > 0:
                next_generation.append(cell)
        
        population = next_generation

        if tick % 100 == 0 or tick == 1:
            size, accuracy, avg_mass, avg_genome = calculate_stats(population)
            print(f"--- Tick {tick} ---")
            print(f"Population Size: {size}")
            print(f"Average Accuracy: {accuracy:.2f}%")
            print(f"Average Mass: {avg_mass:.2f}")
            print(f"Average AI Genome: {avg_genome}")
            
    return population

# mvp_hybrid_cells/main.py

# from cell import HybridCell # assuming cell.py is in the same directory
# from simulation import run_simulation # assuming simulation.py is in the same directory

if __name__ == "__main__":
    print("--- Starting Hybrid Cell Evolution Simulation ---")
    
    # Initialize a starting population
    initial_cells = [HybridCell() for _ in range(5)]
    
    # Capture initial stats
    size, accuracy, _, avg_genome = calculate_stats(initial_cells)
    print("\n--- Initial Population (Generation 1) ---")
    print(f"Initial Population Size: {size}")
    print(f"Initial Average Accuracy: {accuracy:.2f}%")
    print(f"Initial Average AI Genome: {avg_genome}")
    
    final_population = run_simulation(initial_cells)
    
    print("\n--- Simulation Complete ---")
    if final_population:
        size, accuracy, avg_mass, avg_genome = calculate_stats(final_population)
        print("\n--- Final Population Stats ---")
        print(f"Final Population Size: {size}")
        print(f"Final Average Accuracy: {accuracy:.2f}%")
        print(f"Final Average Mass: {avg_mass:.2f}")
        print(f"Final Average AI Genome: {avg_genome}")
        
        best_cell = max(final_population, key=lambda c: c.energy)
        print("\n--- Best Performing Cell ---")
        print(f"Bio Genome: {best_cell.bio_genome} (Metabolism: {best_cell.metabolism}, Growth: {best_cell.growth_rate})")
        print(f"AI Genome: [{', '.join(f'{w:.2f}' for w in best_cell.ai_genome)}]")
        print(f"Final Energy: {best_cell.energy:.2f}, Final Mass: {best_cell.mass:.2f}")
    else:
        print("The population did not survive.")