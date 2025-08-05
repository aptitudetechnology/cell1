"""
Visualize Hybrid Bio-AI Cells
Creates matplotlib visualizations of hybrid cells showing both biological and AI components.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon
from matplotlib.collections import LineCollection
import random

from hybrid_integrated_cell import HybridEvolutionaryCell, AIGenome
from genome import ComputationalGenome


def draw_organelle(ax, x, y, organelle_type, color, size=1.0):
    """Draw different organelle types."""
    if organelle_type == 'nucleus':
        # Draw nucleus as a large circle with inner structure
        nucleus = Circle((x, y), 0.15 * size, color=color, alpha=0.7)
        ax.add_patch(nucleus)
        # Add nucleolus
        nucleolus = Circle((x, y), 0.05 * size, color='darkred', alpha=0.9)
        ax.add_patch(nucleolus)
        
    elif organelle_type == 'mitochondrion':
        # Draw mitochondrion as an oval with cristae
        mito = patches.Ellipse((x, y), 0.15 * size, 0.08 * size, 
                              angle=random.randint(0, 180),
                              color=color, alpha=0.8)
        ax.add_patch(mito)
        
    elif organelle_type == 'ribosome':
        # Draw ribosome as small circles
        ribo = Circle((x, y), 0.03 * size, color=color, alpha=0.9)
        ax.add_patch(ribo)


def draw_neural_network(ax, x_offset, y_offset, weights, scale=0.3):
    """Draw the AI neural network component."""
    # Network architecture: 2 inputs -> 2 outputs
    input_nodes = [(x_offset - 0.15 * scale, y_offset + 0.1 * scale),
                   (x_offset - 0.15 * scale, y_offset - 0.1 * scale)]
    output_nodes = [(x_offset + 0.15 * scale, y_offset + 0.1 * scale),
                    (x_offset + 0.15 * scale, y_offset - 0.1 * scale)]
    
    # Draw connections with weights as line thickness
    for i, inp in enumerate(input_nodes):
        for j, out in enumerate(output_nodes):
            weight = weights[i * 2 + j]
            # Normalize weight for visualization
            line_width = abs(weight) * 2 + 0.5
            color = 'blue' if weight > 0 else 'red'
            alpha = min(abs(weight), 1.0)
            
            ax.plot([inp[0], out[0]], [inp[1], out[1]], 
                   color=color, linewidth=line_width, alpha=alpha)
    
    # Draw nodes
    for node in input_nodes:
        circle = Circle(node, 0.03 * scale, color='green', zorder=5)
        ax.add_patch(circle)
        
    for node in output_nodes:
        circle = Circle(node, 0.03 * scale, color='orange', zorder=5)
        ax.add_patch(circle)
    
    # Add labels
    ax.text(x_offset - 0.2 * scale, y_offset + 0.15 * scale, 'a', 
            fontsize=8, ha='right', va='center')
    ax.text(x_offset - 0.2 * scale, y_offset - 0.15 * scale, 'b', 
            fontsize=8, ha='right', va='center')
    ax.text(x_offset + 0.2 * scale, y_offset + 0.15 * scale, 'sum', 
            fontsize=8, ha='left', va='center')
    ax.text(x_offset + 0.2 * scale, y_offset - 0.15 * scale, 'diff', 
            fontsize=8, ha='left', va='center')


def draw_hybrid_cell(ax, cell, x=0, y=0, size=1.0, show_network=True):
    """Draw a complete hybrid bio-AI cell."""
    # Cell membrane
    cell_radius = 0.5 * size
    membrane = Circle((x, y), cell_radius, fill=False, 
                     edgecolor='black', linewidth=2)
    ax.add_patch(membrane)
    
    # Cell interior (cytoplasm)
    cytoplasm = Circle((x, y), cell_radius * 0.98, 
                      color='lightblue', alpha=0.3)
    ax.add_patch(cytoplasm)
    
    # Draw nucleus (contains both genomes)
    draw_organelle(ax, x, y, 'nucleus', 'purple', size)
    
    # Draw mitochondria
    mito_count = cell.mitochondrion_count
    for i in range(min(mito_count, 8)):  # Limit visual clutter
        angle = i * 2 * np.pi / mito_count
        mx = x + 0.3 * size * np.cos(angle)
        my = y + 0.3 * size * np.sin(angle)
        draw_organelle(ax, mx, my, 'mitochondrion', 'red', size)
    
    # Draw ribosomes
    ribo_count = cell.ribosome_count
    for i in range(min(ribo_count, 12)):  # Limit visual clutter
        # Random distribution within cell
        angle = random.random() * 2 * np.pi
        r = random.random() * 0.35 * size
        rx = x + r * np.cos(angle)
        ry = y + r * np.sin(angle)
        draw_organelle(ax, rx, ry, 'ribosome', 'darkblue', size)
    
    # Draw AI neural network component
    if show_network:
        # Position network in lower part of cell
        draw_neural_network(ax, x, y - 0.2 * size, cell.ai_genome.weights, size)
    
    # Add cell info
    info_text = f"Gen: {cell.generation}\n" \
                f"ATP: {cell.atp:.0f}\n" \
                f"Accuracy: {cell.problem_accuracy:.0f}%"
    ax.text(x, y + cell_radius + 0.1, info_text, 
            fontsize=8, ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add AI chip indicator
    chip_size = 0.08 * size
    chip = Rectangle((x + cell_radius - chip_size * 1.5, 
                     y + cell_radius - chip_size * 1.5),
                    chip_size, chip_size,
                    color='gold', alpha=0.9, zorder=10)
    ax.add_patch(chip)
    ax.text(x + cell_radius - chip_size, y + cell_radius - chip_size,
            'AI', fontsize=6, ha='center', va='center', zorder=11)


def visualize_hybrid_cell_population(population, num_cells=6):
    """Visualize multiple hybrid cells from a population."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get top cells by fitness
    top_cells = population.get_fittest_cells(num_cells)
    
    # Arrange cells in a grid
    cols = 3
    rows = (num_cells + cols - 1) // cols
    
    for i, cell in enumerate(top_cells):
        row = i // cols
        col = i % cols
        x = -1.5 + col * 1.5
        y = 1.0 - row * 1.5
        draw_hybrid_cell(ax, cell, x, y, size=0.8)
    
    # Set axis properties
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title and legend
    plt.title('Hybrid Bio-AI Cells: Evolution of Intelligence', 
              fontsize=16, fontweight='bold')
    
    # Create legend
    legend_elements = [
        patches.Patch(color='purple', label='Nucleus (DNA + AI)'),
        patches.Patch(color='red', label='Mitochondria (ATP)'),
        patches.Patch(color='darkblue', label='Ribosomes (Protein)'),
        patches.Patch(color='gold', label='AI Processor'),
        patches.Line2D([0], [0], color='blue', lw=2, label='Positive weight'),
        patches.Line2D([0], [0], color='red', lw=2, label='Negative weight')
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
              bbox_to_anchor=(0, 1), fontsize=10)
    
    plt.tight_layout()
    return fig


def visualize_single_hybrid_cell(cell=None):
    """Create a detailed view of a single hybrid cell."""
    if cell is None:
        # Create a sample cell
        genome = ComputationalGenome()
        ai_genome = AIGenome()
        cell = HybridEvolutionaryCell(genome, ai_genome)
        # Simulate some evolution
        for _ in range(50):
            cell.update(1.0, 1.0)
            if random.random() < 0.3:
                cell.solve_math_problem(5, 3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left panel: Cell structure
    draw_hybrid_cell(ax1, cell, size=1.5, show_network=False)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Biological Components', fontsize=14, fontweight='bold')
    
    # Right panel: Neural network detail
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('AI Neural Network', fontsize=14, fontweight='bold')
    
    # Draw large neural network
    draw_neural_network(ax2, 0, 0, cell.ai_genome.weights, scale=2.0)
    
    # Add weight values
    weights_text = "Network Weights:\n"
    for i, w in enumerate(cell.ai_genome.weights):
        weights_text += f"w{i}: {w:.3f}\n"
    ax2.text(-0.9, -0.7, weights_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    
    # Add performance stats
    stats_text = f"Problem Solving Stats:\n" \
                f"Problems attempted: {cell.total_problems}\n" \
                f"Problems solved: {cell.problems_solved}\n" \
                f"Accuracy: {cell.problem_accuracy:.1f}%\n" \
                f"Fitness: {cell.get_fitness():.2f}"
    ax2.text(0.3, -0.7, stats_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    
    # Main title
    fig.suptitle(f'Hybrid Bio-AI Cell (Generation {cell.generation})',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def visualize_evolution_timeline(population):
    """Visualize the evolution of problem-solving ability over time."""
    if not population.population_accuracy_history:
        print("No evolution history available")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Extract data
    times = [record['time'] for record in population.population_accuracy_history]
    accuracies = [record['accuracy'] for record in population.population_accuracy_history]
    pop_sizes = [record['population_size'] for record in population.population_accuracy_history]
    
    # Plot accuracy evolution
    ax1.plot(times, accuracies, 'b-', linewidth=2, label='Population Average')
    ax1.set_ylabel('Problem-Solving Accuracy (%)', fontsize=12)
    ax1.set_title('Evolution of Intelligence Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot population size
    ax2.plot(times, pop_sizes, 'g-', linewidth=2)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Population Size', fontsize=12)
    ax2.set_title('Population Dynamics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    from hybrid_population import HybridPopulation, run_hybrid_evolution_experiment
    from environment import create_competitive_environment
    
    print("Creating hybrid cell visualization...")
    
    # Create and run a short evolution
    env = create_competitive_environment()
    population = run_hybrid_evolution_experiment(
        environment=env,
        max_time=100.0,
        max_generations=20,
        snapshot_interval=25.0,
        problem_frequency=5
    )
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Single cell detail
    best_cell = population.get_best_problem_solver()
    if best_cell:
        fig1 = visualize_single_hybrid_cell(best_cell)
        plt.savefig('hybrid_cell_detail.png', dpi=300, bbox_inches='tight')
        print("Saved: hybrid_cell_detail.png")
    
    # 2. Population view
    fig2 = visualize_hybrid_cell_population(population)
    plt.savefig('hybrid_cell_population.png', dpi=300, bbox_inches='tight')
    print("Saved: hybrid_cell_population.png")
    
    # 3. Evolution timeline
    fig3 = visualize_evolution_timeline(population)
    if fig3:
        plt.savefig('hybrid_evolution_timeline.png', dpi=300, bbox_inches='tight')
        print("Saved: hybrid_evolution_timeline.png")
    
    plt.show()
