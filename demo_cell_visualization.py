"""
Demo script to visualize hybrid bio-AI cells using matplotlib
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Ellipse, Rectangle, FancyBboxPatch
import numpy as np

from hybrid_integrated_cell import HybridEvolutionaryCell, AIGenome
from genome import ComputationalGenome
from hybrid_population import HybridPopulation
from environment import create_competitive_environment


def draw_hybrid_cell_diagram():
    """Create a detailed diagram of a hybrid bio-AI cell."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Cell membrane - large circle
    cell_membrane = Circle((0.5, 0.5), 0.35, 
                          facecolor='lightblue', 
                          edgecolor='darkblue', 
                          linewidth=3,
                          alpha=0.3)
    ax.add_patch(cell_membrane)
    
    # Nucleus - central purple circle
    nucleus = Circle((0.5, 0.5), 0.12, 
                    facecolor='purple', 
                    edgecolor='indigo', 
                    linewidth=2,
                    alpha=0.6)
    ax.add_patch(nucleus)
    
    # Nucleolus
    nucleolus = Circle((0.5, 0.5), 0.04, 
                      facecolor='#4B0082',
                      alpha=0.8)
    ax.add_patch(nucleolus)
    
    # Add nucleus label
    ax.text(0.5, 0.38, 'Nucleus\n(32-bit genome)', 
           ha='center', va='top', fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Mitochondria - red ovals at various positions
    mito_positions = [(0.35, 0.6), (0.65, 0.4), (0.4, 0.35), (0.6, 0.65)]
    for x, y in mito_positions:
        # Outer membrane
        mito = Ellipse((x, y), 0.12, 0.06, 
                      angle=np.random.randint(0, 180),
                      facecolor='red', 
                      edgecolor='darkred', 
                      linewidth=2,
                      alpha=0.6)
        ax.add_patch(mito)
        
        # Cristae (internal structures)
        for i in range(3):
            offset = (i - 1) * 0.015
            ax.plot([x - 0.04, x + 0.04], [y + offset, y + offset], 
                   color='darkred', linewidth=1, alpha=0.8)
    
    # Add mitochondria label
    ax.annotate('Mitochondria\n(ATP production)', 
               xy=(0.65, 0.4), xytext=(0.8, 0.3),
               arrowprops=dict(arrowstyle='->', lw=1.5),
               fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='pink', alpha=0.8))
    
    # Ribosomes - small blue dots scattered around
    ribosome_positions = [(0.3, 0.45), (0.7, 0.55), (0.45, 0.7), 
                         (0.55, 0.3), (0.35, 0.7), (0.65, 0.7),
                         (0.3, 0.3), (0.7, 0.35)]
    for x, y in ribosome_positions:
        # Large subunit
        ribo1 = Circle((x - 0.01, y), 0.02, 
                      facecolor='blue', 
                      edgecolor='darkblue',
                      alpha=0.8)
        # Small subunit
        ribo2 = Circle((x + 0.01, y), 0.015, 
                      facecolor='lightblue', 
                      edgecolor='darkblue',
                      alpha=0.8)
        ax.add_patch(ribo1)
        ax.add_patch(ribo2)
    
    # Add ribosome label
    ax.annotate('Ribosomes\n(protein synthesis)', 
               xy=(0.3, 0.45), xytext=(0.15, 0.6),
               arrowprops=dict(arrowstyle='->', lw=1.5),
               fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # Neural Network visualization (outside the cell)
    nn_x_start = 0.9
    nn_y_start = 0.3
    
    # Draw neural network box
    nn_box = FancyBboxPatch((nn_x_start - 0.05, nn_y_start - 0.05), 
                           0.3, 0.4,
                           boxstyle="round,pad=0.02",
                           facecolor='lightyellow',
                           edgecolor='orange',
                           linewidth=2,
                           alpha=0.3)
    ax.add_patch(nn_box)
    
    # Input layer
    input_nodes = [(nn_x_start, nn_y_start + 0.3), 
                   (nn_x_start, nn_y_start + 0.1)]
    
    # Hidden layer
    hidden_nodes = [(nn_x_start + 0.1, nn_y_start + 0.3),
                    (nn_x_start + 0.1, nn_y_start + 0.1)]
    
    # Output layer
    output_node = (nn_x_start + 0.2, nn_y_start + 0.2)
    
    # Draw connections
    for inp in input_nodes:
        for hid in hidden_nodes:
            ax.plot([inp[0], hid[0]], [inp[1], hid[1]], 
                   'g-', linewidth=2, alpha=0.6)
    
    for hid in hidden_nodes:
        ax.plot([hid[0], output_node[0]], [hid[1], output_node[1]], 
               'r-', linewidth=2, alpha=0.6)
    
    # Draw nodes
    for node in input_nodes:
        circle = Circle(node, 0.025, facecolor='lightgreen', 
                       edgecolor='darkgreen', linewidth=2)
        ax.add_patch(circle)
    
    for node in hidden_nodes:
        circle = Circle(node, 0.025, facecolor='yellow', 
                       edgecolor='orange', linewidth=2)
        ax.add_patch(circle)
    
    output_circle = Circle(output_node, 0.025, facecolor='lightcoral', 
                          edgecolor='darkred', linewidth=2)
    ax.add_patch(output_circle)
    
    # Neural network labels
    ax.text(nn_x_start - 0.05, nn_y_start + 0.35, 'Input', fontsize=9, ha='right')
    ax.text(nn_x_start, nn_y_start + 0.3, 'a', fontsize=8, ha='center', va='center')
    ax.text(nn_x_start, nn_y_start + 0.1, 'b', fontsize=8, ha='center', va='center')
    ax.text(nn_x_start + 0.2, nn_y_start + 0.05, 'sum', fontsize=8, ha='center')
    ax.text(nn_x_start + 0.1, nn_y_start - 0.1, 'AI Brain\n(Neural Network)', 
           fontsize=10, ha='center', fontweight='bold')
    
    # Connection from cell to neural network
    ax.annotate('', xy=(nn_x_start - 0.05, nn_y_start + 0.2), 
               xytext=(0.85, 0.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='purple', 
                             connectionstyle="arc3,rad=0.3"))
    ax.text(0.87, 0.4, 'Problem\nSolving', fontsize=9, ha='center', 
           color='purple', fontweight='bold')
    
    # Title and labels
    ax.set_title('Hybrid Bio-AI Cell Structure', fontsize=18, fontweight='bold', pad=20)
    
    # Cell parameters box
    param_text = "Cell Parameters:\n• Energy efficiency\n• Growth rate\n• Division threshold\n• Mutation rate"
    ax.text(0.05, 0.95, param_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # AI capabilities box
    ai_text = "AI Capabilities:\n• Solve math problems\n• Evolve weights\n• Improve accuracy\n• Pass to offspring"
    ax.text(0.95, 0.95, ai_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    # Set axis properties
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig


def draw_cell_population_grid():
    """Draw a grid of cells showing diversity in a population."""
    # Create a small population
    env = create_competitive_environment()
    population = HybridPopulation(environment=env)
    
    # Evolve for a bit to get diversity
    print("Evolving population for visualization...")
    for i in range(100):
        population.update(1.0)
        if i % 20 == 0:
            print(f"  Generation {i}...")
    
    # Create grid visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hybrid Bio-AI Cell Population Diversity', fontsize=18, fontweight='bold')
    
    cells_to_show = population.cells[:6] if len(population.cells) >= 6 else population.cells
    
    for idx, (ax, cell) in enumerate(zip(axes.flat, cells_to_show)):
        # Draw simplified cell
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # Cell membrane
        membrane = Circle((0.5, 0.5), 0.3, 
                        facecolor='lightblue', 
                        edgecolor='darkblue', 
                        linewidth=2,
                        alpha=0.3)
        ax.add_patch(membrane)
        
        # Nucleus
        nucleus = Circle((0.5, 0.5), 0.1, 
                        facecolor='purple', 
                        alpha=0.6)
        ax.add_patch(nucleus)
        
        # Mitochondria (number based on cell)
        n_mito = min(getattr(cell.components, 'mitochondria', 6), 6)
        for i in range(n_mito):
            angle = i * 2 * np.pi / n_mito if n_mito > 0 else 0
            x = 0.5 + 0.2 * np.cos(angle)
            y = 0.5 + 0.2 * np.sin(angle)
            mito = Ellipse((x, y), 0.08, 0.04, 
                          angle=np.degrees(angle),
                          facecolor='red', 
                          alpha=0.6)
            ax.add_patch(mito)
        
        # Ribosomes (number based on cell)
        n_ribo = min(getattr(cell.components, 'ribosomes', 8), 8)
        for i in range(n_ribo):
            angle = i * 2 * np.pi / n_ribo + np.pi / n_ribo if n_ribo > 0 else 0
            x = 0.5 + 0.15 * np.cos(angle)
            y = 0.5 + 0.15 * np.sin(angle)
            ribo = Circle((x, y), 0.02, 
                         facecolor='blue', 
                         alpha=0.8)
            ax.add_patch(ribo)
        
        # Cell info
        ax.text(0.5, 0.05, f'Gen {cell.generation}', 
               ha='center', fontsize=10, fontweight='bold')
        ax.text(0.5, 0.95, f'ATP: {cell.atp:.1f}', 
               ha='center', fontsize=9)
        ax.text(0.5, 0.9, f'Accuracy: {cell.problem_accuracy:.1f}%', 
               ha='center', fontsize=9, color='green')
        
        # Mini neural network indicator
        nn_indicator = Rectangle((0.85, 0.85), 0.1, 0.1,
                               facecolor='yellow',
                               edgecolor='orange',
                               alpha=0.5)
        ax.add_patch(nn_indicator)
        ax.text(0.9, 0.9, 'AI', ha='center', va='center', 
               fontsize=8, fontweight='bold')
        
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(cells_to_show), 6):
        axes.flat[idx].axis('off')
    
    return fig


def draw_evolution_timeline():
    """Show how cells evolve over time."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Timeline data
    stages = [
        ("Generation 0", "Random weights\n0% accuracy", "red"),
        ("Generation 10", "Learning begins\n~25% accuracy", "orange"),
        ("Generation 25", "Selection pressure\n~50% accuracy", "yellow"),
        ("Generation 50", "Optimization\n~75% accuracy", "lightgreen"),
        ("Generation 100", "Expert solvers\n~95% accuracy", "green")
    ]
    
    y_pos = 0.5
    x_positions = np.linspace(0.1, 0.9, len(stages))
    
    # Draw timeline
    ax.plot([0.05, 0.95], [y_pos, y_pos], 'k-', linewidth=3)
    
    for i, (x, (gen, desc, color)) in enumerate(zip(x_positions, stages)):
        # Draw cell
        cell = Circle((x, y_pos), 0.08, 
                     facecolor=color, 
                     edgecolor='black', 
                     linewidth=2)
        ax.add_patch(cell)
        
        # Draw mini brain
        brain = Circle((x + 0.05, y_pos + 0.05), 0.02, 
                      facecolor='yellow', 
                      edgecolor='orange')
        ax.add_patch(brain)
        
        # Add text
        ax.text(x, y_pos - 0.15, gen, ha='center', fontweight='bold', fontsize=11)
        ax.text(x, y_pos - 0.22, desc, ha='center', fontsize=9)
        
        # Add arrow
        if i < len(stages) - 1:
            ax.annotate('', xy=(x_positions[i+1] - 0.08, y_pos), 
                       xytext=(x + 0.08, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.set_title('Evolution of Problem-Solving Ability in Hybrid Cells', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add description
    desc_text = ("Cells evolve both biological traits (metabolism, growth) and AI capabilities (neural networks).\n"
                "Natural selection favors cells that can solve problems correctly, leading to smarter populations.")
    ax.text(0.5, 0.85, desc_text, ha='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    return fig


if __name__ == "__main__":
    print("Creating hybrid bio-AI cell visualizations...")
    
    # Create detailed cell diagram
    print("1. Drawing detailed cell structure...")
    fig1 = draw_hybrid_cell_diagram()
    plt.savefig('hybrid_cell_structure.png', dpi=300, bbox_inches='tight')
    print("   Saved: hybrid_cell_structure.png")
    
    # Create population grid
    print("2. Drawing cell population diversity...")
    fig2 = draw_cell_population_grid()
    plt.savefig('hybrid_cell_population.png', dpi=300, bbox_inches='tight')
    print("   Saved: hybrid_cell_population.png")
    
    # Create evolution timeline
    print("3. Drawing evolution timeline...")
    fig3 = draw_evolution_timeline()
    plt.savefig('hybrid_cell_evolution_timeline.png', dpi=300, bbox_inches='tight')
    print("   Saved: hybrid_cell_evolution_timeline.png")
    
    print("\nAll visualizations complete!")
    
    # Show all figures
    plt.show()
