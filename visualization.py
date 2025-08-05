"""
Visualization System
Provides tools for visualizing evolution, population dynamics, and genetic changes.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
from collections import defaultdict

from population import Population, PopulationStatistics
from genome import ComputationalGenome


class EvolutionVisualizer:
    """
    Visualizes evolutionary dynamics and population statistics.
    """
    
    def __init__(self, population: Population):
        """
        Initialize visualizer with a population.
        
        Args:
            population: Population object to visualize
        """
        self.population = population
        self.figure_size = (12, 8)
        plt.style.use('seaborn-v0_8')
    
    def plot_population_dynamics(self, save_path: str = None):
        """
        Plot population size and fitness over time.
        
        Args:
            save_path: Path to save figure (optional)
        """
        if not self.population.statistics_history:
            print("No statistics history to plot")
            return
        
        stats = self.population.statistics_history
        times = [s.time for s in stats]
        pop_sizes = [s.alive_cells for s in stats]
        avg_fitness = [s.average_fitness for s in stats]
        max_fitness = [s.max_fitness for s in stats]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, sharex=True)
        
        # Population size
        ax1.plot(times, pop_sizes, 'b-', linewidth=2, label='Population Size')
        ax1.set_ylabel('Population Size')
        ax1.set_title('Population Dynamics Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Fitness
        ax2.plot(times, avg_fitness, 'g-', linewidth=2, label='Average Fitness')
        ax2.plot(times, max_fitness, 'r--', linewidth=2, label='Maximum Fitness')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Fitness Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_genetic_diversity(self, save_path: str = None):
        """
        Plot genetic diversity and mutation rate evolution.
        
        Args:
            save_path: Path to save figure (optional)
        """
        if not self.population.statistics_history:
            print("No statistics history to plot")
            return
        
        stats = self.population.statistics_history
        times = [s.time for s in stats]
        diversity = [s.genetic_diversity for s in stats]
        mutation_rates = [s.mutation_rate_avg for s in stats]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, sharex=True)
        
        # Genetic diversity
        ax1.plot(times, diversity, 'purple', linewidth=2)
        ax1.set_ylabel('Genetic Diversity')
        ax1.set_title('Genetic Diversity Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Mutation rate evolution
        ax2.plot(times, mutation_rates, 'orange', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Average Mutation Rate')
        ax2.set_title('Mutation Rate Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_parameter_evolution(self, save_path: str = None):
        """
        Plot evolution of all genetic parameters.
        
        Args:
            save_path: Path to save figure (optional)
        """
        if not self.population.statistics_history:
            print("No statistics history to plot")
            return
        
        stats = self.population.statistics_history
        times = [s.time for s in stats]
        
        parameters = {
            'Energy Efficiency': [s.energy_efficiency_avg for s in stats],
            'Growth Rate': [s.growth_rate_avg for s in stats],
            'Division Threshold': [s.division_threshold_avg for s in stats],
            'Mutation Rate': [s.mutation_rate_avg * 1000 for s in stats]  # Scale for visibility
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, (param_name, values) in enumerate(parameters.items()):
            axes[i].plot(times, values, color=colors[i], linewidth=2)
            axes[i].set_title(f'{param_name} Evolution')
            axes[i].set_xlabel('Time')
            
            if param_name == 'Mutation Rate':
                axes[i].set_ylabel('Mutation Rate (×1000)')
            else:
                axes[i].set_ylabel(param_name)
            
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Genetic Parameter Evolution', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_genome_heatmap(self, num_genomes: int = 20, save_path: str = None):
        """
        Plot heatmap showing genome bit patterns for current population.
        
        Args:
            num_genomes: Number of genomes to display
            save_path: Path to save figure (optional)
        """
        alive_cells = [c for c in self.population.cells if c.is_alive]
        
        if not alive_cells:
            print("No living cells to visualize")
            return
        
        # Get most frequent genomes
        genome_counts = defaultdict(int)
        for cell in alive_cells:
            genome_counts[cell.genome.genome] += 1
        
        # Sort by frequency and take top N
        top_genomes = sorted(genome_counts.items(), key=lambda x: x[1], reverse=True)
        top_genomes = top_genomes[:num_genomes]
        
        # Convert to binary matrix
        genome_matrix = []
        genome_labels = []
        
        for genome_value, count in top_genomes:
            binary_str = format(genome_value, '032b')
            genome_matrix.append([int(bit) for bit in binary_str])
            genome_labels.append(f'0x{genome_value:08X} (n={count})')
        
        genome_matrix = np.array(genome_matrix)
        
        # Create heatmap
        plt.figure(figsize=(16, max(8, num_genomes * 0.4)))
        
        # Create custom colormap
        colors = ['white', 'darkblue']
        n_bins = 2
        cmap = plt.cm.colors.ListedColormap(colors)
        
        ax = sns.heatmap(genome_matrix, 
                        cmap=cmap,
                        cbar_kws={'label': 'Bit Value', 'ticks': [0, 1]},
                        yticklabels=genome_labels,
                        xticklabels=False,
                        linewidths=0.1,
                        linecolor='gray')
        
        # Add gene boundaries
        gene_boundaries = [8, 16, 24, 32]
        gene_names = ['Energy\nEfficiency', 'Growth\nRate', 'Division\nThreshold', 'Mutation\nRate']
        
        for boundary in gene_boundaries[:-1]:
            ax.axvline(boundary, color='red', linewidth=2)
        
        # Add gene labels at top
        gene_positions = [4, 12, 20, 28]
        for pos, name in zip(gene_positions, gene_names):
            ax.text(pos, -1.5, name, ha='center', va='center', fontsize=10, weight='bold')
        
        plt.title('Genome Bit Patterns in Current Population', fontsize=16, pad=20)
        plt.xlabel('Genome Position (32 bits)', fontsize=12)
        plt.ylabel('Genotype (Frequency)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fitness_landscape(self, parameter1: str = 'energy_efficiency', 
                             parameter2: str = 'growth_rate', save_path: str = None):
        """
        Plot 2D fitness landscape for two parameters.
        
        Args:
            parameter1: First parameter for x-axis
            parameter2: Second parameter for y-axis
            save_path: Path to save figure (optional)
        """
        alive_cells = [c for c in self.population.cells if c.is_alive]
        
        if len(alive_cells) < 10:
            print("Not enough cells for fitness landscape")
            return
        
        # Extract parameter values and fitness
        param1_values = [c.parameters[parameter1] for c in alive_cells]
        param2_values = [c.parameters[parameter2] for c in alive_cells]
        fitness_values = [c.get_fitness() for c in alive_cells]
        
        # Create scatter plot
        plt.figure(figsize=self.figure_size)
        
        scatter = plt.scatter(param1_values, param2_values, 
                            c=fitness_values, cmap='viridis', 
                            alpha=0.7, s=50)
        
        plt.colorbar(scatter, label='Fitness')
        plt.xlabel(parameter1.replace('_', ' ').title())
        plt.ylabel(parameter2.replace('_', ' ').title())
        plt.title(f'Fitness Landscape: {parameter1.replace("_", " ").title()} vs {parameter2.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_lineage_tree(self, max_generations: int = 10, save_path: str = None):
        """
        Plot simplified lineage tree showing evolutionary relationships.
        
        Args:
            max_generations: Maximum generations to show
            save_path: Path to save figure (optional)
        """
        if not self.population.lineage_tree:
            print("No lineage data to plot")
            return
        
        # Filter to recent generations
        current_gen = max(node.generation for node in self.population.lineage_tree.values())
        min_gen = max(0, current_gen - max_generations)
        
        relevant_nodes = {
            cell_id: node for cell_id, node in self.population.lineage_tree.items()
            if node.generation >= min_gen
        }
        
        if not relevant_nodes:
            print("No recent lineage data to plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot nodes by generation
        generation_positions = defaultdict(list)
        for node in relevant_nodes.values():
            generation_positions[node.generation].append(node)
        
        # Calculate positions
        node_positions = {}
        y_spacing = 1.0
        
        for gen, nodes in generation_positions.items():
            x_pos = gen - min_gen
            nodes_in_gen = len(nodes)
            
            if nodes_in_gen == 1:
                y_positions = [0]
            else:
                y_positions = np.linspace(-nodes_in_gen/2, nodes_in_gen/2, nodes_in_gen)
            
            for i, node in enumerate(nodes):
                node_positions[node.cell_id] = (x_pos, y_positions[i] * y_spacing)
        
        # Draw connections
        for node in relevant_nodes.values():
            if node.parent_id and node.parent_id in node_positions:
                parent_pos = node_positions[node.parent_id]
                child_pos = node_positions[node.cell_id]
                
                ax.plot([parent_pos[0], child_pos[0]], 
                       [parent_pos[1], child_pos[1]], 
                       'k-', alpha=0.6, linewidth=1)
        
        # Draw nodes
        for node in relevant_nodes.values():
            pos = node_positions[node.cell_id]
            
            # Color by fitness
            if node.death_time is None:  # Still alive
                color = 'green'
                alpha = 0.8
            else:
                color = 'red'
                alpha = 0.4
            
            circle = plt.Circle(pos, 0.1, color=color, alpha=alpha)
            ax.add_patch(circle)
            
            # Add cell ID label
            ax.text(pos[0], pos[1], str(node.cell_id), 
                   ha='center', va='center', fontsize=8, weight='bold')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Lineage Branch')
        ax.set_title(f'Lineage Tree (Last {max_generations} Generations)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='Living'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, alpha=0.4, label='Dead')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_selection_coefficients(self, save_path: str = None):
        """
        Plot selection coefficients for different traits.
        
        Args:
            save_path: Path to save figure (optional)
        """
        coefficients = self.population.calculate_selection_coefficients()
        
        if not coefficients:
            print("No selection coefficients to plot")
            return
        
        traits = list(coefficients.keys())
        values = list(coefficients.values())
        
        plt.figure(figsize=(10, 6))
        
        colors = ['blue' if v > 0 else 'red' for v in values]
        bars = plt.bar(traits, values, color=colors, alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.xlabel('Trait')
        plt.ylabel('Selection Coefficient')
        plt.title('Selection Coefficients for Genetic Traits')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.02),
                    f'{value:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_evolution_dashboard(self, save_path: str = None):
        """
        Create comprehensive dashboard with multiple plots.
        
        Args:
            save_path: Path to save figure (optional)
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Population dynamics
        plt.subplot(3, 3, 1)
        if self.population.statistics_history:
            stats = self.population.statistics_history
            times = [s.time for s in stats]
            pop_sizes = [s.alive_cells for s in stats]
            plt.plot(times, pop_sizes, 'b-', linewidth=2)
            plt.title('Population Size')
            plt.xlabel('Time')
            plt.ylabel('Alive Cells')
            plt.grid(True, alpha=0.3)
        
        # Fitness evolution
        plt.subplot(3, 3, 2)
        if self.population.statistics_history:
            avg_fitness = [s.average_fitness for s in stats]
            max_fitness = [s.max_fitness for s in stats]
            plt.plot(times, avg_fitness, 'g-', label='Average')
            plt.plot(times, max_fitness, 'r--', label='Maximum')
            plt.title('Fitness Evolution')
            plt.xlabel('Time')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Genetic diversity
        plt.subplot(3, 3, 3)
        if self.population.statistics_history:
            diversity = [s.genetic_diversity for s in stats]
            plt.plot(times, diversity, 'purple', linewidth=2)
            plt.title('Genetic Diversity')
            plt.xlabel('Time')
            plt.ylabel('Diversity')
            plt.grid(True, alpha=0.3)
        
        # Parameter evolution
        param_plots = [
            ('Energy Efficiency', [s.energy_efficiency_avg for s in stats], 'blue'),
            ('Growth Rate', [s.growth_rate_avg for s in stats], 'green'),
            ('Division Threshold', [s.division_threshold_avg for s in stats], 'red'),
            ('Mutation Rate', [s.mutation_rate_avg * 1000 for s in stats], 'orange')
        ]
        
        for i, (name, values, color) in enumerate(param_plots, 4):
            plt.subplot(3, 3, i)
            if self.population.statistics_history:
                plt.plot(times, values, color=color, linewidth=2)
                plt.title(name)
                plt.xlabel('Time')
                plt.ylabel(name)
                plt.grid(True, alpha=0.3)
        
        # Population summary stats
        plt.subplot(3, 3, 8)
        summary = self.population.get_population_summary()
        
        stats_text = f"""
        Generation: {summary.get('generation', 'N/A')}
        Population: {summary.get('alive_cells', 'N/A')}
        Avg Fitness: {summary.get('average_fitness', 'N/A'):.3f}
        Diversity: {summary.get('genetic_diversity', 'N/A'):.3f}
        Total Births: {summary.get('total_births', 'N/A')}
        Total Deaths: {summary.get('total_deaths', 'N/A')}
        Environment: {summary.get('environment', 'None')}
        """
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        plt.axis('off')
        plt.title('Population Summary')
        
        # Selection coefficients
        plt.subplot(3, 3, 9)
        coefficients = self.population.calculate_selection_coefficients()
        if coefficients:
            traits = list(coefficients.keys())
            values = list(coefficients.values())
            colors = ['blue' if v > 0 else 'red' for v in values]
            plt.bar(traits, values, color=colors, alpha=0.7)
            plt.title('Selection Coefficients')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.suptitle('Evolution Dashboard', fontsize=20, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing Visualization System")
    print("=" * 40)
    
    from population import run_evolution_experiment
    from environment import create_competitive_environment
    
    # Run a short experiment
    env = create_competitive_environment()
    population = run_evolution_experiment(
        environment=env,
        max_time=100.0,
        max_generations=10,
        snapshot_interval=25.0
    )
    
    # Create visualizations
    visualizer = EvolutionVisualizer(population)
    
    print("\nGenerating visualizations...")
    visualizer.plot_population_dynamics()
    visualizer.plot_parameter_evolution()
    visualizer.plot_genome_heatmap(num_genomes=10)
    visualizer.create_evolution_dashboard()
