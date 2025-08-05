"""
rich_viz.py - Rich-based Terminal Visualization for Cell Evolution System

This module provides terminal visualization tools using the Rich library for
displaying cell states, populations, and live simulation dashboards.

Dependencies:
    pip install rich

Author: Generated for Computational Genome Cell Evolution System
"""

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box
import time
from typing import List, Any, Optional, Callable
from datetime import datetime


class CellVisualizer:
    """Main visualizer class for cell evolution system."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the visualizer with a Rich console."""
        self.console = console or Console()
        
    def display_cell_table(self, cell, title: str = "Cell State") -> None:
        """
        Display a single cell's state as a Rich table.
        
        Args:
            cell: Cell object with attributes like id, generation, mass, etc.
            title: Title for the table
        """
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        # Basic properties
        table.add_row("Cell ID", str(getattr(cell, 'id', 'N/A')))
        table.add_row("Generation", str(getattr(cell, 'generation', 'N/A')))
        
        # Cellular resources
        mass = getattr(cell, 'mass', 0)
        table.add_row("Mass", f"{mass:.3f}" if isinstance(mass, (int, float)) else str(mass))
        
        atp = getattr(cell, 'atp', 0)
        table.add_row("ATP", f"{atp:.3f}" if isinstance(atp, (int, float)) else str(atp))
        
        ribosomes = getattr(cell, 'ribosomes', 0)
        table.add_row("Ribosomes", str(ribosomes))
        
        proteins = getattr(cell, 'proteins', 0)
        table.add_row("Proteins", f"{proteins:.3f}" if isinstance(proteins, (int, float)) else str(proteins))
        
        # Fitness and status
        fitness = getattr(cell, 'fitness', 0)
        table.add_row("Fitness", f"{fitness:.6f}" if isinstance(fitness, (int, float)) else str(fitness))
        
        can_divide = getattr(cell, 'can_divide', False)
        table.add_row("Can Divide", "✅ Yes" if can_divide else "❌ No")
        
        is_alive = getattr(cell, 'is_alive', True)
        table.add_row("Is Alive", "✅ Alive" if is_alive else "💀 Dead")
        
        # Additional properties if they exist
        if hasattr(cell, 'age'):
            table.add_row("Age", str(cell.age))
        if hasattr(cell, 'genome_length'):
            table.add_row("Genome Length", str(cell.genome_length))
        if hasattr(cell, 'mutations'):
            table.add_row("Mutations", str(cell.mutations))
            
        self.console.print(table)
    
    def display_population_table(self, cells: List[Any], title: str = "Population Overview") -> None:
        """
        Display a table of all cells in the population.
        
        Args:
            cells: List of cell objects
            title: Title for the table
        """
        if not cells:
            self.console.print(Panel("No cells in population", style="red"))
            return
            
        table = Table(title=title, box=box.ROUNDED)
        
        # Add columns
        table.add_column("ID", style="cyan", justify="center")
        table.add_column("Gen", style="blue", justify="center")
        table.add_column("Mass", style="green", justify="right")
        table.add_column("ATP", style="yellow", justify="right")
        table.add_column("Ribosomes", style="magenta", justify="right")
        table.add_column("Proteins", style="white", justify="right")
        table.add_column("Fitness", style="bright_green", justify="right")
        table.add_column("Divide", style="bright_blue", justify="center")
        table.add_column("Status", style="bright_red", justify="center")
        
        # Add rows for each cell
        for cell in cells:
            cell_id = str(getattr(cell, 'id', 'N/A'))
            generation = str(getattr(cell, 'generation', 'N/A'))
            
            mass = getattr(cell, 'mass', 0)
            mass_str = f"{mass:.2f}" if isinstance(mass, (int, float)) else str(mass)
            
            atp = getattr(cell, 'atp', 0)
            atp_str = f"{atp:.2f}" if isinstance(atp, (int, float)) else str(atp)
            
            ribosomes = str(getattr(cell, 'ribosomes', 0))
            
            proteins = getattr(cell, 'proteins', 0)
            proteins_str = f"{proteins:.2f}" if isinstance(proteins, (int, float)) else str(proteins)
            
            fitness = getattr(cell, 'fitness', 0)
            fitness_str = f"{fitness:.4f}" if isinstance(fitness, (int, float)) else str(fitness)
            
            can_divide = getattr(cell, 'can_divide', False)
            divide_str = "✅" if can_divide else "❌"
            
            is_alive = getattr(cell, 'is_alive', True)
            status_str = "🟢" if is_alive else "🔴"
            
            table.add_row(
                cell_id, generation, mass_str, atp_str, ribosomes,
                proteins_str, fitness_str, divide_str, status_str
            )
        
        self.console.print(table)
    
    def create_population_summary_panel(self, cells: List[Any]) -> Panel:
        """Create a summary panel with population statistics."""
        if not cells:
            return Panel("No cells in population", title="Population Summary", style="red")
        
        alive_cells = [c for c in cells if getattr(c, 'is_alive', True)]
        can_divide_cells = [c for c in cells if getattr(c, 'can_divide', False)]
        
        total = len(cells)
        alive = len(alive_cells)
        can_divide = len(can_divide_cells)
        
        # Calculate averages for alive cells
        if alive_cells:
            avg_mass = sum(getattr(c, 'mass', 0) for c in alive_cells) / len(alive_cells)
            avg_atp = sum(getattr(c, 'atp', 0) for c in alive_cells) / len(alive_cells)
            avg_fitness = sum(getattr(c, 'fitness', 0) for c in alive_cells) / len(alive_cells)
            max_generation = max(getattr(c, 'generation', 0) for c in alive_cells)
        else:
            avg_mass = avg_atp = avg_fitness = max_generation = 0
        
        summary_text = f"""
[bold cyan]Total Cells:[/bold cyan] {total}
[bold green]Alive:[/bold green] {alive}
[bold blue]Can Divide:[/bold blue] {can_divide}
[bold yellow]Max Generation:[/bold yellow] {max_generation}

[bold magenta]Averages (Alive Cells):[/bold magenta]
  Mass: {avg_mass:.3f}
  ATP: {avg_atp:.3f}
  Fitness: {avg_fitness:.6f}
"""
        
        return Panel(summary_text.strip(), title="Population Summary", box=box.ROUNDED)
    
    def live_population_dashboard(
        self, 
        simulation_generator: Callable,
        refresh_rate: float = 0.5,
        max_steps: Optional[int] = None
    ) -> None:
        """
        Show a live-updating dashboard for the population over simulation steps.
        
        Args:
            simulation_generator: Generator function that yields (step, cells) tuples
            refresh_rate: Update frequency in seconds
            max_steps: Maximum number of steps (for progress bar)
        """
        layout = Layout()
        
        # Create layout structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="summary", ratio=1),
            Layout(name="table", ratio=2)
        )
        
        with Live(layout, refresh_per_second=1/refresh_rate, screen=True):
            step_count = 0
            start_time = time.time()
            
            for step, cells in simulation_generator:
                step_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Update header
                header_text = f"[bold blue]Cell Evolution Simulation Dashboard[/bold blue]\n"
                header_text += f"Step: {step} | Elapsed: {elapsed:.1f}s | "
                header_text += f"Rate: {step_count/elapsed:.2f} steps/s"
                layout["header"].update(Panel(header_text, style="blue"))
                
                # Update summary
                layout["summary"].update(self.create_population_summary_panel(cells))
                
                # Update main table
                if cells:
                    # Show top 20 cells by fitness for performance
                    sorted_cells = sorted(
                        cells, 
                        key=lambda c: getattr(c, 'fitness', 0), 
                        reverse=True
                    )[:20]
                    
                    table = Table(title=f"Top Cells (Step {step})", box=box.MINIMAL)
                    table.add_column("ID", style="cyan", width=8)
                    table.add_column("Gen", style="blue", width=5)
                    table.add_column("Mass", style="green", width=8)
                    table.add_column("ATP", style="yellow", width=8)
                    table.add_column("Fitness", style="bright_green", width=12)
                    table.add_column("Status", style="white", width=8)
                    
                    for cell in sorted_cells:
                        cell_id = str(getattr(cell, 'id', 'N/A'))[:7]
                        generation = str(getattr(cell, 'generation', 'N/A'))
                        
                        mass = getattr(cell, 'mass', 0)
                        mass_str = f"{mass:.2f}" if isinstance(mass, (int, float)) else str(mass)[:7]
                        
                        atp = getattr(cell, 'atp', 0)
                        atp_str = f"{atp:.2f}" if isinstance(atp, (int, float)) else str(atp)[:7]
                        
                        fitness = getattr(cell, 'fitness', 0)
                        fitness_str = f"{fitness:.6f}" if isinstance(fitness, (int, float)) else str(fitness)[:11]
                        
                        is_alive = getattr(cell, 'is_alive', True)
                        can_divide = getattr(cell, 'can_divide', False)
                        
                        if not is_alive:
                            status = "💀 Dead"
                        elif can_divide:
                            status = "🟢 Ready"
                        else:
                            status = "🟡 Growing"
                        
                        table.add_row(cell_id, generation, mass_str, atp_str, fitness_str, status)
                    
                    layout["table"].update(table)
                else:
                    layout["table"].update(Panel("No cells in population", style="red"))
                
                # Update footer
                footer_text = f"[dim]Press Ctrl+C to exit | Refresh rate: {refresh_rate}s[/dim]"
                layout["footer"].update(Panel(footer_text, style="dim"))
                
                if max_steps and step >= max_steps:
                    break
                
                time.sleep(refresh_rate)


# Convenience functions for easy import and use
def display_cell_table(cell, title: str = "Cell State", console: Optional[Console] = None) -> None:
    """
    Display a single cell's state as a Rich table.
    
    Args:
        cell: Cell object with attributes like id, generation, mass, etc.
        title: Title for the table
        console: Optional Rich console instance
    """
    visualizer = CellVisualizer(console)
    visualizer.display_cell_table(cell, title)


def display_population_table(cells: List[Any], title: str = "Population Overview", console: Optional[Console] = None) -> None:
    """
    Display a table of all cells in the population.
    
    Args:
        cells: List of cell objects
        title: Title for the table
        console: Optional Rich console instance
    """
    visualizer = CellVisualizer(console)
    visualizer.display_population_table(cells, title)


def live_population_dashboard(
    simulation_generator: Callable,
    refresh_rate: float = 0.5,
    max_steps: Optional[int] = None,
    console: Optional[Console] = None
) -> None:
    """
    Show a live-updating dashboard for the population over simulation steps.
    
    Args:
        simulation_generator: Generator function that yields (step, cells) tuples
        refresh_rate: Update frequency in seconds
        max_steps: Maximum number of steps (for progress bar)
        console: Optional Rich console instance
    """
    visualizer = CellVisualizer(console)
    visualizer.live_population_dashboard(simulation_generator, refresh_rate, max_steps)


# Example usage and demonstration
if __name__ == "__main__":
    # Mock cell class for demonstration
    class MockCell:
        def __init__(self, cell_id, generation=0, mass=1.0, atp=100.0, ribosomes=5, 
                     proteins=10.0, fitness=0.5, can_divide=True, is_alive=True):
            self.id = cell_id
            self.generation = generation
            self.mass = mass
            self.atp = atp
            self.ribosomes = ribosomes
            self.proteins = proteins
            self.fitness = fitness
            self.can_divide = can_divide
            self.is_alive = is_alive
            self.age = generation * 10
    
    # Create mock cells for demonstration
    mock_cells = [
        MockCell("cell_001", 0, 1.2, 95.5, 5, 12.3, 0.654321, True, True),
        MockCell("cell_002", 1, 0.8, 87.2, 4, 8.9, 0.543210, False, True),
        MockCell("cell_003", 1, 1.5, 110.0, 6, 15.2, 0.789012, True, True),
        MockCell("cell_004", 0, 0.5, 45.0, 2, 4.1, 0.234567, False, False),
        MockCell("cell_005", 2, 2.1, 156.7, 8, 22.4, 0.890123, True, True),
    ]
    
    print("\n=== Rich Visualization Demo ===\n")
    
    # Demo 1: Single cell display
    print("1. Single Cell Display:")
    display_cell_table(mock_cells[0], "Cell #001 Details")
    
    input("\nPress Enter to continue to population table...")
    
    # Demo 2: Population table
    print("\n2. Population Table:")
    display_population_table(mock_cells, "Current Population")
    
    input("\nPress Enter to continue to live dashboard demo...")
    
    # Demo 3: Mock live dashboard
    def mock_simulation():
        """Mock simulation generator for demonstration."""
        import random
        cells = mock_cells.copy()
        
        for step in range(50):
            # Simulate some changes
            for cell in cells:
                if cell.is_alive:
                    cell.mass += random.uniform(-0.1, 0.2)
                    cell.atp += random.uniform(-5, 10)
                    cell.fitness += random.uniform(-0.01, 0.02)
                    cell.can_divide = cell.mass > 1.0 and cell.atp > 80
                    
                    # Occasionally add new cells (division)
                    if random.random() < 0.1 and cell.can_divide:
                        new_cell = MockCell(
                            f"cell_{len(cells)+1:03d}",
                            cell.generation + 1,
                            cell.mass * 0.6,
                            cell.atp * 0.7,
                            cell.ribosomes,
                            cell.proteins * 0.8,
                            cell.fitness * random.uniform(0.8, 1.2),
                            False,
                            True
                        )
                        cells.append(new_cell)
                        cell.mass *= 0.6
                        cell.atp *= 0.7
            
            yield step, cells
            
    print("\n3. Live Dashboard (will run for a few seconds):")
    print("Starting live dashboard in 3 seconds...")
    time.sleep(3)
    
    try:
        live_population_dashboard(mock_simulation(), refresh_rate=0.2, max_steps=30)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    
    print("\n=== Demo Complete ===")
    print("\nTo use in your simulation:")
    print("  from rich_viz import display_cell_table, display_population_table, live_population_dashboard")
    print("  display_cell_table(my_cell)")
    print("  display_population_table(my_population)")
    print("  live_population_dashboard(my_simulation_generator)")