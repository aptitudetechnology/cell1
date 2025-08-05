# Instructions for Claude: Create rich_viz.py for Rich-based Terminal Visualization

## Objective
Create a Python module named `rich_viz.py` that uses the Rich library to visualize the state of cells and populations in the computational genome cell evolution system. The goal is to provide clear, live-updating terminal dashboards and tables for simulation runs.

## Requirements
1. **Dependencies:**
   - Use the `rich` library (install with `pip install rich`).

2. **Features to Implement:**
   - **Cell Table:**
     - Function to display a single cell's state as a Rich table (fields: id, generation, mass, ATP, ribosomes, proteins, fitness, can_divide, is_alive, etc.).
   - **Population Table:**
     - Function to display a summary table for a list of cells (one row per cell, key stats as columns).
   - **Live Dashboard:**
     - Function to show a live-updating dashboard during simulation (e.g., using `Live` from Rich).
     - Optionally, include a progress bar for simulation steps.
   - **Integration:**
     - Functions should accept cell or population objects from the main simulation code.
     - Should be easy to call from main.py or examples.py at any simulation step.

3. **Example Usage:**
   - Provide example code snippets for displaying a cell, a population, and a live dashboard.

4. **Documentation:**
   - Add docstrings and comments for clarity.

## Example Table Columns
- Cell ID
- Generation
- Mass
- ATP
- Ribosomes
- Proteins
- Fitness
- Can Divide
- Is Alive

## Example Function Signatures
```python
def display_cell_table(cell):
    """Display a single cell's state as a Rich table."""
    ...

def display_population_table(cells):
    """Display a table of all cells in the population."""
    ...

def live_population_dashboard(cells, steps):
    """Show a live-updating dashboard for the population over simulation steps."""
    ...
```

## Notes
- Use Rich's Table, Live, and Progress features.
- Make the module self-contained and importable.
- Do not include simulation logic—only visualization.
- Ensure compatibility with the existing cell and population classes.

---

**End of instructions.**
