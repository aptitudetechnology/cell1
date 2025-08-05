"""
Environment System
Controls selection pressures, resource availability, and environmental conditions.
"""
import random
import math
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class SelectionType(Enum):
    """Types of selection pressure."""
    DIRECTIONAL = "directional"  # Favors one extreme
    STABILIZING = "stabilizing"  # Favors intermediate values
    DISRUPTIVE = "disruptive"    # Favors extremes
    FREQUENCY_DEPENDENT = "frequency_dependent"  # Selection depends on genotype frequency


@dataclass
class EnvironmentalConditions:
    """Environmental parameters affecting cell survival and reproduction."""
    temperature: float = 37.0  # Celsius
    ph: float = 7.0           # pH level
    oxygen_level: float = 1.0  # Relative oxygen availability
    nutrient_density: float = 1.0  # Overall nutrient availability
    toxin_level: float = 0.0   # Environmental toxicity
    carrying_capacity: int = 1000  # Maximum population size


class SelectionPressure:
    """Represents a specific selection pressure acting on the population."""
    
    def __init__(self, name: str, target_parameter: str, 
                 selection_type: SelectionType = SelectionType.DIRECTIONAL,
                 strength: float = 1.0, optimal_value: float = None):
        """
        Initialize selection pressure.
        
        Args:
            name: Name of the selection pressure
            target_parameter: Cell parameter being selected for
            selection_type: Type of selection
            strength: Strength of selection (0-1)
            optimal_value: Optimal value for stabilizing selection
        """
        self.name = name
        self.target_parameter = target_parameter
        self.selection_type = selection_type
        self.strength = strength
        self.optimal_value = optimal_value
        self.active = True
    
    def calculate_fitness_modifier(self, parameter_value: float, 
                                 population_stats: Dict = None) -> float:
        """
        Calculate fitness modifier based on selection type.
        
        Args:
            parameter_value: Value of the target parameter
            population_stats: Population statistics for frequency-dependent selection
            
        Returns:
            Fitness modifier (0-2, where 1 is neutral)
        """
        if not self.active:
            return 1.0
        
        modifier = 1.0
        
        if self.selection_type == SelectionType.DIRECTIONAL:
            # Higher values are better
            modifier = 1.0 + (parameter_value - 1.0) * self.strength
        
        elif self.selection_type == SelectionType.STABILIZING:
            if self.optimal_value is not None:
                # Closer to optimal is better
                distance = abs(parameter_value - self.optimal_value)
                max_distance = max(abs(0 - self.optimal_value), abs(2 - self.optimal_value))
                normalized_distance = distance / max_distance
                modifier = 1.0 + (1.0 - normalized_distance) * self.strength - self.strength
        
        elif self.selection_type == SelectionType.DISRUPTIVE:
            # Extreme values are better
            if self.optimal_value is not None:
                distance = abs(parameter_value - self.optimal_value)
                max_distance = max(abs(0 - self.optimal_value), abs(2 - self.optimal_value))
                normalized_distance = distance / max_distance
                modifier = 1.0 + normalized_distance * self.strength
        
        elif self.selection_type == SelectionType.FREQUENCY_DEPENDENT:
            # Rare genotypes have advantage (negative frequency dependence)
            if population_stats and 'genotype_frequencies' in population_stats:
                # This would need more sophisticated implementation
                modifier = 1.0  # Placeholder
        
        return max(0.1, min(2.0, modifier))  # Clamp between 0.1 and 2.0


class Environment:
    """
    Manages environmental conditions and selection pressures.
    """
    
    def __init__(self, name: str = "Default Environment"):
        """
        Initialize environment with default conditions.
        
        Args:
            name: Name of the environment
        """
        self.name = name
        self.conditions = EnvironmentalConditions()
        self.selection_pressures: List[SelectionPressure] = []
        self.time = 0.0
        self.history = []
        
        # Environmental change parameters
        self.change_frequency = 0.0  # How often environment changes
        self.change_magnitude = 0.1  # How much it changes
        self.cyclic_changes = {}     # Cyclic environmental variables
    
    def add_selection_pressure(self, pressure: SelectionPressure):
        """Add a selection pressure to the environment."""
        self.selection_pressures.append(pressure)
    
    def remove_selection_pressure(self, pressure_name: str):
        """Remove a selection pressure by name."""
        self.selection_pressures = [p for p in self.selection_pressures 
                                   if p.name != pressure_name]
    
    def get_nutrient_availability(self, population_size: int) -> float:
        """
        Calculate nutrient availability based on population density.
        
        Args:
            population_size: Current population size
            
        Returns:
            Nutrient availability factor (0-1)
        """
        base_nutrients = self.conditions.nutrient_density
        
        # Carrying capacity effects
        if population_size >= self.conditions.carrying_capacity:
            density_factor = self.conditions.carrying_capacity / population_size
        else:
            density_factor = 1.0
        
        # Competition reduces available nutrients
        competition_factor = max(0.1, 1.0 - (population_size / (self.conditions.carrying_capacity * 2)))
        
        return base_nutrients * density_factor * competition_factor
    
    def calculate_environmental_stress(self) -> float:
        """
        Calculate overall environmental stress factor.
        
        Returns:
            Stress factor (0-1, where 0 is no stress)
        """
        stress_factors = []
        
        # Temperature stress
        optimal_temp = 37.0
        temp_stress = abs(self.conditions.temperature - optimal_temp) / 50.0
        stress_factors.append(min(1.0, temp_stress))
        
        # pH stress
        optimal_ph = 7.0
        ph_stress = abs(self.conditions.ph - optimal_ph) / 7.0
        stress_factors.append(min(1.0, ph_stress))
        
        # Oxygen stress
        oxygen_stress = max(0, 1.0 - self.conditions.oxygen_level)
        stress_factors.append(oxygen_stress)
        
        # Toxin stress
        toxin_stress = self.conditions.toxin_level
        stress_factors.append(min(1.0, toxin_stress))
        
        # Overall stress is average of individual stresses
        return sum(stress_factors) / len(stress_factors)
    
    def apply_selection(self, cell_parameters: Dict, population_stats: Dict = None) -> float:
        """
        Apply all selection pressures to calculate fitness modifier.
        
        Args:
            cell_parameters: Dictionary of cell parameters
            population_stats: Population statistics
            
        Returns:
            Combined fitness modifier
        """
        total_modifier = 1.0
        
        for pressure in self.selection_pressures:
            if pressure.target_parameter in cell_parameters:
                parameter_value = cell_parameters[pressure.target_parameter]
                modifier = pressure.calculate_fitness_modifier(
                    parameter_value, population_stats
                )
                total_modifier *= modifier
        
        return total_modifier
    
    def update(self, time_step: float):
        """
        Update environmental conditions over time.
        
        Args:
            time_step: Time elapsed
        """
        self.time += time_step
        
        # Apply cyclic changes
        for var_name, (amplitude, period, phase) in self.cyclic_changes.items():
            if hasattr(self.conditions, var_name):
                base_value = getattr(self.conditions, var_name)
                cyclic_value = amplitude * math.sin(2 * math.pi * self.time / period + phase)
                setattr(self.conditions, var_name, base_value + cyclic_value)
        
        # Random environmental fluctuations
        if self.change_frequency > 0 and random.random() < self.change_frequency * time_step:
            self._apply_random_change()
        
        # Record environmental state
        self.history.append({
            'time': self.time,
            'conditions': {
                'temperature': self.conditions.temperature,
                'ph': self.conditions.ph,
                'oxygen_level': self.conditions.oxygen_level,
                'nutrient_density': self.conditions.nutrient_density,
                'toxin_level': self.conditions.toxin_level
            },
            'stress_level': self.calculate_environmental_stress()
        })
    
    def _apply_random_change(self):
        """Apply random environmental changes."""
        # Randomly modify one environmental parameter
        parameters = ['temperature', 'ph', 'oxygen_level', 'nutrient_density', 'toxin_level']
        param_to_change = random.choice(parameters)
        
        current_value = getattr(self.conditions, param_to_change)
        change = random.uniform(-self.change_magnitude, self.change_magnitude)
        new_value = current_value + change
        
        # Apply reasonable bounds
        if param_to_change == 'temperature':
            new_value = max(0, min(100, new_value))
        elif param_to_change == 'ph':
            new_value = max(0, min(14, new_value))
        elif param_to_change in ['oxygen_level', 'nutrient_density']:
            new_value = max(0, min(2.0, new_value))
        elif param_to_change == 'toxin_level':
            new_value = max(0, min(1.0, new_value))
        
        setattr(self.conditions, param_to_change, new_value)
    
    def add_cyclic_change(self, parameter: str, amplitude: float, 
                         period: float, phase: float = 0.0):
        """
        Add cyclic changes to an environmental parameter.
        
        Args:
            parameter: Name of parameter to change
            amplitude: Amplitude of change
            period: Period of cycle
            phase: Phase shift
        """
        self.cyclic_changes[parameter] = (amplitude, period, phase)
    
    def create_nutrient_limitation_scenario(self):
        """Create a nutrient limitation scenario."""
        self.conditions.nutrient_density = 0.3
        self.conditions.carrying_capacity = 200
        
        # Add selection pressure for energy efficiency
        energy_pressure = SelectionPressure(
            name="nutrient_scarcity",
            target_parameter="energy_efficiency",
            selection_type=SelectionType.DIRECTIONAL,
            strength=0.8
        )
        self.add_selection_pressure(energy_pressure)
    
    def create_toxic_environment_scenario(self):
        """Create a toxic environment scenario."""
        self.conditions.toxin_level = 0.6
        
        # Add selection pressure for slower growth (conserve energy)
        toxin_pressure = SelectionPressure(
            name="toxin_resistance",
            target_parameter="growth_rate",
            selection_type=SelectionType.STABILIZING,
            strength=0.6,
            optimal_value=0.8  # Moderate growth rate is optimal
        )
        self.add_selection_pressure(toxin_pressure)
    
    def create_variable_environment_scenario(self):
        """Create an environment with regular changes."""
        self.change_frequency = 0.05  # Change every 20 time units on average
        self.change_magnitude = 0.2
        
        # Add cyclic nutrient availability
        self.add_cyclic_change('nutrient_density', 0.3, 100.0)
        
        # Selection for moderate mutation rates (evolvability)
        mutation_pressure = SelectionPressure(
            name="environmental_variability",
            target_parameter="mutation_rate",
            selection_type=SelectionType.STABILIZING,
            strength=0.4,
            optimal_value=0.005  # Moderate mutation rate
        )
        self.add_selection_pressure(mutation_pressure)
    
    def to_dict(self) -> Dict:
        """Convert environment to dictionary for serialization."""
        return {
            'name': self.name,
            'time': self.time,
            'conditions': {
                'temperature': self.conditions.temperature,
                'ph': self.conditions.ph,
                'oxygen_level': self.conditions.oxygen_level,
                'nutrient_density': self.conditions.nutrient_density,
                'toxin_level': self.conditions.toxin_level,
                'carrying_capacity': self.conditions.carrying_capacity
            },
            'selection_pressures': [
                {
                    'name': p.name,
                    'target_parameter': p.target_parameter,
                    'selection_type': p.selection_type.value,
                    'strength': p.strength,
                    'optimal_value': p.optimal_value,
                    'active': p.active
                }
                for p in self.selection_pressures
            ],
            'environmental_stress': self.calculate_environmental_stress(),
            'change_frequency': self.change_frequency,
            'change_magnitude': self.change_magnitude
        }
    
    def __str__(self) -> str:
        """String representation of environment."""
        return f"Environment('{self.name}', stress={self.calculate_environmental_stress():.2f})"


# Predefined environment scenarios
def create_benign_environment() -> Environment:
    """Create a benign environment with no selection pressures."""
    env = Environment("Benign Laboratory Conditions")
    return env


def create_competitive_environment() -> Environment:
    """Create an environment with resource competition."""
    env = Environment("Competitive Environment")
    env.conditions.carrying_capacity = 500
    env.conditions.nutrient_density = 0.8
    
    # Selection for efficient growth
    growth_pressure = SelectionPressure(
        name="resource_competition",
        target_parameter="growth_rate",
        selection_type=SelectionType.DIRECTIONAL,
        strength=0.6
    )
    env.add_selection_pressure(growth_pressure)
    
    return env


def create_harsh_environment() -> Environment:
    """Create a harsh environment with multiple stressors."""
    env = Environment("Harsh Environment")
    env.conditions.temperature = 45.0  # High temperature
    env.conditions.toxin_level = 0.4   # Moderate toxicity
    env.conditions.nutrient_density = 0.4  # Limited nutrients
    env.conditions.carrying_capacity = 200
    
    # Multiple selection pressures
    energy_pressure = SelectionPressure(
        name="energy_conservation",
        target_parameter="energy_efficiency",
        selection_type=SelectionType.DIRECTIONAL,
        strength=0.9
    )
    env.add_selection_pressure(energy_pressure)
    
    division_pressure = SelectionPressure(
        name="stress_tolerance",
        target_parameter="division_threshold",
        selection_type=SelectionType.STABILIZING,
        strength=0.5,
        optimal_value=120  # Larger cells survive better
    )
    env.add_selection_pressure(division_pressure)
    
    return env


if __name__ == "__main__":
    # Test the environment system
    print("Testing Environment System")
    print("=" * 40)
    
    # Create test environment
    env = create_competitive_environment()
    print(f"Environment: {env}")
    print(f"Conditions: {env.to_dict()['conditions']}")
    
    # Test selection pressures
    test_parameters = {
        'energy_efficiency': 1.5,
        'growth_rate': 2.0,
        'division_threshold': 100,
        'mutation_rate': 0.005
    }
    
    fitness_modifier = env.apply_selection(test_parameters)
    print(f"Fitness modifier for test parameters: {fitness_modifier:.3f}")
    
    # Test environmental changes
    print("\nTesting environmental changes...")
    for step in range(10):
        env.update(1.0)
        if step % 3 == 0:
            nutrients = env.get_nutrient_availability(population_size=100)
            stress = env.calculate_environmental_stress()
            print(f"Step {step}: Nutrients={nutrients:.3f}, Stress={stress:.3f}")
