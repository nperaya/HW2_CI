import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
from dataclasses import dataclass

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class LightingConfig:
    brightness_range: tuple = (0, 120)  # Extended range to accommodate out-of-range values
    time_of_day_range: tuple = (0, 24)
    light_intensity_range: tuple = (0, 100)

class FuzzyLightingControlSystem:
    def __init__(self, config: LightingConfig):
        self.config = config
        self._initialize_fuzzy_variables()

    def _initialize_fuzzy_variables(self):
        # Initialize fuzzy variables
        self.brightness = ctrl.Antecedent(np.arange(*self.config.brightness_range, 1), 'brightness')
        self.time_of_day = ctrl.Antecedent(np.arange(*self.config.time_of_day_range, 1), 'time_of_day')
        self.light_intensity = ctrl.Consequent(np.arange(*self.config.light_intensity_range, 1), 'light_intensity')

        # Custom membership functions
        self.brightness['dark'] = fuzz.trimf(self.brightness.universe, [0, 0, 30])
        self.brightness['dim'] = fuzz.trimf(self.brightness.universe, [20, 50, 80])
        self.brightness['bright'] = fuzz.trimf(self.brightness.universe, [60, 90, 100])
        self.brightness['very_bright'] = fuzz.trimf(self.brightness.universe, [90, 120, 120])  #out-of-range values

        self.time_of_day['night'] = fuzz.trimf(self.time_of_day.universe, [0, 3, 6])
        self.time_of_day['morning'] = fuzz.trimf(self.time_of_day.universe, [5, 9, 12])
        self.time_of_day['afternoon'] = fuzz.trimf(self.time_of_day.universe, [11, 15, 18])
        self.time_of_day['evening'] = fuzz.trimf(self.time_of_day.universe, [17, 21, 24])

        self.light_intensity['very_low'] = fuzz.trimf(self.light_intensity.universe, [0, 0, 25])
        self.light_intensity['low'] = fuzz.trimf(self.light_intensity.universe, [15, 35, 55])
        self.light_intensity['medium'] = fuzz.trimf(self.light_intensity.universe, [45, 60, 75])
        self.light_intensity['high'] = fuzz.trimf(self.light_intensity.universe, [65, 85, 100])
        self.light_intensity['very_high'] = fuzz.trimf(self.light_intensity.universe, [90, 100, 100])

        self._initialize_rules()

    def _initialize_rules(self):
        # Define expanded set of fuzzy rules
        rule1 = ctrl.Rule(self.brightness['dark'] & self.time_of_day['night'], self.light_intensity['very_high'])
        rule2 = ctrl.Rule(self.brightness['dark'] & self.time_of_day['morning'], self.light_intensity['high'])
        rule3 = ctrl.Rule(self.brightness['dim'] & self.time_of_day['afternoon'], self.light_intensity['medium'])
        rule4 = ctrl.Rule(self.brightness['bright'], self.light_intensity['low'])
        rule5 = ctrl.Rule(self.brightness['dark'] & self.time_of_day['evening'], self.light_intensity['very_high'])
        rule6 = ctrl.Rule(self.brightness['dim'] & self.time_of_day['night'], self.light_intensity['high'])
        rule7 = ctrl.Rule(self.brightness['dim'] & self.time_of_day['morning'], self.light_intensity['medium'])
        rule8 = ctrl.Rule(self.brightness['bright'] & self.time_of_day['afternoon'], self.light_intensity['very_low'])
        rule9 = ctrl.Rule(self.brightness['bright'] & self.time_of_day['evening'], self.light_intensity['very_low'])
        rule10 = ctrl.Rule(self.brightness['dim'] & self.time_of_day['evening'], self.light_intensity['high'])
        rule11 = ctrl.Rule(self.brightness['dark'] & self.time_of_day['afternoon'], self.light_intensity['medium'])
        rule12 = ctrl.Rule(self.brightness['bright'] & self.time_of_day['night'], self.light_intensity['low'])
        rule13 = ctrl.Rule(self.brightness['bright'] & self.time_of_day['morning'], self.light_intensity['very_low'])
        
        # rule for out-of-range brightness
        rule14 = ctrl.Rule(self.brightness['very_bright'], self.light_intensity['very_low'])

        # Create control system
        self.light_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14])
        self.light_simulation = ctrl.ControlSystemSimulation(self.light_ctrl)

    def compute_light_intensity(self, brightness_value, time_of_day_value):
        try:
            brightness_value, time_of_day_value = self._validate_input(brightness_value, time_of_day_value)

            self.light_simulation.input['brightness'] = brightness_value
            self.light_simulation.input['time_of_day'] = time_of_day_value
            self.light_simulation.compute()

            intensity = self.light_simulation.output['light_intensity']
            logging.info(f"Computed light intensity: {intensity:.2f} for brightness: {brightness_value} and time_of_day: {time_of_day_value}")
            return intensity

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None

    def _validate_input(self, brightness, time_of_day):
        if brightness < self.config.brightness_range[0]:
            msg = f"Brightness value {brightness} is below minimum, adjusting to {self.config.brightness_range[0]}."
            logging.warning(msg)
            brightness = self.config.brightness_range[0]
        elif brightness > self.config.brightness_range[1]:
            msg = f"Brightness value {brightness} is above maximum, adjusting to {self.config.brightness_range[1]}."
            logging.warning(msg)
            brightness = self.config.brightness_range[1]

        if time_of_day < self.config.time_of_day_range[0]:
            msg = f"Time of Day value {time_of_day} is below minimum, adjusting to {self.config.time_of_day_range[0]}."
            logging.warning(msg)
            time_of_day = self.config.time_of_day_range[0]
        elif time_of_day > self.config.time_of_day_range[1]:
            msg = f"Time of Day value {time_of_day} is above maximum, adjusting to {self.config.time_of_day_range[1]}."
            logging.warning(msg)
            time_of_day = self.config.time_of_day_range[1]

        return brightness, time_of_day

    def plot_membership_functions(self):
        self.brightness.view()
        self.time_of_day.view()
        self.light_intensity.view()
        plt.show()

    def plot_results(self, results):
        df = pd.DataFrame(results, columns=['Light Intensity'])
        sns.lineplot(data=df, marker='o')
        plt.title("Computed Light Intensity Over Test Cases")
        plt.xlabel("Test Case")
        plt.ylabel("Light Intensity")
        plt.grid(True)
        plt.show()

# Example of configuration and usage
config = LightingConfig()
fuzzy_control_system = FuzzyLightingControlSystem(config)

# Expanded test cases including out-of-range value
test_cases = [
    {'brightness': 20, 'time_of_day': 22},  # Dark, Evening
    {'brightness': 60, 'time_of_day': 10},  # Dim, Morning
    {'brightness': 90, 'time_of_day': 15},  # Bright, Afternoon
    {'brightness': 30, 'time_of_day': 20},  # Dim, Evening
    {'brightness': 10, 'time_of_day': 3},   # Dark, Night
    {'brightness': 80, 'time_of_day': 8},   # Bright, Morning
    {'brightness': 40, 'time_of_day': 13},  # Dim, Afternoon
    {'brightness': 70, 'time_of_day': 18},  # Bright, Evening
    {'brightness': 105, 'time_of_day': 12}, # Very Bright (out of range), 
]

results = []
for i, case in enumerate(test_cases):
    intensity = fuzzy_control_system.compute_light_intensity(case['brightness'], case['time_of_day'])
    if intensity is not None:
        results.append(intensity)

if results:
    fuzzy_control_system.plot_results(results)

# Show membership functions
fuzzy_control_system.plot_membership_functions() 