import gc
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# States:
# 1 - tree
# 2 - burning tree
# 3 - burned tree
# 4 - water

# Evolution rules:
# - A tree becomes a burning tree with probability p if it has a burning tree in its neighborhood
# - A burning tree becomes a burned tree in the next generation
# - A burned tree regenerates after k iterations
# - Tree self-ignition occurs with probability ps (relatively low)
# - Consider water, which acts as a barrier to fire
# - Consider wind changing the probabilities of fire spread in different directions,
#   the direction should change every few iterations

class ForestFireSimulation:
    def __init__(self, forest_size, water, tree_ignition_probability, tree_self_ignition_probability,
                 burned_tree_regeneration_period, wind_direction_change_period, wind_strength, neighborhood_type,
                 simulation_speed_ms):

        self.forest = np.ones(shape=(forest_size, forest_size), dtype=int)

        if water:
            self.add_water()

        self.tree_ignition_probability = tree_ignition_probability
        self.tree_self_ignition_probability = tree_self_ignition_probability
        self.burned_tree_regeneration_period = burned_tree_regeneration_period
        self.wind_direction_change_period = wind_direction_change_period
        self.wind_strength = wind_strength
        self.neighborhood_type = neighborhood_type
        self.simulation_speed_ms = simulation_speed_ms
        self.wind_duration = 0
        self.wind_direction = np.random.randint(0, 4)

        self.burned_trees = np.zeros(shape=(forest_size, forest_size),
                                     dtype=int)

    def add_water(self):
        y = np.random.randint(0, self.forest.shape[0])
        x = np.random.randint(0, self.forest.shape[1])

        # algorytm - bladzenie losowe
        for _ in range(self.forest.size // 5):
            for i in range(-3, 4):
                for j in range(-3, 4):
                    new_y = (y + i) % self.forest.shape[0]
                    new_x = (x + j) % self.forest.shape[1]
                    self.forest[new_y, new_x] = 4

            move_index = np.random.randint(0, 8)
            move = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)][move_index]

            y = (y + move[0]) % self.forest.shape[0]
            x = (x + move[1]) % self.forest.shape[1]

    def start_the_fire(self):
        trees = np.where(self.forest == 1)
        if len(trees[0]) != 0:
            random_index = np.random.randint(0, len(trees[0]))
            y = trees[0][random_index]
            x = trees[1][random_index]
            self.forest[y, x] = 2

    def change_wind_direction(self):
        while True:
            new_wind_direction = np.random.randint(0, 4)

            if new_wind_direction != self.wind_direction:
                break

        self.wind_direction = new_wind_direction
        self.wind_duration = 0

    def next_generation(self):

        self.forest = np.vstack(
            [np.zeros((1, self.forest.shape[1]), dtype=int),self.forest,
                                 np.zeros((1, self.forest.shape[1]), dtype=int)])

        self.forest = np.hstack([np.zeros((self.forest.shape[0], 1), dtype=int),
                                 self.forest,
                                 np.zeros((self.forest.shape[0], 1), dtype=int)])

        if self.burned_tree_regeneration_period:
            self.burned_trees = np.vstack([np.zeros((1, self.burned_trees.shape[1]), dtype=int),
                                           self.burned_trees,
                                           np.zeros((1, self.burned_trees.shape[1]), dtype=int)])

            self.burned_trees = np.hstack([np.zeros((self.burned_trees.shape[0], 1), dtype=int),
                                           self.burned_trees,
                                           np.zeros((self.burned_trees.shape[0], 1), dtype=int)])

        next_generation_forest = self.forest.copy()

        trees = np.where(self.forest == 1)

        y = trees[0]
        x = trees[1]

        if self.neighborhood_type == 'Moore':

            # N
            if self.wind_direction == 0:
                wind_neighbors = np.vstack([self.forest[y+1, x-1], self.forest[y+1, x], self.forest[y+1, x+1]])

                regular_neighbors = np.vstack([
                    self.forest[y-1, x-1], self.forest[y-1, x], self.forest[y-1, x+1],
                    self.forest[y, x-1], self.forest[y, x+1],
                ])

            # W
            elif self.wind_direction == 1:
                wind_neighbors = np.vstack([
                    self.forest[y+1, x+1],
                    self.forest[y, x+1],
                    self.forest[y-1, x+1]
                ])

                regular_neighbors = np.vstack([
                    self.forest[y+1, x-1], self.forest[y+1, x],
                    self.forest[y, x-1],
                    self.forest[y-1, x-1], self.forest[y-1, x]
                ])

            # E
            elif self.wind_direction == 2:
                wind_neighbors = np.vstack([
                    self.forest[y+1, x-1],
                    self.forest[y, x-1],
                    self.forest[y-1, x-1]
                ])

                regular_neighbors = np.vstack([
                    self.forest[y+1, x], self.forest[y+1, x+1],
                    self.forest[y, x+1],
                    self.forest[y-1, x], self.forest[y-1, x+1]
                ])

            # S
            elif self.wind_direction == 3:
                wind_neighbors = np.vstack([self.forest[y-1, x-1], self.forest[y-1, x], self.forest[y-1, x+1]])

                regular_neighbors = np.vstack([
                    self.forest[y, x-1], self.forest[y, x+1],
                    self.forest[y+1, x-1], self.forest[y+1, x], self.forest[y+1, x+1]
                ])

        elif self.neighborhood_type == 'Von Neumann':

            # N
            if self.wind_direction == 0:
                wind_neighbors = np.vstack([self.forest[y+1, x]])

                regular_neighbors = np.vstack([self.forest[y-1, x], self.forest[y, x-1], self.forest[y, x+1]])

            # W
            elif self.wind_direction == 1:
                wind_neighbors = np.vstack([self.forest[y, x+1]])

                regular_neighbors = np.vstack([self.forest[y+1, x], self.forest[y, x-1], self.forest[y-1, x]])

            # E
            elif self.wind_direction == 2:
                wind_neighbors = np.vstack([self.forest[y, x-1]])

                regular_neighbors = np.vstack([self.forest[y+1, x], self.forest[y-1, x], self.forest[y, x+1]])

            # S
            elif self.wind_direction == 3:
                wind_neighbors = np.vstack([self.forest[y-1, x]])

                regular_neighbors = np.vstack([self.forest[y, x-1], self.forest[y+1, x], self.forest[y, x+1]])

        tree_ignition_probability_wind = (self.tree_ignition_probability +
                                          self.tree_ignition_probability * (self.wind_strength / 2))

        tree_ignition_probability_regular = (self.tree_ignition_probability -
                                             self.tree_ignition_probability * (self.wind_strength / 2))


        burn_condition_wind = np.any(wind_neighbors == 2, axis=0) & (tree_ignition_probability_wind > np.random.random(len(x)))
        next_generation_forest[y[burn_condition_wind], x[burn_condition_wind]] = 2

        burn_condition_regular = np.any(regular_neighbors == 2, axis=0) & np.all(wind_neighbors != 2, axis=0) & (tree_ignition_probability_regular > np.random.random(len(x)))
        next_generation_forest[y[burn_condition_regular], x[burn_condition_regular]] = 2

        burning_trees = np.where(self.forest == 2)
        next_generation_forest[burning_trees[0], burning_trees[1]] = 3

        if self.burned_tree_regeneration_period:
            self.burned_trees[self.burned_trees != 0] += 1
            self.burned_trees[burning_trees[0], burning_trees[1]] = 1

            trees_to_restore = np.where(self.burned_trees == self.burned_tree_regeneration_period + 1)
            if len(trees_to_restore[0]) != 0:
                self.burned_trees[trees_to_restore[0], trees_to_restore[1]] = 0
                next_generation_forest[trees_to_restore[0], trees_to_restore[1]] = 1

            self.burned_trees = self.burned_trees[1:-1, 1:-1]

        next_generation_forest = next_generation_forest[1:-1, 1:-1]

        self.forest = next_generation_forest

    def simulate_forest_fire(self):

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.set_facecolor('#f0f0f0')
        plt.axis('off')

        colors = ['#8B4513', '#228B22', '#FF4500', '#6B6B6B', '#4682B4']
        custom_cmap = ListedColormap(colors)

        def update(frame):

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            wind_labels = ['N ⬆', 'W ⬅', 'E ➡', 'S ⬇']

            if frame == 0:
                ax.imshow(self.forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
                ax.set_title(f'Initial forest state\nWind: {wind_labels[self.wind_direction]}\n', fontweight='bold', color='#111111')
            elif frame == 1:
                self.start_the_fire()
                ax.clear()
                plt.axis('off')
                ax.imshow(self.forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
                ax.set_title(f'Start of the fire\nWind: {wind_labels[self.wind_direction]}\n', fontweight='bold', color='#111111')
            else:
                if self.wind_duration == self.wind_direction_change_period:
                    self.change_wind_direction()

                if self.tree_self_ignition_probability > np.random.random():
                    self.start_the_fire()

                self.next_generation()
                self.wind_duration += 1

                ax.clear()
                plt.axis('off')
                ax.imshow(self.forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')

                ax.set_title(f'Generation: {frame - 1}\nWind: {wind_labels[self.wind_direction]}\n', fontweight='bold', color='#111111')

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                burning_trees = np.where(self.forest == 2)

                if len(burning_trees[0]) == 0:
                    animation.event_source.stop()

        max_frames = 999999999999999999999999999999999999999
        animation = FuncAnimation(fig, update, frames=max_frames, interval=self.simulation_speed_ms, repeat=False)

        return animation


class GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Forest Fire Simulation")

        self.animation = None

        self.forest_size_var = tk.IntVar(value=100)
        self.water_var = tk.BooleanVar(value=True)
        self.tree_ignition_prob_var = tk.DoubleVar(value=0.7)
        self.tree_self_ignition_prob_var = tk.DoubleVar(value=0.0001)
        self.burned_tree_reg_period_var = tk.IntVar(value=100)
        self.wind_dir_change_period_var = tk.IntVar(value=30)
        self.wind_strength_var = tk.DoubleVar(value=0.5)
        self.neighborhood_type_var = tk.StringVar(value='Moore')
        self.simulation_speed_var = tk.IntVar(value=200)

        self.font_tuple = ('DejaVu Sans', 10, 'bold')

        self.start_button = tk.Button(
            self,
            text='Start the simulation',
            command=self.start_simulation,
            wraplength=70,
            font=self.font_tuple,
            fg='#111111'
        )

        self.stop_button = tk.Button(
            self,
            text='Stop the simulation',
            command=self.stop_simulation,
            wraplength=70,
            font=self.font_tuple,
            fg='#111111'
        )

        self.create_widgets()

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.set_facecolor('#f0f0f0')
        self.ax.axis('off')
        self.ax.set_title('Forest Fire Simulation\n', fontweight='bold', color='#111111')
        self.ax.imshow(np.ones((10, 10)), cmap='gray', vmin=0, vmax=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, rowspan=16, padx=0, pady=0, sticky='nsew')

        self.toolbar = Toolbar(self.canvas, self, pack_toolbar=False)
        self.toolbar.grid(row=15, column=0, pady=0)

        self.protocol('WM_DELETE_WINDOW', self.exit)

    def create_widgets(self):

        vcmd_numeric = (self.register(self.validate_numeric), '%P')

        label_wrap_length = 100

        tk.Label(self, text='Forest size', wraplength=label_wrap_length, font=self.font_tuple, fg='#111111').grid(
            row=6, column=2, sticky='nsew', padx=5, pady=0)
        tk.Scale(self, from_=50, to=1000, orient='horizontal', variable=self.forest_size_var).grid(
            row=6, column=3, padx=(5, 30), pady=0, sticky='nsew')

        tk.Label(self, text='Water', wraplength=label_wrap_length, font=self.font_tuple, fg='#111111').grid(
            row=7, column=2, sticky='nsew', padx=5, pady=0)
        tk.Checkbutton(self, variable=self.water_var).grid(
            row=7, column=3, padx=(5, 30), pady=0, sticky='nsew')
                            
        tk.Label(self, text='Tree ignation probability', wraplength=label_wrap_length, font=self.font_tuple, fg='#111111').grid(
            row=8, column=2, sticky='nsew', padx=5, pady=0)
        tk.Scale(self, from_=0, to=1, resolution=0.01, orient='horizontal', variable=self.tree_ignition_prob_var).grid(
            row=8, column=3, padx=(5, 30), pady=0, sticky='nsew')

        tk.Label(self, text='Tree self ignation probability', wraplength=label_wrap_length, font=self.font_tuple, fg='#111111').grid(
            row=9, column=2, sticky='nsew', padx=5, pady=0)
        tk.Scale(self, from_=0, to=0.1, resolution=0.0001, orient='horizontal', variable=self.tree_self_ignition_prob_var).grid(
            row=9, column=3, padx=(5, 30), pady=0, sticky='nsew')

        tk.Label(self, text=f'Burned tree regeneration period [iter]\n(0 = OFF)', wraplength=label_wrap_length, font=self.font_tuple, fg='#111111').grid(
            row=10, column=2, sticky='nsew', padx=5, pady=0)
        tk.Entry(self, textvariable=self.burned_tree_reg_period_var, validate='all', validatecommand=vcmd_numeric).grid(
            row=10, column=3, padx=(5, 30), pady=0, sticky='ew')

        tk.Label(self, text='Wind direction change period [iter]', wraplength=label_wrap_length, font=self.font_tuple, fg='#111111').grid(
            row=11, column=2, sticky='nsew', padx=5, pady=0)
        tk.Entry(self, textvariable=self.wind_dir_change_period_var, validate='all', validatecommand=vcmd_numeric).grid(
            row=11, column=3, padx=(5, 30), pady=0, sticky='ew')

        tk.Label(self, text='Wind strength', wraplength=label_wrap_length, font=self.font_tuple, fg='#111111').grid(
            row=12, column=2, sticky='nsew', padx=5, pady=0)
        tk.Scale(self, from_=0, to=1, resolution=0.01, orient='horizontal', variable=self.wind_strength_var).grid(
            row=12, column=3, padx=(5, 30), pady=0, sticky='nsew')

        tk.Label(self, text='Neighborhood type', wraplength=label_wrap_length, font=self.font_tuple, fg='#111111').grid(
            row=13, column=2, sticky='nsew', padx=5, pady=0)
        neighborhood_options = ['Moore', 'Von Neumann']
        tk.OptionMenu(self, self.neighborhood_type_var, *neighborhood_options).grid(
            row=13, column=3, padx=(5, 30), pady=0, sticky='ew')

        tk.Label(self, text='Simulation speed [ms]', wraplength=label_wrap_length, font=self.font_tuple, fg='#111111').grid(
            row=14, column=2, sticky='nsew', padx=5, pady=0)
        tk.Entry(self, textvariable=self.simulation_speed_var, validate='all', validatecommand=vcmd_numeric).grid(
            row=14, column=3, padx=(5, 30), pady=0, sticky='ew')

        self.start_button.grid(row=15, column=2, pady=15)
        self.stop_button.grid(row=15, column=3, pady=15)

    @staticmethod
    def validate_numeric(value):
        if not value:
            return True
        try:
            int(value)
            return True
        except ValueError:
            return False

    def start_simulation(self):

        gc.collect()

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            plt.close()
            self.canvas = None

        if self.animation:
            if self.animation.event_source:
                self.animation.event_source.stop()
                self.animation = None

        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None

        gc.collect()

        forest_size = int(self.forest_size_var.get())
        water = bool(self.water_var.get())
        tree_ignition_prob = float(self.tree_ignition_prob_var.get())
        tree_self_ignition_prob = float(self.tree_self_ignition_prob_var.get())
        burned_tree_reg_period = int(self.burned_tree_reg_period_var.get()) if self.burned_tree_reg_period_var.get() else False
        wind_dir_change_period = int(self.wind_dir_change_period_var.get()) if self.wind_dir_change_period_var.get() else 50
        wind_strength = float(self.wind_strength_var.get())
        neighborhood_type = str(self.neighborhood_type_var.get())
        simulation_speed = int(self.simulation_speed_var.get()) if self.simulation_speed_var.get() else 200

        simulation = ForestFireSimulation(
            forest_size=forest_size,
            water=water,
            tree_ignition_probability=tree_ignition_prob,
            tree_self_ignition_probability=tree_self_ignition_prob,
            burned_tree_regeneration_period=burned_tree_reg_period,
            wind_direction_change_period=wind_dir_change_period,
            wind_strength=wind_strength,
            neighborhood_type=neighborhood_type,
            simulation_speed_ms=simulation_speed
        )

        self.animation = simulation.simulate_forest_fire()
        self.canvas = FigureCanvasTkAgg(self.animation._fig, self)
        self.canvas_widget = self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=16, padx=0, pady=0, sticky='nsew')

        self.toolbar = Toolbar(self.canvas, self, pack_toolbar=False)
        self.toolbar.grid(row=15, column=0)

        gc.collect()

    def stop_simulation(self):

        gc.collect()

        if self.animation:
            if self.animation.event_source:
                self.animation.event_source.stop()
                plt.close()
                self.animation = None

        gc.collect()

    def exit(self):
        self.stop_simulation()
        self.destroy()
        self.quit()


class Toolbar(NavigationToolbar2Tk):
    def set_message(self, s):
        pass


if __name__ == "__main__":
    app = GUI()
    app.mainloop()
