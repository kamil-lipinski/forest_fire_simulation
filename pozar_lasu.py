import gc
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Stany:
# 1 - drzewo
# 2 - plonace drzewo
# 3 - spalone drzewo
# 4 - woda

# Zasady ewolucji:
# - drzewo staje sie plonacym drzewem z prawdopodobienstwem p, jesli ma w sasiedztwie plonace drzewo
# - plonace drzewo w nastepnej generacji staje sie spalonym drzewem
# - spalone drzewo odnawia sie po k iteracjach
# - samozaplon drzewa nastepuje z prawdopodobienstwem ps (odpowiednio male)
# - uwzglednic wode, która stanowi bariere dla ognia
# - uwzglednic wiatr zmieniajacy prawdopodobienstwa rozprzestrzeniania sie pozaru w róznych kierunkach,
#   kierunek powinien zmieniac sie co kilka iteracji


class ForestFireSimulation:
    def __init__(self, forest_size, water, tree_ignition_probability, tree_self_ignition_probability,
                 burned_tree_regeneration_period, wind_direction_change_period, wind_strength, neighborhood_type,
                 simulation_speed_ms):

        if water:
            self.forest = np.ones(shape=(forest_size, forest_size), dtype=int)
            self.add_water()
        else:
            self.forest = np.ones(shape=(forest_size, forest_size), dtype=int)

        self.tree_ignition_probability = tree_ignition_probability
        self.tree_self_ignition_probability = tree_self_ignition_probability
        self.burned_tree_regeneration_period = burned_tree_regeneration_period
        self.wind_direction_change_period = wind_direction_change_period
        self.wind_strength = wind_strength
        self.neighborhood_type = neighborhood_type
        self.simulation_speed_ms = simulation_speed_ms

        self.wind_duration = 0
        self.wind_direction = np.random.randint(0, 4)
        self.burned_trees = np.zeros(shape=(forest_size, forest_size), dtype=int)

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

        # waruneki brzegowe - zamknięte pochłaniające
        self.forest = np.vstack([np.zeros((1, self.forest.shape[1]), dtype=int),
                                 self.forest,
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

        # sasiedztwo - moore
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

        # sasiedztwo - von neumann
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

        burn_condition_regular = np.any(regular_neighbors == 2, axis=0) & (tree_ignition_probability_regular > np.random.random(len(x)))
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

            self.burned_trees = self.burned_trees[1:-1, :]
            self.burned_trees = self.burned_trees[:, 1:-1]

        next_generation_forest = next_generation_forest[1:-1, :]
        next_generation_forest = next_generation_forest[:, 1:-1]

        self.forest = next_generation_forest

    def simulate_forest_fire(self):

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.set_facecolor('#f0f0f0')
        plt.axis('off')

        colors = ['#8B4513', '#228B22', '#FF4500', '#6B6B6B', '#4682B4']
        custom_cmap = ListedColormap(colors)


        def update(frame):
            wind_labels = ['N ⬆', 'W ⬅', 'E ➡', 'S ⬇']
            if frame == 0:
                ax.imshow(self.forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
                ax.set_title(f'Początkowy stan lasu\nWiatr: {wind_labels[self.wind_direction]}')
            elif frame == 1:
                self.start_the_fire()
                ax.clear()
                plt.axis('off')
                ax.imshow(self.forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
                ax.set_title(f'Rozpoczęcie pożaru\nWiatr: {wind_labels[self.wind_direction]}')
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

                ax.set_title(f'Generacja: {frame - 1}\nWiatr: {wind_labels[self.wind_direction]}')

                burning_trees = np.where(self.forest == 2)

                if len(burning_trees[0]) == 0:
                    animation.event_source.stop()

        max_frames = 999999999999999999999999999999999999999
        animation = FuncAnimation(fig, update, frames=max_frames, interval=self.simulation_speed_ms, repeat=False)

        return animation


class GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Pożar lasu - Symulacja")

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

        self.start_button = tk.Button(
            self,
            text="Rozpocznij symulację",
            command=self.start_simulation,
            wraplength=70
        )

        self.stop_button = tk.Button(
            self,
            text="Zatrzymaj symulację",
            command=self.stop_simulation,
            state=tk.DISABLED,
            wraplength=70
        )

        self.simulation_label = tk.Label(self, text="")

        self.simulation_running = False
        self.create_widgets()

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.set_facecolor('#f0f0f0')
        self.ax.axis('off')
        self.ax.imshow(np.ones((10, 10)), cmap='gray', vmin=0, vmax=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, rowspan=14, padx=0, pady=0, sticky='nsew')

        self.protocol("WM_DELETE_WINDOW", self.exit)

    def create_widgets(self):

        vcmd_numeric = (self.register(self.validate_numeric), '%P')

        label_wrap_length = 100

        tk.Label(self, text="Rozmiar lasu", wraplength=label_wrap_length).grid(
            row=2, column=2, sticky="nsew", padx=5, pady=0)
        tk.Scale(self, from_=50, to=1000, orient='horizontal', variable=self.forest_size_var).grid(
            row=2, column=3, padx=(5, 30), pady=0, sticky="nsew")

        tk.Label(self, text="Woda", wraplength=label_wrap_length).grid(
            row=3, column=2, sticky="nsew", padx=5, pady=0)
        tk.Checkbutton(self, variable=self.water_var).grid(
            row=3, column=3, padx=(5, 30), pady=0, sticky="nsew")

        tk.Label(self, text="P-stwo zapłonu drzewa", wraplength=label_wrap_length).grid(
            row=4, column=2, sticky="nsew", padx=5, pady=0)
        tk.Scale(self, from_=0, to=1, resolution=0.01, orient='horizontal', variable=self.tree_ignition_prob_var).grid(
            row=4, column=3, padx=(5, 30), pady=0, sticky="nsew")

        tk.Label(self, text="P-stwo samozapłonu drzewa", wraplength=label_wrap_length).grid(
            row=5, column=2, sticky="nsew", padx=5, pady=0)
        tk.Scale(self, from_=0, to=0.1, resolution=0.0001, orient='horizontal', variable=self.tree_self_ignition_prob_var).grid(
            row=5, column=3, padx=(5, 30), pady=0, sticky="nsew")

        tk.Label(self, text="Okres regeneracji spalonego drzewa [iter] (0 = OFF)", wraplength=label_wrap_length).grid(
            row=6, column=2, sticky="nsew", padx=5, pady=0)
        tk.Entry(self, textvariable=self.burned_tree_reg_period_var, validate='all', validatecommand=vcmd_numeric).grid(
            row=6, column=3, padx=(5, 30), pady=0, sticky="ew")

        tk.Label(self, text="Okres zmiany kierunku wiatru [iter]", wraplength=label_wrap_length).grid(
            row=7, column=2, sticky="nsew", padx=5, pady=0)
        tk.Entry(self, textvariable=self.wind_dir_change_period_var, validate='all', validatecommand=vcmd_numeric).grid(
            row=7, column=3, padx=(5, 30), pady=0, sticky="ew")

        tk.Label(self, text="Siła wiatru", wraplength=label_wrap_length).grid(
            row=8, column=2, sticky="nsew", padx=5, pady=0)
        tk.Scale(self, from_=0, to=1, resolution=0.01, orient='horizontal', variable=self.wind_strength_var).grid(
            row=8, column=3, padx=(5, 30), pady=0, sticky="nsew")

        tk.Label(self, text="Typ sąsiedztwa", wraplength=label_wrap_length).grid(
            row=9, column=2, sticky="nsew", padx=5, pady=0)
        neighborhood_options = ['Moore', 'Von Neumann']
        tk.OptionMenu(self, self.neighborhood_type_var, *neighborhood_options).grid(
            row=9, column=3, padx=(5, 30), pady=0, sticky="ew")

        tk.Label(self, text="Długość wyświetlania klatki [ms]", wraplength=label_wrap_length).grid(
            row=10, column=2, sticky="nsew", padx=5, pady=0)
        tk.Entry(self, textvariable=self.simulation_speed_var, validate='all', validatecommand=vcmd_numeric).grid(
            row=10, column=3, padx=(5, 30), pady=0, sticky="ew")

        self.start_button.grid(row=11, column=2, pady=10)
        self.stop_button.grid(row=11, column=3, pady=10)

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

        if self.simulation_running:
            return

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

        self.simulation_running = True
        self.stop_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)

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

        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()
            plt.close(self.canvas.figure)

        self.animation = simulation.simulate_forest_fire()
        self.canvas = FigureCanvasTkAgg(self.animation._fig, self)
        self.canvas_widget = self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=14, padx=0, pady=0, sticky='nsew')

        self.update_idletasks()

    def stop_simulation(self):
        gc.collect()

        if self.animation:
            self.animation.event_source.stop()
            plt.close(self.animation._fig)
            self.animation = None

        self.simulation_running = False
        self.simulation_label.config(text="")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

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
