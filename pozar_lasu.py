import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import Canvas, Button, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.axis('off')

        colors = ['#8B4513', '#228B22', '#FF4500', '#6B6B6B', '#4682B4']
        custom_cmap = ListedColormap(colors)

        def update(frame):
            if frame == 0:
                ax.imshow(self.forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
                ax.set_title('Las - początkowy stan lasu')
            elif frame == 1:
                self.start_the_fire()
                ax.clear()
                plt.axis('off')
                ax.imshow(self.forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
                ax.set_title('Las - rozpoczęcie pożaru')
            else:
                if self.wind_duration == self.wind_direction_change_period:
                    self.change_wind_direction()

                if self.tree_self_ignition_probability > np.random.random():
                    self.start_the_fire()

                self.next_generation()
                self.wind_duration += 1

                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                ax.clear()
                plt.axis('off')
                ax.imshow(self.forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
                ax.set_title(f'Las - generacja {frame - 1}')

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                burning_trees = np.where(self.forest == 2)

                if len(burning_trees[0]) == 0:
                    animation.event_source.stop()

            labels = ['Ziemia', 'Drzewo', 'Płonące drzewo', 'Spalone drzewo', 'Woda']
            handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10) for color in colors]

            wind_labels = ['N', 'W', 'E', 'S']
            labels.append(f'Wiatr ({wind_labels[self.wind_direction]})')

            wind_markers = ['^', '<', '>', 'v']
            wind_handle = plt.Line2D([0], [0], marker=wind_markers[self.wind_direction], color='w', markerfacecolor='black', markersize=10)
            handles.append(wind_handle)

            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

        max_frames = 999999999999999999999999999999999999999
        animation = FuncAnimation(fig, update, frames=max_frames, interval=self.simulation_speed_ms, repeat=False)

        plt.show()


class GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Pożar lasu - Symulacja")

        self.forest_size_var = tk.IntVar(value=100)
        self.water_var = tk.BooleanVar(value=True)
        self.tree_ignition_prob_var = tk.DoubleVar(value=0.7)
        self.tree_self_ignition_prob_var = tk.DoubleVar(value=0.0001)
        self.burned_tree_reg_period_var = tk.StringVar(value='100')
        self.wind_dir_change_period_var = tk.StringVar(value='30')
        self.wind_strength_var = tk.DoubleVar(value=0.5)
        self.neighborhood_type_var = tk.StringVar(value='Moore')
        self.simulation_speed_var = tk.StringVar(value='200')

        self.start_button = tk.Button(
            self,
            text="Rozpocznij symulację",
            command=self.start_simulation,
            disabledforeground='grey',
            highlightbackground='grey'
        )

        self.simulation_running = False
        self.create_widgets()


    def create_widgets(self):

        vcmd_numeric = (self.register(self.validate_numeric), '%P')

        tk.Label(self, text="Rozmiar lasu:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Scale(self, from_=50, to=1000, orient='horizontal', variable=self.forest_size_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self, text="Woda:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Checkbutton(self, variable=self.water_var).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="P-stwo zapłonu drzewa:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        tk.Scale(self, from_=0, to=1, resolution=0.01, orient='horizontal', variable=self.tree_ignition_prob_var).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self, text="P-stwo samozapłonu drzewa:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        tk.Scale(self, from_=0, to=0.1, resolution=0.0001, orient='horizontal', variable=self.tree_self_ignition_prob_var).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self, text="Okres regeneracji spalonego drzewa:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self, textvariable=self.burned_tree_reg_period_var, validate='all', validatecommand=vcmd_numeric).grid(row=4, column=1, padx=5, pady=5)

        tk.Label(self, text="Okres zmiany kierunku wiatru:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self, textvariable=self.wind_dir_change_period_var, validate='all', validatecommand=vcmd_numeric).grid(row=5, column=1, padx=5, pady=5)

        tk.Label(self, text="Siła wiatru:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        tk.Scale(self, from_=0, to=1, resolution=0.01, orient='horizontal', variable=self.wind_strength_var).grid(row=6, column=1, padx=5, pady=5)

        tk.Label(self, text="Typ sąsiedztwa:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        neighborhood_options = ['Moore', 'Von Neumann']
        tk.OptionMenu(self, self.neighborhood_type_var, *neighborhood_options).grid(row=7, column=1, padx=5, pady=5)

        tk.Label(self, text="Długość wyświetlania klatki (ms):").grid(row=8, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self, textvariable=self.simulation_speed_var, validate='all', validatecommand=vcmd_numeric).grid(row=8, column=1, padx=5, pady=5)

        self.start_button.grid(row=9, column=0, columnspan=2, pady=10)

    def validate_numeric(self, value):
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

        simulation.simulate_forest_fire()

        self.simulation_running = False
        self.start_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    app = GUI()
    app.mainloop()



