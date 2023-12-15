import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

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


def create_the_forest(forest_size, water):
    forest = np.ones(shape=(forest_size, forest_size), dtype=int)

    if water:
        add_water(forest)

    return forest


def add_water(forest):
    y = np.random.randint(0, forest.shape[0])
    x = np.random.randint(0, forest.shape[1])

    # algorytm - bladzenie losowe
    for _ in range(forest.size // 5):
        for i in range(-3, 4):
            for j in range(-3, 4):
                new_y = (y + i) % forest.shape[0]
                new_x = (x + j) % forest.shape[1]
                forest[new_y, new_x] = 4

        move_index = np.random.randint(0, 8)
        move = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)][move_index]

        y = (y + move[0]) % forest.shape[0]
        x = (x + move[1]) % forest.shape[1]


def start_the_fire(forest):
    trees = np.where(forest == 1)
    random_index = np.random.randint(0, len(trees[0]))
    y = trees[0][random_index]
    x = trees[1][random_index]
    forest[y, x] = 2


def next_generation(forest, burned_trees, tree_ignition_probability, burned_tree_regeneration_period, wind_direction,
                    wind_strength, neighborhood_type):

    # waruneki brzegowe - zamknięte pochłaniające
    forest = np.vstack([np.zeros((1, forest.shape[1]), dtype=int), forest, np.zeros((1, forest.shape[1]), dtype=int)])
    forest = np.hstack([np.zeros((forest.shape[0], 1), dtype=int), forest, np.zeros((forest.shape[0], 1), dtype=int)])

    if burned_tree_regeneration_period:
        burned_trees = np.vstack([np.zeros((1, burned_trees.shape[1]), dtype=int), burned_trees, np.zeros((1, burned_trees.shape[1]), dtype=int)])
        burned_trees = np.hstack([np.zeros((burned_trees.shape[0], 1), dtype=int), burned_trees, np.zeros((burned_trees.shape[0], 1), dtype=int)])

    next_generation_forest = forest.copy()

    trees = np.where(forest == 1)

    y = trees[0]
    x = trees[1]

    # sasiedztwo - moore
    if neighborhood_type == 'moore':

        # N
        if wind_direction == 0:
            wind_neighbors = np.vstack([forest[y+1, x-1], forest[y+1, x], forest[y+1, x+1]])

            regular_neighbors = np.vstack([
                forest[y-1, x-1], forest[y-1, x], forest[y-1, x+1],
                forest[y, x-1], forest[y, x+1],
            ])

        # W
        elif wind_direction == 1:
            wind_neighbors = np.vstack([
                forest[y+1, x+1],
                forest[y, x+1],
                forest[y-1, x+1]
            ])

            regular_neighbors = np.vstack([
                forest[y+1, x-1], forest[y+1, x],
                forest[y, x-1],
                forest[y-1, x-1], forest[y-1, x]
            ])

        # E
        elif wind_direction == 2:
            wind_neighbors = np.vstack([
                forest[y+1, x-1],
                forest[y, x-1],
                forest[y-1, x-1]
            ])

            regular_neighbors = np.vstack([
                forest[y+1, x], forest[y+1, x+1],
                forest[y, x+1],
                forest[y-1, x], forest[y-1, x+1]
            ])

        # S
        elif wind_direction == 3:
            wind_neighbors = np.vstack([forest[y-1, x-1], forest[y-1, x], forest[y-1, x+1]])

            regular_neighbors = np.vstack([
                forest[y, x-1], forest[y, x+1],
                forest[y+1, x-1], forest[y+1, x], forest[y+1, x+1]
            ])

    # sasiedztwo - von neumann
    elif neighborhood_type == 'von_neumann':

        # N
        if wind_direction == 0:
            wind_neighbors = np.vstack([forest[y+1, x]])

            regular_neighbors = np.vstack([forest[y-1, x], forest[y, x-1], forest[y, x+1]])

        # W
        elif wind_direction == 1:
            wind_neighbors = np.vstack([forest[y, x+1]])

            regular_neighbors = np.vstack([forest[y+1, x], forest[y, x-1], forest[y-1, x]])

        # E
        elif wind_direction == 2:
            wind_neighbors = np.vstack([forest[y, x-1]])

            regular_neighbors = np.vstack([forest[y+1, x], forest[y-1, x], forest[y, x+1]])

        # S
        elif wind_direction == 3:
            wind_neighbors = np.vstack([forest[y-1, x]])

            regular_neighbors = np.vstack([forest[y, x-1], forest[y+1, x], forest[y, x+1]])

    tree_ignition_probability_wind = tree_ignition_probability + tree_ignition_probability * (wind_strength / 2)
    tree_ignition_probability_regular = tree_ignition_probability - tree_ignition_probability * (wind_strength / 2)

    burn_condition_wind = np.any(wind_neighbors == 2, axis=0) & (tree_ignition_probability_wind > np.random.random(len(x)))
    next_generation_forest[y[burn_condition_wind], x[burn_condition_wind]] = 2

    burn_condition_regular = np.any(regular_neighbors == 2, axis=0) & (tree_ignition_probability_regular > np.random.random(len(x)))
    next_generation_forest[y[burn_condition_regular], x[burn_condition_regular]] = 2

    burning_trees = np.where(forest == 2)
    next_generation_forest[burning_trees[0], burning_trees[1]] = 3

    if burned_tree_regeneration_period:
        burned_trees[burned_trees != 0] += 1
        burned_trees[burning_trees[0], burning_trees[1]] = 1

        trees_to_restore = np.where(burned_trees == burned_tree_regeneration_period+1)
        if len(trees_to_restore[0]) != 0:
            burned_trees[trees_to_restore[0], trees_to_restore[1]] = 0
            next_generation_forest[trees_to_restore[0], trees_to_restore[1]] = 1

        burned_trees = burned_trees[1:-1, :]
        burned_trees = burned_trees[:, 1:-1]

    next_generation_forest = next_generation_forest[1:-1, :]
    next_generation_forest = next_generation_forest[:, 1:-1]

    return next_generation_forest, burned_trees


def simulate_forest_fire(forest_size, water, tree_ignition_probability, tree_self_ignition_probability,
                         burned_tree_regeneration_period, wind_direction_change_period, wind_strength,
                         neighborhood_type, simulation_speed_ms):

    forest = create_the_forest(forest_size=forest_size, water=water)

    burned_trees = np.zeros_like(forest)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.axis('off')

    colors = ['#8B4513', '#228B22', '#FF4500', '#6B6B6B', '#4682B4']
    custom_cmap = ListedColormap(colors)

    # ['N', 'W', 'E', 'S']
    # [(1, 0), (0, -1), (0, 1), (-1, 0)]
    wind_direction = np.random.randint(0, 4)
    wind_duration = 0

    def update(frame):
        nonlocal forest, wind_direction, wind_duration, burned_trees
        if frame == 0:
            ax.imshow(forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
            ax.set_title('Las - początkowy stan lasu')
        elif frame == 1:
            start_the_fire(forest)
            ax.clear()
            plt.axis('off')
            ax.imshow(forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
            ax.set_title('Las - rozpoczęcie pożaru')
        else:
            if wind_duration == wind_direction_change_period:
                while True:
                    new_wind_direction = np.random.randint(0, 4)

                    if new_wind_direction != wind_direction:
                        break

                wind_direction = new_wind_direction
                wind_duration = 0

            if tree_self_ignition_probability > np.random.random():
                start_the_fire(forest)

            forest, burned_trees = next_generation(forest, burned_trees,
                                                   tree_ignition_probability=tree_ignition_probability,
                                                   burned_tree_regeneration_period=burned_tree_regeneration_period,
                                                   wind_direction=wind_direction, wind_strength=wind_strength,
                                                   neighborhood_type=neighborhood_type)

            wind_duration += 1

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            ax.clear()
            plt.axis('off')
            ax.imshow(forest, cmap=custom_cmap, vmin=0, vmax=4, interpolation='nearest')
            ax.set_title(f'Las - generacja {frame-1}')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            burning_trees = np.where(forest == 2)

            if len(burning_trees[0]) == 0:
                animation.event_source.stop()

        labels = ['Ziemia', 'Drzewo', 'Płonące drzewo', 'Spalone drzewo', 'Woda']
        handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10) for color in colors]

        wind_labels = ['N', 'W', 'E', 'S']
        labels.append(f'Wiatr ({wind_labels[wind_direction]})')

        wind_markers = ['^', '<', '>', 'v']
        wind_handle = plt.Line2D([0], [0], marker=wind_markers[wind_direction], color='w', markerfacecolor='black', markersize=10)
        handles.append(wind_handle)

        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    max_frames = 999999999999999999999999999999999999999
    animation = FuncAnimation(fig, update, frames=max_frames, interval=simulation_speed_ms, repeat=False)

    plt.show()


simulate_forest_fire(
    forest_size=500,
    water=True,
    tree_ignition_probability=0.7,
    tree_self_ignition_probability=0.0001,
    burned_tree_regeneration_period=500,
    wind_direction_change_period=30,
    wind_strength=0.5,
    neighborhood_type='moore',
    # neighborhood_type='von_neumann',
    simulation_speed_ms=200
)
