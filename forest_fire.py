import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

# Stany:
# 0 - ziemia
# 1 - drzewo
# 2 - plonace drzewo
# 3 - spalone drzewo
# 4 - woda

# Zasady ewolucji:
# - drzewo staje sie plonacym drzewem z prawdopodobienstwem p, jesli ma w sasiedztwie plonace drzewo
# - plonace drzewo w nastepnej generacji staje sie spalonym drzewem
# - spalone drzewo pozostaje spalonym
# - ziemia pozostaje ziemia


def create_the_forest(forest_dimension, water):
    forest = np.random.choice([0, 1], size=[forest_dimension, forest_dimension], p=[0.1, 0.9])

    if water:
        forest = add_water(forest)

    return forest


def add_water(forest):
    size = forest.shape[0] ** 2 // 2
    random_y = np.random.randint(0, forest.shape[0])
    random_x = np.random.randint(0, forest.shape[1])

    forest[random_y, random_x] = 4

    current_position = (random_y, random_x)

    for _ in range(size):
        direction_index = np.random.randint(0, 8)

        direction_mapping = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        direction = direction_mapping[direction_index]

        new_position = (
            (current_position[0] + direction[0]) % forest.shape[0],
            (current_position[1] + direction[1]) % forest.shape[1]
        )

        forest[new_position] = 4
        for i in range(-1, 2):
            for j in range(-1, 2):
                x = (new_position[1] + i) % forest.shape[1]
                y = (new_position[0] + j) % forest.shape[0]
                forest[y, x] = 4

        current_position = new_position

    return forest


def start_the_fire(forest):
    trees = np.where(forest == 1)
    random_index = np.random.randint(0, len(trees[0]))
    y = trees[0][random_index]
    x = trees[1][random_index]
    forest[y, x] = 2


def next_generation(forest, tree_ignition_probability, neighborhood_type):

    # warunek brzegowy - zamknięte pochłaniające
    forest = np.vstack([np.zeros((1, forest.shape[1]), dtype=int), forest, np.zeros((1, forest.shape[1]), dtype=int)])
    forest = np.hstack([np.zeros((forest.shape[0], 1), dtype=int), forest, np.zeros((forest.shape[0], 1), dtype=int)])

    next_generation_forest = forest.copy()

    trees = np.where(forest == 1)

    y = trees[0]
    x = trees[1]

    # sasiedztwo - moore
    if neighborhood_type == 'moore':
        neighbors = np.vstack([
            forest[y-1, x-1], forest[y, x-1], forest[y+1, x-1],
            forest[y-1, x], forest[y+1, x],
            forest[y-1, x+1], forest[y, x+1], forest[y+1, x+1]
        ])

    # sasiedztwo - von neumann
    elif neighborhood_type == 'von_neumann':
        neighbors = np.vstack([
            forest[y, x-1],
            forest[y-1, x], forest[y+1, x],
            forest[y, x+1]
        ])

    burn_condition = np.any(neighbors == 2, axis=0) & (tree_ignition_probability > np.random.random(len(x)))

    next_generation_forest[y[burn_condition], x[burn_condition]] = 2

    burning_trees = np.where(forest == 2)
    burned_condition = forest[burning_trees] == 2
    next_generation_forest[burning_trees[0][burned_condition], burning_trees[1][burned_condition]] = 3

    next_generation_forest = next_generation_forest[1:-1, :]
    next_generation_forest = next_generation_forest[:, 1:-1]

    return next_generation_forest


def simulate_forest_fire(forest_dimension, water, tree_ignition_probability, tree_self_ignition_probability, neighborhood_type, simulation_speed_ms):
    forest = create_the_forest(forest_dimension=forest_dimension, water=water)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.axis('off')

    colors = ['#8B4513', '#228B22', '#FF4500', '#6B6B6B', '#3F64BA']
    custom_cmap = ListedColormap(colors)

    def update(frame):
        nonlocal forest
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
            forest = next_generation(forest, tree_ignition_probability=tree_ignition_probability, neighborhood_type=neighborhood_type)

            if tree_self_ignition_probability > np.random.random():
                start_the_fire(forest)

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
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    max_frames = 999999999999999999999999999999999999999
    animation = FuncAnimation(fig, update, frames=max_frames, interval=simulation_speed_ms, repeat=False)

    plt.show()


simulate_forest_fire(
    forest_dimension=500,
    water=True,
    tree_ignition_probability=0.6,
    tree_self_ignition_probability=0.005,
    neighborhood_type='moore',
    # neighborhood_type='von_neumann',
    simulation_speed_ms=200
)
