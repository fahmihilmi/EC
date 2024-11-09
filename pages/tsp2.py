import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Streamlit input to get city names and coordinates
st.title("Traveling Salesman Problem (TSP) with Genetic Algorithm")

# Collect city names
num_cities = st.number_input("Enter number of cities:", min_value=2, value=10, step=1)

city_names = []
city_coords = {}

for i in range(num_cities):
    city_name = st.text_input(f"Enter name for city {i+1}:")
    city_names.append(city_name)
    x_coord = st.number_input(f"Enter x-coordinate for {city_name}:", key=f"x_{i}")
    y_coord = st.number_input(f"Enter y-coordinate for {city_name}:", key=f"y_{i}")
    city_coords[city_name] = (x_coord, y_coord)

# Visualization of the cities entered by user
if city_coords:
    x = [coord[0] for coord in city_coords.values()]
    y = [coord[1] for coord in city_coords.values()]
    cities_names = list(city_coords.keys())

    # Pastel Pallete for different cities
    colors = sns.color_palette("pastel", len(cities_names))

    # City Icons
    city_icons = {city: f"★" for city in cities_names}  # You can adjust these icons

    fig, ax = plt.subplots()
    ax.grid(False)  # Disable grid

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons[city]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')

        # Connect cities with opaque lines
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

# Parameters for the genetic algorithm
n_population = st.number_input("Enter population size:", min_value=10, value=250, step=10)
crossover_per = st.slider("Enter crossover rate (0-1):", min_value=0.0, max_value=1.0, value=0.8)
mutation_per = st.slider("Enter mutation rate (0-1):", min_value=0.0, max_value=1.0, value=0.2)
n_generations = st.number_input("Enter number of generations:", min_value=10, value=200, step=10)

# Genetic algorithm functions

# Initial Population
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

# Distance between two cities
def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual):
    total_dist = 0
    for i in range(0, len(individual)):
        if(i == len(individual) - 1):
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

# Fitness probability function
def fitness_prob(population):
    total_dist_all_individuals = []
    for i in range(0, len(population)):
        total_dist_all_individuals.append(total_dist_individual(population[i]))

    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - total_dist_all_individuals
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = population_fitness / population_fitness_sum
    return population_fitness_probs

# Roulette Wheel Selection
def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
    return population[selected_individual_index]

# Crossover
def crossover(parent_1, parent_2):
    n_cities_cut = len(cities_names) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1 = parent_1[0:cut]
    offspring_1 += [city for city in parent_2 if city not in offspring_1]

    offspring_2 = parent_2[0:cut]
    offspring_2 += [city for city in parent_1 if city not in offspring_2]

    return offspring_1, offspring_2

# Mutation
def mutation(offspring):
    n_cities_cut = len(cities_names) - 1
    index_1 = round(random.uniform(0, n_cities_cut))
    index_2 = round(random.uniform(0, n_cities_cut))

    temp = offspring[index_1]
    offspring[index_1] = offspring[index_2]
    offspring[index_2] = temp
    return offspring

# Main Genetic Algorithm
def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    fitness_probs = fitness_prob(population)

    parents_list = []
    for i in range(0, int(crossover_per * n_population)):
        parents_list.append(roulette_wheel(population, fitness_probs))

    offspring_list = []
    for i in range(0, len(parents_list), 2):
        offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i + 1])
        mutate_threashold = random.random()
        if (mutate_threashold > (1 - mutation_per)):
            offspring_1 = mutation(offspring_1)
        mutate_threashold = random.random()
        if (mutate_threashold > (1 - mutation_per)):
            offspring_2 = mutation(offspring_2)

        offspring_list.append(offspring_1)
        offspring_list.append(offspring_2)

    mixed_offspring = parents_list + offspring_list
    fitness_probs = fitness_prob(mixed_offspring)
    sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
    best_fitness_indices = sorted_fitness_indices[0:n_population]
    best_mixed_offspring = []
    for i in best_fitness_indices:
        best_mixed_offspring.append(mixed_offspring[i])

    for i in range(0, n_generations):
        fitness_probs = fitness_prob(best_mixed_offspring)
        parents_list = []
        for i in range(0, int(crossover_per * n_population)):
            parents_list.append(roulette_wheel(best_mixed_offspring, fitness_probs))

        offspring_list = []
        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i + 1])

            mutate_threashold = random.random()
            if (mutate_threashold > (1 - mutation_per)):
                offspring_1 = mutation(offspring_1)

            mutate_threashold = random.random()
            if (mutate_threashold > (1 - mutation_per)):
                offspring_2 = mutation(offspring_2)

            offspring_list.append(offspring_1)
            offspring_list.append(offspring_2)

        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_prob(mixed_offspring)
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        best_fitness_indices = sorted_fitness_indices[0:int(0.8 * n_population)]

        best_mixed_offspring = []
        for i in best_fitness_indices:
            best_mixed_offspring.append(mixed_offspring[i])

        old_population_indices = [random.randint(0, (n_population - 1)) for j in range(int(0.2 * n_population))]
        for i in old_population_indices:
            best_mixed_offspring.append(population[i])

        random.shuffle(best_mixed_offspring)

    return
