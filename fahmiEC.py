import streamlit as st
import random

# Set page configuration
st.set_page_config(page_title="Genetic Algorithm")

# Add header
st.header("Genetic Algorithm", divider="gray")

# Default genetic algorithm parameters
POP_SIZE = 500  # Population size
MUT_RATE = 0.2  # Mutation rate
GENES = ' abcdefghijklmnopqrstuvwxyz'  # Gene pool (alphabet + space)

# Get user input for target string
TARGET = st.text_input("Enter your target string:", "fahmi")

# Initialization
def initialize_pop(TARGET):
    population = list()
    tar_len = len(TARGET)

    for i in range(POP_SIZE):
        temp = list()
        for j in range(tar_len):
            temp.append(random.choice(GENES))
        population.append(temp)

    return population

# Fitness calculation: Returns the difference from the target string
def fitness_cal(TARGET, chromo_from_pop):
    difference = 0
    for tar_char, chromo_char in zip(TARGET, chromo_from_pop):
        if tar_char != chromo_char:
            difference += 1
    return [chromo_from_pop, difference]

# Selection: Select top 50% based on fitness
def selection(population, TARGET):
    sorted_chromo_pop = sorted(population, key=lambda x: x[1])
    return sorted_chromo_pop[:int(0.5 * POP_SIZE)]

# Crossover: Mate parents to create offspring
def crossover(selected_chromo, CHROMO_LEN, population):
    offspring_cross = []
    for i in range(int(POP_SIZE)):
        parent1 = random.choice(selected_chromo)
        parent2 = random.choice(population[:int(POP_SIZE * 0.5)])

        p1 = parent1[0]
        p2 = parent2[0]

        crossover_point = random.randint(1, CHROMO_LEN - 1)
        child = p1[:crossover_point] + p2[crossover_point:]
        offspring_cross.extend([child])
    return offspring_cross

# Mutation: Mutate offspring based on mutation rate
def mutate(offspring, MUT_RATE):
    mutated_offspring = []

    for arr in offspring:
        for i in range(len(arr)):
            if random.random() < MUT_RATE:
                arr[i] = random.choice(GENES)
        mutated_offspring.append(arr)
    return mutated_offspring

# Replacement: Replace the least fit individuals with new generation
def replace(new_gen, population):
    for _ in range(len(population)):
        if population[_][1] > new_gen[_][1]:
            population[_][0] = new_gen[_][0]
            population[_][1] = new_gen[_][1]
    return population

# Main function to run the genetic algorithm
def main(POP_SIZE, MUT_RATE, TARGET, GENES):
    # 1) Initialize population
    initial_population = initialize_pop(TARGET)
    found = False
    population = []
    generation = 1

    # 2) Calculate the fitness for the current population
    for _ in range(len(initial_population)):
        population.append(fitness_cal(TARGET, initial_population[_]))

    # Now population has 2 things: [chromosome, fitness]
    # 3) Loop until the target string is found
    while not found:
        # 3.1) Select best individuals from the current population
        selected = selection(population, TARGET)

        # 3.2) Mate parents to create a new generation
        population = sorted(population, key=lambda x: x[1])
        crossovered = crossover(selected, len(TARGET), population)

        # 3.3) Mutate the offspring to diversify the new generation
        mutated = mutate(crossovered, MUT_RATE)

        new_gen = []
        for _ in mutated:
            new_gen.append(fitness_cal(TARGET, _))

        # 3.4) Replace the least fit individuals with new generation
        population = replace(new_gen, population)

        # Check if target is found
        if population[0][1] == 0:
            st.write('Target found!')
            st.write(f"String: {''.join(population[0][0])} Generation: {generation} Fitness: {population[0][1]}")
            break
        
        # Display the current best string and its fitness
        st.write(f"String: {''.join(population[0][0])} Generation: {generation} Fitness: {population[0][1]}")
        generation += 1

# Run the algorithm
if TARGET:
    result = main(POP_SIZE, MUT_RATE, TARGET, GENES)
