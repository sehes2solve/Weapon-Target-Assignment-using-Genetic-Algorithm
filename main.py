import numpy
import GARI

temp_weapon_names = []
temp_weapon_num = []
print('\nEnter the weapon types and the number of instances of each type, press x to stop')
while True:    
    user_input = input()
    if (user_input == 'x'):
        break
    weapon_name, weapon_type = user_input.split()
    temp_weapon_names.append(weapon_name)  
    temp_weapon_num.append(int(weapon_type))


weapon_names = numpy.array(temp_weapon_names)
weapon_types_num = len(weapon_names)
temp_weapon_types = []
for i in range(weapon_types_num):
    for j in range(temp_weapon_num[i]):
        temp_weapon_types.append(i+1)

weapon_types = numpy.array(temp_weapon_types)
weapon_num = len(weapon_types)

target_num = int(input('\nEnter the number of targets:\n'))

print('\nEnter the threat coefficient of each target:')
threat_coeff = numpy.zeros(target_num)
for i in range(target_num):
    threat_coeff[i] = int(input())

sucess_prob = numpy.zeros((weapon_types_num, target_num))
print('\nEnter the weapons’ success probabilities matrix:')
for i in range(weapon_types_num):
    for j in range(target_num):
        sucess_prob[i][j] = float(input())

print('\nPlease wait while running the GA...')

population_size = 6
parents_num = int((population_size+1) / 2)
offsprings_num = int((population_size) / 2)
num_of_generations = 10
mutation_prob = 0.5

population = numpy.zeros((population_size, weapon_num))
for i in range (population_size):
    population[i, :] = numpy.random.randint(1, target_num+1, weapon_num)

for generation in range(num_of_generations):
    # Measing the fitness of each chromosome in the population.
    fitness_values = GARI.calc_population_fitness(population, target_num, weapon_num, sucess_prob, threat_coeff, weapon_types)
    
    # Selecting the best parents in the population for mating.
    parents = GARI.fill_mating_pool(population, fitness_values, parents_num)

    #crossover
    offsprings = GARI.crossover(parents, offsprings_num)

    #mutation
    mutants = GARI.mutation(offsprings, mutation_prob, target_num)

    #replacement by Elitist Strategy
    population = GARI.replacement(parents, mutants, population_size, weapon_num)

final_fitness_values = GARI.calc_population_fitness(population, target_num, weapon_num, sucess_prob, threat_coeff, weapon_types)
optimal_result = numpy.min(final_fitness_values)
optimal_chromosome_idx = numpy.where(final_fitness_values == optimal_result)
optimal_chromosome_idx = optimal_chromosome_idx[0][0]
optimal_chromosome = population[optimal_chromosome_idx]
print('\nThe final WTA solution is:\n',optimal_chromosome)
print(GARI.chromosome_encode(optimal_chromosome, target_num))
print('\nThe expected total threat of the surviving targets is', optimal_result)
