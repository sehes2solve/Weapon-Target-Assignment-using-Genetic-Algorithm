import numpy
import random

#weapon assigned to target
def chromosome_encode_2D(chromosome, target_num):
	rows = len(chromosome)
	cols = target_num
	result = numpy.zeros((rows, cols))
	for i in range (rows):
		for j in range (cols):
			result[i][j] = (chromosome[i] == j+1) * 1  # we multiply * 1 to convert the var from bool to int
	return result

#calculating fitness value for certain chromosome
def calc_chromosome_fitness(chromosome, target_num, weapon_num, sucess_prob, threat_coeff, weapon_types):
	x = chromosome_encode_2D(chromosome, target_num)
	result = 0
	for j in range(target_num):
		product = 1
		for i in range(weapon_num):
			product *= (1-sucess_prob[weapon_types[i]-1][j])**x[i][j]
		result += threat_coeff[j]*product
	return result

def calc_population_fitness(population, target_num, weapon_num, sucess_prob, threat_coeff, weapon_types):
	fitness_values = numpy.zeros(population.shape[0])	#shape[0] return num of elements(which is arrays in this case) inside population array, shape[1] return size of each element (which is size of each array in this case)
	for i_chromosome in range(population.shape[0]):
		fitness_values[i_chromosome] = calc_chromosome_fitness(population[i_chromosome, :], target_num, weapon_num, sucess_prob, threat_coeff, weapon_types)
	return fitness_values

def fill_mating_pool(population, fitness_values, parents_num):
	parents = numpy.empty((parents_num, population.shape[1]))
	for parent in range(parents_num):
		best_chromosome_idx = numpy.where(fitness_values == numpy.min(fitness_values))
		best_chromosome_idx = best_chromosome_idx[0][0]
		parents[parent, :] = population[best_chromosome_idx]
		fitness_values[best_chromosome_idx] = float('inf')
	return parents

def crossover(parents, offsprings_num):
	offsprings = numpy.empty((offsprings_num, parents.shape[1]))
	mid = int((parents.shape[1]+1) / 2)
	for i in range(offsprings_num):
		offsprings[i, :mid] = parents[i, :mid]
		offsprings[i, mid:] = parents[(i+1)%parents.shape[0], mid:]
	return offsprings

def mutation(offsprings, mutation_prob, target_num):
	for i in range(offsprings.shape[0]):
		for j in range (offsprings.shape[1]):
			rand = random.uniform(0, 1)
			if(rand<=mutation_prob):
				offsprings[i][j] = random.randrange(1, target_num+1)
	return offsprings

def replacement(parents, mutants, population_size, weapon_num):
	new_population = numpy.zeros((population_size, weapon_num))
	new_population[:parents.shape[0]] = parents
	new_population[parents.shape[0]:] = mutants
	return new_population

def chromosome_encode(chromosome, target_num):	
	chromosome_2d = chromosome_encode_2D(chromosome, target_num)
	chromosome_1d = chromosome_2d.flatten()		#flatten() convert 2D array to 1D
	return chromosome_1d