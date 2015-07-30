__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import matplotlib.pyplot as plt
from code.Algorithms import onePlusOneES, CMSA_ES, onePlusOneCholeskyCMAES, onePlusOneActiveCMAES

# Constant fitness function
def constantFitness(individual):
    individual.fitness = 0.5

# Random fitness function
def randomFitness(individual):
    individual.fitness = np.random.random(1)

# Sum fitness (minimize all parameters)
def sumFitness(individual):
    individual.fitness = np.sum(individual.dna)

# Sphere fitness function
def sphereFitness(individual):
    individual.fitness = np.sqrt(np.sum(np.square(individual.dna)))

# Rastrigin fitness function
def rastriginFitness(individual):
    individual.fitness = np.sum(np.square(individual.dna) + 10*np.cos(2*np.pi*individual.dna) + 10)


fitnes_functions = {'const': constantFitness, 'random': randomFitness, 'sum': sumFitness,
                    'sphere': sphereFitness, 'rastrigin': rastriginFitness, }
algorithms = {'1+1': onePlusOneES, 'CMSA': CMSA_ES, 'Cholesky': onePlusOneCholeskyCMAES,
              'Active': onePlusOneActiveCMAES}


def run_tests():

    # Set parameters
    n = 10
    budget = 1000
    num_runs = 3
    # fitnesses_to_test = ['const', 'random', 'sum', 'sphere', 'rastrigin']
    fitnesses_to_test = ['const', 'random', 'sphere', 'rastrigin']
    # fitnesses_to_test = ['const', 'sphere']

    # algorithms_to_test = ['1+1']
    # algorithms_to_test = ['CMSA']
    algorithms_to_test = ['1+1', 'CMSA', 'Cholesky', 'Active']

    # 'Catch' results
    results = {}
    sigmas = {}
    fitnesses = {}

    fig = plt.figure(figsize=(12, 8))
    num_rows = len(algorithms_to_test)  # One row per algorithm
    num_colums = 2  # Fitness and Sigma

    # Run algorithms
    for i, alg_name in enumerate(algorithms_to_test):

        algorithm = algorithms[alg_name]

        for fitness_name in fitnesses_to_test:

            results[fitness_name] = []
            sigmas[fitness_name] = None
            fitnesses[fitness_name] = None

            for _ in range(num_runs):
                results[fitness_name].append(algorithm(n, fitnes_functions[fitness_name], budget))

            # Preprocess/unpack results
            _, sigmas[fitness_name], fitnesses[fitness_name] = (list(x) for x in zip(*results[fitness_name]))
            sigmas[fitness_name] = np.mean(np.array(sigmas[fitness_name]), axis=0)
            fitnesses[fitness_name] = np.mean(np.array(fitnesses[fitness_name]), axis=0)


        # Plot results for this algorithm
        x_range = np.array(range(len(sigmas[fitnesses_to_test[0]])))
        nil_line = np.zeros(budget)

        sigma_plot = fig.add_subplot(num_rows, num_colums, num_colums*i + 1)
        sigma_plot.set_title('Sigma')
        fitness_plot = fig.add_subplot(num_rows, num_colums, num_colums*i + 2)
        fitness_plot.set_title('Fitness')

        for fitness_name in fitnesses_to_test:
            sigma_plot.plot(x_range, sigmas[fitness_name], label=fitness_name)
            fitness_plot.plot(x_range, fitnesses[fitness_name], label=fitness_name)

        sigma_plot.legend(loc=0, fontsize='small')
        sigma_plot.set_title("Sigma over time")
        sigma_plot.set_xlabel('Evaluations')
        sigma_plot.set_ylabel('Sigma')
        sigma_plot.set_yscale('log')

        fitness_plot.legend(loc=0, fontsize='small')
        fitness_plot.set_title("Fitness over time")
        fitness_plot.set_xlabel('Evaluations')
        fitness_plot.set_ylabel('Fitness value')
        fitness_plot.set_yscale('log')

    fig.tight_layout()
    fig.show()
    input("Done, press enter to close")  # Prevents the plot from closing prematurely




if __name__ == '__main__':
    run_tests()