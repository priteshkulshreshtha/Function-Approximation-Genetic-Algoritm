import numpy as np


def function(x, y, z):
    return 93 * x + (z / 2) + y ** 5 + 50


class GeneticAlgorithm:

    def __init__(self, pop_size, n_generations: int, mutation_chance, mutation_strength) -> None:
        self.mutation_chance = mutation_chance
        self.mutation_strength = mutation_strength
        self.pop_size = pop_size
        self.n_generations = n_generations

        self.population = self.generate_population()

    def generate_population(self):
        return [(np.random.uniform(-1, 1, 3)) for _ in range(self.pop_size)]

    @staticmethod
    def fitness(x, y, z):
        ans = function(x, y, z) - 25

        if ans == 0:
            return 9999
        else:
            return abs(1 / ans)

    def mutate(self, member):
        member[0] += np.random.uniform(1 - self.mutation_strength, 1 + self.mutation_strength)
        member[1] += np.random.uniform(1 - self.mutation_strength, 1 + self.mutation_strength)
        member[2] += np.random.uniform(1 - self.mutation_strength, 1 + self.mutation_strength)
        return member

    def reproduce(self, population):
        new_population = []
        traits = np.array(population).flatten()
        for _ in range(self.pop_size):
            if np.random.uniform() < self.mutation_chance:
                new_population.append(self.mutate(np.random.choice(traits, 3)))
            else:
                new_population.append(np.random.choice(traits, 3))
        self.population = new_population

    def evolve(self):
        top_half_index = []

        for generation in range(self.n_generations):
            scores = np.zeros(self.pop_size)
            for i, member in enumerate(self.population):
                scores[i] = self.fitness(member[0], member[1], member[2])

            sorted_scores_idx = np.argsort(scores)[::-1]
            top_half_index = sorted_scores_idx[:self.pop_size // 2]
            top_half_pop = [self.population[idx] for idx in top_half_index]

            if generation % 10 == 0:
                print(f"===== Generation {generation} =====")
                print(scores[top_half_index[0]])

            if scores[top_half_index[0]] > 100:
                print(f'Early Stopping at Generation {generation}, score: {scores[top_half_index[0]]}')
                return self.population[top_half_index[0]]

            self.reproduce(top_half_pop)

        print(f"Evolution Reached")

        return self.population[top_half_index[0]]


g = GeneticAlgorithm(1000, 50, 0.1, 0.05)
n1, n2, n3 = g.evolve()
print(function(n1, n2, n3))
