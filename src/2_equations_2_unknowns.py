import random
import math
import time


def equations(x, y, a, b, c, d, q1, q2):
    eq1 = a * x + b * y
    eq2 = c * x + d * y
    return eq1, eq2


def fitness(x, y, a, b, c, d, q1, q2):
    eq1, eq2 = equations(x, y, a, b, c, d, q1, q2)
    return abs(eq1 - q1) + abs(eq2 - q2)


def crossover(p1, p2):
    p = random.random()
    x = p * p1[0] + (1 - p) * p2[0]
    y = p * p1[1] + (1 - p) * p2[1]
    return (x, y)


def mutate(individual, mutation_rate=0.1):
    x, y = individual
    if random.random() < mutation_rate:
        x += random.uniform(-1, 1)
    if random.random() < mutation_rate:
        y += random.uniform(-1, 1)
    return (x, y)


import random
import time


def genetic_algorithm(
    a,
    b,
    c,
    d,
    q1,
    q2,
    pop_size=1000,
    generations=200,
    mutation_rate=0.1,
    fitness_threshold=10,
):
    population = []
    while len(population) < pop_size:
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        f = fitness(x, y, a, b, c, d, q1, q2)
        if f < fitness_threshold:
            population.append((x, y))
    for generation in range(generations):
        scored = [
            (ind, fitness(ind[0], ind[1], a, b, c, d, q1, q2)) for ind in population
        ]
        scored.sort(key=lambda x: x[1])
        best = scored[0]
        print(
            f"Gen {generation+1}: best fitness = {best[1]:.6f}, x = {best[0][0]:.3f}, y = {best[0][1]:.3f}"
        )
        time.sleep(0.2)
        if best[1] < 1e-6:
            break
        new_population = [ind for ind, _ in scored[: int(0.1 * pop_size)]]
        while len(new_population) < pop_size:
            p1 = random.choice(scored[: pop_size // 2])[0]
            p2 = random.choice(scored[: pop_size // 2])[0]
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            if fitness(child[0], child[1], a, b, c, d, q1, q2) < fitness_threshold:
                new_population.append(child)
        population = new_population
    final_scored = [
        (ind, fitness(ind[0], ind[1], a, b, c, d, q1, q2)) for ind in population
    ]
    final_scored.sort(key=lambda x: x[1])
    print("\nTop solutions:")
    for i in range(min(5, len(final_scored))):
        x, y = final_scored[i][0]
        loss = final_scored[i][1]
        print(f"x = {x:.6f}, y = {y:.6f}, loss = {loss:.6f}")


if __name__ == "__main__":
    print("Enter a, b, q1:")
    a, b, q1 = map(float, input().split())
    print("Enter c, d, q2:")
    c, d, q2 = map(float, input().split())
    genetic_algorithm(a, b, c, d, q1, q2)
