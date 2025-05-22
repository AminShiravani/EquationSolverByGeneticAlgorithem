import numpy as np
import random
from sympy import symbols, Eq, sympify, lambdify


def initialize_population():
    return np.random.uniform(VAL_RANGE_MIN, VAL_RANGE_MAX, size=(POP_SIZE, 3))


def parse_input_equation(equation_text):
    lhs, rhs = equation_text.split("=")
    return Eq(sympify(lhs.strip()), float(rhs.strip()))


def generate_objective_functions(equations):
    return [lambdify((x, y, z), eq.lhs - eq.rhs, "numpy") for eq in equations]


def crossover(p1, p2):
    w = random.random()
    return w * p1 + (1 - w) * p2


def apply_mutation(vector):
    if random.random() > MUTATION_PROB:
        return vector
    offset = np.random.normal(0, MUTATION_STDDEV, size=3)
    return np.clip(vector + offset, VAL_RANGE_MIN, VAL_RANGE_MAX)


def compute_loss(individual, funcs):
    try:
        return sum(abs(f(*individual)) for f in funcs)
    except:
        return float("inf")


def select_candidate(pop, scores):
    group = random.sample(range(PARENT_POOL), 20)
    return pop[min(group, key=lambda idx: scores[idx])]


x, y, z = symbols("x y z")

MAX_GENERATIONS = 10000
POP_SIZE = 5000
MUTATION_PROB = 0.4
MUTATION_STDDEV = 10000
ELITE_COUNT = 1000
PARENT_POOL = 1000
EQ_COUNT = 3
STALL_THRESHOLD = 30
VAL_RANGE_MAX = 1e7
VAL_RANGE_MIN = -1e7

user_input_equations = []
for i in range(EQ_COUNT):
    eq = input(f"Equation {i+1} : ")
    user_input_equations.append(eq)

parsed = [parse_input_equation(e) for e in user_input_equations]
functions = generate_objective_functions(parsed)

population = initialize_population()
best_score = float("inf")
last_best = None
stagnant_rounds = 0

for gen in range(MAX_GENERATIONS):
    losses = np.array([compute_loss(ind, functions) for ind in population])
    sort_idx = np.argsort(losses)
    population = population[sort_idx]
    losses = losses[sort_idx]

    best_individual = population[0]
    current_loss = losses[0]

    if current_loss <= 1e-7:
        print("Converged to a precise solution!")
        break

    if last_best is not None and np.allclose(
        best_individual, last_best, rtol=1e-5, atol=1e-5
    ):
        stagnant_rounds += 1
    else:
        stagnant_rounds = 0

    if stagnant_rounds >= STALL_THRESHOLD:
        print(
            f"No improvement for {STALL_THRESHOLD} generations. Ending early at Gen {gen+1}."
        )
        break

    last_best = best_individual.copy()
    new_generation = population[:ELITE_COUNT].tolist()

    while len(new_generation) < POP_SIZE:
        p1 = select_candidate(population, losses)
        p2 = select_candidate(population, losses)
        child = crossover(p1, p2)
        mutated = apply_mutation(child)
        new_generation.append(mutated)

    population = np.array(new_generation)

    print(
        f"Gen {gen+1}: x={best_individual[0]:.4f}, y={best_individual[1]:.4f}, z={best_individual[2]:.4f}, Error={current_loss:.10f}"
    )

final_answer = population[0]
final_error = compute_loss(final_answer, functions)

print("\nSolution summary:")
print(f"x = {final_answer[0]:.6f}")
print(f"y = {final_answer[1]:.6f}")
print(f"z = {final_answer[2]:.6f}")
print(f"Final error (L1 norm): {final_error:.10f}")
