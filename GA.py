from numba import njit
import pandas as pd
import numpy as np
from random import uniform, randint, randrange
from tqdm import tqdm


df = pd.read_csv(r"C:\Users\Corey\OneDrive - University of Toledo\SRN DSGN\py-data.csv")
preference_columns = []
for x in range(8):
    preference_columns.append(str(x))


preference_matrix = df[preference_columns].to_numpy()
required = df["Required"].to_numpy()
#required[:] = np.nan
projects = np.array(pd.unique(df[preference_columns].values.ravel()), dtype='float')
projects = np.sort(projects)
class_size = np.size(preference_matrix, 0)


@njit
def init_population(max_population, num_students, projects, req):
    population = np.zeros((max_population, num_students))
    for i in range(max_population):
        for j in range(num_students):
            if np.isnan(req[j]): 
                population[i][j] = np.random.choice(projects)
            else:
                population[i][j] = req[j]
    return population


@njit
def cost(population, pref_matrix):
    max_population = np.shape(population)[0]
    num_students = np.shape(pref_matrix)[0]
    num_of_choices = np.shape(pref_matrix)[1]
    t_cost = np.zeros(max_population)
    for i in range(max_population):
        solution = population[i].copy()
        cost = 0
        for j in range(num_students):
            choice = np.where(preference_matrix[j] == solution[j])[0]
            # Some students had single projects selected multiple times or not at all
            if len(choice) == 0:
                choice = num_of_choices
            elif len(choice) > 1:
                choice = np.amin(choice)
            else:
                choice = choice.item()
            cost += choice
        t_cost[i] = cost
    return t_cost


def select_parents(population, fitness):
    rng = np.random.default_rng()
    parents = rng.choice(
        np.shape(population)[0],
        size=np.shape(population)[0],
        p=fitness,
        axis=0
    )
    return parents


@njit
def cross_over(population, parents, num_of_mutations, projects, req):
    num_of_students = np.shape(population)[1]
    num_of_solutions = np.shape(population)[0]
    for i in range(0,num_of_solutions,2):
        x = randint(0, num_of_students - 1)
        p1 = population[parents[i]].copy()
        p2 = population[parents[i+1]].copy()
        tmp = p2[:x].copy()
        p2[:x], p1[:x] = p1[:x], tmp
        population[parents[i]] = p1.copy()
        population[parents[i+1]] = p2.copy()
    for i in range(num_of_solutions):
        for _ in range(num_of_mutations):
            index = randrange(num_of_students)
            if np.isnan(req[index]):
                population[i][index] = np.random.choice(projects)
    return population


new_pop = init_population(100, class_size, projects, required)
itermax = 10000
F_min_arr = np.zeros(itermax)
F_avg_arr = np.zeros(itermax)
starting_cost = cost(new_pop, preference_matrix).min()

for _ in tqdm(range(itermax), ncols=100):
    t_fit = cost(new_pop, preference_matrix)
    F_min_arr[_] = t_fit.min()
    F_avg_arr[_] = t_fit.mean()
    least_fit = t_fit.max()
    for i in range(np.shape(new_pop)[0]):
        t_fit[i] = least_fit - t_fit[i]
    tot = np.sum(t_fit)
    t_fit = t_fit / tot

    parents = select_parents(new_pop, t_fit)
    new_pop = cross_over(new_pop, parents, 7, projects, required)
print(" - - - - Genetic Algorithm - - - - ")
print("Starting cost: ", starting_cost)
last_cost = cost(new_pop, preference_matrix)
print("Ending cost: ", last_cost.min())

spare_seats = 0
best_sol = new_pop[last_cost.argmin()]
instance_arr = []
for proj in projects:
    instances = np.sum(best_sol == proj)
    instance_arr.append(instances)
    alpha = (instances%4)%3
    spare_seats += alpha
print("Spare_seats: ", spare_seats)
print("Largest team: ", max(instance_arr), "Smallest Team: ", min(instance_arr))

data = {'min': F_min_arr,
        'avg': F_avg_arr
    }
df = pd.DataFrame(data, columns=["min", "avg"])
df.to_csv("ga.csv")
