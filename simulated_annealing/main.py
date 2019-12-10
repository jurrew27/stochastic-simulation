import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from itertools import permutations


class SimulatedAnnealing:
    def __init__(self, cities, chain_length, trials, control_start, cooling_param, cooling_method='linear'):
        self.cities = cities
        self.control_start = control_start
        self.chain_length = chain_length
        self.trials = trials
        self.cooling_param = cooling_param

        if cooling_method == 'continuous':
            self.calculate_new_control = self.continuous_cooling
        elif cooling_method == 'stationary':
            self.calculate_new_control = self.stationary_cooling
        else:
            self.calculate_new_control = self.linear_cooling

    def ratio_accepted(self, samples):
        samples = min(samples, np.math.factorial(len(cities)))
        tours = np.zeros((samples, len(self.cities)), dtype=np.int32)
        perms = set()
        for i in range(samples):
            while True:
                perm = np.random.permutation(range(1, len(self.cities)))
                key = tuple(perm)
                if key not in perms:
                    perms.update(key)
                    break

            tours[i, 1:] = perm

        cost_tours = np.array([self.length_tour(tour) for tour in tours])

        sum_probabilities = 0
        for i in range(samples):
            for j in range(samples):
                if i == j:
                    continue
                sum_probabilities += min(1, np.exp(-(cost_tours[i]-cost_tours[j]) / self.control_start))
        return sum_probabilities / (samples * (samples - 1))

    def linear_cooling(self, all_tours, all_costs, all_controls):
        if len(all_tours) % self.chain_length == 0:
            return all_controls[-1] * self.cooling_param
        else:
            return all_controls[-1]

    def continuous_cooling(self, all_tours, all_costs, all_controls):
        return all_controls[-1] * self.cooling_param

    def stationary_cooling(self, all_tours, all_costs, all_controls):
        if len(all_tours) % self.chain_length == 0:
            std = np.std(all_costs[-self.chain_length:])
            return all_controls[-1] * (1 + (np.log(1 + self.cooling_param) * all_controls[-1] / 3 * std))**-1
        else:
            return all_controls[-1]

    def length_tour(self, tour):
        route = self.cities[tour]
        shifted_route = np.concatenate((route[1:], route[:1]))
        return np.sum(np.sqrt(np.sum(np.square(route - shifted_route), axis=1)))

    def mutate_tour(self, tour):
        tour = tour.copy()
        a = 0
        b = 0
        while a == b:
            a, b = np.random.randint(1, len(tour), 2)
        lower = min(a, b)
        upper = max(a, b) + 1
        tour[lower:upper] = tour[lower:upper][::-1]
        return tour

    def do_markov_step(self, all_tours, all_costs, all_controls):
        tour = all_tours[-1]
        cost = all_costs[-1]
        new_tour = self.mutate_tour(tour)
        new_cost = self.length_tour(new_tour)

        probability_accept = min(1, np.exp(-(new_cost-cost) / all_controls[-1]))
        if np.random.uniform() <= probability_accept:
            tour = new_tour
            cost = new_cost

        return tour, cost

    def run(self):
        tour = np.concatenate(([0], np.random.permutation(range(1, len(self.cities)))))

        all_tours = [tour]
        all_costs = [self.length_tour(tour)]
        all_controls = [self.control_start]

        for step in range(self.trials * self.chain_length - 1):
            tour, cost = self.do_markov_step(all_tours, all_costs, all_controls)
            control = self.calculate_new_control(all_tours, all_costs, all_controls)

            all_tours.append(tour)
            all_costs.append(cost)
            all_controls.append(control)

        return np.array(all_tours), np.array(all_costs), np.array(all_controls)

    def run_multiple(self, runs):
        results_per_run = mp.Pool().starmap(self.run, [() for _ in range(runs)])
        results = [np.array(t) for t in zip(*results_per_run)]
        return results[0], results[1], results[2]


def import_configuration(tsp_filename):
    with open(tsp_filename) as tsp_file:
        for _ in range(3):
            next(tsp_file)

        dimensions = int(tsp_file.readline().split()[2])
        cities = np.zeros((dimensions, 2))

        for _ in range(2):
            next(tsp_file)

        for i, line in enumerate(tsp_file):
            if line.strip() == 'EOF':
                break
            cities[i] = line.split()[1:]
    return cities


def plot_cost(costs):
    costs_mean = np.mean(costs, axis=0)
    costs_std = np.std(costs, axis=0)

    plt.figure()
    plt.plot(costs_mean)
    plt.fill_between(range(len(costs_mean)), costs_mean - costs_std, costs_mean + costs_std, alpha=0.3)
    plt.show()


def plot_cost_control(costs, controls):
    costs_mean = np.mean(costs, axis=0)
    costs_std = np.std(costs, axis=0)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Markov Steps')
    ax1.set_ylabel('Tour distance', color='tab:blue')
    ax1.plot(costs_mean, color='tab:blue')
    ax1.fill_between(range(len(costs_mean)), costs_mean - costs_std, costs_mean + costs_std, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Control', color='tab:red')
    ax2.plot(np.mean(controls,axis=0), color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.show()


def plot_frequencies(tours, controls):
    if tours.shape[2] > 10:
        print('Too many states to count')
        return

    all_perms = np.array(list((permutations(range(1, tours.shape[2])))))
    all_perms = np.concatenate((np.zeros((len(all_perms), 1), dtype=np.int32), all_perms), axis=1)
    all_perms = all_perms[all_perms[:, 1].argsort()]  # might not be necessary ?

    states_table = {}
    for i, perm in enumerate(all_perms):
        states_table[perm.tostring()] = i

    control_values, control_counts = np.unique(controls[0], return_counts=True)
    chain_length = control_counts[0]
    for i, control in enumerate(control_values[::-1]):
        lower_bound = i * chain_length
        upper_bound = (i + 1) * chain_length
        tours_control = tours[:, lower_bound:upper_bound, :].reshape(-1, tours.shape[-1])

        frequencies = [states_table[state.tostring()] for state in tours_control]
        plt.figure()
        plt.hist(frequencies, bins=len(all_perms), range=(0, len(all_perms)))
        plt.xlabel('Tour number')
        plt.ylabel('Frequency')
        plt.title(f'Control: {control}')
        plt.show()


if __name__ == '__main__':
    cities = import_configuration('TSP-Configurations/eil51.tsp.txt')
    sa = SimulatedAnnealing(cities, chain_length=5000, trials=10, control_start=75, cooling_param=0.66)
    # print(np.mean(mp.Pool().map(sa.ratio_accepted, [1000]*5)))
    tours, costs, controls = sa.run_multiple(100)
    plot_cost_control(costs, controls)


