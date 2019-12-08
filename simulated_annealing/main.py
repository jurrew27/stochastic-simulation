import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


class SimulatedAnnealing:
    def __init__(self, cities, chain_length, trials, control_start, cooling_param, cooling_method):
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

    def ratio_accepted(self):
        # Needs to check (n!)^2 pairs, quite impossible, need to think of something smarter
        pass

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


if __name__ == '__main__':
    cities = import_configuration('TSP-Configurations/eil51.tsp.txt')
    sa = SimulatedAnnealing(cities, chain_length=100, trials=100, control_start=5, cooling_param=0.95)
    tours, costs, controls = sa.run_multiple(5)
    print(tours)
    print(costs)
