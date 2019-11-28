import simpy
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm

def patch_resource(resource, pre=None, post=None):
    """Patch *resource* so that it calls the callable *pre* before each
    put/get/request/release operation and the callable *post* after each
    operation.  The only argument to these functions is the resource
    instance.
    """
    def get_wrapper(func):
        # Generate a wrapper for put/get/request/release
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is the actual wrapper
            # Call "pre" callback
            if pre:
                pre(resource)

            # Perform actual operation
            ret = func(*args, **kwargs)

            # Call "post" callback
            if post:
                post(resource)

            return ret
        return wrapper

    # Replace the original operations with our wrapper
    for name in ['get']:
        if hasattr(resource, name):
            setattr(resource, name, get_wrapper(getattr(resource, name)))


class QueueSimulation:
    def __init__(self, _lambda=1, mu=0.9, n_servers=1, degenerate_mu=False, shortest_job_first=False, debug=False):
        self._lambda = _lambda
        self.mu = mu
        self.degenerate_mu = degenerate_mu
        self.n_servers = n_servers
        self.debug = debug

        self.env = simpy.Environment()

        if shortest_job_first:
            self.customers = simpy.PriorityStore(self.env)
        else:
            self.customers = simpy.Store(self.env)

        patch_resource(self.customers, pre=self.monitor_wait_times)

    def create_customers(self, n_customers):
        for i in range(n_customers):
            if self.degenerate_mu:
                job_time = self.mu
            else:
                if type(self.mu) is tuple:
                    job_time = np.random.exponential(1 / self.mu[0 if np.random.uniform() < 0.75 else 1])
                else:
                    job_time = np.random.exponential(1 / self.mu)

            customer = {
                'number': i,
                'arrival_time': self.env.now,
                'job_time': job_time
            }

            if type(self.customers) is simpy.PriorityStore:
                customer = simpy.PriorityItem(job_time, customer)

            self.customers.put(customer)

            if self.debug:
                print(f'{round(self.env.now,2)}: Customer {i} arrives')

            yield self.env.timeout(np.random.exponential(1 / self._lambda))

    def server(self):
        while True:
            customer = yield self.customers.get()
            if type(customer) is simpy.PriorityItem:
                customer = customer[1]

            service_time = self.env.now

            if self.debug:
                print(f'{round(self.env.now,2)}: Customer {customer["number"]} is serviced, wait time: {round(service_time - customer["arrival_time"],2)}')

            yield self.env.timeout(customer["job_time"])

            if self.debug:
                print(f'{round(self.env.now,2)}: Customer {customer["number"]} is finished, service time: {round(self.env.now - service_time,2)}')

    def monitor_wait_times(self, resource):
        if len(resource.items) == 0:
            return

        customer = resource.items[0]
        if type(customer) is simpy.PriorityItem:
            customer = customer[1]

        self.wait_times[customer["number"]] = self.env.now - customer["arrival_time"]

    def run(self, n_customers):
        self.wait_times = np.zeros(n_customers)
        self.env.process(self.create_customers(n_customers))
        [self.env.process(self.server()) for _ in range(self.n_servers)]
        self.env.run()

        return np.array(self.wait_times)


def run_multiple_simulations(runs, n_customers, *args, **kwargs):
    wait_times = np.zeros((runs,n_customers))
    for run in range(runs):
        wait_times[run] = QueueSimulation(*args, **kwargs).run(n_customers=n_customers)

    return wait_times

def plot_distribution(wait_times):
    plt.figure()
    plt.hist(wait_times.flatten(), 100, density=True)
    # plt.title('Histogram of wait times per customer')
    plt.xlabel('Seconds')
    plt.ylabel('Density')
    plt.show()

    plt.figure()
    plt.hist(np.mean(wait_times, axis=1), 100, density=True)
    # plt.title('Histogram of mean wait times')
    plt.xlabel('Seconds')
    plt.ylabel('Density')
    plt.show()

def test_normality(x):
    fig = sm.qqplot(x, line='s')
    fig.show()

    print('Kolmogorov-Smirnov, H0: x=normal')
    print(scs.kstest(x, 'norm'))

    print('Jarque-Bera, H0: x=normal')
    print(scs.jarque_bera(x))

def test_difference(x, y):
    print('Wilcoxon, H0: same distribution, H1: x > y')
    print(scs.wilcoxon(x, y, alternative='greater', correction=True))

    print('Mann-Whitney U, H0: same distribution, H1: x > y')
    print(scs.mannwhitneyu(x, y, alternative='greater', use_continuity=True))

if __name__ == '__main__':
    runs = 10
    n_customers = 10000

    for lambda_ in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        wait_times = run_multiple_simulations(runs=runs, n_customers=n_customers, _lambda=lambda_, mu=1, n_servers=1)
        np.savetxt(f'rho={lambda_},r={runs},n={n_customers},s=1,fifo.txt', wait_times)
        means = np.mean(wait_times, axis=1)
        mean = np.mean(means)
        std = np.std(means)
        print(f'Mean: {mean}')
        print(f'Std: {std}')

        wait_times = run_multiple_simulations(runs=runs, n_customers=n_customers, _lambda=lambda_ * 2, mu=1, n_servers=2)
        np.savetxt(f'rho={lambda_}r={runs},n={n_customers},s=2,fifo.txt', wait_times)
        means = np.mean(wait_times, axis=1)
        mean = np.mean(means)
        std = np.std(means)
        print(f'Mean: {mean}')
        print(f'Std: {std}')

        wait_times = run_multiple_simulations(runs=runs, n_customers=n_customers, _lambda=lambda_ * 4, mu=1, n_servers=4)
        np.savetxt(f'rho={lambda_}r={runs},n={n_customers},s=4,fifo.txt', wait_times)
        means = np.mean(wait_times, axis=1)
        mean = np.mean(means)
        std = np.std(means)
        print(f'Mean: {mean}')
        print(f'Std: {std}')

        print('-------------------')