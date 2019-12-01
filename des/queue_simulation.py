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


def plot_distribution(wait_times_c_1, wait_times_c_2, wait_times_c_4, range=()):
    if not range:
        min_wait_time = np.min([np.min(wait_times_c_1), np.min(wait_times_c_2), np.min(wait_times_c_4)])
        max_wait_time = np.max([np.max(wait_times_c_1), np.max(wait_times_c_2), np.max(wait_times_c_4)])
        range = (min_wait_time, max_wait_time)

    plt.figure()
    weights = np.ones_like(wait_times_c_1) / float(len(wait_times_c_1))
    plt.hist(wait_times_c_1, bins=100, range=range, weights=weights, density=False, alpha=0.5, label='#Servers = 1')
    weights = np.ones_like(wait_times_c_2) / float(len(wait_times_c_2))
    plt.hist(wait_times_c_2, bins=100, range=range, weights=weights, density=False, alpha=0.5, label='#Servers = 2')
    weights = np.ones_like(wait_times_c_4) / float(len(wait_times_c_4))
    plt.hist(wait_times_c_4, bins=100, range=range, weights=weights, density=False, alpha=0.5, label='#Servers = 4')
    plt.legend()
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

    print('Shapiro-Wilk, H0: x=normal')
    print(scs.shapiro(x))


def test_difference(x, y):
    print('Mann-Whitney U, H0: same distribution, H1: x > y')
    print(scs.mannwhitneyu(x, y, alternative='greater', use_continuity=True))

    print('Mann-Whitney U, H0: same distribution, H1: x =/= y')
    print(scs.mannwhitneyu(x, y, alternative='two-sided', use_continuity=True))


def plot_rho_measurements_dependence(s_measurements_x, m_measurements_x, l_measurements_x,
                                     s_measurements_y, m_measurements_y, l_measurements_y):
    s_significance = np.zeros(len(s_measurements_x))
    m_significance = np.zeros(len(m_measurements_x))
    l_significance = np.zeros(len(l_measurements_x))
    for i in range(len(s_measurements_x)):
        s_significance[i] = scs.mannwhitneyu(s_measurements_x[i], s_measurements_y[i], alternative='greater')[0]
        m_significance[i] = scs.mannwhitneyu(m_measurements_x[i], m_measurements_y[i], alternative='greater')[0]
        l_significance[i] = scs.mannwhitneyu(l_measurements_x[i], l_measurements_y[i], alternative='greater')[0]

    plt.figure()
    plt.plot(np.arange(0.1, 1, 0.1), s_significance, label='#Measurements=10')
    plt.plot(np.arange(0.1, 1, 0.1), m_significance, label='#Measurements=100')
    plt.plot(np.arange(0.1, 1, 0.1), l_significance, label='#Measurements=1000')
    plt.legend()
    plt.xlabel('Rho')
    plt.ylabel('Test statistic')
    plt.show()


def plot_boxplots(fifo, sjf, deg, hyp):
    plt.figure()
    plt.boxplot(fifo, positions=[0], sym='')
    plt.boxplot(sjf, positions=[1], sym='')
    plt.boxplot(deg, positions=[2], sym='')
    plt.boxplot(hyp, positions=[3], sym='')
    plt.xticks([0, 1, 2, 3], ['FIFO', 'Shortest job first', 'Constant mu', 'Hyperexponential mu'])
    plt.ylabel('Average wait time')
    plt.show()


def setBoxColors(bp, color):
    for offset in [0, 1, 2]:
        plt.setp(bp['boxes'][0+offset], color=color)
        plt.setp(bp['caps'][0+offset*2], color=color)
        plt.setp(bp['caps'][1+offset*2], color=color)
        plt.setp(bp['whiskers'][0+offset*2], color=color)
        plt.setp(bp['whiskers'][1+offset*2], color=color)
        plt.setp(bp['medians'][0+offset], color=color)


def plot_all_boxplots(fifo, sjf, deg, hyp):
    plt.figure()

    bp = plt.boxplot(fifo, positions=[0, 3, 6], sym='', widths=0.75)
    setBoxColors(bp, 'blue')
    bp = plt.boxplot(sjf, positions=[1, 4, 7], sym='', widths=0.75)
    setBoxColors(bp, 'green')
    ax = plt.gca()
    ax.set_xticklabels('')
    ax.set_xticks([0.5, 3.5, 6.5], minor=True)
    ax.set_xticklabels([1, 2, 4,], minor=True)
    plt.ylabel('Average wait time')
    plt.xlabel('Number of servers')
    h1, = plt.plot([1, 1],'b-')
    h2, = plt.plot([1, 1],'g-')
    plt.legend((h1, h2),('FIFO, exp mu', 'Shortest job first, exp mu'))
    h1.set_visible(False)
    h2.set_visible(False)
    plt.show()

    plt.figure()
    bp = plt.boxplot(fifo, positions=[0, 3, 6], sym='', widths=0.7)
    setBoxColors(bp, 'blue')
    bp = plt.boxplot(deg, positions=[1, 4, 7], sym='', widths=0.7)
    setBoxColors(bp, 'red')
    ax = plt.gca()
    ax.set_xticklabels('')
    ax.set_xticks([0.5, 3.5, 6.5], minor=True)
    ax.set_xticklabels([1, 2, 4,], minor=True)
    plt.ylabel('Average wait time')
    plt.xlabel('Number of servers')
    h1, = plt.plot([1, 1],'b-')
    h2, = plt.plot([1, 1],'r-')
    plt.legend((h1, h2),('FIFO, exp mu', 'FIFO, const mu'))
    h1.set_visible(False)
    h2.set_visible(False)
    plt.show()

    plt.figure()
    bp = plt.boxplot(fifo, positions=[0, 3, 6], sym='', widths=0.7)
    setBoxColors(bp, 'blue')
    bp = plt.boxplot(hyp, positions=[1, 4, 7], sym='', widths=0.7)
    setBoxColors(bp, 'm')
    ax = plt.gca()
    ax.set_xticklabels('')
    ax.set_xticks([0.5, 3.5, 6.5], minor=True)
    ax.set_xticklabels([1, 2, 4,], minor=True)
    plt.ylabel('Average wait time')
    plt.xlabel('Number of servers')
    h1, = plt.plot([1, 1],'b-')
    h2, = plt.plot([1, 1],'m-')
    plt.legend((h1, h2),('FIFO, exp mu', 'FIFO, hyperexp mu'))
    h1.set_visible(False)
    h2.set_visible(False)
    plt.show()


if __name__ == '__main__':
    x = run_multiple_simulations(runs=1000, n_customers=10000, _lambda=0.9, mu=1, n_servers=1)
    test_normality(x)
    plot_distribution(x)