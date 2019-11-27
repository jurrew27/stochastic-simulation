import simpy
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, _lambda=1, mu=0.9, n_servers=1, degenerate_mu=True, shortest_job_first=False, debug=False):
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


if __name__ == '__main__':
    wait_times = run_multiple_simulations(runs=100, n_customers=1000, _lambda=0.9, mu=1, n_servers=1)
    print(f'Mean: {np.mean(wait_times)}')
    print(f'Std: {np.mean(np.std(wait_times, axis=1))}')
