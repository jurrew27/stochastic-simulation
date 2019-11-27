import random, simpy
from functools import partial, wraps
import numpy as np
import matplotlib as plt


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
    def __init__(self, arrival_rate, capacity, n_servers=1, shortest_job_first=False, debug=False):
        self.arrival_rate = arrival_rate
        self.capacity = capacity
        self.n_servers = n_servers
        self.debug = debug

        self.env = simpy.Environment()

        if shortest_job_first:
            self.customers = simpy.PriorityStore(self.env)
        else:
            self.customers = simpy.Store(self.env)

        self.wait_times = []

        patch_resource(self.customers, pre=self.monitor_wait_times)

    def create_customers(self, n_customers):
        for i in range(n_customers):
            job_time = random.expovariate(self.capacity)

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

            yield self.env.timeout(random.expovariate(self.arrival_rate))

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
        wait_time = 0
        if len(resource.items) > 0:
            customer = resource.items[0]

            if type(customer) is simpy.PriorityItem:
                customer = customer[1]

            wait_time = self.env.now - customer["arrival_time"]
        self.wait_times.append(wait_time)

    def run(self, n_customers):
        self.wait_times = []
        self.env.process(self.create_customers(n_customers))
        [self.env.process(self.server()) for _ in range(self.n_servers)]
        self.env.run()

        return np.mean(np.array(self.wait_times)), np.std(np.array(self.wait_times))


def run_multiple_simulations(runs=100):
    mean_wait_times = np.zeros(runs)
    std_wait_times = np.zeros(runs)
    for run in range(runs):
        sim = QueueSimulation(5, 2, 2)
        mean_wait_times[run], std_wait_times[run] = sim.run(n_customers=1000)

    return mean_wait_times, std_wait_times


if __name__ == '__main__':
    means, stds = run_multiple_simulations(10)
    print(f'Mean: {np.mean(means)}')
    print(f'Std: {np.mean(stds)}')
