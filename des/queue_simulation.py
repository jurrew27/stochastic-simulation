import random

import simpy


class QueueSimulation:
    def __init__(self, arrival_rate, capacity, n_servers=1):
        self.arrival_rate = arrival_rate
        self.capacity = capacity
        self.n_servers = n_servers

        self.env = simpy.Environment()
        self.customers = simpy.Store(self.env)

    def create_customers(self, n_customers):
        customer_number = 0

        for i in range(n_customers):
            customer = {
                'number': i,
                'arrival_time': self.env.now,
                'job_time': random.expovariate(self.capacity)
            }
            self.customers.put(customer)

            print(f'{round(self.env.now,2)}: Customer {customer_number} arrives')

            customer_number += 1
            yield self.env.timeout(random.expovariate(self.arrival_rate))

    def server(self):
        while True:
            customer = yield self.customers.get()

            service_time = self.env.now

            print(f'{round(self.env.now,2)}: Customer {customer["number"]} is serviced, wait time: {round(service_time - customer["arrival_time"],2)}')

            yield self.env.timeout(random.expovariate(customer["job_time"]))

            print(f'{round(self.env.now,2)}: Customer {customer["number"]} is finished, service time: {round(self.env.now - service_time,2)}')

    def run(self, n_customers):
        self.env.process(self.create_customers(n_customers))
        [self.env.process(self.server()) for _ in range(self.n_servers)]
        self.env.run()


if __name__ == '__main__':
    sim = QueueSimulation(5, 2, 2)
    sim.run(100)