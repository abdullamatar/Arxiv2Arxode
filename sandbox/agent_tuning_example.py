
import random


# A very simple agent with a tunable parameter
class SimpleAgent:
    def __init__(self, parameter):
        self.parameter = parameter

    def perform_task(self):
        performance = random.random() * self.parameter  # Dummy performance metric
        return performance

# Tune the agent to maximize performance
best_parameter = None
best_performance = 0
search_space = [0.1 * i for i in range(1, 11)]  # Example search space for the parameter

for param in search_space:
    agent = SimpleAgent(parameter=param)
    performance = agent.perform_task()
    
    if performance > best_performance:
        best_performance = performance
        best_parameter = param

print(f"Best parameter found: {best_parameter}, with performance: {best_performance}")
