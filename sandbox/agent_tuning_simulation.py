# filename: agent_tuning_simulation.py

import random

# Simulate a very basic agent interaction environment
class SimulatedAgentEnvironment:
    def __init__(self):
        self.state = "initial state"

    def apply_instruction(self, instruction):
        # Apply the instruction to the environment and return a reward
        self.state = f"state after {instruction}"
        return random.random()  # Random reward score for simplicity

# Simulate AgentInstruct dataset with simple instructions
agent_instructions = [
    "move forward",
    "turn left",
    "collect item",
    "avoid obstacle",
    "jump over",
    "find exit"
]

# Assume a minimal number of trajectories
num_trajectories = 5

# Simulated interaction trajectories
interaction_trajectories = []

# Simulated agent environment
environment = SimulatedAgentEnvironment()

for _ in range(num_trajectories):
    trajectory = []
    for instruction in agent_instructions:
        reward = environment.apply_instruction(instruction)
        trajectory.append((instruction, reward))
    
    # Filter trajectories with a reward threshold (for simplicity, we just take the trajectory as is)
    interaction_trajectories.append(trajectory)

# Save interaction trajectories to file
with open("interaction_trajectories.txt", "w") as file:
    for trajectory in interaction_trajectories:
        file.write(f"{trajectory}\n")

print("Interaction trajectories simulated and saved to 'interaction_trajectories.txt'")