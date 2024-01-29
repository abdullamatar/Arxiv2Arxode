# filename: minimal_agent_tuning.py

class SimpleAgent:
    def __init__(self, task_descriptions):
        self.task_descriptions = task_descriptions

    def generate_rationale(self, task):
        # This is a placeholder for what would be the model's Chain-of-Thought (CoT) rationale
        rationale = "Considering the pros and cons, based on the given instructions."
        return rationale

    def decide_action(self, task):
        # This is a placeholder for the agent's decision-making process
        action = f"Performing the best action for {task}."
        return action

    def perform_task(self, instruction):
        task = self.task_descriptions.get(instruction, "Unknown Task")
        if task == "Unknown Task":
            return f"No rationale or action possible for {task}."

        rationale = self.generate_rationale(task)
        action = self.decide_action(task)
        return f"Task: {task}\nRationale: {rationale}\nAction: {action}"

# Sample tasks that the agent can perform
task_descriptions = {
    "task1": "Solve a math problem",
    "task2": "Write a poem",
    "task3": "Play a game of chess"
}

# Initiating the agent with sample tasks
agent = SimpleAgent(task_descriptions)

# Agent performing a task based on an instruction example
instruction_example = "task1"
result = agent.perform_task(instruction_example)

# This would typically be logged to a file, but for simplicity, we print it
print(result)

# Save this script and run it to simulate a very basic agent tuning example.