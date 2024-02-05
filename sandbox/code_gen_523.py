import random

# Part 1: Dataset Creation
# Simulating a very basic form of the AgentInstruct dataset
agent_instruct = [
    (["What's the weather like today?"], ["It's sunny with a slight chance of rain."], 1.0),
    (["Set a timer for 10 minutes."], ["Timer set for 10 minutes."], 1.0),
    (["Remind me to call John tomorrow."], ["Reminder set to call John tomorrow."], 1.0)
]

general_domain_instructions = [
    (["Tell me a joke."], ["Why don't scientists trust atoms? Because they make up everything."]),
    (["What's the capital of France?"], ["The capital of France is Paris."])
]

# Part 2: Instruction-Tuning Strategy
# Simulating a simple mixing of specific AgentInstruct data with general-domain instructions
mixed_instructions = agent_instruct + general_domain_instructions

# Part 3: Model Simulation - Basic decision-making algorithm to simulate agent behavior
def simple_agent(user_input):
    # Prioritize agent-specific instructions
    for question, answer, _ in mixed_instructions:
        if user_input in question:
            return random.choice(answer)
    # If no specific instruction is found, respond with a general statement
    return "I'm not sure, but I can find out!"

# Part 4: Evaluation - Simple Task to evaluate agent on unseen task
new_task = "What's the weather like today?"
print("Evaluating new task:", new_task)
print("Agent's response:", simple_agent(new_task))

# This code is a highly simplified and abstract conceptual demonstration of the AgentTuning approach.
# It uses a simulated dataset and a basic model simulation instead of fine-tuning actual LLMs.