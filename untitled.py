# RC: https://www.youtube.com/watch?v=PUPO2tTyPOo&t=2s
from autogen import GroupChat, GroupChatManager
from agents import EmbeddingRetrieverAgent
from autogen import UserProxyAgent, AssistantAgent
from utils.misc import create_llm_config

"""
From: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb
Call RetrieveUserProxyAgent while init chat with another user proxy agent
Sometimes, there might be a need to use RetrieveUserProxyAgent in group chat without initializing the chat with it. In such scenarios, it becomes essential to create a function that wraps the RAG agents and allows them to be called from other agents.
"""


SEED = 22
PROBLEM = "I want to understand the agent tuning paper and come out with a minimal implementation of some of the core ideas in the paper the code must be executable."
# TODO: creating llm conf isnt really misc...
lmconf = create_llm_config(model="gpt-4-1106-preview", temperature=1, seed=SEED)

termination_msg = (
    lambda x: isinstance(x, dict)
    and "TERMINATE" == str(x.get("content", ""))[-9:].upper()
)
# def is_termination_msg(termination_msg

#         return x.get("page_content", " ").rstrip().endswith("TERMINATE")
#     else:
#         return False

lmconf = {
    **lmconf,
    "cache_seed": SEED,
}

# usrproxagent can exe code, feedback provider to other agents.
# To modify the way to execute code blocks, single code block, or function call, override execute_code_blocks, run_code, and execute_function methods respectively. (https://microsoft.github.io/autogen/docs/reference/agentchat/user_proxy_agent)

agent0 = UserProxyAgent(
    name="coordinator",
    human_input_mode="NEVER",
    is_termination_msg=termination_msg,
    code_execution_config=False,
    system_message="Your role is to coordinate the completion of tasks related to generating code based off of machine learning and AI research. You must be diligent and operate in a step by step manner to pinpoint potentially implementable parts of the research in accordance with the over all task at hand. With the goals of either simplifying the ideas in a paper for better understanding, merging ideas, improving upon or otherwise manipulating them where you see fit.",
    default_auto_reply="Reply `TERMINATE` if the task is complete."
    # llm_config={
    #     **lmconf,
    # },
)

agent1 = EmbeddingRetrieverAgent(
    name="info_hoarder",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=4,
    system_message="You play a pivotal role in the progression of the task at hand as you have access to databases that store embeddings of research papers and their associated code if it exists. Your main job is to provide a detailed and step by step understanding of the relevant research paper(s) and pieces of code to the other agents. You should focus on the core ideas of the paper and the code base and how they relate to each other, with the end goal of helping in providing an implementation plan.",
    code_execution_config=False,
    is_termination_msg=termination_msg,
)

agent2 = AssistantAgent(
    name="code_designer",
    system_message="Your role is to design an interface of functions and/or classes that will be implemented based on the research paper(s) and code you are provided. You should focus on the implementation of the ideas in the research paper(s) and code base according to the task at hand, with the end goal of making it clear and simple for the coding agent to implement the interface. When you are done reply with 'TERMINATE'",
    llm_config=lmconf,
    is_termination_msg=termination_msg,
)

agent3 = AssistantAgent(
    name="machinelearning_engineer",
    system_message="Your role is to implement the interface designed by the code designer. You should focus on the implementation of the interface according to the task at hand, with the end goal of making it clear and simple for the coding agent to implement the interface. When you are done reply with 'TERMINATE'",
    is_termination_msg=termination_msg,
    llm_config=lmconf,
)


# rc: https://microsoft.github.io/autogen/blog/2023/10/18/RetrieveChat/


def _reset_agents():
    agent0.reset()
    agent1.reset()
    agent2.reset()
    agent3.reset()


def rag_chat():
    _reset_agents()
    groupchat = GroupChat(
        agents=[agent1, agent2, agent3],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=lmconf)

    # Start chatting with boss_aid as this is the user proxy agent.
    agent1.initiate_chat(
        manager,
        problem=PROBLEM,
        n_results=3,
    )


def call_rag_chat():
    _reset_agents()

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.
    def retrieve_content(message, n_results=4):
        agent1.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = agent1._check_update_context(
            message
        )
        if (update_context_case1 or update_context_case2) and agent1.update_context:
            agent1.problem = (
                message if not hasattr(agent1, "problem") else agent1.problem
            )
            _, ret_msg = agent1._generate_retrieve_user_reply(message)
        else:
            ret_msg = agent1.generate_init_message(message, n_results=n_results)
        return ret_msg if ret_msg else message

    agent1.human_input_mode = (
        "NEVER"  # Disable human input for boss_aid since it only retrieves content.
    )
    nested_conf = {
        "functions": [
            {
                "name": "retrieve_content",
                "description": "retrieve page_content for code generation and question answering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Refined message which keeps the original meaning and can be used to retrieve page_content for code generation and research summarization.",
                        }
                    },
                    "required": ["message"],
                },
            },
        ],
        "config_list": lmconf,
        "timeout": 60,
    }

    for a in [agent2, agent3]:
        a.llm_config.update(nested_conf)

    for agent in [agent0, agent2, agent3]:
        # register functions for all agents.
        agent.register_function(
            function_map={
                "retrieve_content": retrieve_content,
            }
        )

    groupchat = GroupChat(
        agents=[agent0, agent2, agent3],
        messages=[],
        max_round=13,
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=lmconf)
    # manager_conf = nested_conf.copy()
    # manager_conf.pop("functions")
    manager = GroupChatManager(groupchat=groupchat, llm_config=lmconf)
    agent0.initiate_chat(
        manager,
        message=PROBLEM,
    )


if __name__ == "__main__":
    call_rag_chat()
