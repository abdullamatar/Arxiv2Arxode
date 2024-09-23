import os

# from autogen import config_list_from_json
# from autogen import config_list_from_models
# TODO: Conform to their way of cfg.......

config_list = [
    {
        "model": "gpt-4o",
        "api_key": os.environ.get("OPENAI_APIKEY2"),
        # "temperature": 1.0,
    },
    # {
    #     "model": "gpt-4-turbo",
    #     "api_key": os.environ.get("OPENAI_APIKEY2"),
    #     # "temperature": 1.0,
    # },
    # {
    #     "model": "gpt-4-1106-preview",
    #     "api_key": os.environ.get("OPENAI_APIKEY2"),
    # },
]

claude_config_list = [
    {
        "model": "claude-3-5-sonnet-20240620",
        "api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "api_type": "anthropic",
        "temperature": 1.0,
    },
    {
        "model": "claude-3-sonnet-20240229",
        "api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "api_type": "anthropic",
        "temperature": 1.0,
    },
    {
        "model": "claude-3-opus-20240229",
        "api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "api_type": "anthropic",
    },
]

base_cfg = {
    # "use_cache": False,
    # "seed": 22,
    "config_list": config_list,
    "temperature": 0.7,
}

base_claude_cfg = {"config_list": claude_config_list}


gcconf = {
    "config_list": config_list,
    # "cache_seed": 45,  # quarter
    # "cache_seed": 48,  # full
    "cache_seed": 78,  # CLAUDE
}

retrieve_conf = {
    "config_list": [
        {
            "model": "gpt-4-turbo",
            "api_key": os.environ.get("OPENAI_APIKEY2"),
        }
    ],
    "functions": [
        {
            "name": "retrieve_content",
            "description": "retrieve content for code generation and question answering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Detailed, yet concise message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                    }
                },
                "required": ["message"],
            },
        },
    ],
    "customized_prompt": """
You are an agent tasked with retrieving information from research papers to aid in the task of code generation. Given the following problem do your best to retrieve the most relevant piece of information required for building the program given the current state of the conversation. Here is the problem:
{input_question}

The context is:
{input_context}
""",
}


claude_retr_conf = {
    **retrieve_conf,
    "config_list": claude_config_list,  # overwrite prev config_list ^
}
