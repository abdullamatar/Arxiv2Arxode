import os

# from autogen import config_list_from_json

# from autogen import config_list_from_models
# TODO: Conform to their way of cfg.......

config_list = [
    {
        "model": "gpt-4o",
        "api_key": os.environ.get("OPENAI_APIKEY2"),
    },
    {
        "model": "gpt-4-turbo",
        "api_key": os.environ.get("OPENAI_APIKEY2"),
    },
    # {
    #     "model": "gpt-4-1106-preview",
    #     "api_key": os.environ.get("OPENAI_APIKEY2"),
    # },
]

base_cfg = {
    # "use_cache": False,
    # "seed": 22,
    "config_list": config_list,
    "temperature": 1.0,
}

gcconf = {
    "config_list": config_list,
    # "config_list": [
    #     {
    #         "model": "gpt-3.5-turbo",
    #         "api_key": os.environ.get("OPENAI_APIKEY2"),
    #     }
    # ],
    "cache_seed": 44,  # quarter
    # "cache_seed": 48,  # full
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
    # "timeout": 180,
    # "max_retries": 5,
    # "seed": 42,
}

exec_py_conf = {
    **base_cfg,
    "functions": [
        {
            "name": "exec_py",
            "description": "Execute generated python code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python str to be executed.",
                    }
                },
                "required": ["code"],
            },
        }
    ],
}

write_file_config = {
    **base_cfg,
    "functions": [
        {
            "name": "write_file",
            "description": "Save accepted python code to file",
            "parameters": {
                "type": "object",
                "properties": {
                    "fname": {
                        "type": "string",
                        "description": "The name of the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the file to write",
                    },
                },
                "required": ["fname", "content"],
            },
        }
    ],
}
