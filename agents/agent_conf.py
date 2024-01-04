import os

# from autogen import config_list_from_models

config_list = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": os.environ.get("OPENAI_APIKEY"),
    },
    {
        "model": "gpt-3.5-turbo",
        "api_key": os.environ.get("OPENAI_APIKEY"),
    },
    {
        "model": "gpt-3.5-turbo-16k",
        "api_key": os.environ.get("OPENAI_APIKEY"),
    },
]

base_cfg = {
    # "use_cache": False,
    # "seed": 22,
    "config_list": config_list,
    "temperature": 1.0,
}

retrieve_conf = {
    **base_cfg,
    "functions": [
        {
            "name": "retrieve_content",
            "description": "retrieve content for code generation and question answering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                    }
                },
                "required": ["message"],
            },
        },
    ],
    "timeout": 60,
    "seed": 42,
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
