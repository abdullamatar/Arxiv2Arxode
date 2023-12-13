import os


def ftos(fd):
    with open(fd, "r") as f:
        return f.read()


def create_llm_config(model, seed, temperature=1.0) -> dict:
    """
    args: model, temperature, seed
    --------
    """
    # Models to fall back to if token limit is reached
    config_list = [
        {
            "model": model,
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

    llm_config = {
        "seed": int(seed),
        "config_list": config_list,
        "temperature": float(temperature),
    }

    return llm_config
