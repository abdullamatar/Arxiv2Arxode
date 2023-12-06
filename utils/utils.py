import os


def ftos(fd):
    with open(fd, "r") as f:
        return f.read()


def create_llm_config(model, temperature, seed) -> dict:
    """
    args: model, temperature, seed
    --------
    """
    config_list = [
        {
            "model": model,
            "api_key": os.environ.get("OPENAI_APIKEY"),
        },
    ]

    llm_config = {
        "seed": int(seed),
        "config_list": config_list,
        "temperature": float(temperature),
    }

    return llm_config
