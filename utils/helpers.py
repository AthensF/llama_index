import os


def set_openai_key(key):
    if key:
        os.environ["OPENAI_API_KEY"] = key
    else:
        os.environ["OPENAI_API_KEY"] = "sk-..."  # default fallback key
