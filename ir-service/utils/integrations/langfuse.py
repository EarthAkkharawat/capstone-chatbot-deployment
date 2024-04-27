import os

from langfuse.callback import CallbackHandler

def get_langfuse_callback_handler():
    try:
        handler = CallbackHandler(
            public_key=os.environ.get("ENV_PUBLIC_KEY"),
            secret_key=os.environ.get("ENV_SECRET_KEY"),
            host=os.environ.get("ENV_HOST"),
        )
    except: raise Exception("Please set ENV_PUBLIC_KEY, ENV_SECRET_KEY, ENV_HOST for Langfuse in .env file")
    return handler