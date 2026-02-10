import os
import time
import openai
from openai import OpenAI
import tiktoken


client = OpenAI(
    # This is the default and can be omitted
    api_key="EMPTY",
    base_url="http://172.16.49.123:8000/v1"
)

def res_is_valid(res):
    try:
        if res.choices[0].message.content:
            return True
        else:
            return False
    except:
        return False


def get_model():
    models = client.models.list()
    model = models.data[0].id
    return model


def call_api(
    msg, 
    model_str, 
    max_return_tokens=100,
    temperature=0.2
):
    res = None

    while True:
        try:
            res = client.chat.completions.create(
                model=model_str,
                messages=msg,
                max_tokens=max_return_tokens,
                temperature=temperature
            )
            # print(res)
        except openai.APIError as e:
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
        except openai.Timeout as e:
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
        except openai.RateLimitError as e:
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
        except openai.APIConnectionError as e: 
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
        except openai.InternalServerError as e:
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
            
        if not res_is_valid(res):
            print("Got an empty response, sleep for 3 seconds.")
            time.sleep(3)
        else:
            break

    return res.choices[0].message.content


def call_embedding_api(
    sentences,
    model_str,
):
    responses = None

    if len(sentences) == 0:
        return []

    while True:
        try:
            responses = client.embeddings.create(
                input=sentences,
                model=model_str,
            )

        except openai.APIError as e:
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
        except openai.Timeout as e:
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
        except openai.RateLimitError as e:
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
        except openai.APIConnectionError as e: 
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
        except openai.InternalServerError as e:
            time.sleep(5)
            print(f"{e}... Sleep for 5 seconds.")
            
        if not responses.data:
            print("Got an empty response, sleep for 3 seconds.")
            time.sleep(3)
        else:
            break

    return [x.embedding for x in responses.data]
