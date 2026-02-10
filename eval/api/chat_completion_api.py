import os
import time
import openai
from openai import OpenAI
import tiktoken


client = OpenAI(
    # This is the default and can be omitted
    api_key="OPENAI_API_KEY",
)

def res_is_valid(res):
    try:
        if res.choices[0].message.content:
            return True
        else:
            return False
    except:
        return False


def num_tokens_from_string(string: str, encoding_name: str = 'gpt=3.5-turbo') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


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
    
    print(res)

    return res.choices[0].message.content


