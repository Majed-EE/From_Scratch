import os 
import requests
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv, dotenv_values
import json
load_dotenv()
env_keys = dotenv_values().keys()
print("Loaded keys:", list(env_keys))
print(f"os.getenv('OPENAI_API_KEY'): {os.getenv('OPENAI_API_KEY')}")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



#---------------------------------------------------------
# Define the tools (Functions) you want to call
#---------------------------------------------------------



def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]["temperature_2m"]


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    raise ValueError(f"Unknown function: {name}")


def intelligence_with_tools(prompt: str) -> str:
    client = OpenAI()

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    input_messages = [{"role": "user", "content": prompt}]

    # Step 1: Call model with tools
    response = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        tools=tools,
    )

    # Step 2: Handle function calls
    for tool_call in response.output:
        if tool_call.type == "function_call":
            # Step 3: Execute function
            name = tool_call.name
            args = json.loads(tool_call.arguments)
            result = call_function(name, args)

            # Step 4: Append function call and result to messages
            input_messages.append(tool_call)
            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(result),
                }
            )

    # Step 5: Get final response with function results
    final_response = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        tools=tools,
    )

    return final_response.output_text


if __name__ == "__main__":
    result = intelligence_with_tools(prompt="What's the weather like in Paris today?")
    print("Tool Calling Output:")
    print(result)
# Step1: Define response format in a pydantic model
#---------------------------------------------------


# class CalenderEvent(BaseModel):
#     name: str
#     date: str
#     participatns: list[str]


# class CalenderEvent(BaseModel):
#     name: str
#     date: str
#     participatns: list[str]


# def call_function(name,args):
#     if name == "get_weather":
#         return get_weather(**args)
#     else:
#         raise ValueError(f"Function {name} not recognized.")

# for tool_call in completion.choices[0].message.tool_calls or []:
#     function_response = call_function(
#         name=tool_call.name,
#         args=tool_call.arguments,
#     )
#     print(f"Function response: {function_response}")



# # Step 2 call the model


# completion = client.chat.completions.parse (
#     model = "gpt-4o", 
#     messages = [
#         {"role": "system", "content": "you are a helpful weather assistent."},
#         {
#             "role": "user",
#             "content": "What is the weather like in Paris right now?" ,

#         },

#     ],

#     tools = tools# response_format= CalenderEvent,
# )

# # completion = client.chat.completions.parse (
# #     model = "gpt-4o", 
# #     messages = messages


# response= completion.model_dump()
# # response = completion.choices[0].message.content
# print(response)

# # event= completion.choices[0].message.parsed
# # print(f"Event Name: {event.name}")
# # print(f"Event Date: {event.date}")
# # print(f"Event Participants: {', '.join(event.participatns)}")

