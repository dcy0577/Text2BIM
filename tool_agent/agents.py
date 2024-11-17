
import builtins
import json
import os
from typing import Callable
import openai
from transformers.tools import Agent
from transformers.utils import is_openai_available
from transformers.tools.agents import resolve_tools, get_tool_creation_code, BASE_PYTHON_TOOLS
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import dotenv
from itertools import zip_longest

import vs
from tool_agent.python_interpreter import evaluate
from tool_agent.utils import delete_all_in_model, local_image_to_data_url, clean_code_for_chat_extend

import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
from vertexai.generative_models._generative_models import ToolConfig

# TODO: might consider to switch to gemini api in the future
# import google.generativeai as genai
# genai.configure(api_key=GEMINI_API_KEY)

# set the project root path
PROJECT_ROOT = r"C:\Users\ge25yak\Desktop\Text2BIM"
# load api key from .env file, if it doesn't work, need to set the keys below manually
dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# config the architect agent prompt template
ARCHITECT_PROMPT_PATH = os.path.join(PROJECT_ROOT, "tool_agent", "muti_agent_prompt", "floor_plan_designer_chat_prompt_few_shots.txt")

# enter your api key here, in case the dotenv doesn't work
OPENAI_API_KEY=""
MISTRAL_API_KEY=""

# vertex ai config for gemini agent
VERTEX_PROJECT = ""
VERTEX_LOCATION = "europe-west3"


# the architect agent is wrapped in this function
def plan_designer(query:str, 
                  model:str="gpt", # model can be "gpt" or "gemini" or "mistral"
                  floor_plan_designer_path=ARCHITECT_PROMPT_PATH): 

    with open(floor_plan_designer_path, "r", encoding="utf-8") as f:
        floor_plan_designer_prompt_str = f.read()
    prompt = floor_plan_designer_prompt_str.replace("<<task>>", query)

    if "gpt" in model:
        openai.api_key = OPENAI_API_KEY
        messages = [{"role": "user", "content": prompt}]
        model_name = "o1-preview" # "gpt-4o", "gpt-4o-2024-05-13"
        if "o1" in model_name:
            # o1 api currently doesnt support other args
            second_response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages
            )
        else:
            second_response = openai.ChatCompletion.create(
                    # model="gpt-4o",
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    stop=["User:"],
                )
        
        return second_response["choices"][0]["message"]["content"]

    if "gemini" in model:
        vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        model = GenerativeModel("gemini-1.5-pro")
        user_prompt_content = [
            Content(role="user", parts=[Part.from_text(prompt)])
        ]
        second_response = model.generate_content(
            user_prompt_content,
            generation_config=GenerationConfig(temperature=0, stop_sequences=["User:"]),
        )
       
        return second_response.candidates[0].content.parts[0].text
    
    if "mistral" in model:
        api_key = os.environ.get("MISTRAL_API_KEY", MISTRAL_API_KEY)
        client = MistralClient(api_key=api_key)
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        chat_response = client.chat(
            model="mistral-large-latest",
            # model = "open-mixtral-8x22b",
            messages=messages,
            temperature=0,
        )
        
        return chat_response.choices[0].message.content


FUNCTION_DESCRIPTION = [
        {
            "type": "function",
            "function": {
                "name": "plan_designer",
                "description": "Generate a floor or building plan in structured text based on the query, providing architectural knowledge.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The requirements for the plan, the more detailed the better.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
]

AVALIABLE_FUNCTIONS = {"plan_designer" : plan_designer}

FUNCTION_DECLARATIONS_GEMINI = [
    FunctionDeclaration(
        name="plan_designer",
        description="Generate a floor or building plan in structured text based on the query, providing architectural knowledge.",
        parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The requirements for the plan, the more detailed the better.",
                    },
                },
                "required": ["query"],
            },
    ),
]


class MyAgent(Agent):
    """
    change the base class of Agent. We dont need the default tools from huggingface.
    """
    def __init__(self, chat_prompt_template=None, run_prompt_template=None, additional_tools=None):

        agent_name = self.__class__.__name__
        self.chat_prompt_template = chat_prompt_template
        self.run_prompt_template = run_prompt_template
        self._toolbox = {}
        self.log = print
        self.response_counter = 0
        self._histroy_response= []
        if additional_tools is not None:
            if isinstance(additional_tools, (list, tuple)):
                additional_tools = {t.name: t for t in additional_tools}
            elif not isinstance(additional_tools, dict):
                additional_tools = {additional_tools.name: additional_tools}

            self._toolbox.update(additional_tools)

        self.prepare_for_new_chat()
    
    def log_new(self, msg):
        self._histroy_response.append(msg)
        print(msg)
    
    def generate_with_function_call(self, prompt:str, stop:str, available_functions:dict[str, Callable], function_description:list[dict]):
        raise NotImplementedError
    
    def generate_with_img(self, prompt:str, stop:str, img_path:str):
        raise NotImplementedError
    
    def chat(self, task, chat_history, *, hint_string=None, return_code=False, remote=False, is_reviewing=False, **kwargs):
    
        # BASE_PYTHON_TOOLS = {name: attr for name, attr in vars(builtins).items() if callable(attr)}
        for name, attr in vars(builtins).items():
            if callable(attr):
                BASE_PYTHON_TOOLS[name] = attr

        # update chat history from frontend befor call format_prompt
        description = "\n".join([f"- {name}: {tool.description}" for name, tool in self.toolbox.items()])
        # replace method will return original string if no match is found
        prompt = self.chat_prompt_template.replace("<<all_tools>>", description)
        if hint_string:
            # get the textual representation of the floor plan from floor plan designer
            prompt = prompt.replace("<<floor_plan_text>>", hint_string.strip())

        if chat_history:
            self.chat_history = prompt.replace("<<chat_history>>", chat_history.strip() + "\n")
        else:
            self.chat_history = prompt.replace("<<chat_history>>", "")
        
        prompt = self.chat_history

        prompt = prompt.replace("<<task>>", task)

        # prompt = self.format_prompt(task, chat_mode=True)
        original_result = self.generate_one(prompt, stop=["User:", "Programmer:", "Product Owner:"])
        # vs.AlrtDialog(f"original answer from llm: {original_result}")
        explanations, codes = clean_code_for_chat_extend(original_result)
        # init chat state in current chat session
        self.chat_state.update(kwargs)
        
        if self.response_counter == 0:
            # extract the role string from the last line of the prompt. e.g. "Programmer:" or "Assistant:"
            role = prompt.split("\n")[-1]
            self.log(f"{role}")
        
        for explanation, code in zip_longest(explanations, codes):
            
            # only show result if documentation_retrieval_tool is used
            if code is not None and "documentation_retrieval_tool" in code:
                try:
                    self.cached_tools = resolve_tools(code, self.toolbox, remote=remote, cached_tools=self.cached_tools)
                    # the result is the return value of the last tool function invoked in the generated code
                    result, state = evaluate(code, self.cached_tools, state=self.chat_state, chat_mode=True) # state=kwargs.copy()
                    # update the chat state
                    self.chat_state.update(state)
                    self.log(f"{result}")
                    
                except Exception as e:
                    self.log(f"Error while excuting code: {e}")
            else:
                # explanation from agent
                if explanation:
                    self.log(f"{explanation}")
                if code is not None:
                    self.log(f"\n```py\n{code}\n```")
                    if not return_code:
                        self.log("\n==Result==\n")
                        try:
                            self.cached_tools = resolve_tools(code, self.toolbox, remote=remote, cached_tools=self.cached_tools)
                            # the result is the return value of the last tool function invoked in the generated code
                            result, state = evaluate(code, self.cached_tools, state=self.chat_state, chat_mode=True) # state=kwargs.copy()
                            # update the chat state
                            self.chat_state.update(state)
                            self.log(f"Code executed successfully!\n")
                            # rest the counter
                            self.response_counter = 0
                        except Exception as e:
                            self.response_counter += 1
                            self.log(f"Error while excuting code: {e}\n")
                            # recursively call the chat function to solve the error, we only try 3 times
                            if self.response_counter < 3:
                                vs.AlrtDialog(f"Try to fix the error: {e}, iteration: {self.response_counter}")
                                # we delete the elements that are created from the error code
                                # TODO: this is not a good solution, might be dangerous
                                if not is_reviewing:
                                    delete_all_in_model()
                                self.chat_state, original_result = self.chat(f"Could you please fix the error by revising your code: Error while excuting code: {e}", original_result, **self.chat_state)
                            else:
                                self.log(f"I can't fix the error by myself: {e}. Please give me some hints.")

                    else:
                        tool_code = get_tool_creation_code(code, self.toolbox, remote=remote)
                        # return f"{tool_code}\n{code}"
        
        return self.chat_state, original_result
    
    def chat_with_function_call(self, task:str, chat_history:str, stop=["User:","Programmer:","Product Owner:"]):
        # BASE_PYTHON_TOOLS = {name: attr for name, attr in vars(builtins).items() if callable(attr)}
        # Get all Python built-in functions
        for name, attr in vars(builtins).items():
            if callable(attr):
                BASE_PYTHON_TOOLS[name] = attr

        description = "\n".join([f"- {name}: {tool.description}" for name, tool in self.toolbox.items()])
        # replace method will return original string if no match is found
        prompt = self.chat_prompt_template.replace("<<all_tools>>", description)
         # update chat history from frontend
        if chat_history:
            self.chat_history = prompt.replace("<<chat_history>>", chat_history.strip() + "\n")
        else:
            self.chat_history = prompt.replace("<<chat_history>>", "")
        
        prompt = self.chat_history
        prompt = prompt.replace("<<task>>", task)
        # check which agent class
        if isinstance(self, OpenAiAgent):
            result = self.generate_with_function_call(prompt, stop=stop, available_functions=AVALIABLE_FUNCTIONS, function_description=FUNCTION_DESCRIPTION)
        if isinstance(self, GeminiAgent):
            result = self.generate_with_function_call(prompt, stop=stop, available_functions=AVALIABLE_FUNCTIONS, function_declaration=FUNCTION_DECLARATIONS_GEMINI)
        if isinstance(self, MistralAgent):
            result = self.generate_with_function_call(prompt, stop=stop, available_functions=AVALIABLE_FUNCTIONS, function_description=FUNCTION_DESCRIPTION)
        # extract the role string from the last line of the prompt. e.g. "Programmer:" or "Assistant:"
        role = prompt.split("\n")[-1]
        self.log(f"{role}{result}")
        return result
    
    # this method is designed for reviewer agent
    def chat_check(self, code: str = None, issues: str = None, stop=["User:","Programmer:","Product Owner:", "Reviewer:"]):
        description = "\n".join([f"- {name}: {tool.description}" for name, tool in self.toolbox.items()])
        # replace method will return original string if no match is found
        prompt = self.chat_prompt_template.replace("<<all_tools>>", description)
        if code:
            prompt = prompt.replace("<<code>>", code)
        if issues:
            prompt = prompt.replace("<<issues>>", issues)

        result = self.generate_one(prompt, stop)
        # extract the role string from the last line of the prompt. e.g. "Programmer:" or "Assistant:"
        role = prompt.split("\n")[-1]
        self.log(f"{role}{result}")
        return result

class MistralAgent(MyAgent):
    def __init__(
        self,
        model = "open-mixtral-8x22b",
        api_key=None,
        chat_prompt_template=None,
        run_prompt_template=None,
        additional_tools=None,
    ):

        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY", MISTRAL_API_KEY)
        else:
            api_key = api_key
        self.model = model
        self.client = MistralClient(api_key=api_key)
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_many(self, prompts, stop):
        return self._completion_generate(prompts, stop)

    def generate_one(self, prompt, stop):
        return self._completion_generate(prompt, stop)

    def _completion_generate(self, prompts, stop):
        # No streaming
        messages = [
            ChatMessage(role="user", content=prompts)
        ]
        chat_response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return chat_response.choices[0].message.content
    
    def generate_with_function_call(self, prompt:str, stop:str, available_functions:dict[str, Callable], function_description:list[dict]):
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0,
            tools=function_description,
            tool_choice="auto",
            # tool_choice="any",
        )
        response_message = response.choices[0].message
        if hasattr(response_message, "tool_calls"):
            if response_message.tool_calls:
                tool_calls = response_message.tool_calls[0]
            else:
                return response_message.content
        else:
            return response_message.content
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            function_name = tool_calls.function.name
            if function_name not in available_functions:
                # if the function is not available, most likely the model is trying to call a function that is belong to the programmer's toolset
                # in this case, we can just disable the function call and give the model second chance to generate the response
                return self.generate_one(prompt, stop)
            
            messages.append(response_message)
            function_params = json.loads(tool_calls.function.arguments)
            function_to_call = available_functions[function_name]
            function_response = function_to_call(
                function_params.get("query"),
            )
            messages.append(ChatMessage(role="tool", name=function_name, content=function_response))
            response = self.client.chat(model=self.model, messages=messages, temperature=0)
            # also send the function call response to the frontend, currently hard coded the role as "Architect:"
            self.log(f"Architect: {function_response}")
            return response.choices[0].message.content

class GeminiAgent(MyAgent):
    def __init__(
        self,
        model="gemini-1.5-pro",
        chat_prompt_template=None,
        run_prompt_template=None,
        additional_tools=None,
    ):
        # Initialize Vertex AI
        vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        self.model = GenerativeModel(model)
        # self.api_model = genai.GenerativeModel(
        #     'gemini-1.5-pro',
        #     generation_config=genai.GenerationConfig(
        #         temperature=0,
        #     ))
     
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_many(self, prompts, stop):
        return self._completion_generate(prompts, stop)

    def generate_one(self, prompt, stop):
        return self._completion_generate(prompt, stop)
        # return self._completion_generate_api(prompt, stop)

    def _completion_generate(self, prompts, stop):
        user_prompt_content = Content(
        role="user",
        parts=[
            Part.from_text(prompts),
        ],
        )
        
        chat_response = self.model.generate_content(
            user_prompt_content,
            generation_config=GenerationConfig(temperature=0, stop_sequences=stop),
        )
        return chat_response.candidates[0].content.parts[0].text
    
    # def _completion_generate_api(self, prompts, stop):
    #     response = self.api_model.generate_content(
    #         prompts,
    #         generation_config = genai.GenerationConfig(stop_sequences=stop, temperature=0),
    #     )
    #     return response.text
    
    def generate_with_function_call(self, prompt:str, stop:list[str], available_functions:dict[str, Callable], function_declaration:list[dict]):
        user_prompt_content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.model.generate_content(
            user_prompt_content,
            generation_config=GenerationConfig(temperature=0, stop_sequences=stop),
            tools=[Tool(function_declarations=function_declaration)],
            tool_config=ToolConfig(
                function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY, # force call a function
                # mode=ToolConfig.FunctionCallingConfig.Mode.AUTO,
                allowed_function_names=["plan_designer"],
            ))
        )
        response_function_call_content = response.candidates[0].content
        if hasattr(response_function_call_content.parts[0], "function_call"):
            # TODO: how about multiple function calls?
            # if function calling list is not emtpy
            if response_function_call_content.parts[0].function_call:
                tool_calls = response_function_call_content.parts[0].function_call
            else:
                return response_function_call_content.parts[0].text
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            function_name = tool_calls.name
            if function_name not in available_functions:
                # if the function is not available, most likely the model is trying to call a function that is belong to the programmer's toolset
                # in this case, we can just disable the function call and give the model second chance to generate the response
                return self.generate_one(prompt, stop)
            
            function_to_call = available_functions[function_name]
            function_args = tool_calls.args["query"]
            function_response = function_to_call(
                function_args,
            )
            # Return the API response to Gemini so it can generate a model response or request another function call
            second_response = self.model.generate_content(
            [
                user_prompt_content,  # User prompt
                response_function_call_content,  # Function call response
                Content(
                    parts=[
                        Part.from_function_response(
                            name = function_name,
                            response={
                                "content": function_response,  # Return the API response to Gemini
                            },
                        )
                    ],
                ),
            ],
            generation_config=GenerationConfig(temperature=0, stop_sequences=stop),
            tools=[Tool(function_declarations=function_declaration)],
            tool_config=ToolConfig(
                function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.NONE, # force call a function
            ))
            )
            # Get the model summary response
            summary = second_response.candidates[0].content.parts[0].text
            # also send the function call response to the frontend, currently hard coded the role as "Architect:"
            self.log(f"Architect: {function_response}")
            return summary
    
    # def generate_with_function_call_api(self, prompt:str, stop:str, available_functions:dict[str, Callable], function_declaration:list[dict]):
    #     response = self.api_model.generate_content(
    #         prompt,
    #         generation_config = genai.GenerationConfig(stop_sequences=stop),
    #         tools=[genai.Tool(function_declarations=function_declaration)],
    #         tool_config=genai.ToolConfig(
    #             function_calling_config=genai.ToolConfig.FunctionCallingConfig(
    #             mode=genai.ToolConfig.FunctionCallingConfig.Mode.AUTO,
    #             allowed_function_names=["plan_designer"],
    #         ))
    #     )
    #     return response.text
        

class OpenAiAgent(MyAgent):
    def __init__(
        self,
        model="gpt-4o",
        api_key=None,
        chat_prompt_template=None,
        run_prompt_template=None,
        additional_tools=None,
    ):
        if not is_openai_available():
            raise ImportError("Using `OpenAiAgent` requires `openai`: `pip install openai`.")

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
        if api_key is None:
            raise ValueError(
                "You need an openai key to use `OpenAIAgent`. You can get one here: Get one here "
                "https://openai.com/api/`. If you have one, set it in your env with `os.environ['OPENAI_API_KEY'] = "
                "xxx."
            )
        else:
            openai.api_key = api_key
        self.model = model
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_many(self, prompts, stop):
        return [self._chat_generate(prompt, stop) for prompt in prompts]

    def generate_one(self, prompt, stop):
        return self._chat_generate(prompt, stop)
        
    def generate_with_img(self, prompt, stop, img_path):
        data_url = local_image_to_data_url(img_path)
        messages = [{ "role": "user", "content": [  
            { 
                "type": "text", 
                "text": prompt 
            },
            { 
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
            ] } 
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0,
            stop=stop,
        )
        return response.choices[0].message.content
    
    def generate_with_function_call(self, prompt:str, stop:str, available_functions:dict[str, Callable], function_description:list[dict]):
        messages=[{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0,
            tools=function_description,
            tool_choice = "auto",
            # tool_choice = "required",
            stop=stop,
        )
        response_message = response.choices[0].message
        if hasattr(response_message, "tool_calls"):
            tool_calls = response_message.tool_calls
        else:
            return response_message.content
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
    
            messages.append(response_message)  # extend conversation with assistant's reply
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                if function_name not in available_functions:
                    # if the function is not available, most likely the model is trying to call a function that is belong to the programmer's toolset
                    # in this case, we can just disable the function call and give the model second chance to generate the response
                    return self.generate_one(prompt, stop)
                    
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    function_args.get("query"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
            second_response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0,
                stop=stop,
            )
            # also send the function call response to the frontend, currently hard coded the role as "Architect:"
            self.log(f"Architect: {function_response}")
            return second_response["choices"][0]["message"]["content"]
        else:
            return response_message.content

    def _chat_generate(self, prompt, stop):
        if "o1" in self.model:
            # o1 api currently doesnt support other args
            result = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
        else:
            result = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stop=stop,
            )
        return result["choices"][0]["message"]["content"]








    