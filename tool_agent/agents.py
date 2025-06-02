
import builtins
import json
import os
from typing import Callable
from openai import OpenAI
from transformers.tools import Agent
from transformers.utils import is_openai_available
from transformers.tools.agents import resolve_tools, get_tool_creation_code, BASE_PYTHON_TOOLS
from mistralai import Mistral

import dotenv
from itertools import zip_longest

import anthropic 

import vs
from tool_agent.python_interpreter import evaluate
from tool_agent.utils import delete_all_in_model, local_image_to_data_url, clean_code_for_chat_extend



# TODO: might consider to switch to gemini api in the future
from google import genai
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    HttpOptions,
    Tool,
    ToolConfig,
    FunctionCallingConfig,
    Content,
    Part,   
    ThinkingConfig,
)

# set the project root path
PROJECT_ROOT = r"C:\Users\dell\Desktop\Text2BIM"
# load api key from .env file, if it doesn't work, need to set the keys below manually
dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# config the architect agent prompt template
ARCHITECT_PROMPT_PATH = os.path.join(PROJECT_ROOT, "tool_agent", "muti_agent_prompt", "floor_plan_designer_chat_prompt_few_shots.txt")

# enter your api key here, in case the dotenv doesn't work
OPENAI_API_KEY=""
MISTRAL_API_KEY=""
CLAUDE_API_KEY=""  
GEMINI_API_KEY="" # this is very unstable, better use vertexai api
VERTEX_PROJECT = ""  # Your Google Cloud project ID
# VERTEX_LOCATION = "us-central1"  # Your Google Cloud region
VERTEX_LOCATION = "global"  # Your Google Cloud region


# the architect agent is wrapped in this function
def plan_designer(query: str, 
                  model: str = "claude",  # model can be "gpt" or "gemini" or "mistral" or "claude"
                  floor_plan_designer_path=ARCHITECT_PROMPT_PATH): 

    with open(floor_plan_designer_path, "r", encoding="utf-8") as f:
        floor_plan_designer_prompt_str = f.read()
    prompt = floor_plan_designer_prompt_str.replace("<<task>>", query)

    if "gpt" in model:
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = [{"role": "user","content": str(prompt)}]
        model_name="o4-mini-2025-04-16" # "gpt-4.1","o4-mini","gpt-4o-2024-05-13","o1-preview" model_name = "o4-mini"  # "gpt-4o", "gpt-4o-2024-05-13" "o1-preview" "gpt-4.1"
        if "o1" or "o4" in model_name:
            # o1 api currently doesnt support other args
            second_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                reasoning_effort="high",  # "low", "medium", "high"
            )
        else:
            second_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                stop=["User:"],
            )
        
        return second_response.choices[0].message.content

    if "gemini" in model:
        # Configure Google GenAI API (new library)
        api_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
        if api_key is None:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        
        from google import genai
        from google.genai.types import GenerateContentConfig, HttpOptions
        
        client = genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION, http_options=HttpOptions(api_version='v1')
        )
        
        second_response = client.models.generate_content(
            model="gemini-2.5-pro-preview-05-06", # "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-05-20"
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.1, 
                stop_sequences=["User:"],
                max_output_tokens=65535,  # Set a reasonable limit for output tokens
            ),
        )
       
        return second_response.text
    
    if "mistral" in model:
        api_key = os.environ.get("MISTRAL_API_KEY", MISTRAL_API_KEY)
        client = Mistral(api_key=api_key)
        messages = [{"role": "user", "content": prompt}]
        
        chat_response = client.chat.complete(
            model="mistral-medium-2505",
            messages=messages,
            temperature=0,
        )
        return chat_response.choices[0].message.content

    if "claude" in model:
        api_key = os.environ.get("CLAUDE_API_KEY", CLAUDE_API_KEY)
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514", # "claude-4-opus-20250514", "claude-sonnet-4-20250514"
            max_tokens=4096,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stop_sequences=["User:"]
        )
        return response.content[0].text

FUNCTION_DESCRIPTION = [{
        "type": "function",
        "function": {
            "name": "plan_designer",
            "description": "Generate a floor or building plan in structured text based on the query, providing architectural knowledge",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The requirements for the plan, the more detailed the better"
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }]


AVALIABLE_FUNCTIONS = {"plan_designer" : plan_designer}

# Updated function declarations for google-genai
FUNCTION_DECLARATIONS_GEMINI = [
    FunctionDeclaration(
        name="plan_designer",
        description="Generate a floor or building plan in structured text based on the query, providing architectural knowledge.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "The requirements for the plan, the more detailed the better.",
                },
            },
            "required": ["query"],
        },
    )
]

# Claude function declaration (uses same format as OpenAI/Mistral)
FUNCTION_DECLARATIONS_CLAUDE = [
    {
        "name": "plan_designer",
        "description": "Generate a floor or building plan in structured text based on the query, providing architectural knowledge.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The requirements for the plan, the more detailed the better.",
                },
            },
            "required": ["query"],
        },
    }
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
                                    # delete_all_in_model()
                                    pass
                                self.chat_state, original_result = self.chat(f"Could you please fix the error by revising your code: Error while excuting code: {e}. In the script where you made the mistake, the code before the line where the error occurred has already been executed. It may have created or modified some objects, so please avoid repeating the same actions. If you really need to, please delete the objects created before in the script where you made mistake", original_result, **self.chat_state)
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
        if isinstance(self, ClaudeAgent):  # Add Claude support
            result = self.generate_with_function_call(prompt, stop=stop, available_functions=AVALIABLE_FUNCTIONS, function_description=FUNCTION_DECLARATIONS_CLAUDE)
        # extract the role string from the last line of the prompt. e.g. "Programmer:" or "Assistant:"
        role = prompt.split("\n")[-1]
        # TODO: lets reduce this for gemini agent, when the output string is too long, it will cause the webplatte to crash
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


class ClaudeAgent(MyAgent):
    def __init__(
        self,
        model="claude-4-opus-20250514", # "claude-4-opus-20250514", "claude-sonnet-4-20250514"
        api_key=None,
        chat_prompt_template=None,
        run_prompt_template=None,
        additional_tools=None,
    ):
        if api_key is None:
            api_key = os.environ.get("CLAUDE_API_KEY", CLAUDE_API_KEY)
        if api_key is None:
            raise ValueError(
                "You need a Claude API key to use `ClaudeAgent`. You can get one here: "
                "https://console.anthropic.com/. If you have one, set it in your env with "
                "`os.environ['CLAUDE_API_KEY'] = xxx` or pass it as api_key parameter."
            )
        
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_many(self, prompts, stop):
        return [self._completion_generate(prompt, stop) for prompt in prompts]

    def generate_one(self, prompt, stop):
        return self._completion_generate(prompt, stop)

    def _completion_generate(self, prompt, stop):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            stop_sequences=stop if stop else None,
        )
        return response.content[0].text
    
    def generate_with_img(self, prompt, stop, img_path):
        """Generate response with image input using Claude's vision capabilities"""
        import base64
        
        # Read and encode the image
        with open(img_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine image type
        image_ext = os.path.splitext(img_path)[1].lower()
        if image_ext in ['.jpg', '.jpeg']:
            media_type = "image/jpeg"
        elif image_ext == '.png':
            media_type = "image/png"
        elif image_ext == '.gif':
            media_type = "image/gif"
        elif image_ext == '.webp':
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"  # default
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    }
                ]
            }],
            stop_sequences=stop if stop else None,
        )
        return response.content[0].text
    
    def generate_with_function_call(self, prompt: str, stop: list[str], available_functions: dict[str, Callable], function_description: list[dict]):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            tools=function_description,
            stop_sequences=stop if stop else None,
        )
        
        # Check if Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Find the tool use block
            tool_use = None
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_use = content_block
                    break
            
            if tool_use:
                function_name = tool_use.name
                if function_name not in available_functions:
                    # Function not available, generate regular response
                    return self.generate_one(prompt, stop)
                
                function_to_call = available_functions[function_name]
                function_args = tool_use.input
                function_response = function_to_call(function_args.get("query"))
                
                # Continue conversation with function result
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": function_response,
                            }
                        ],
                    },
                ]
                
                second_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0,
                    messages=messages,
                    stop_sequences=stop if stop else None,
                )
                
                # Log the function response
                self.log(f"Architect: {function_response}")
                return second_response.content[0].text
        
        # No tool use, return regular response
        return response.content[0].text


class MistralAgent(MyAgent):
    def __init__(
        self,
        model="mistral-large-latest",  # Updated to use latest model
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
        self.client = Mistral(api_key=api_key)
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_many(self, prompts, stop):
        return [self._completion_generate(prompt, stop) for prompt in prompts]

    def generate_one(self, prompt, stop):
        return self._completion_generate(prompt, stop)

    def _completion_generate(self, prompts, stop):
        # Convert to dict format for consistency
        messages = [{"role": "user", "content": prompts}]
        
        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return chat_response.choices[0].message.content
    
    def generate_with_function_call(self, prompt: str, stop: list[str], available_functions: dict[str, Callable], function_description: list[dict]):
        # Convert to the message format that Mistral expects
        messages = [{"role": "user", "content": prompt}]
        
        # Initial request with tools
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0,
            tools=function_description,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        
        # Check if the response has tool calls
        if not hasattr(response_message, "tool_calls") or not response_message.tool_calls:
            return response_message.content
        
        # Add the assistant's response to messages (convert to dict format)
        messages.append({
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in response_message.tool_calls
            ]
        })
        
        # Process each tool call
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            
            if function_name not in available_functions:
                # If the function is not available, disable function call and generate regular response
                return self.generate_one(prompt, stop)
            
            # Execute the function
            function_to_call = available_functions[function_name]
            function_params = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                function_params.get("query"),
            )
            
            # Add tool response using the exact format from Mistral documentation
            messages.append({
                "role": "tool",
                "name": function_name,
                "content": function_response,
                "tool_call_id": tool_call.id
            })
            
            # Log the function response
            self.log(f"Architect: {function_response}")
        
        # Get the final response after tool execution
        final_response = self.client.chat.complete(
            model=self.model, 
            messages=messages, 
            temperature=0
        )
        
        return final_response.choices[0].message.content

class GeminiAgent(MyAgent):
    def __init__(
        self,
        model="gemini-2.0-flash-001",  # Using the latest model
        api_key=None,
        chat_prompt_template=None,
        run_prompt_template=None,
        additional_tools=None,
    ):
        # Configure Google GenAI
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
        if api_key is None:
            raise ValueError(
                "You need a Gemini API key to use `GeminiAgent`. You can get one here: "
                "https://makersuite.google.com/app/apikey. If you have one, set it in your env with "
                "`os.environ['GEMINI_API_KEY'] = xxx` or pass it as api_key parameter."
            )
        
        self.client = genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION, http_options=HttpOptions(api_version='v1')
        )
        self.model_name = model
     
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    def generate_many(self, prompts, stop):
        return [self._completion_generate(prompt, stop) for prompt in prompts]

    def generate_one(self, prompt, stop):
        return self._completion_generate(prompt, stop)

    def _completion_generate(self, prompts, stop):
        client = genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION, http_options=HttpOptions(api_version='v1')
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=[prompts],
            config=GenerateContentConfig(
                # stop_sequences=stop if stop else None,
                temperature=0,
                max_output_tokens=65535,  # Set a reasonable limit for output tokens
            ),
        )
        return response.text
    
    def generate_with_img(self, prompt, stop, img_path):
        """Generate response with image input using Gemini's vision capabilities"""
        import base64
        
        # Read and encode the image
        with open(img_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine image type
        image_ext = os.path.splitext(img_path)[1].lower()
        if image_ext in ['.jpg', '.jpeg']:
            media_type = "image/jpeg"
        elif image_ext == '.png':
            media_type = "image/png"
        elif image_ext == '.gif':
            media_type = "image/gif"
        elif image_ext == '.webp':
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"  # default
        
        # Create content with text and image
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": media_type,
                            "data": image_data
                        }
                    }
                ]
            }
        ]
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=GenerateContentConfig(
                stop_sequences=stop if stop else None,
                temperature=0
            ),
        )
        return response.text
    
    def generate_with_function_call(self, prompt: str, stop: list[str], available_functions: dict[str, Callable], function_declaration: list[dict]):
        # Create the tool with function declarations
        tools = Tool(function_declarations=FUNCTION_DECLARATIONS_GEMINI)

        # Configure tool usage mode
        tool_config = ToolConfig(
            function_calling_config=FunctionCallingConfig(
                mode="AUTO", # allowed_function_names=["plan_designer"] # "ANY" or "AUTO" or "REQUIRED" or "DISABLED"
            )
        )

        thinking_config = ThinkingConfig(
            include_thoughts=False,
            thinking_budget=1024,  # "AUTO" or "REQUIRED" or "DISABLED"
        )

        contents = [
            Content(
                role="user", parts=[Part(text=prompt)]
            )
        ]
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=GenerateContentConfig(
                stop_sequences=stop if stop else None,
                temperature=0.1,
                tool_config=tool_config,
                tools=[tools],
                # max_output_tokens=65535,  # Set a reasonable limit for output tokens
            ),
        )
        
        # Check if the model made a function call
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            function_name = function_call.name
                            
                            if function_name not in available_functions:
                                # If the function is not available, generate a regular response
                                return self.generate_one(prompt, stop)
                            
                            # Execute the function
                            function_to_call = available_functions[function_name]
                            function_args = dict(function_call.args) if hasattr(function_call, 'args') else {}
                            function_response = function_to_call(function_args.get("query", ""))
                            

                            function_response_part = Part.from_function_response(
                                name=function_name,
                                response={"result": function_response},
                            )

                            contents.append(Content(role="model", parts=[Part(function_call=function_call)])) # Append the model's function call message
                            contents.append(Content(role="user", parts=[function_response_part])) # Append the function response
                            
                            # Get the final response
                            second_response = self.client.models.generate_content(
                                model=self.model_name,
                                contents=contents,
                                config=GenerateContentConfig(
                                    # stop_sequences=stop if stop else None,
                                    temperature=0.1,
                                    max_output_tokens=8096,  # Set a reasonable limit for output tokens
                                ),
                            )
                            
                            # Log the function response
                            self.log(f"Architect: {function_response}")
                            return second_response.text
        
        # No function call, return regular response
        return response.text
        

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

        self.client = OpenAI(api_key=api_key)
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
        response =  self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            stop=stop,
        )
        return response.choices[0].message.content
    
    def generate_with_function_call(self, prompt:str, stop:str, available_functions:dict[str, Callable], function_description:list[dict]):
        messages=[{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # temperature=0,
            tools=function_description,
            tool_choice = "auto",
            # tool_choice = "required",
            # reasoning_effort="high",  # "low", "medium", "high"
            # stop=stop,
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
                        "content": str(function_response),
                    }
                )  # extend conversation with function response
            second_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                reasoning_effort="high",  # "low", "medium", "high"
                # temperature=0,
                # stop=stop,
            )
            # also send the function call response to the frontend, currently hard coded the role as "Architect:"
            self.log(f"Architect: {function_response}")
            return second_response.choices[0].message.content
        else:
            return response_message.content

    def _chat_generate(self, prompt, stop):
        if "o1" or "o4" in self.model:
            # o1 api currently doesnt support other args
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="high",  # "low", "medium", "high"
            )
        else:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stop=stop,
            )
        return result.choices[0].message.content








    