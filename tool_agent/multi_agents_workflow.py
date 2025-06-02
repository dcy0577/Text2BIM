from datetime import datetime
import importlib
import os
import threading
import uuid
import dotenv
import json

import vs
from tool_agent.utils import remove_last_human_message_with_regex, safe_encode
from tool_agent.agents import ClaudeAgent, OpenAiAgent, GeminiAgent, MistralAgent, PROJECT_ROOT
from tool_agent.solibri_checker import *
import tool_agent.vw_tools_extend
importlib.reload(tool_agent.vw_tools_extend)
from tool_agent.vw_tools_extend import *

# load the api key from the .env file
dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# config the prompt template paths
PRODUCT_OWNER_PROMPT_PATH = os.path.join(PROJECT_ROOT, "tool_agent", "muti_agent_prompt", "po_chat_prompt_temp.txt")
CODER_PROMPT_PATH = os.path.join(PROJECT_ROOT, "tool_agent", "muti_agent_prompt", "coder_chat_prompt_temp.txt")
REVIEWER_PROMPT_PATH = os.path.join(PROJECT_ROOT, "tool_agent", "muti_agent_prompt", "checker_chat_prompt_temp.txt")

# config state path
STATE_PATH = os.path.join(PROJECT_ROOT, "data", "state.json")

# config data/ifc output folder path
MAIN_OUTPUT_FOLDER_PATH = os.path.join(PROJECT_ROOT, "tool_agent", "Ifc_test_data")

# config the solibri paths
SOLIBRI_PATH = r"C:\Program Files\Solibri\SOLIBRI\Solibri.exe"
SOLIBRI_WORKFLOW_PATH = os.path.join(PROJECT_ROOT, "tool_agent", "Solibri_workflow_xml.xml")
BCF_EXPORT_PATH= os.path.join(PROJECT_ROOT, "tool_agent", "solibri_data","autorun_issues.bcfzip")
MUSTER_IFC_PATH = os.path.join(PROJECT_ROOT, "tool_agent", "Ifc_test_data", "sample_model.ifc")
SOLIBRI_SMC_PATH = os.path.join(PROJECT_ROOT, "tool_agent", "solibri_data", "LLM-checking-FullExample.smc")


MODEL = "claude"  # default model, can be changed to "gpt", "gemini", "claude", "mistral"
TASK_NUMBER = f"Prompt_NR.{1}"
TASK_ROUND = "1"
OUTPUT_FOLDER_PATH = os.path.join(MAIN_OUTPUT_FOLDER_PATH, MODEL, TASK_NUMBER, TASK_ROUND)
if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

EXPERIMENT_LOG_PATH = os.path.join(PROJECT_ROOT, f"experiment_log_{MODEL}.json")



# # global variable holding the outputs from the agents
# output_sum = ''
# def streamer(output):
#     global output_sum
#     output_sum = output_sum + output
#     print(output)


output_chunks = []
output_lock = threading.Lock()

# Replace your streamer function with:
def streamer(output):
    global output_chunks
    with output_lock:
        output_chunks.append(output)
    print(output)

# Add helper functions:
def get_output_sum():
    with output_lock:
        return ''.join(output_chunks)

def clear_output():
    global output_chunks
    with output_lock:
        output_chunks = []


# init tools
def init_vw_tools():
    create_wall_tool = CreateWallTool()
    set_wall_thickness_tool = SetWallThickness()
    set_wall_height_tool = SetWallHeight()
    set_wall_style_tool = SetWallStyle()
    get_wall_elevation_tool = GetWallElevation()
    get_wall_thickness_tool = GetWallThickness()
    add_window_tool = AddWindowToWall()
    add_door_tool = AddDoorToWall()
    move_tool = Move()
    delete_tool = DeleteTool()
    get_selected_tool = FindSelect()
    create_polygon_tool = CreatePolygon()
    get_polygon_vertex_tool = GetPolygonVertex()
    get_polygon_vertex_count_tool = GetVertNum()
    create_slab_tool = CreateSlab()
    set_slab_height_tool = SetSlabHeight()
    get_slab_height_tool = GetSlabHeight()
    set_slab_style_tool = SetSlabStyle()
    duplicate_object_tool = DuplicateObj()
    rotate_object_tool = RotateObj()
    create_roof_tool = CreateRoof()
    set_roof_attributes_tool = SetRoofAttributes()
    set_roof_style_tool = SetRoofStyle()
    create_story_layer = CreateStoryLayer()
    set_active_story_layer = SetStoryLayerActive()
    create_space = CreateSpace()
    

    tool_list = [
        create_wall_tool, 
        set_wall_height_tool, 
        set_wall_thickness_tool, 
        set_wall_style_tool, 
        get_wall_elevation_tool, 
        get_wall_thickness_tool, 
        add_window_tool, 
        add_door_tool, 
        move_tool, 
        delete_tool, 
        get_selected_tool, 
        create_polygon_tool, 
        get_polygon_vertex_tool, 
        get_polygon_vertex_count_tool, 
        create_slab_tool, 
        set_slab_height_tool, 
        get_slab_height_tool,
        set_slab_style_tool, 
        duplicate_object_tool, 
        rotate_object_tool, 
        create_roof_tool, 
        set_roof_attributes_tool,
        set_roof_style_tool,
        create_story_layer,
        set_active_story_layer,
        create_space
    ]

    return tool_list

def log_experiment_data(model, task_number, task_round, 
                       issue_fixing_counter=None, 
                       issue_count=None, 
                       issue_type_info_dict=None, 
                       file_name=None,
                       experiment_phase=None):
    """
    Log the experiment data to a file and keep updating it.
    
    Args:
        model (str): The model used ("gpt", "gemini", "mistral", "claude")
        task_number (str): Task number (e.g., "prompt_NR.11")
        task_round (str): Task round (e.g., "1")
        issue_fixing_counter (int, optional): Current iteration counter
        issue_count (int, optional): Number of issues found
        issue_type_info_dict (dict, optional): Issue type information
        file_name (str, optional): Generated file name
        experiment_phase (str, optional): Phase of experiment ("initial", "checking", "final")
    """
    
    # Create experiment key
    experiment_key = f"{model}_{task_number}_{task_round}"
    
    # Create log directory if not exists
    log_dir = os.path.dirname(EXPERIMENT_LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Load existing log data
    if os.path.exists(EXPERIMENT_LOG_PATH):
        try:
            with open(EXPERIMENT_LOG_PATH, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            log_data = {}
    else:
        log_data = {}
    
    # Initialize experiment entry if doesn't exist
    if experiment_key not in log_data:
        log_data[experiment_key] = {
            "model": model,
            "task_number": task_number,
            "task_round": task_round,
            "created_at": datetime.now().isoformat(),
            "iterations": []
        }
    
    # Add iteration data
    iteration_data = {
        "timestamp": datetime.now().isoformat(),
        "issue_fixing_counter": issue_fixing_counter,
        "issue_count": issue_count,
        "issue_type_info_dict": issue_type_info_dict,
        "file_name": file_name,
        "experiment_phase": experiment_phase
    }
    
    # Remove None values
    iteration_data = {k: v for k, v in iteration_data.items() if v is not None}
    
    log_data[experiment_key]["iterations"].append(iteration_data)
    log_data[experiment_key]["last_updated"] = datetime.now().isoformat()
    
    # Save updated log data
    with open(EXPERIMENT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"Logged experiment data for {experiment_key}, iteration {issue_fixing_counter}")


# function for user instruction->product owner(->architect)->programmer workflow 
def run_po_coder_agents(query: str, 
                        chat_history_front_end: str, 
                        model=MODEL): # model = "gpt" or "gemini" or "mistral" or "claude"
    # init tools
    tool_list = init_vw_tools()

    with open(PRODUCT_OWNER_PROMPT_PATH, "r", encoding="utf-8") as f:
        po_prompt_str = f.read()

    with open(CODER_PROMPT_PATH, "r", encoding="utf-8") as f:
        coder_prompt_str = f.read()

    if model == "gpt":
        agent_po = OpenAiAgent(
            model="o4-mini-2025-04-16", # "gpt-4.1","o4-mini","gpt-4o-2024-05-13","o1-preview",o4-mini-2025-04-16
            chat_prompt_template=po_prompt_str,
            additional_tools=tool_list
        )

        agent_coder = OpenAiAgent(
            model="o4-mini-2025-04-16", # "gpt-4.1","o4-mini","gpt-4o-2024-05-13","o1-preview"
            chat_prompt_template=coder_prompt_str,
            additional_tools=tool_list
        )
    
    elif model == "gemini":
        agent_po = GeminiAgent(
            model="gemini-2.5-pro-preview-05-06", # "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-05-20"
            chat_prompt_template=po_prompt_str,
            additional_tools=tool_list
        )
        agent_coder = GeminiAgent(
            model="gemini-2.5-pro-preview-05-06", # "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-05-20"
            chat_prompt_template=coder_prompt_str,
            additional_tools=tool_list
        )
    
    elif model == "mistral":
        agent_po = MistralAgent(
            model = "mistral-medium-2505",
            chat_prompt_template=po_prompt_str,
            additional_tools=tool_list
        )
        agent_coder = MistralAgent(
            model = "mistral-medium-2505",
            chat_prompt_template=coder_prompt_str,
            additional_tools=tool_list
        )
    
    elif model == "claude":
        agent_po = ClaudeAgent(
            model="claude-sonnet-4-20250514", # "claude-4-opus-20250514", "claude-sonnet-4-20250514"
            chat_prompt_template=po_prompt_str,
            additional_tools=tool_list
        )
        agent_coder = ClaudeAgent(
            model="claude-sonnet-4-20250514", # "claude-4-opus-20250514", "claude-sonnet-4-20250514"
            chat_prompt_template=coder_prompt_str,
            additional_tools=tool_list
        )

    agent_coder.set_stream(streamer)
    agent_po.set_stream(streamer)

    # create a state dictionary file if not exist
    state_path = STATE_PATH
    if not os.path.exists(state_path):
        with open(state_path, "w") as f:
            json.dump({}, f)

    # load the previous state from file
    try:
        with open(state_path, "r") as f:
            previous_state = json.load(f)
    except json.JSONDecodeError:
        previous_state = {} 
            
    chat_history = remove_last_human_message_with_regex(chat_history_front_end)

    answer = agent_po.chat_with_function_call(query, chat_history) # consider the function call to connect with architect agent
    state, code_result = agent_coder.chat(answer, chat_history, **previous_state) # chat will evaluate the code

    # update the state dictionary
    if isinstance(state, dict):
        previous_state.update(state)

    # save the state to file
    with open(state_path, "w") as f:
        json.dump(previous_state, f, default=safe_encode)

    output_sum = get_output_sum()
    return output_sum, code_result

# function for exporting the ifc file and agents' chat records to disk
def export_ifc_and_stuffs(output_sum: str, issue_fixing_counter: int, chat_history_front_end: str, query: str):
    # export the ifc
    if output_sum.split("\n")[-2] == "Code executed successfully!":
        uuids = uuid.uuid4()

        if vs.YNDialog(f"Do you want to export the IFC file? (iteration: {issue_fixing_counter})") == 0:
            # 0 for beak the loop
            return "break"
        else:
            if str(chat_history_front_end).count("User: ") == 1:
                file_name = "prompt_" + str(uuids)
            else:
                file_name = "follow_up_prompt_" + str(uuids)
            
            if issue_fixing_counter == 0:
                # have to predifine the ifc export settings in project
                new_ifc_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}.ifc")
                prompt_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}.txt")
                response_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_from_agents.txt")
                # also store the corresponding prompt data
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(str(query))
                # also store the response data
                with open(response_path, "w", encoding="utf-8") as f:
                    f.write(str(output_sum))
                
                # Log initial experiment data
                log_experiment_data(
                    model=MODEL,
                    task_number=TASK_NUMBER,
                    task_round=TASK_ROUND,
                    issue_fixing_counter=issue_fixing_counter,
                    file_name=file_name,
                    experiment_phase="initial"
                )
            else:
                new_ifc_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_checking_{issue_fixing_counter}.ifc")
                issue_str_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_checking_{issue_fixing_counter}.txt")

                # fake file just for get the ifc name easily, the content will be overwritten anyway after the checking
                with open(issue_str_path, "w") as f:
                    f.write("fake string")
                
            # vs.IFC_ExportNoUI(new_ifc_path)
            # it seems that the ifc export without ui is not working for automatic story mapping, so we use vs.IFC_ExportWithUI instead
            # remember to set the ifc path as file_name.ifc
            vs.IFC_ExportWithUI(False)
            # also export the vw project file
            project_file_path = os.path.join(OUTPUT_FOLDER_PATH, f"{new_ifc_path}.vwx")
            vs.SaveActiveDocument(project_file_path)
            if issue_fixing_counter < 3: # we will need the output_sum for the final export
                clear_output()
            return file_name
    else:
        return "break"

# function for the quality optimization workflow (model checker -> reviewer -> programmer), the for loop is implemented in the web palette backend
def run_agent_checking_loop(issue_fixing_counter: int, 
                            original_code_result: str, 
                            code_result: str, 
                            file_name: str,
                            model=MODEL): # model = "gpt" or "gemini" or "mistral" or "claude"

    # init tools
    tool_list = init_vw_tools()

    with open(CODER_PROMPT_PATH, "r", encoding="utf-8") as f:
        coder_prompt_str = f.read()
    
    with open(REVIEWER_PROMPT_PATH, "r", encoding="utf-8") as f:
        review_prompt_str = f.read()

    if model == "gpt":
        agent_reviewer = OpenAiAgent(
            model="o4-mini-2025-04-16", # "gpt-4.1","o4-mini","gpt-4o-2024-05-13","o1-preview"
            chat_prompt_template=review_prompt_str,
            additional_tools=tool_list
        )
        agent_coder = OpenAiAgent(
            model="o4-mini-2025-04-16", # "gpt-4.1","o4-mini","gpt-4o-2024-05-13","o1-preview"
            chat_prompt_template=coder_prompt_str,
            additional_tools=tool_list
        )
    
    elif model == "gemini":
        agent_reviewer = GeminiAgent(
            model="gemini-2.5-pro-preview-05-06", # "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-05-20"
            chat_prompt_template=review_prompt_str,
            additional_tools=tool_list
        )
        agent_coder = GeminiAgent(
            model="gemini-2.5-pro-preview-05-06", # "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-05-20"
            chat_prompt_template=coder_prompt_str,
            additional_tools=tool_list
        )
    
    elif model == "mistral":
        agent_reviewer = MistralAgent(
            model = "mistral-medium-2505",
            chat_prompt_template=review_prompt_str,
            additional_tools=tool_list
        )
        agent_coder = MistralAgent(
            model = "mistral-medium-2505",
            chat_prompt_template=coder_prompt_str,
            additional_tools=tool_list
        )
    
    elif model == "claude":
        agent_reviewer = ClaudeAgent(
            model="claude-sonnet-4-20250514", # "claude-4-opus-20250514", "claude-sonnet-4-20250514"
            chat_prompt_template=review_prompt_str,
            additional_tools=tool_list
        )
        agent_coder = ClaudeAgent(
            model="claude-sonnet-4-20250514", # "claude-4-opus-20250514", "claude-sonnet-4-20250514"
            chat_prompt_template=coder_prompt_str,
            additional_tools=tool_list
        )

    agent_coder.set_stream(streamer)
    agent_reviewer.set_stream(streamer)

    # create a state dictionary file if not exist
    state_path = STATE_PATH
    # load the previous state from file
    try:
        with open(state_path, "r") as f:
            state = json.load(f)
    except json.JSONDecodeError:
       state = {}

    # run checker
    if vs.YNDialog(f"Do you want to run the checking? (iteration: {issue_fixing_counter})") == 0:
        # break for beak the loop
        return "break", "No code"
    else:
        if issue_fixing_counter == 0:
            new_ifc_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}.ifc")
        else:
            new_ifc_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_checking_{issue_fixing_counter}.ifc")
        workflow_path = SOLIBRI_WORKFLOW_PATH
        bcf_export_path = BCF_EXPORT_PATH
        set_xml_path(workflow_path,
                    smc_path=SOLIBRI_SMC_PATH,
                    old_ifc_path=MUSTER_IFC_PATH, # currently we dont save the smc model so the old ifc path is always the one we set at the beginning
                    new_ifc_path=new_ifc_path, 
                    bcf_export_path=bcf_export_path)
        check_model(workflow_path, solibri_path=SOLIBRI_PATH)
        issue_str, issue_count, issue_type_info_dict = process_bcf_report(bcf_export_path, new_ifc_path)

        # Log checking results
        log_experiment_data(
            model=model,
            task_number=TASK_NUMBER,
            task_round=TASK_ROUND,
            issue_fixing_counter=issue_fixing_counter,
            issue_count=issue_count,
            issue_type_info_dict=issue_type_info_dict,
            file_name=file_name,
            experiment_phase="checking"
        )

        # if no issues are found, we can stop the loop
        if issue_str == "":
            vs.AlrtDialog("No issues found! The model is compliant!")
            # break for beak the loop
            return "break", "No code"
        vs.AlrtDialog(issue_str)
        
        # I think it would make sense to let the reviewer see all the code during the fixing loop?
        # may helps the agent to gain more spatial context?
        review_result = agent_reviewer.chat_check(code = code_result, issues = issue_str)

        # call the programmer agent to fix the issues
        # this agent can only see the history within the loop
        # with the local chat history, the agent tends to rewrite the previous code instead of adding paches as standalone functions
        # TODO: I think for more complex issues this may makes sense. But we need to delete the previous generated building.
        # delete_all_in_model() # if we want to delete the previous generated building. BUT this will make the uuids of the elements in model change!!!
        
        local_chat_history = "" 
        input_review_result = "The Python code that generates the building with some issues: \n" + f"{code_result}" + "Please write a patch to solve the issues based on the review suggestions. Do not write diff but a excutable python script in a code block. You can reuse the variables defined in the original code, but DO NOT rewrite the original code." + "\n" + "Review suggestions: " + review_result

        state, fix_code_result = agent_coder.chat(input_review_result, local_chat_history, **state, is_reviewing=True)
        # update the local chat history
        local_chat_history += "\n" + fix_code_result
        # update the code result by adding the fixing code
        code_result += "\n" + "Previously executed code patch that attempted to fix the issues:" + "\n" + fix_code_result

        issue_str_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_checking_{issue_fixing_counter}.txt")
        fixing_output_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_checking_{issue_fixing_counter}_from_agents.txt")

        # save the output from reviewer and coder to file
        with open(fixing_output_path, "w", encoding="utf-8") as f:
            f.write("Reviewer: " + review_result + "\n" + "Programmer: " + fix_code_result + "\n")
        # save the issue string to file
        with open(issue_str_path, "w", encoding="utf-8") as f:
            f.write(issue_str)
        
        output_sum = get_output_sum()  # Get string when needed

        return output_sum, code_result

# function for exporting the final ifc file and the checking results to disk  
def export_final_ifc_and_checks(output_sum: str, issue_fixing_counter: int):
    # export the ifc
    if output_sum.split("\n")[-2] == "Code executed successfully!":

        uuids = uuid.uuid4()

        if vs.YNDialog(f"Do you want to export the IFC file? (iteration: {issue_fixing_counter})") == 0:
            # 0 for beak the loop
            return "break"
        else:

            file_name = "prompt_" + str(uuids)
            issue_str_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_checking_{issue_fixing_counter}_final.txt")

            # fake file just for get the ifc name easily, the content will be overwritten anyway after the checking
            with open(issue_str_path, "w") as f:
                f.write("fake string")
                
            # vs.IFC_ExportNoUI(new_ifc_path)
            # it seems that the ifc export without ui is not working for automatic story mapping
            # remember to set the ifc path as file_name.ifc
            vs.IFC_ExportWithUI(False)
            # also export the vw project file
            project_file_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_checking_{issue_fixing_counter}_final.vwx")
            vs.SaveActiveDocument(project_file_path)

            clear_output()
            return file_name
    else:
        return "break"

# function for the final model checking
def pure_checking(file_name, issue_fixing_counter):

    new_ifc_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_checking_{issue_fixing_counter}_final.ifc")
    workflow_path = SOLIBRI_WORKFLOW_PATH
    bcf_export_path = BCF_EXPORT_PATH
    set_xml_path(workflow_path,
                smc_path=SOLIBRI_SMC_PATH,
                old_ifc_path=MUSTER_IFC_PATH, # currently we dont save the smc model so the old ifc path is always the one we set at the beginning
                new_ifc_path=new_ifc_path, 
                bcf_export_path=bcf_export_path)
    check_model(workflow_path, solibri_path=SOLIBRI_PATH)
    issue_str, issue_count, issue_type_info_dict = process_bcf_report(bcf_export_path, new_ifc_path)

     # Log final checking results
    log_experiment_data(
        model=MODEL,
        task_number=TASK_NUMBER,
        task_round=TASK_ROUND,
        issue_fixing_counter=issue_fixing_counter,
        issue_count=issue_count,
        issue_type_info_dict=issue_type_info_dict,
        file_name=file_name,
        experiment_phase="final"
    )
    # if no issues are found, we can stop the loop
    if issue_str == "":
        vs.AlrtDialog("No issues found! The model is compliant!")
        return
    vs.AlrtDialog(issue_str)
    issue_str_path = os.path.join(OUTPUT_FOLDER_PATH, f"{file_name}_checking_{issue_fixing_counter}_final.txt")
    with open(issue_str_path, "w", encoding="utf-8") as f:
        f.write(issue_str)


