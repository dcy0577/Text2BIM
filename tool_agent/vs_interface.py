######
# This file contains functions that will be called by the web palette backend
######
import vs
import importlib
import ptvsd
import tool_agent.multi_agents_workflow
importlib.reload(tool_agent.multi_agents_workflow)
from tool_agent.multi_agents_workflow import STATE_PATH, run_po_coder_agents, export_ifc_and_stuffs, run_agent_checking_loop, export_final_ifc_and_checks, pure_checking
class DebugServer:
    _instance = None

    @staticmethod
    def getInstance():
        if DebugServer._instance == None:
            DebugServer()
        return DebugServer._instance

    def __init__(self):
        if DebugServer._instance != None:
            raise Exception("This class is a singleton!")
        else:
            DebugServer._instance = self
            svr_addr = str(ptvsd.options.host) + ":" + str(ptvsd.options.port)
            print(" -> Hosting debug server ... (" + svr_addr + ")")
            ptvsd.enable_attach()
            ptvsd.wait_for_attach(0.3)

# Below are the endpoints functions that will be excuted by the web palette backend

def excute_webpalette_po_coder(input_str, chat_history):
    # debug attach
    DEBUG = True
    if DEBUG:
        DebugServer.getInstance()

    chat_history = chat_history.replace("\\n", "\n")
    output_str, code_result = run_po_coder_agents(str(input_str), str(chat_history))
    
    return output_str, code_result

def excute_webpalette_export(output_str, issue_fixing_counter, chat_history, query):
    # debug attach
    DEBUG = True
    if DEBUG:
        DebugServer.getInstance()

    chat_history = chat_history.replace("\\n", "\n")
    output_str = output_str.replace("\\n", "\n")
    file_name = export_ifc_and_stuffs(str(output_str), int(issue_fixing_counter), str(chat_history), str(query))
    
    return file_name

def excute_webpalette_checking_loop(issue_fixing_counter, original_code_result, code_result, file_name):
    # debug attach
    DEBUG = True
    if DEBUG:
        server = DebugServer.getInstance()
    code_result = code_result.replace("\\n", "\n")
    original_code_result = original_code_result.replace("\\n", "\n")
    output_str, output_code_result = run_agent_checking_loop(int(issue_fixing_counter), str(original_code_result), str(code_result), str(file_name))

    return output_str, output_code_result

def excute_final_ifc_export(output_str, issue_fixing_counter):
    # debug attach
    DEBUG = True
    if DEBUG:
        server = DebugServer.getInstance()
    output_str = output_str.replace("\\n", "\n")
    file_name = export_final_ifc_and_checks(str(output_str), int(issue_fixing_counter))

    return file_name

def excute_pure_checking(file_name, issue_fixing_counter):
    # debug attach
    DEBUG = True
    if DEBUG:
        server = DebugServer.getInstance()
    pure_checking(str(file_name), int(issue_fixing_counter))

def excute_state_clean():
    # clean the content of state.json when reload the frontend page
    with open(STATE_PATH, "w") as f:
        f.write("{}")

