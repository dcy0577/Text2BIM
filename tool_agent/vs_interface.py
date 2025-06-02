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

DEBUG = False
def excute_webpalette_po_coder(input_str, chat_history):
    # debug attach
    if DEBUG:
        DebugServer.getInstance()

    chat_history = chat_history.replace("\\n", "\n")
    output_str, code_result = run_po_coder_agents(str(input_str), str(chat_history))
    
    return output_str, code_result

def excute_webpalette_export(output_str, issue_fixing_counter, chat_history, query):
    # debug attach
    if DEBUG:
        DebugServer.getInstance()

    chat_history = chat_history.replace("\\n", "\n")
    output_str = output_str.replace("\\n", "\n")
    file_name = export_ifc_and_stuffs(str(output_str), int(issue_fixing_counter), str(chat_history), str(query))
    
    return file_name

def excute_webpalette_checking_loop(issue_fixing_counter, original_code_result, code_result, file_name):
    # debug attach
    if DEBUG:
        server = DebugServer.getInstance()
    code_result = code_result.replace("\\n", "\n")
    original_code_result = original_code_result.replace("\\n", "\n")
    output_str, output_code_result = run_agent_checking_loop(int(issue_fixing_counter), str(original_code_result), str(code_result), str(file_name))

    return output_str, output_code_result

def excute_final_ifc_export(output_str, issue_fixing_counter):
    # debug attach
    if DEBUG:
        server = DebugServer.getInstance()
    output_str = output_str.replace("\\n", "\n")
    file_name = export_final_ifc_and_checks(str(output_str), int(issue_fixing_counter))

    return file_name

def excute_pure_checking(file_name, issue_fixing_counter):
    # debug attach
    if DEBUG:
        server = DebugServer.getInstance()
    pure_checking(str(file_name), int(issue_fixing_counter))

def excute_state_clean():
    # clean the content of state.json when reload the frontend page
    with open(STATE_PATH, "w") as f:
        f.write("{}")


# BUG: This is not working with IFC output in Vectorworks as the geometry is not ready before output!
def faceless_execution(msg="Construct a residential building with a rectangular footprint (15m x 10m), a pitched roof and two floors. Create balconies by extending the floor slab outwards from the exterior walls on the first floor. Add doors and windows to each floor. Make sure that the balconies are accessible from the inside.", 
                       history="User: "):
    """
    BUG: This doesnt work with ifc output. as the geometry are not ready before output!
    """
    # Initial processing
    output_str, code_result = excute_webpalette_po_coder(msg, history)
    # final_output = output_str
    
    # Issue fixing loop (max 3 iterations)
    issue_fixing_counter = 0
    fix_code = code_result
    
    while issue_fixing_counter < 3:
        # Export step
        file_name = excute_webpalette_export(
            output_str, issue_fixing_counter, history, msg
        )
        
        if file_name in ("break", "", None):
            break
            
        # Checking loop step
        current_code = code_result if issue_fixing_counter == 0 else fix_code
        output_str2, fix_code = excute_webpalette_checking_loop(
            issue_fixing_counter, code_result, current_code, file_name
        )
        
        if output_str2 in ("break", "", None):
            break
            
        # Update for next iteration
        issue_fixing_counter += 1
        # final_output += output_str2
    
    # Final processing if all 3 iterations completed
    if issue_fixing_counter == 3:
        file_name_final = excute_final_ifc_export(output_str2, issue_fixing_counter)
        
        if file_name_final not in ("break", "", None):
            excute_pure_checking(file_name_final, issue_fixing_counter)
    
    excute_state_clean()

    final_output = "Execution completed successfully. Please check the output files and logs for details."
    vs.AlrtDialog(final_output)
    
    # return final_output