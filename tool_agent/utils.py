import base64
from mimetypes import guess_type
import types
import vs
import re

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def safe_encode(obj):
    if isinstance(obj, types.ModuleType):
        return None  # or return a string like '<<module>>'
    try:
        return obj.__dict__
    except AttributeError:
        return str(obj)  # or return None 

def delete_all_in_model():
    # delete all objects in the model
    h_list = []
    def get_all_handles(h):
        h_list.append(h)
    # https://developer.vectorworks.net/index.php/VS:Search_Criteria#Multiple_Search_Terms
    vs.ForEachObject(get_all_handles, "ALL")
    for h in h_list:
        vs.DelObject(h)

def remove_last_human_message_with_regex(text):
    # Regex pattern to find a 'Human:' message followed by any text, including new lines up to the end of the string
    # and ensuring it's the last thing in the string with optional preceding whitespace or line breaks
    pattern = r"\s*\nUser: [^\n]*\s*\Z|\A\s*User: [^\n]*\s*\Z"
    
    # Use re.sub to replace the found pattern with an empty string
    new_text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    
    return new_text

def clean_code_for_chat_extend(result):
    lines = result.split("\n")
    idx = 0
    explanations = []
    codes = []
    while idx < len(lines):
        while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
            idx += 1
        explanation = "\n".join(lines[:idx]).strip()
        if explanation:  # Only append if there is content
            explanations.append(explanation)
        if idx == len(lines):
            break
        idx += 1
        start_idx = idx
        while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
            idx += 1
        code = "\n".join(lines[start_idx:idx]).strip()
        if code:  # Only append if there is content
            codes.append(code)
        lines = lines[idx+1:]
        idx = 0
    return explanations, codes
 
        



    