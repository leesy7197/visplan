from collections import defaultdict
from copy import deepcopy
import fnmatch
import logging
import re
from typing import Dict
import base64
import os
import io
import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def change_directory(directory):
    current_directory = os.getcwd()
    
    try:

        print("Changing directory to:", directory)
        os.chdir(directory)
        yield 
    finally:

        print("Changing directory back to:", current_directory)
        os.chdir(current_directory)

def get_workspace_structure(work_directory):

    def generate_directory_structure(path):
        result = {'name':os.path.basename(path)}
        if os.path.isdir(path):
            result['type'] = 'directory'
            result['children'] = [generate_directory_structure(os.path.join(path,child)) for child in os.listdir(path)]
        else:
            result['type'] = 'file'
        return result
    return generate_directory_structure(work_directory)

def fill_in_placeholders(prompt_messages,placeholders: dict):

    filled_messages = deepcopy(prompt_messages)
    
    for key, value in placeholders.items():
        if value is not None:
            filled_messages = filled_messages.replace("{{" + str(key) + "}}", str(value))
    return filled_messages

def _check_ignorement(path:str,ignored_list)->bool:
    for pattern in ignored_list:
        if fnmatch.fnmatch(path,pattern):
            return True
    return False

def print_filesys_struture(work_directory,return_root=False,max_entry_nums_for_level=100,ignored_list=[])->str:
    full_repr = ''
    if return_root:
        full_repr += f'Global Root Work Directory: {work_directory}\n'

    folder_counts =  defaultdict(lambda: 0)
    for root, dirs, files in os.walk(work_directory):
        if _check_ignorement(root,ignored_list):
            continue
        level = root.replace(work_directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        
        folder_counts[root] += 1
        if folder_counts[root] > max_entry_nums_for_level:
            full_repr += f'{indent}`wrapped`\n'
        
        full_repr += f'{indent}- {os.path.basename(root)}/\n'
        
        idx = 0
        subindent = ' ' * 4 * (level + 1) + '- '
        for f in files:
            if _check_ignorement(f,ignored_list):
                continue
            
            idx += 1
            if idx > max_entry_nums_for_level:
                full_repr += f'{subindent}`wrapped`\n'
                break
            full_repr += f'{subindent}{f}\n'


    return full_repr


def get_code(response):

    all_python_code_blocks_pattern = re.compile(r'```python\s*([\s\S]+?)\s*```', re.MULTILINE)

    all_code_blocks = all_python_code_blocks_pattern.findall(response)
    all_code_blocks_combined = '\n'.join(all_code_blocks)
    return all_code_blocks_combined


def run_code(workspace, code_file, log_file=None)->str:
    if log_file is None:
        log_file = code_file + '.log'
    with change_directory(workspace):

        os.system(f'python "{code_file}" > "{log_file}" 2>&1')
        with open(log_file,'r') as f:
            log = f.read()

    return log


def is_run_code_success(log):
    if 'Traceback (most recent call last):' in log or 'Error:' in log:
        return False
    else:
        return True

def get_error_message(log):
    if 'Traceback (most recent call last):' in log:
        return log.split('Traceback (most recent call last):')[1]
    elif 'Error:' in log:
        return log.split('Error:')[1]
    else:
        return 'Unknown Error'

def print_chat_message(messages):
    for message in messages:
        logging.info(f"{message['role']}: {message['content']}")
        
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def base64_to_image(base64_string, output_path):
    with open(output_path, "wb") as img_file:
        img_file.write(base64.b64decode(base64_string))

# def code_to_image(code, img_save_path):
#     import matplotlib.pyplot as plt
#     exec_globals = {"plt": plt, "io": io}
#     exec_locals = {}
#     code +"\nplt.savefig({img_save_path}) \nplt.close"
#     try:
#         plt.figure()
#         exec(code, exec_globals, exec_locals)
        
#         img_buffer = io.BytesIO()
#         plt.savefig(img_buffer, format='png') 
#         plt.close()         

#         img_buffer.seek(0)
#         img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
#         return img_base64
#     except Exception as e:
#         return str(e)

def code_to_image(code, id, img_save_path):
    import matplotlib.pyplot as plt
    exec_globals = {"plt": plt, "io": io}
    exec_locals = {}
    # code_n = code + f"\nplt.savefig(r'{img_save_path}/{id}.png') \nplt.close()"
    code_n = code.replace("plt.show()", f"plt.savefig('{img_save_path}')")
    # print(code_n)
    try:
        exec(code_n, exec_globals, exec_locals)
        print("Save Image Successfully!")
        return True
    except Exception as e:
        print(f"Error during Save : {str(e)}")
        return False