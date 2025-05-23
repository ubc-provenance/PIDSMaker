import yaml
import os

from pidsmaker.config import TASK_ARGS

def convert_types_to_strings(obj):
    if isinstance(obj, dict):
        return {key: convert_types_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, type):
        return obj.__name__
    return obj

def dict_to_markdown_list(d, indent=0):
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append("    " * indent + f"- <span class=\"key\">**{key}**</span>")
            lines.extend(dict_to_markdown_list(value, indent + 1))
        else:
            lines.append("    " * indent + f"- <span class=\"key\">**{key}**</span>: <span class=\"value\">{value}</span>")
    return lines

config_dict = convert_types_to_strings(TASK_ARGS)

for task, d in config_dict.items():
    markdown_output_path = f"./args/args_{task}.md"
    os.makedirs(os.path.dirname(markdown_output_path), exist_ok=True)
    markdown_lines = dict_to_markdown_list(d)
    with open(markdown_output_path, "w") as f:
        f.write("\n".join(markdown_lines))
