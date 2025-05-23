import yaml
import os

from pidsmaker.config import TASK_ARGS, AND, OR

def convert_types_to_strings(obj, counter=None):
    if counter is None:
        counter = [0]
    if isinstance(obj, dict):
        return {key: convert_types_to_strings(value, counter) for key, value in obj.items()}
    elif isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], type):
        type_name = obj[0].__name__
        values = obj[1]
        counter[0] += 1
        annotation_id = counter[0]
      
        if not hasattr(convert_types_to_strings, "annotations"):
            convert_types_to_strings.annotations = []
        separator = "<br>"
        values_str = separator.join(values)
        if isinstance(values, AND):
            values_str = "<b>Available options (multi selection)</b>:<br><br>" + values_str
        else:
            values_str = "<b>Available options (one selection)</b>:<br><br>" + values_str
        convert_types_to_strings.annotations.append((annotation_id, values_str))
        return f"{type_name} ({annotation_id})"
    elif isinstance(obj, type):
        return obj.__name__
    return obj

def dict_to_markdown_list(d, indent=0):
    lines = []
    
    lines.append("    " * indent + "<ul>")
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append("    " * indent + f"    <li class='bullet'><span class=\"key\">{key}</span>")
            lines.extend(dict_to_markdown_list(value, indent + 1))
            lines.append("    " * indent + "    </li>")
        else:
            lines.append("    " * indent + f"    <li class='no-bullet'><span class=\"key-leaf\">{key}</span>: <span class=\"value\">{value}</span></li>")

    lines.append("    " * indent + "</ul>")
    return lines

for task, d in TASK_ARGS.items():
    convert_types_to_strings.annotations = []
    counter = [0]
    task_dict = convert_types_to_strings(d, counter)
    
    pwd = os.path.dirname(os.path.realpath(__file__))
    markdown_output_path = os.path.join(pwd, f"args/args_{task}.md")
    os.makedirs(os.path.dirname(markdown_output_path), exist_ok=True)
    markdown_lines = dict_to_markdown_list(task_dict)
    with open(markdown_output_path, "w") as f:
        # class="annotate" required to use md in html
        f.write('<div class="annotate">\n\n')
        f.write("\n".join(markdown_lines))
        f.write('\n\n</div>\n\n')
        # Append annotations in i. format as in mkdocs format
        if hasattr(convert_types_to_strings, "annotations") and convert_types_to_strings.annotations:
            for annotation_id, values_str in convert_types_to_strings.annotations:
                f.write(f"{annotation_id}. {values_str}\n")
