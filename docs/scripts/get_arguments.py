import os

from pidsmaker.config import (
    AND,
    DECODERS_CFG,
    ENCODERS_CFG,
    FEATURIZATIONS_CFG,
    OBJECTIVES_CFG,
    TASK_ARGS,
    Arg,
)


def convert_types_to_strings(obj, counter=None, ignore=[]):
    if counter is None:
        counter = [0]
    if isinstance(obj, dict):
        return {
            key: convert_types_to_strings(value, counter, ignore)
            for key, value in obj.items()
            if key not in ignore
        }
    elif isinstance(obj, Arg):
        type_name = obj.type.__name__
        values = obj.vals
        desc = obj.desc

        if values is None and desc is None:
            return type_name

        counter[0] += 1
        annotation_id = counter[0]

        if not hasattr(convert_types_to_strings, "annotations"):
            convert_types_to_strings.annotations = []

        separator = "<br>"
        values_str = ""
        if desc is not None:
            values_str += desc + separator

        if values is not None:
            if isinstance(values, AND):
                values_str += "<br><b>Available options (multi selection)</b>:<br>"
            else:
                values_str += "<br><b>Available options (one selection)</b>:<br>"
            values_str += separator.join([f"`{v}`" for v in values])

        convert_types_to_strings.annotations.append((annotation_id, values_str))
        return f"{type_name} ({annotation_id})"

    return obj


def dict_to_markdown_list(d, indent=0):
    lines = []

    lines.append("    " * indent + ("<ul>" if len(d) > 0 else ""))
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(
                "    " * indent + f"    <li class='bullet'><span class=\"key\">{key}</span>"
            )
            lines.extend(dict_to_markdown_list(value, indent + 1))
            lines.append("    " * indent + "    </li>")
        else:
            lines.append(
                "    " * indent
                + f'    <li class=\'no-bullet\'><span class="key-leaf">{key}</span>: <span class="value">{value}</span></li>'
            )

    lines.append("    " * indent + ("</ul>" if len(d) > 0 else ""))
    return lines


def write_md(d, filename, ignore=[]):
    convert_types_to_strings.annotations = []
    counter = [0]
    task_dict = convert_types_to_strings(d, counter, ignore=ignore)

    pwd = os.path.dirname(os.path.realpath(__file__))
    markdown_output_path = os.path.join(pwd, f"args/args_{filename}.md")
    os.makedirs(os.path.dirname(markdown_output_path), exist_ok=True)
    markdown_lines = dict_to_markdown_list(task_dict)
    with open(markdown_output_path, "w") as f:
        # class="annotate" required to use md in html
        f.write('<div class="annotate">\n\n')
        f.write("\n".join(markdown_lines))
        f.write("\n\n</div>\n\n")
        # Append annotations in i. format as in mkdocs format
        if (
            hasattr(convert_types_to_strings, "annotations")
            and convert_types_to_strings.annotations
        ):
            for annotation_id, values_str in convert_types_to_strings.annotations:
                f.write(f"{annotation_id}. {values_str}\n")


write_md(FEATURIZATIONS_CFG, "featurizations")
write_md(ENCODERS_CFG, "encoders")
write_md(DECODERS_CFG, "decoders")
write_md(OBJECTIVES_CFG, "objectives", ignore=list(DECODERS_CFG.keys()))

for task, d in TASK_ARGS.items():
    write_md(
        d,
        task,
        ignore=list(DECODERS_CFG.keys())
        + list(ENCODERS_CFG.keys())
        + list(OBJECTIVES_CFG.keys())
        + list(FEATURIZATIONS_CFG.keys()),
    )
