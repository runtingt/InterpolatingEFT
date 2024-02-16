"""
Converts a .yaml file to a html table
"""

import yaml
from typing import Dict, Any

css_style = """
<style>
    .custom-table {
        margin: 10px;
        padding: 10px;
        border: 1px solid #dddfe1;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    table {
        border-collapse: collapse;
        font-family: Tahoma, Geneva, sans-serif;
        width: 100%;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    table td {
        padding: 15px;
    }
    table thead td {
        background-color: #54585d;
        color: #ffffff;
        font-weight: bold;
        font-size: 13px;
        border: 1px solid #54585d;
    }
    table tbody td {
        color: #636363;
        border: 1px solid #dddfe1;
    }
    table tbody tr {
        background-color: #f9fafb;
    }
    table tbody tr:nth-child(odd) {
        background-color: #ffffff;
    }
</style>
"""
wrapper = f"<div class='custom-table'>{css_style}"

def dictToHTML(data: Dict[str, Any], indent: int=0, 
               from_top: bool=True) -> str:
    """
    Converts a dictionary to html

    Args:
        data (Dict[str, Any]): The data to convert
        indent (int, optional): The indent level. Defaults to 0.
        from_top (bool, optional): Whether the calling function had a zero indent.\
            Defaults to True.

    Returns:
        str: The dictionary as html
    """
    html = ''
    for key, value in data.items():
        # Header
        if indent == 0 and from_top:
            html += "\n<table border=1>\n"
            html += r"<tr>"
            html += f"<td style='font-weight: bold;' colspan='2'>{key}</td>"
            html += r"</tr>"+'\n'
        elif isinstance(value, dict):
            html += r"<tr>"
            html += f"<td colspan='2'>{'&nbsp'*indent*6}{key}</td>"
        else:
            html += r"<tr>"
            html += f"<td>{'&nbsp'*indent*6}{key}</td>"
        if isinstance(value, dict):
            html += dictToHTML(value, indent=indent+1, from_top=False)
        else:
            html += f"<td>{value}</td>"
            html += r"</tr>"+'\n'
    
    if from_top:
        html += r"</table>"+'\n'

    return html

def toTable(file, name) -> None:
    """
    Converts a .yaml file to a html table

    Args:
        file (str): The path to the file
        name (str): The name of the output directory
    """
    with open(file, 'r') as yaml_file:
        try:
            yaml_data = yaml.safe_load(yaml_file)
            if isinstance(yaml_data, dict):
                html_table = dictToHTML(yaml_data)
                with open(f"out/{name}/config.html", 'w') as html_file:
                    html_file.write(wrapper+html_table+"</div>")
                print("HTML table generated successfully.")
            else:
                print("Invalid YAML format. Please provide a dictionary.")
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")

