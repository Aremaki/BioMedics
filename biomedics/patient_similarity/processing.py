import re
from pathlib import Path


def parse_clinical_case(file_path: Path):
    """
    Parses a clinical case file to extract the clinical text, CIM-10 codes, and axes.

    Args:
        file_path (Path): The path to the clinical case file.

    Returns:
        tuple: A tuple containing the clinical text, a list of CIM-10 codes, and a list of axes.
               Returns (None, None, None) if the file cannot be parsed.
    """
    text = file_path.read_text(encoding="utf-8")

    # Split the text to isolate the clinical notes
    parts = re.split(r"-----Codes CIM-10-----", text, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None, None, None

    clinical_text = parts[0].strip()

    # Further split to get CIM-10 codes and Axes
    remaining_text = parts[1]
    sub_parts = re.split(r"---Axes---", remaining_text, flags=re.IGNORECASE)
    if len(sub_parts) < 2:
        return clinical_text, [], []

    cim10_section = sub_parts[0].strip()
    axes_section = sub_parts[1].strip()

    # Extract CIM-10 codes
    cim10_codes = [line.strip() for line in cim10_section.split("\n") if line.strip()]

    # Extract axes
    axes = [line.strip() for line in axes_section.split("\n") if line.strip()]

    return clinical_text, cim10_codes, axes
