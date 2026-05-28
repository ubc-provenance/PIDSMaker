C = {
    # Benign (Semantic Coloring) - Brightened
    "benign_process": [0.8, 0.5, 1.0, 0.35],  # Bright Purple
    "benign_file": [0.3, 1.0, 0.7, 0.30],  # Bright Teal
    "benign_netflow": [0.5, 0.8, 1.0, 0.40],  # Bright Blue
    # Detected Malicious
    "det_process": [1.0, 0.2, 0.2, 0.9],  # Red
    "det_file": [1.0, 0.8, 0.0, 0.9],  # Yellow
    "det_netflow": [0.2, 0.8, 1.0, 0.9],  # Cyan
    # Undetected Malicious
    "undet_process": [1.0, 0.6, 0.2, 0.9],  # Orange
    "undet_file": [1.0, 1.0, 0.4, 0.9],  # Light Yellow
    "undet_netflow": [0.2, 0.8, 1.0, 0.9],  # Cyan
    "default": [0.5, 0.5, 0.5, 0.1],
}

def get_color(p):
    ptype = (p.get("type") or "").lower()
    label = p.get("label", 0)
    det = p.get("detection_status", 0)

    is_process = "process" in ptype or "subject" in ptype
    is_file = "file" in ptype
    is_netflow = "netflow" in ptype

    if label == 0:
        if is_process:
            return C["benign_process"]
        if is_file:
            return C["benign_file"]
        if is_netflow:
            return C["benign_netflow"]
        return C["default"]

    if det in (0, 1):  # Detected (or ground truth only)
        if is_process:
            return C["det_process"]
        if is_file:
            return C["det_file"]
        if is_netflow:
            return C["det_netflow"]
        return C["det_process"]

    if det == 2:  # Undetected
        if is_process:
            return C["undet_process"]
        if is_file:
            return C["undet_file"]
        if is_netflow:
            return C["undet_netflow"]
        return C["undet_process"]

    return C["default"]
