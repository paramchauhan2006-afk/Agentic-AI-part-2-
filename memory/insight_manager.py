import os
import json

# Ensure path is always absolute relative to the location of this script
INSIGHTS_FILE = os.path.join(os.path.dirname(__file__), "insights.json")

def load_insights() -> list[str]:
    """
    Reads the permanent insights rules list from memory/insights.json.
    If the file does not exist, or is empty or invalid, returns an empty list.
    """
    if not os.path.exists(INSIGHTS_FILE):
        return []
    
    try:
        with open(INSIGHTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            return []
    except (json.JSONDecodeError, IOError):
        return []

def save_insight(new_rule: str):
    """
    Appends a new engineering rule to memory/insights.json, safely
    creating parent directories and files if necessary.
    """
    # Load current rules
    rules = load_insights()
    
    # Avoid duplicate rules
    if new_rule not in rules:
        rules.append(new_rule)
        
    # Write back to JSON file
    os.makedirs(os.path.dirname(INSIGHTS_FILE), exist_ok=True)
    try:
        with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[ERROR] Failed to save rule to {INSIGHTS_FILE}: {e}")
