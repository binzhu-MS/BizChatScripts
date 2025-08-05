import os.path
from . import mirror

def get(project_name, version):
    """
    Load a prompt by project name and version.
    
    Args:
        project_name: Name of the project (e.g., 'text_summarizer', 'simple_scorer')
        version: Version string (e.g., '0.1.0')
    
    Returns:
        Parsed prompt dictionary with messages
    """
    # Get the project root directory (go up from llms/prompts to project root)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    prompts_dir = os.path.join(project_root, "prompts", project_name)
    
    prompt_file = os.path.join(prompts_dir, f"v{version}.md")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, encoding="utf-8") as f:
        return mirror.de_mirror(f.read())
