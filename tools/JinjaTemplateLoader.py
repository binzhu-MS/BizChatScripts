import os
from jinja2 import Environment, FileSystemLoader

def construct_prompt(input_meta_template_file, input_prompt_file, input_utterance=None):
    """
    It loads meta template and the prompt to analyze/revise and construct a final prompt for 
    analyzing/revising the prompt.

    Parameters:
    input_meta_template_file (str): The path to the input meta template file.
    input_prompt_file (str): The path to prompt file to be analyzed/revised.
    output_prompt_file (str): The path to the output prompt file for analyzing/revising the input prompt.
    """
    
    # Extract the directory from the input_meta_template_file
    template_dir = os.path.dirname(input_meta_template_file)
    template_file = os.path.basename(input_meta_template_file)
    
    # Create an Environment object with a FileSystemLoader
    env = Environment(loader=FileSystemLoader(template_dir))

    # Load a template file by its name
    template = env.get_template(template_file)

    # load and read the prompt file
    with open(input_prompt_file, 'r', encoding='utf-8') as file:
        input_prompt = file.read()

    # Replace the variable "utterance" in input_prompt with input_utterance
    if input_utterance is not None:
        prompt_template = Environment().from_string(input_prompt)
        input_prompt = prompt_template.render(utterance=input_utterance)

    # Render the template with input prompt
    output = template.render(prompt=input_prompt)

    return output

def main():
    """
    Main function to call the construct_prompt function with specified input and output file paths.
    """
    meta_template = "Templates/Meta_prompt.jinja"
    prompt_file = "Prompts/Experiment_Calendar_cand_7.md"
    output_file = "Prompts/Output_Prompt.txt"
    utterance =  "List the messages where I was mentioned in the objectives"
    
    output = construct_prompt(meta_template, prompt_file, utterance)

    # Write the output prompt to a file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(output)

    print(f"Done Successfully! Saved output prompt to: {output_file}.")

if __name__ == "__main__":
    main()
    