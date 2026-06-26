import os
import json
import pandas as pd
 
def process_json(input_file, evaluation_key):
    """
    Extract results from each item in the JSON data and save to an excel file.

    Parameters:
    input_file (str): The path to the input JSON file.
    evaluation_key (str): The evaluation_key keyword in the input JSON file where scores reside.
    """
    # Open and read the JSON file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
 
    # Initialize DataFrame columns
    columns = ['input_parameters']
    for item in data:
        for result in item['results']:
            name = result['name']
            if f'{name}_output' not in columns:
                columns.append(f'{name}_output')
            if f'{name}_score' not in columns:
                columns.append(f'{name}_score')
 
    # Create an empty DataFrame
    df = pd.DataFrame(columns=columns)
 
    # Loop through specific elements and store them in the DataFrame
    for item in data:
        input_parameters = item['input']['parameters']
        # Initialize all column values to empty strings
        row_data = {col: '' for col in columns}
        row_data['input_parameters'] = input_parameters

        # Extract data and store it in the DataFrame
        for result in item['results']:
            output = result['output'].strip().replace('\n', ' ')
            name = result['name']
            row_data[f'{name}_output'] = output
            if evaluation_key in result['evaluations'] and 'score' in result['evaluations'][evaluation_key]:
                row_data[f'{name}_score'] = result['evaluations'][evaluation_key]['score']
        
        # Append the row data to the DataFrame        
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
 
    # Derive output file name from input file name
    output_file = os.path.splitext(input_file)[0] + '.xlsx'
    
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
 
    print(f"Data has been successfully saved to {output_file}")
 
def main():
    """
    Main function to call the process_json with specified input and output file paths.
    """
    # Call the function to extract results
    input_file = r'E:\Cloud\OneDrive - Microsoft\Documents\Working\M365Copilot\MultiTurn\Results\Multiturn_Scen_1_results_3.json'
    #evaluation_key = 'Accuracy'
    evaluation_key = r'ReasoningLeoEnterprise (0.2.7 - Foundry Hops)'
    process_json(input_file, evaluation_key)

if __name__ == "__main__":
    main()