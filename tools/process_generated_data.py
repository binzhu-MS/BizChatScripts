import os
import json
import pandas as pd
import csv
 

# Unfinished code

def process_csv(input_file, output_file):
    """
    Read a CSV file with specified format and process it into a DataFrame.

    Parameters:
    input_file (str): The path to the input CSV file.
    output_file (str): The path to the output file where the processed data will be saved.
    """
    # Initialize DataFrame columns
    expected_columns = ['query', 'segment']
    
    # Create an empty DataFrame
    df = pd.DataFrame(columns=expected_columns)
    
    # Open and read the CSV file
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Read the header row
        header = next(reader) 
        columns = [col.strip() for col in header]  # Remove leading/trailing whitespaces

        # Check if the input file has the expected columns
        if columns != expected_columns:
            raise ValueError(f"Expected columns: {expected_columns}, but got: {columns}")
        
        # Loop through the input data and store them in the DataFrame
        for row in reader:
            # Ignore empty lines
            if not row:
                continue
            
            # Trim leading and trailing whitespaces for each column value
            row = [col.strip() for col in row]

            # Initialize all column values to empty strings or suitable default values
            row_data = {col: '' for col in columns}
            
             # Assign values from the CSV row to the DataFrame row
            query = row[0]
            if query.startswith('"') and query.endswith('"'):
                query = query[1:-1].replace('""', '"')  # Remove surrounding quotes and handle escaped quotes
            row_data['query'] = query
            row_data['segment'] = row[1]
            
            # Append the row data to the DataFrame
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Data has been successfully saved to {output_file}")




 
def main():
    """
    Main function to call the process_json with specified input file path.
    """
    # Call the function to extract utterances
    #input_file = r'E:\Cloud\OneDrive - Microsoft\Documents\Working\M365Copilot\MultiTurn\Results\Raw_Testdata\Multiturn_Scen1_raw.json'
    input_file = r'C:\Users\binzhu\Downloads\test.json'
    num_repeat = 1
    process_json(input_file, num_repeat)


input_file = 'path/to/your/input_file.csv'
output_file = 'path/to/your/output_file.csv'
process_csv(input_file, output_file)


if __name__ == "__main__":
    main()