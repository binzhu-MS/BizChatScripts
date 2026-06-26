import os
import pandas as pd
import csv


def get_max_spaces(num_words, max_num_generated_queries):
    """
    Get the maximum number of spaces to insert between words to generate the specified number of queries.

    Parameters:
    num_words (int): The number of words in the query.
    max_num_generated_queries (int): The maximum number of generated queries.

    Returns:
    int: The maximum number of spaces to insert between words.
    """
    max_spaces = 1
    while max_spaces ** (num_words - 1) < max_num_generated_queries:
        max_spaces += 1

    return max_spaces


def process_query(query, max_num_generated_queries=100):
    """
    Generate queries with varying numbers of space characters between adjacent words.

    Parameters:
    query (str): The original query.
    max_num_generated_queries (int): The maximum number of generated queries.

    Returns:
    list: A list of generated queries.
    """
    words = query.split()
    num_words = len(words)

    if (max_num_generated_queries <= 1 or num_words <= 1):
        return [query]

    # Determine the maximum number of spaces to insert between words
    max_spaces = get_max_spaces(num_words, max_num_generated_queries)

    def generate_queries(partial_queries, words, max_spaces, max_num_generated_queries):
        if not words:
            return partial_queries

        if len(partial_queries) >= max_num_generated_queries:
            max_spaces = 1

        if not partial_queries:
            partial_queries = [words[0]]
            return generate_queries(partial_queries, words[1:], max_spaces, max_num_generated_queries)
        else:
            new_partial_queries = []
            num_remaining_queries = len(partial_queries)
            num_generated_queries = 0
            for pq in partial_queries:
                num_remaining_queries -= 1
                for space_count in range(1, max_spaces + 1):
                    new_query = pq + ' ' * space_count + words[0]
                    new_partial_queries.append(new_query)
                    num_generated_queries += 1
                    if max_spaces > 1 and num_generated_queries + num_remaining_queries >= max_num_generated_queries:
                        max_spaces = 1   
                        break

            return generate_queries(new_partial_queries, words[1:], max_spaces,max_num_generated_queries)

    initial_partial_queries = []
    generated_queries = generate_queries(initial_partial_queries, words, max_spaces, max_num_generated_queries)

    return generated_queries

def process_tsv(input_file, output_file, max_num_generated_queries=100):
    """
    Read a TSV file with specified format and repeat each query with variable space chars (to make SEVAL consider them different to work around its restriction of not allowing repeated queries).

    Parameters:
    input_file (str): The path to the input TSV file.
    output_file (str): The path to the output file where the processed data will be saved (in the same format as the input file).
    max_num_generated_queries (int): The maximum number of generated queries.
    """
    # Initialize DataFrame columns
    expected_columns = ['query', 'segment']
    
    # Create an empty DataFrame for output
    dfout = pd.DataFrame(columns=expected_columns)
    
    # Open and read the TSV file
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        
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
            
            query, segment = row
            
            # Generate all possible queries with varying spaces
            processed_queries = process_query(query, max_num_generated_queries)
            print(f"Generated {len(processed_queries)} queries for the original query: {query}")

            for processed_query in processed_queries:
                dfout = pd.concat([dfout, pd.DataFrame([{'query': processed_query, 'segment': segment}])], ignore_index=True)

    # Save the processed DataFrame to the output file
    dfout.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    print(f"Data has been successfully saved to {output_file}")

def main():
    """
    Main function to call the process_tsv with specified input file path.
    """
    # Call the function to extract utterances
    input_file = r'E:\Cloud\OneDrive - Microsoft\Documents\Working\M365Copilot\MultiTurn\Results\Raw_Testdata\Multiturn_Scen1_Personalized_For_BZ.tsv'
    base_name, ext = os.path.splitext(input_file)
    output_file = base_name + "_out" + ext
    max_num_generated_queries = 100
    process_tsv(input_file, output_file, max_num_generated_queries)

if __name__ == "__main__":
    main()