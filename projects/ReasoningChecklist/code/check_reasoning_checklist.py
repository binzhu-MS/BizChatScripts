def get_access_token_msal():
    """Get access token using MSAL"""
    import msal
    
    # client_id = "de90c7b7-0c85-44e4-ae41-50e87bd34553"  # Replace with your client ID
    client_id = "cf5c3336-f1f1-435e-9d31-b32a6638ba15"  # Replace with your client ID
    tenant_id = "72f988bf-86f1-41af-91ab-2d7cd011db47"  # Replace with your tenant ID
    # scopes = ["api://de90c7b7-0c85-44e4-ae41-50e87bd34553/FoundryToolkit.Use"]  # Replace with your scopes
    scopes = ["api://cf5c3336-f1f1-435e-9d31-b32a6638ba15/FoundryToolkit.Use"]  # Replace with your scopes

    # Special redirect URI for Windows broker integration
    redirect_uri = f"ms-appx-web://Microsoft.AAD.BrokerPlugin/{client_id}"
    
    # For a public client application (desktop app)
    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
        allow_broker=True  # Enable broker usage for device compliance
    )
    
    # Try to get token silently first using cached tokens
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(scopes, account=accounts[0])
        if result:
            return result['access_token']
    
    # If silent acquisition fails, do interactive login
    result = app.acquire_token_interactive(scopes, redirect_uri=redirect_uri, port=44436, parent_window_handle=msal.PublicClientApplication.CONSOLE_WINDOW_HANDLE)

    if "access_token" in result:
        return result["access_token"]
    else:
        error = result.get("error")
        error_description = result.get("error_description")
        raise Exception(f"Authentication failed: {error} - {error_description}")

def save_session_results(run_eval_results, original_file_path):
    import json

    if not original_file_path or not run_eval_results:
        print("Error: Missing original file path or runEvalResults.")
        return

    try:
        # Load the original JSON file
        with open(original_file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # Replace the "resultsV2" property with runEvalResults
        combine_results(original_data, run_eval_results)

        # Save the updated JSON back to the file
        with open(original_file_path, 'w', encoding='utf-8') as f:
            json.dump(original_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Error updating 'resultsv2': {str(e)}")

def combine_results(original_data, run_eval_results):
    """
    Combine original data with run evaluation results.
    This function assumes that original_data is a dictionary and run_eval_results is a list of dictionaries.
    """
    if not isinstance(original_data, dict) or not isinstance(run_eval_results, list):
        raise ValueError("Invalid input types: expected dict for original_data and list for run_eval_results")

    session_input_indices = {}
    for result in original_data.get("resultsv2", []):
        session_input_index = result.get("sessionInputIndex")
        if session_input_index is not None:
            session_input_indices[session_input_index] = result

    for result in run_eval_results:
        sessionInputIndex = result.get("sessionInputIndex")

        if sessionInputIndex is not None and sessionInputIndex not in session_input_indices:
            original_data.setdefault("resultsv2", []).append(result)
        else:
            session_input_result = session_input_indices.get(sessionInputIndex)

            # If no matching entry exists (e.g. sessionInputIndex is None), just append
            if session_input_result is None:
                original_data.setdefault("resultsv2", []).append(result)
                continue

            # Merge Foundry results (keyed by promptId)
            for key in ["inference", "evaluation"]:
                existing = session_input_result.get(key)
                incoming = result.get(key)
                if existing is None or incoming is None:
                    continue
                original_prompt_ids = {item.get("promptId") for item in existing if "promptId" in item}

                for item in incoming:
                    prompt_id = item.get("promptId")
                    if prompt_id not in original_prompt_ids:
                        existing.append(item)
                    else:
                        for original_item in existing:
                            if original_item.get("promptId") == prompt_id:
                                original_item.update(item)

            # Merge Sydney results (keyed by sydneyId)
            for key in ["sydneyInference", "sydneyEvaluation"]:
                incoming = result.get(key)
                if incoming is None:
                    continue
                existing = session_input_result.get(key)
                if existing is None:
                    session_input_result[key] = incoming
                else:
                    original_sydney_ids = {item.get("sydneyId") for item in existing if "sydneyId" in item}

                    for item in incoming:
                        sydney_id = item.get("sydneyId")
                        if sydney_id not in original_sydney_ids:
                            existing.append(item)
                        else:
                            for original_item in existing:
                                if original_item.get("sydneyId") == sydney_id:
                                    original_item.update(item)
                                
    return original_data

def extract_number_start_or_end(s):
    """Extract a number from the start or end of a string."""
    if s is None:
        return None
    try:
        # Try to parse the first line
        first_line_value = float(s.split('\n')[0].strip())
        if not first_line_value != first_line_value:  # Check for NaN
            return first_line_value
    except ValueError:
        pass  # Ignore parsing errors for the first line

    # If first line parsing fails, try the last line
    lines = [line.strip() for line in s.split('\n') if line.strip()]
    if not lines:
        return None
    try:
        last_line_value = float(lines[-1])
        if not last_line_value != last_line_value:  # Check for NaN
            return last_line_value
    except ValueError:
        pass  # Ignore parsing errors for the last line

    return None


def generate_results_md(output_folder, sessions_info=None, combined_sessions_folder=None):
    """
    Generate results.md file from completed sessions.
    
    Args:
        output_folder: Path to the folder containing session result JSON files and sessions_info.json
        sessions_info: Optional dict with session info. If None, will be loaded from sessions_info.json
        combined_sessions_folder: Optional path to combined sessions folder. If None, will look for it relative to output_folder
    """
    import os
    import json
    from collections import defaultdict
    
    # Load sessions_info if not provided
    if sessions_info is None:
        sessions_info_path = os.path.join(output_folder, "sessions_info.json")
        if not os.path.exists(sessions_info_path):
            print(f"Error: sessions_info.json not found in {output_folder}")
            return
        try:
            with open(sessions_info_path, "r", encoding="utf-8") as f:
                sessions_info = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing sessions_info.json: {str(e)}")
            return
    
    session_info_list = sessions_info.get("sessions", [])
    if not session_info_list:
        print("Error: No session information found in sessions_info.json")
        return
    
    # Determine combined_sessions_folder if not provided
    if combined_sessions_folder is None:
        combined_sessions_folder = os.path.join(os.path.dirname(output_folder), "combined-sessions")
    
    # Read prompt and criteria data from combined sessions
    prompt_data = {}
    criteria_data = {}
    
    if os.path.exists(combined_sessions_folder):
        for file in os.listdir(combined_sessions_folder):
            if file.endswith('.json'):
                file_path = os.path.join(combined_sessions_folder, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        combined_data_file = json.load(f)
                        if combined_data_file:
                            prompt_list = combined_data_file.get("prompts", [])
                            for prompt in prompt_list:
                                prompt_id = prompt.get("id")
                                if prompt_id and prompt_id not in prompt_data:
                                    prompt_data[prompt_id] = {"title": prompt.get("title")}

                            eval_strat = combined_data_file.get("evaluationStrategy")
                            if eval_strat:
                                criteria_list = eval_strat.get("criteriaList", [])
                                for criteria in criteria_list:
                                    criteria_id = criteria.get("id")
                                    if criteria_id and criteria_id not in criteria_data:
                                        criteria_data[criteria_id] = {
                                            "name": criteria.get("name"),
                                            "weight": criteria.get("weight")
                                        }
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
    else:
        print(f"Warning: Combined sessions folder not found at {combined_sessions_folder}")
    
    # Build session_info dict for quick lookup
    session_info = {}
    for session in session_info_list:
        session_id = session.get("id")
        if session_id:
            session_info[session_id] = {
                "filename": session.get("name"),
                "originalFilePath": session.get("originalFilePath")
            }
    
    # Process results from JSON files
    results_table = []
    for file in os.listdir(output_folder):
        if not file.endswith('.json') or file == "sessions_info.json":
            continue

        file_path = os.path.join(output_folder, file)
        file_base_name = os.path.splitext(file)[0]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                session_response_content = json.load(f)

            eval_results = session_response_content.get("runEvalResults", [])
            original_file_path = session_info.get(file_base_name, {}).get("originalFilePath")
            
            # Save results to original file if path exists
            if original_file_path and eval_results:
                save_session_results(eval_results, original_file_path)
                # Read the updated results from original file
                try:
                    with open(original_file_path, 'r', encoding='utf-8') as f:
                        original_data = json.load(f)
                    eval_results = original_data.get("resultsv2", [])
                except Exception as e:
                    print(f"Warning: Could not read updated original file: {e}")
            
            # Process evaluation results
            for result in eval_results:
                # Process Foundry evaluations (keyed by promptId)
                evaluations = result.get("evaluation", [])
                for evaluation in evaluations:
                    for criteria in evaluation.get('criteria', []):
                        dataResults = criteria.get('dataItemsEvaluationOutputs', {})
                        count = 0
                        total = 0.0
                        for _, value in dataResults.items():
                            output = value.get("scriptOutput") if isinstance(value, dict) else None
                            try:
                                output_value = extract_number_start_or_end(output)
                                if output_value is not None:
                                    count += 1
                                    total += output_value
                            except (ValueError, TypeError):
                                pass  # Silently ignore parsing errors
                    
                        average = total / max(1, count)
                        results_table.append({
                            "Name": session_info.get(file_base_name, {}).get('filename', 'Unknown'),
                            "Criteria": criteria_data.get(criteria.get('criteriaId'), {}).get('name', 'Unknown'),
                            "Link": f"https://m365playground.prod.substrateai.microsoft.net/eval/rating-based/v2?session={file_base_name}",
                            prompt_data.get(evaluation.get('promptId'), {}).get('title', 'Unknown'): f"{average:.2f} ({count} rows)",
                        })

                # Process Sydney evaluations (keyed by sydneyId)
                sydney_evaluations = result.get("sydneyEvaluation", [])
                for sydney_eval in sydney_evaluations:
                    sydney_id = sydney_eval.get('sydneyId', 'Unknown')
                    column_title = sydney_id.capitalize() if sydney_id else 'Unknown'
                    for criteria in sydney_eval.get('criteria', []):
                        dataResults = criteria.get('dataItemsEvaluationOutputs', {})
                        count = 0
                        total = 0.0
                        for _, value in dataResults.items():
                            output = value.get("scriptOutput") if isinstance(value, dict) else None
                            try:
                                output_value = extract_number_start_or_end(output)
                                if output_value is not None:
                                    count += 1
                                    total += output_value
                            except (ValueError, TypeError):
                                pass  # Silently ignore parsing errors
                    
                        average = total / max(1, count)
                        results_table.append({
                            "Name": session_info.get(file_base_name, {}).get('filename', 'Unknown'),
                            "Criteria": criteria_data.get(criteria.get('criteriaId'), {}).get('name', 'Unknown'),
                            "Link": f"https://m365playground.prod.substrateai.microsoft.net/eval/rating-based/v2?session={file_base_name}",
                            column_title: f"{average:.2f} ({count} rows)",
                        })
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Merge results by Criteria + Name
    combined = defaultdict(dict)
    for entry in results_table:
        name = entry.get("Criteria", "") + " - " + entry.get("Name", "")
        combined[name].update(entry)
    
    merged_data = list(combined.values())
    
    # Group sessions by directory
    grouped_sessions = defaultdict(list)
    for session in session_info_list:
        original_file_path = session.get("originalFilePath")
        if original_file_path:
            directory = os.path.dirname(original_file_path)
            grouped_sessions[directory].append(session)
    
    # Write results.md file
    output_file = os.path.join(output_folder, "results.md")
    if os.path.exists(output_file):
        os.remove(output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for directory, sessions in grouped_sessions.items():
            f.write(f"# Directory: {os.path.basename(directory)}\n\n")

            # Get headers from first session's data
            first_session = sessions[0] if sessions else {}
            merged_data_for_first_session = [item for item in merged_data if item.get("Name") == first_session.get("name")]
            headers = list(merged_data_for_first_session[0].keys()) if merged_data_for_first_session else []
            
            # Ensure standard columns come first
            headers = ["Name", "Criteria", "Link"] + [key for key in headers if key not in ["Name", "Criteria", "Link"]]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")

            # Write rows for sessions in the current group
            for session in sessions:
                for item in merged_data:
                    if item.get("Name") == session.get("name"):
                        row = [item.get("Name", ""), item.get("Criteria", ""), f"[link]({item.get('Link', '')})"]
                        for key in headers[3:]:
                            value = item.get(key)
                            row.append(str(value) if value else "")
                        f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
    
    print(f"Results written to: {output_file}")

def process_sessions(input_folder):
    import os
    import sys
    import time
    import json
    import requests
    from datetime import datetime
    from pathlib import Path
    from collections import defaultdict

    bearer_token = get_access_token_msal() 
    
    sessions_info_path = os.path.join(input_folder, "sessions_info.json")
    
    if not os.path.exists(sessions_info_path):
        print(f"Error: sessions_info.json not found in {input_folder}")
        sys.exit(1)
    
    # Read all session information from the JSON file
    session_info_list = []
    with open(sessions_info_path, "r", encoding="utf-8") as f:
        try:
            sessions_info = json.load(f)
            session_info_list = sessions_info.get("sessions", [])
            if not session_info_list:
                print("Error: No session information found in sessions_info.json")
                sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing sessions_info.json: {str(e)}")
            sys.exit(1)
    
    # Extract session IDs and their corresponding filenames and original file paths
    session_info = {}
    for session in session_info_list:
        session_id = session.get("id")
        filename = session.get("name")
        original_filePath = session.get("originalFilePath")
        if session_id:
            session_info[session_id] = {
                "filename": filename,
                "originalFilePath": original_filePath
            }
    
    session_ids = list(session_info.keys())
    
    # Track statuses
    session_statuses = {id: "Pending" for id in session_ids}
    final_statuses = {}

    print("\nStarting live status polling every 1 minute...")

    while "Running" in session_statuses.values() or "Queued" in session_statuses.values() or "Pending" in session_statuses.values():
        for id in session_statuses:
            if id in final_statuses:
                continue
            
            try:
                response = requests.get(
                    f"https://m365playground.prod.substrateai.microsoft.net/api/v2/evaluation/async?jobId={id}",
                    headers={"Authorization": f"Bearer {bearer_token}"}
                )
                response.raise_for_status()
                
                parsed = response.json()
                status = parsed.get("status")
                
                session_statuses[id] = status
                
                if status not in ["Running", "Queued"]:
                    final_statuses[id] = status
                    jobId = parsed.get("jobId")
                    new_file_name = f"{jobId}.json"
                    new_file_path = os.path.join(input_folder, new_file_name)
                    os.remove(new_file_path) if os.path.exists(new_file_path) else None
                    if not os.path.exists(new_file_path):
                        json_body = json.dumps(parsed)
                        with open(new_file_path, 'w', encoding='utf-8') as f:
                            f.write(json_body)
                    
            except Exception:
                session_statuses[id] = "ERROR"
                final_statuses[id] = "ERROR"
        
        # Clear screen (cross-platform)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"Live Session Status (updated: {current_time})\n")
        
        # Print the statuses as a table
        max_id_len = max(len(id) for id in session_statuses)
        max_filename_len = max(len(str(session_info.get(id, {}).get("filename", "N/A"))) for id in session_statuses)
        # Cap filename length to prevent excessively wide tables
        max_filename_len = min(max_filename_len, 25)

        print(f"{'FILENAME':<{max_filename_len+2}} {'ID':<{max_id_len+2}} {'STATUS':<10} LINK")
        print("-" * (max_id_len + max_filename_len + 75))
        for id, status in sorted(session_statuses.items()):
            filename = session_info.get(id, {}).get("filename", "N/A")
            # If filename is too long, truncate it and add ...
            if filename and len(filename) > max_filename_len:
                display_filename = filename[:max_filename_len-3] + "..."
            else:
                display_filename = filename or "N/A"

            sessionlink = f"https://m365playground.prod.substrateai.microsoft.net/eval/rating-based/v2?session={id}"
            print(f"{display_filename:<{max_filename_len+2}} {id:<{max_id_len+2}} {status:<10} {sessionlink}")
        
        if "Running" in session_statuses.values() or "Queued" in session_statuses.values() or "Pending" in session_statuses.values():
            time.sleep(60)
    
    print("\n✅ All sessions completed.\n")

    # Check for any missing JSON files for completed sessions
    print("Checking for missing JSON files for completed sessions...")
    missing_files = []
    for id in session_ids:
        # Check if a JSON file exists for this session
        json_exists = False
        for file in os.listdir(input_folder):
            if file.endswith('.json') and file.startswith(id):
                json_exists = True
                break
        
        if not json_exists:
            missing_files.append(id)
    
    # Fetch missing files
    if missing_files:
        print(f"Found {len(missing_files)} sessions without JSON files. Fetching data...")
        for id in missing_files:
            try:
                response = requests.get(
                    f"https://m365playground.prod.substrateai.microsoft.net/api/v2/evaluation/async?jobId={id}",
                    headers={"Authorization": f"Bearer {bearer_token}"}
                )
                response.raise_for_status()
                
                parsed = response.json()
                status = parsed.get("status")
                
                if status not in ["Running", "Queued", "Pending"]:
                    jobId = parsed.get("jobId")
                    new_file_name = f"{jobId}.json"
                    new_file_path = os.path.join(input_folder, new_file_name)
                    json_body = json.dumps(parsed)
                    with open(new_file_path, 'w', encoding='utf-8') as f:
                        f.write(json_body)
                    print(f"Downloaded completed session: {jobId} (Status: {status})")
                else:
                    print(f"Session {id} is still in status: {status}")
            
            except Exception as e:
                print(f"Error fetching session {id}: {str(e)}")
    else:
        print("All completed sessions have JSON files.")   

    # Generate the results markdown file using the unified function
    generate_results_md(input_folder)

def check_reasoning_checklist():
    import argparse

    parser = argparse.ArgumentParser(description="Reasoning Change Checklist Runner")
    parser.add_argument('-sessions', type=str, required=True, help='Path to the folder containing session files')
    args = parser.parse_args()

    process_sessions(args.sessions)

if __name__ == "__main__":
    check_reasoning_checklist()