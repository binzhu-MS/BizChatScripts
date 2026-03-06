def get_access_token_msal():
    """Get access token and username using MSAL"""
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
            # Extract username and remove domain if present
            username = extract_username_without_domain(result)
            return result['access_token'], username
    
    # If silent acquisition fails, do interactive login
    result = app.acquire_token_interactive(
        scopes, 
        redirect_uri=redirect_uri, 
        port=44436, 
        parent_window_handle=msal.PublicClientApplication.CONSOLE_WINDOW_HANDLE
    )

    if "access_token" in result:
        # Extract username and remove domain if present
        username = extract_username_without_domain(result)
        return result['access_token'], username
    else:
        error = result.get("error")
        error_description = result.get("error_description")
        raise Exception(f"Authentication failed: {error} - {error_description}")

def extract_username_without_domain(result):
    """Extract username from ID token claims and remove domain if present"""
    username = None
    if "id_token_claims" in result:
        username = result["id_token_claims"].get("preferred_username") or \
                  result["id_token_claims"].get("upn") or \
                  result["id_token_claims"].get("email")
        
        # Remove domain from username if it's in email format
        if username and '@' in username:
            username = username.split('@')[0]
    
    return username

def combine_sessions(prompt_session, session):
    """Combine prompt_session and session by replacing session's prompts with prompt_session's prompts"""
    if not prompt_session or not session:
        return session
        
    # Create a deep copy of the session to avoid modifying the original
    import copy
    combined = copy.deepcopy(session)
    
    # Replace the prompts in the combined session with prompts from prompt_session
    if 'prompts' in prompt_session and 'prompts' in combined:
        combined['prompts'] = prompt_session['prompts']
    
    return combined

def safe_remove_create_folder(folder_path):
    import os
    import shutil

    try:
        # For OneDrive, try to clear contents rather than delete if folder exists
        if os.path.exists(folder_path):
            try:
                # List all entries and delete them individually
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"Warning: Could not remove {item_path}: {e}")
            except Exception as e:
                print(f"Warning: Could not clear folder contents: {e}")
                # Continue anyway
        else:
            os.makedirs(folder_path, exist_ok=True)        
                
        return folder_path
    except Exception as e:
        print(f"Warning: Issue with folder {folder_path}: {e}")

def extract_unique_prompts(input_folder):
    """Extract unique prompts from files starting with 'prompt_' in the given directory and its subfolders."""
    import os
    import json

    unique_prompts = {}
    seen_prompts = set()
    index = 1

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.startswith("prompt_") and file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    
                    prompts = content.get("prompts", [])
                    for prompt in prompts:
                        prompt_id = prompt.get("id")
                        prompt_title = prompt.get("title")
                        if prompt_id and prompt_title:
                            # Use a tuple of id and title to check for duplicates
                            prompt_key = (prompt_id, prompt_title)
                            if prompt_key not in seen_prompts:
                                seen_prompts.add(prompt_key)
                                unique_prompts[index] = {"id": prompt_id, "title": prompt_title}
                                index += 1
                except Exception as e:
                    print(f"Warning: Could not process file {file_path}: {e}")

    return unique_prompts

def check_llm_api_errors(parsed_response):
    """
    Check if the session response contains LLM API errors.
    Returns a tuple: (has_errors: bool, error_messages: list, error_rate: float)
    error_rate = error_count / query_count (0.0 if no queries)
    """
    error_messages = []
    response_error_count = 0
    criteria_error_count = 0
    error_patterns = [
        "ERROR From LLMAPI",
        "StatusCode: InternalServerError",
        "StatusCode: ServiceUnavailable", 
        "StatusCode: BadGateway",
        "StatusCode: GatewayTimeout",
        "StatusCode: TooManyRequests",
        "LLM request failed",
        "Model inference failed"
    ]
    
    # Count queries from sessionInputs -> dataItems
    criteria_count = 0
    query_count = 0
    session_inputs = parsed_response.get("sessionInputs", [])
    for session_input in session_inputs:
        data_items = session_input.get("dataItems", [])
        query_count += len(data_items)
    
    run_eval_results = parsed_response.get("runEvalResults", [])
    
    for result in run_eval_results:
        inferences = result.get("inference", [])
        for inference in inferences:
            has_error_in_inference = False
            
            response_text = inference.get("response", "")
            if response_text:
                for pattern in error_patterns:
                    if pattern in response_text:
                        error_messages.append(f"Inference error: {response_text[:200]}")
                        has_error_in_inference = True
                        break
            
            data_items_outputs = inference.get("dataItemsOutputs", {})
            for key, value in data_items_outputs.items():
                if isinstance(value, str):
                    for pattern in error_patterns:
                        if pattern in value:
                            error_messages.append(f"Inference dataItem {key}: {value[:200]}")
                            has_error_in_inference = True
                            break
            
            error = inference.get("error", "")
            if error:
                error_messages.append(f"Inference error field: {error[:200]}")
                has_error_in_inference = True
            
            if has_error_in_inference:
                response_error_count += 1
        
        evaluations = result.get("evaluation", [])
        for evaluation in evaluations:
            criteria_list = evaluation.get("criteria", [])
            criteria_count += len(criteria_list)
            for criteria in criteria_list:
                data_outputs = criteria.get("dataItemsEvaluationOutputs", {})
                for key, value in data_outputs.items():
                    script_output = value.get("scriptOutput", "") if isinstance(value, dict) else ""
                    if script_output:
                        for pattern in error_patterns:
                            if pattern in script_output:
                                error_messages.append(f"Evaluation error in {key}: {script_output[:200]}")
                                criteria_error_count += 1
                                break
    
    response_error_rate = response_error_count / query_count if query_count > 0 else 0.0
    criteria_error_rate = criteria_error_count / (query_count * criteria_count) if (query_count > 0 and criteria_count > 0) else 0.0
    return len(error_messages) > 0, error_messages, response_error_count, response_error_rate, criteria_error_count, criteria_error_rate


def wait_for_session_completion(session_id, bearer_token, session_name, max_retries=3, poll_interval=60, get_fresh_token_func=None):
    """
    Wait for a session to complete, checking for LLM errors and retrying if needed.
    Returns tuple: (final_session_id, parsed_response, success)
    """
    import time
    import requests
    
    current_session_id = session_id
    retry_count = 0
    current_token = bearer_token
    
    while True:
        try:
            # Refresh token if function provided
            if get_fresh_token_func:
                current_token = get_fresh_token_func()
            
            response = requests.get(
                f"https://m365playground.prod.substrateai.microsoft.net/api/v2/evaluation/async?jobId={current_session_id}",
                headers={"Authorization": f"Bearer {current_token}"}
            )

            # Handle 401 - token expired, force refresh and retry immediately
            if response.status_code == 401:
                print(f"  🔑 Token expired (401), forcing refresh...")
                if get_fresh_token_func:
                    # Force token refresh by getting a new one
                    current_token = get_fresh_token_func(force_refresh=True)
                    continue
                else:
                    print(f"  ❌ No token refresh function available")
                    return current_session_id, None, False

            if response.status_code != 200:
                print(f"  ⚠️ HTTP {response.status_code} for session {current_session_id}")
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"  Waiting {poll_interval}s before retry...")
                    time.sleep(poll_interval)
                    continue
                else:
                    return current_session_id, None, False
            
            parsed = response.json()
            status = parsed.get("status")
            
            print(f"  Status: {status}", end="\r")
            
            if status in ["Running", "Queued"]:
                time.sleep(poll_interval)
                continue
            
            # Session completed - check for failures
            failed_statuses = ["Failed", "Error", "Cancelled", "TimedOut"]
            if status in failed_statuses:
                print(f"  ❌ Session failed with status: {status}")
                return current_session_id, parsed, False
            
            # Check for LLM API errors
            has_llm_errors, llm_error_messages, response_error_count, response_error_rate, criteria_error_count, criteria_error_rate = check_llm_api_errors(parsed)
            
            if has_llm_errors:
                print(f"\n  ⚠️ LLM API errors detected {len(llm_error_messages)} errors.( response error: {response_error_count} errors, rate: {response_error_rate:.1%}; criteria error: {criteria_error_count} errors, rate: {criteria_error_rate:.1%})")
                for err_msg in llm_error_messages[:2]:
                    print(f"    - {err_msg[:100]}...")
                
                # Only retry if response error rate is over 5%
                if response_error_rate > 0.05:
                    print(f"  🔄 Response error rate > 5%, will retry...")
                    return current_session_id, parsed, False
                else:
                    print(f"  ✅ Response error rate <= 5%, treating as success")
                    return current_session_id, parsed, True
            
            # Success!
            print(f"  ✅ Completed successfully")
            return current_session_id, parsed, True
            
        except Exception as e:
            print(f"  ⚠️ Error: {str(e)}")
            if retry_count < max_retries:
                retry_count += 1
                time.sleep(poll_interval)
                continue
            return current_session_id, None, False


def process_sessions(input_folder, selected_prompt_ids, sequential=False, max_retries=3, poll_interval=60, run_type=3):
    import json
    import os
    import requests
    import time
    from datetime import datetime
    # Create new-sessions folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_sessions_folder = os.path.join(input_folder, f"{timestamp}-results")

    if os.path.exists(new_sessions_folder):
        import shutil
        shutil.rmtree(new_sessions_folder)
    
    os.makedirs(new_sessions_folder)

    combined_sessions_folder = os.path.join(input_folder, f"combined-sessions")

    safe_remove_create_folder(combined_sessions_folder)

    bearer_token, username = get_access_token_msal()
    token_obtained_time = time.time()
    TOKEN_REFRESH_INTERVAL = 45 * 60  # Refresh token every 45 minutes (before 1 hour expiry)

    def get_fresh_token(force_refresh=False):
        """Get a fresh token, refreshing if needed or forced"""
        nonlocal bearer_token, token_obtained_time
        if force_refresh or time.time() - token_obtained_time > TOKEN_REFRESH_INTERVAL:
            print("\n🔑 Refreshing access token...")
            bearer_token, _ = get_access_token_msal()
            token_obtained_time = time.time()
            print("✅ Token refreshed successfully")
        return bearer_token

    # Prepare sessions_info.json file
    sessions_info_path = os.path.join(new_sessions_folder, "sessions_info.json")
    sessions_info = {"sessions": []}

    # Get all JSON files, including those in subfolders
    json_files = []
    for root, _, files in os.walk(input_folder):
        # Skip subfolders whose name ends with '-results'
        if os.path.basename(root).endswith('-results'):
            continue
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # Group files by subfolder
    subfolder_files = {}
    for file_path in json_files:
        subfolder = os.path.dirname(file_path)
        if subfolder not in subfolder_files:
            subfolder_files[subfolder] = []
        subfolder_files[subfolder].append(file_path)

    # Process each subfolder separately
    num_sessions = 0
    for subfolder, files in subfolder_files.items():
        # Sort files to process prompt_ files first
        prompt_files = [f for f in files if os.path.basename(f).startswith("prompt_")]
        other_files = [f for f in files if not os.path.basename(f).startswith("prompt_")]
        sorted_files = prompt_files + other_files

        # Initialize prompt_session for the current subfolder
        prompt_session = None

        # Loop through JSON files in the sorted order
        for file_path in sorted_files:
            num_sessions += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_content = json.load(f)

                dataItems = session_content.get("dataItems", [])
                curDataIndex = 0
                for dataItem in dataItems:
                    # Check both for missing index or null/None index
                    if "index" not in dataItem or dataItem["index"] is None:
                        dataItem["index"] = curDataIndex
                        curDataIndex += 1  

                evaluationStrategy = session_content.get("evaluationStrategy") if session_content else {}
                criteriaList = evaluationStrategy.get("criteriaList", []) if evaluationStrategy is not None else []

                sessionInputs = session_content.get("sessionInputs", {}) 
                if sessionInputs:
                    for sessionInput in sessionInputs:
                        sessionInputDataItems = sessionInput.get("dataItems", [])
                        curIndex = 0
                        for sessionInputDataItem in sessionInputDataItems:
                            # Check both for missing index or null/None index
                            if "index" not in sessionInputDataItem or sessionInputDataItem["index"] is None:
                                sessionInputDataItem["index"] = curIndex
                                curIndex += 1

                        sessionInput["criteriaIds"] = []
                        for criteria in criteriaList:
                            if "id" in criteria and criteria["id"] is not None:
                                sessionInput["criteriaIds"].append(criteria["id"])

                if os.path.basename(file_path).startswith("prompt_"):
                    prompt_session = session_content  
                    combined_session = session_content
                else:
                    combined_session = combine_sessions(prompt_session, session_content)
                
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                new_session = {
                    "parameters": {
                        "runType": run_type,
                        "runSelectedPrompts": selected_prompt_ids,
                        "runningState": 0
                    },
                    "sessionHeader": {
                        "id": "",
                        "title": f"{base_name} (Reasoning)",
                        "isPublic": True,
                        "owners": [] if username is None else [f"{username}"] ,
                        "model": "",
                        "chained": False,
                        "templated": False,
                        "dataset": False,
                        "private": True,
                        "lastModified": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "isChatting": False,
                        "isEvaluation": True,
                        "isLarge": True,
                        "isTprompt": False
                    },
                    "session": combined_session
                }

                json_body = json.dumps(combined_session)
                new_file_name = f"{base_name}.json"
                new_file_path = os.path.join(combined_sessions_folder, new_file_name)
                
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(json_body)

                json_body = json.dumps(new_session)    

                headers = {
                    "Authorization": f"Bearer {get_fresh_token()}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    "https://m365playground.prod.substrateai.microsoft.net/api/v2/evaluation/async",
                    data=json_body,
                    headers=headers
                )
                
                    
                response.raise_for_status()
                
                parsed = response.json()

                if 'sessionId' in parsed:
                    session_id = parsed['sessionId']
                    
                    # Add session information to sessions_info
                    sessions_info["sessions"].append({
                        "id": session_id,
                        "name": base_name,
                        "originalFilePath": file_path
                    })

                    print(f"Processed: {file_path} => Session ID: {session_id}")
                    
                    # Sequential mode: wait for completion and retry if needed
                    if sequential:
                        retry_attempt = 0
                        while retry_attempt <= max_retries:
                            final_id, parsed_response, success = wait_for_session_completion(
                                session_id, bearer_token, base_name, max_retries=0, poll_interval=poll_interval,
                                get_fresh_token_func=get_fresh_token
                            )
                            
                            if success:
                                # Save the result JSON
                                result_file_path = os.path.join(new_sessions_folder, f"{final_id}.json")
                                with open(result_file_path, 'w', encoding='utf-8') as f:
                                    json.dump(parsed_response, f, indent=4)
                                
                                # Save results to the original session file immediately
                                eval_results = parsed_response.get("runEvalResults", [])
                                if eval_results:
                                    save_session_results(eval_results, file_path)
                                    print(f"  📝 Results saved to: {file_path}")
                                
                                # Update sessions_info with final session ID
                                sessions_info["sessions"][-1]["id"] = final_id
                                break
                            else:
                                retry_attempt += 1
                                if retry_attempt <= max_retries:
                                    print(f"  🔄 Retry {retry_attempt}/{max_retries} for '{base_name}'...")
                                    
                                    # Resubmit the session
                                    new_session = {
                                        "parameters": {
                                            "runType": run_type,
                                            "runSelectedPrompts": selected_prompt_ids,
                                            "runningState": 0
                                        },
                                        "sessionHeader": {
                                            "id": "",
                                            "title": f"{base_name} (Reasoning - Retry {retry_attempt})",
                                            "isPublic": True,
                                            "owners": [] if username is None else [f"{username}"],
                                            "model": "",
                                            "chained": False,
                                            "templated": False,
                                            "dataset": False,
                                            "private": True,
                                            "lastModified": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                            "isChatting": False,
                                            "isEvaluation": True,
                                            "isLarge": True,
                                            "isTprompt": False
                                        },
                                        "session": combined_session
                                    }
                                    
                                    # Use fresh token for retry
                                    retry_headers = {
                                        "Authorization": f"Bearer {get_fresh_token()}",
                                        "Content-Type": "application/json"
                                    }
                                    
                                    retry_response = requests.post(
                                        "https://m365playground.prod.substrateai.microsoft.net/api/v2/evaluation/async",
                                        data=json.dumps(new_session),
                                        headers=retry_headers
                                    )
                                    
                                    if retry_response.status_code == 200:
                                        retry_parsed = retry_response.json()
                                        if 'sessionId' in retry_parsed:
                                            session_id = retry_parsed['sessionId']
                                            print(f"  ✅ Resubmitted => New Session ID: {session_id}")
                                        else:
                                            print(f"  ❌ Resubmit failed: No sessionId in response")
                                            break
                                    else:
                                        print(f"  ❌ Resubmit failed: HTTP {retry_response.status_code}")
                                        break
                                else:
                                    print(f"  ❌ Max retries reached for '{base_name}'")
                                    # Still save the failed result if we have it
                                    if parsed_response:
                                        result_file_path = os.path.join(new_sessions_folder, f"{final_id}.json")
                                        with open(result_file_path, 'w', encoding='utf-8') as f:
                                            json.dump(parsed_response, f, indent=4)
                                        
                                        # Save results to the original session file even if max retries reached
                                        eval_results = parsed_response.get("runEvalResults", [])
                                        if eval_results:
                                            save_session_results(eval_results, file_path)
                                            print(f"  📝 Results saved to: {file_path}")
                                    sessions_info["sessions"][-1]["id"] = final_id
                        
                        # Update sessions_info.json after each session in sequential mode
                        with open(sessions_info_path, 'w', encoding='utf-8') as f:
                            json.dump(sessions_info, f, indent=4)
                else:
                    warning_message = f"No sessionId found in response for {file_path}"
                    print(f"Warning: {warning_message}")
                    with open(os.path.join(new_sessions_folder, "error-log.txt"), 'a') as f:
                        f.write(warning_message + '\n')
            
            except Exception as e:
                # Capture server response body if available (e.g. from HTTPError)
                response_body = ""
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        response_body = e.response.text
                    except Exception:
                        response_body = "(could not read response body)"

                error_details = f"""
    ===========================
    File: {file_path}
    Status: {getattr(getattr(e, 'response', None), 'status_code', 'N/A')} {str(e)[:200]}
    Error: {str(e)}
    Response Body: {response_body[:2000]}

    Full Request Body:
    {json_body}

    ===========================
    """
                with open(os.path.join(new_sessions_folder, "error-log.txt"), 'a') as f:
                    f.write(error_details)
                print(f"Error: POST failed for {file_path}: {str(e)}")
                if response_body:
                    print(f"  Response body: {response_body[:500]}")
    
    # Write sessions_info to sessions_info.json
    with open(sessions_info_path, 'w', encoding='utf-8') as f:
        json.dump(sessions_info, f, indent=4)

    if sequential:
        print(f'\n✅ All {num_sessions} sessions have been processed sequentially.')
        print(f'Generating results.md...')
        generate_results_md(new_sessions_folder, sessions_info, combined_sessions_folder)
        print(f'Results saved to: {os.path.join(new_sessions_folder, "results.md")}')
    else:
        print(f'All {num_sessions} sessions have been submitted. Run python check_reasoning_checklist.py -sessions "{new_sessions_folder}"')


# Import the unified generate_results_md and save_session_results functions from check_reasoning_checklist
from check_reasoning_checklist import generate_results_md, save_session_results

def run_reasoning_checklist():
    import argparse

    parser = argparse.ArgumentParser(description="Reasoning Change Checklist Runner")
    parser.add_argument('-source', type=str, required=True, help='Path to the folder containing session files')
    parser.add_argument('-list-prompts', action='store_true', help='List available prompts in the source folder')
    parser.add_argument('-run-selected-prompts', type=int, nargs='+', help='List of prompt numbers to run (from the list shown by -list-prompts)')
    parser.add_argument('-sequential', action='store_true', help='Run sessions sequentially (one at a time) instead of submitting all at once')
    parser.add_argument('-max-retries', type=int, default=3, help='Maximum number of retries for failed sessions (default: 3)')
    parser.add_argument('-poll-interval', type=int, default=60, help='Polling interval in seconds for sequential mode (default: 60)')
    parser.add_argument('-run-type', type=int, default=3, choices=[0, 1, 2, 3, 4],
                        help='Run type: 0=Inference, 1=Evaluation, 2=PythonScript, 3=All (default), 4=ParseInference')
    args = parser.parse_args()

    if args.list_prompts:
        unique_prompts = extract_unique_prompts(args.source)
        if unique_prompts:
            print("Available Prompts:")
            for index, prompt in unique_prompts.items():
                print(f"{index}. {prompt['title']}")
        else:
            print("No prompts found in the specified folder.")
        return

    selected_prompt_ids = []
    if args.run_selected_prompts:
        unique_prompts = extract_unique_prompts(args.source)
        if unique_prompts:
            for prompt_number in args.run_selected_prompts:
                if prompt_number in unique_prompts:
                    selected_prompt_ids.append(unique_prompts[prompt_number]['id'])
                else:
                    print(f"Warning: Prompt number {prompt_number} is not valid.")
        else:
            print("No prompts found in the specified folder.")
            return
        
    process_sessions(args.source, selected_prompt_ids, 
                     sequential=args.sequential, 
                     max_retries=args.max_retries,
                     poll_interval=args.poll_interval,
                     run_type=args.run_type)

if __name__ == "__main__":
    run_reasoning_checklist()
