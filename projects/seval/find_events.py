# find_events.py
import json

with open('seval_data/144683_top200_full.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        for arm in ['all_search_results_treatment']:
            data = r.get(arm, {})
            for it_key, plugins in data.items():
                for plugin, result_lists in plugins.items():
                    for rl in result_lists:
                        results = rl.get('Results', [])
                        types = set(res.get('DocType') for res in results)
                        if 'Event' in types or 'EmailMessage' in types:
                            utt = r.get("utterance", "")[:60]
                            print(f'Index {i}: {utt}... Types: {types}')
