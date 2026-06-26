# check_examples.py
import json

with open('seval_data/144683_top200_full.jsonl','r',encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        for arm in ['all_search_results_treatment']:
            data = r.get(arm, {})
            for it_key, plugins in data.items():
                for plugin, result_lists in plugins.items():
                    for rl in result_lists:
                        for res in rl.get('Results', []):
                            if res.get('DocType') == 'PeopleInferenceAnswer':
                                exclude = ['ExtendedProfile','Facts','Skills','AIIdentifiedSkills','RelatedEntities','RelevanceSignals']
                                filtered = {k:v for k,v in res.items() if k not in exclude}
                                print('People example:')
                                print(json.dumps(filtered, indent=2))
                                exit(0)
