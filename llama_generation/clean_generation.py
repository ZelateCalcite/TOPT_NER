import json
import re

for task_name in ['ai',
                  'literature',
                  'music',
                  'politics',
                  'science']:
    data = json.loads(open('{}.json'.format(task_name), 'r', encoding='utf-8').read())
    result = []
    for e in data.values():
        tmp = {'text': e['text'], 'explanations': []}
        for exp in e['entity'].values():
            # remove useless words
            temp = exp['explanation'].replace('!', '.')
            temp = re.sub(r'(Thank you for.*?\.)|(\")|(I\'m here to .*?\.)', '', temp)
            temp = re.sub(r'\n', ' ', temp)
            # remove negative explanations
            if re.search(r'however|However|not appropriate|not accurate|I cannot', temp):
                continue

            tmp['explanations'].append({
                'word': exp['word'],
                'label': exp['entity'],
                'explanation': temp.strip(),
            })
        if tmp['explanations']:
            result.append(tmp)

    with open('{}_cleaned.json'.format(task_name), 'w', encoding='utf-8') as f:
        f.write(json.dumps(result))
