import json
import torch

instruction = '''The task is to label named entities from in the given sentence and the entity should be chosen in [{}].
Please answer in the format (entity type, entity).
Here is the sentence: 
'''


def ner_data_process(path, label_type='BIO'):
    with open(path, 'r', encoding='utf-8') as f:
        if label_type == 'BIO':
            result = {}
            words, labels = [], []
            for line in f.readlines():
                if line.strip():
                    words.append(line.strip().split('\t')[0])
                    labels.append(line.strip().split('\t')[1])
                else:
                    text = ' '.join(words)
                    index = 0
                    entity = {}
                    while index < len(labels):
                        if labels[index].startswith('B-'):
                            t = index + 1
                            while t < len(labels) and labels[t].startswith('I-'):
                                t += 1
                            entity[len(entity)] = {
                                'word': ' '.join(words[index:t]),
                                'entity': labels[index][2:]
                            }
                            index = t
                        else:
                            index += 1
                    result[len(result)] = {
                        'text': text,
                        'entity': entity
                    }
                    words, labels = [], []
            entity_type = set()
            for data in result.values():
                for entity in data['entity'].values():
                    entity_type.add(entity['entity'])
            for i in result.keys():
                result[i]['instruction'] = instruction.format(', '.join(list(entity_type)))
            return result


def entity_process(entity):
    # label_format = '<extra_id_0> {0} <extra_id_5> {1} <extra_id_1>'
    # return '<extra_id_0> {} <extra_id_1>'.format(
    #     ' '.join([label_format.format(e['entity'], e['word']) for e in entity.values()]))
    label_format = '({0}, {1})'
    return ' '.join([label_format.format(e['entity'], e['word']) for e in entity.values()])


def entity_types(path, label_type='BIO'):
    with open(path, 'r', encoding='utf-8') as f:
        if label_type == 'BIO':
            types = set()
            for line in f.readlines():
                if line.strip():
                    types.add(line.strip().split('\t')[1])
            types.remove('O')
            return types


def create_masked_corpus(words, strip_end=False):
    result = []
    mask_word = '<extra_id_{}>'
    for index, word in enumerate(words):
        result.append(word)
        result.append(mask_word.format(str(index)))
    if strip_end:
        result.pop()
    return ' '.join(result)


def masked_corpus(mask: [bool], words: [str]) -> {}:
    if len(mask) != len(words):
        raise ValueError('Mask and words must have the same shape')
    pi = 0
    inputs = []
    labels = []
    tmp = []
    while pi < len(words):
        if mask[pi]:
            if tmp:
                inputs.append(' '.join(tmp))
                tmp = []
            if inputs and not labels:
                labels.append('')
            if pi == 0:
                inputs.append('')
            pj = pi + 1
            while pj < len(words) and mask[pj]:
                pj += 1
            labels.append(' '.join(words[pi: pj]))
            pi = pj
        else:
            tmp.append(words[pi])
            pi += 1
    if tmp:
        inputs.append(' '.join(tmp))

        return {
            'input': create_masked_corpus(inputs, True),
            'label': create_masked_corpus(labels)
        }
    else:
        return {
            'input': create_masked_corpus(inputs),
            'label': create_masked_corpus(labels, True)
        }


def a_dapt_corpus_process(path):
    data = json.loads(open(path, 'r', encoding='utf-8').read())
    result = []
    possibility = torch.tensor([0.15])
    for item in data:
        for exp in item['explanations']:
            raw = list(exp['explanation'].split(' '))
            masks = []
            count = 0
            mask = [False for _ in raw]
            for word in exp['word'].split(' '):
                try:
                    mask[raw.index(word)] = True
                    count += 1
                except ValueError:
                    print('Word not found: ', word)
            try:
                mask[raw.index(exp['label'])] = True
                count += 1
            except ValueError:
                print('Label not found: ', exp['label'])
            if count:
                nums = 9
                masks.append(mask)
            else:
                nums = 10
            for _ in range(nums):
                mask = torch.zeros(1, len(raw), dtype=torch.int)
                mask.bernoulli_(possibility)
                masks.append(mask.squeeze(0).tolist()[:])

            for mask in masks:
                result.append(masked_corpus(mask, raw))
    return result


def dapt_corpus_process(path):
    nums = 1
    possibility = torch.tensor([0.15])
    result = []
    data = [i.strip() for i in open(path, 'r', encoding='utf-8').readlines()]

    for text in data:
        raw = list(text.split(' '))
        masks = []
        for _ in range(nums):
            mask = torch.zeros(1, len(raw), dtype=torch.int)
            mask.bernoulli_(possibility)
            masks.append(mask.squeeze(0).tolist()[:])

        for mask in masks:
            result.append(masked_corpus(mask, raw))
    return result


def dataset_measure(file):
    tokens = 0
    entity = set()
    for line in open(file, 'r', encoding='utf-8').readlines():
        if line.strip():
            e = line.strip().split('\t')[-1]
            entity.add(e.split('-')[-1])
            tokens += 1
    return tokens, len(entity), entity
