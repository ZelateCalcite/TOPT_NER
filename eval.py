from tqdm import tqdm
import re
import os


def generate_result(_tokenizer, _model, _prompts: [str], _device='cuda'):
    _inputs = _tokenizer(_prompts, return_tensors='pt', padding=True).to(_device)
    _model.to(_device)
    _outputs = _model.generate(input_ids=_inputs['input_ids'], attention_mask=_inputs['attention_mask'],
                               do_sample=False, max_new_tokens=128)
    return _tokenizer.batch_decode(_outputs, skip_special_tokens=True)


def out_filter(text):
    pattern = r'\) '
    text = re.sub(pattern, '\n', text)
    if text.endswith(')'):
        text = text[:-1]
    result = []
    for item in text.split('\n'):
        try:
            if ', ' in item:
                item = item[1:]
                result.append({
                    item.split(', ')[1]: item.split(', ')[0]
                })
        except Exception as e:
            continue
    return result


def evaluation(predictions, labels, entity_type, print_all=False, raw_text=None):
    entity_type = set([i[2:] for i in entity_type])
    predictions = [out_filter(p[0]) for p in predictions]
    # token num
    predict_num = {i: 0 for i in entity_type}
    correct_num = {i: 0 for i in entity_type}

    for i, seq in enumerate(predictions):
        for e in seq:
            for key in e:
                if key in raw_text[i]:
                    try:
                        # NER forced callback
                        predict_num[e[key]] += 1
                    except KeyError:
                        continue
    temp = []
    for i in labels:
        tmp = []
        for e in i.values():
            if {e['word']: e['entity']} not in tmp:
                tmp.append({e['word']: e['entity']})
        temp.append(tmp)
    labels = temp

    true_entity = {i: 0 for i in entity_type}
    for entities in labels:
        for e in entities:
            for i in e.values():
                true_entity[i] += 1

    predict_num = {i: 0 for i in entity_type}
    for index in range(len(predictions)):
        temp = labels[index][:]
        for entity in predictions[index]:
            if entity in temp:
                correct_num[list(entity.values())[0]] += len(list(entity.keys())[0].split(' '))
                predict_num[list(entity.values())[0]] += len(list(entity.keys())[0].split(' '))
                true_entity[list(entity.values())[0]] += len(list(entity.keys())[0].split(' ')) - 1
            else:
                if list(entity.keys())[0] in raw_text[index]:
                    # NER forced callback
                    try:
                        predict_num[list(entity.values())[0]] += len(list(entity.keys())[0].split(' '))
                    except KeyError:
                        continue

    precision = {k: (correct_num[k] / predict_num[k] if predict_num[k] != 0 else 0) for k in correct_num.keys()}
    recall = {k: (correct_num[k] / true_entity[k] if true_entity[k] != 0 else 0) for k in correct_num.keys()}

    tp, fp, fn = sum(correct_num.values()), sum(predict_num.values()) - sum(correct_num.values()), sum(
        true_entity.values()) - sum(correct_num.values())
    p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)

    macro_f1 = sum([(2 * precision[k] * recall[k] / (recall[k] + precision[k]))
                    * true_entity[k] / sum(true_entity.values()) if recall[k] + precision[k] != 0 else 0 for k in
                    correct_num.keys()])

    if print_all:
        print("-----precision-----")
        for i in precision:
            print('{}\t{:.4f}'.format(i, precision[i]))
        print("------recall-------")
        for i in recall:
            print('{}\t{:.4f}'.format(i, recall[i]))

    return f, macro_f1


def test(_model, _tokenizer, _data, _types, save_output='', print_all=False):
    _test_seq = list(list(_data.values())[:])

    _sentences = ['{}\n{}'.format(i['instruction'], i['text']) for i in _test_seq]
    _outs = []
    for seq in tqdm(_sentences):
        _outs.append(generate_result(_tokenizer, _model, seq, _device='cuda'))
    if save_output:
        if os.path.isfile(save_output):
            with open(save_output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(_outs, ensure_ascii=False) + '\n')
        else:
            with open(save_output, 'w', encoding='utf-8') as f:
                f.write(json.dumps(_outs, ensure_ascii=False) + '\n')
    _a, _b = evaluation(_outs, [i['entity'] for i in _test_seq], _types, print_all=print_all,
                        raw_text=[i['text'] for i in _test_seq])
    return _a, _b


if __name__ == '__main__':
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from utils.data_processor import *

    for name in ['ai',
                 'literature',
                 'music',
                 'politics',
                 'science']:
        # CrossNER Dateset
        types = entity_types('./ner_data/{}/test.txt'.format(name))
        test_data = ner_data_process('./ner_data/{}/test.txt'.format(name))

        # load your model here
        model = T5ForConditionalGeneration.from_pretrained('Model_path')
        tokenizer = T5Tokenizer.from_pretrained('Model_path')

        test_seq = list(list(test_data.values())[:])
        sentences = [i['text'] for i in test_seq]

        test_result = test(model, tokenizer, test_data, types, save_output='./output/{}.json'.format(name))
        e = evaluation(test_result, [i['entity'] for i in test_seq], types, raw_text=sentences)
        print('{0}\t{1}\t{2}'.format(name, e[0], e[1]))
