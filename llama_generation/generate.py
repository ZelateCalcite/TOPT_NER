import json
import torch
from transformers import AutoTokenizer, pipeline
from utils.data_processor import ner_data_process
from tqdm import tqdm

Prompt = '''[INST]
Instruction: Take the text below and give an explanation of why the text span "{0}" can be labeled \
as "{1}".
{2}
[/INST]'''


def generate(_prompt, _tokenizer, _pipeline, _text, _span, _label):
    _prompt = _prompt.format(_span, _label, _text)
    response = _pipeline(
        _prompt,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=_tokenizer.eos_token_id,
        max_length=4096,
        truncation=True
    )[0]['generated_text']
    return response.split('[/INST]')[-1].strip()


if __name__ == '__main__':
    for task in ['ai',
                 'literature',
                 'music',
                 'politics',
                 'science']:
        train_data = ner_data_process('../ner_data/{}/train.txt'.format(task))
        tokenizer = AutoTokenizer.from_pretrained('Llama path')
        pipe = pipeline(
            'text-generation',
            model='Llama path',
            torch_dtype=torch.float16,
            device_map='auto'
        )

        for index in tqdm(train_data.keys()):
            text = train_data[index]['text']
            train_data[index]['instruction'] = Prompt
            for entity in tqdm(train_data[index]['entity'].values()):
                span, label = entity['word'], entity['entity']
                explanation = generate(Prompt, tokenizer, pipe, text, span, label)
                entity['explanation'] = explanation
        with open('./{}.json'.format(task), 'w', encoding='utf-8') as f:
            f.write(json.dumps(train_data, ensure_ascii=False))
