from transformers import Trainer
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch


def load_lora_config(_model):
    config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )
    return get_peft_model(_model, config)


def create_prompt(_data, data_keyname):
    return '''
    {0}
    {1}
    '''.format(_data['instruction'] if _data.get('instruction') else '', _data[data_keyname])


def get_encodings(_tokenizer, data, data_keyname, max_source_length=256):
    prompt = create_prompt(data, data_keyname=data_keyname)
    encoding = _tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_source_length,
        return_tensors="pt"
    )
    return encoding.input_ids, encoding.attention_mask


def get_labels(_tokenizer, data, max_target_length=128):
    target_encoding = _tokenizer(
        data['label'],
        padding="max_length",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    _labels = target_encoding.input_ids
    _labels[_labels == _tokenizer.pad_token_id] = -100
    return _labels


def collate_fn(batch):
    _input_ids = []
    _attention_mask = []
    _labels = []

    for obj in batch:
        _input_ids.append(obj['input_ids'])
        _labels.append(obj['labels'])
        _attention_mask.append(obj['attention_mask'])

    return {
        'input_ids': torch.stack(_input_ids),
        'attention_mask': torch.stack(_attention_mask),
        'labels': torch.stack(_labels)
    }


def save_tuned_parameters(_model, path, _device):
    saved_params = {
        k: v.to(_device)
        for k, v in _model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


class UniDataset(Dataset):
    def __init__(self, _tokenizer, data, data_keyname):
        super().__init__()
        self.data = data
        self.tokenizer = _tokenizer
        self.data_keyname = data_keyname

    def __getitem__(self, index):
        item_data = self.data[index]
        _tokenizer = self.tokenizer
        _input_ids, _attention_mask = get_encodings(_tokenizer, item_data, self.data_keyname)
        _labels = get_labels(_tokenizer, item_data)

        return {
            "input_ids": _input_ids,
            "labels": _labels,
            "attention_mask": _attention_mask
        }

    def __len__(self):
        return len(self.data)


class ModifiedTrainer(Trainer):
    def compute_loss(self, _model, inputs, return_outputs=False):
        outputs = _model(
            input_ids=inputs["input_ids"].squeeze(1),
            attention_mask=inputs["attention_mask"].squeeze(1),
            labels=inputs["labels"].squeeze(1),
        )
        if return_outputs:
            return outputs.loss, outputs
        else:
            return outputs.loss
