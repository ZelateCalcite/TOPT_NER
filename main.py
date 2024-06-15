from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments
from eval import out_filter
from utils.config import parse_train_config
from utils.data_processor import *
from utils.trainer import collate_fn, save_tuned_parameters, UniDataset, ModifiedTrainer
import os
import numpy as np


def train(args, device):
    epoch = args.epoch
    train_data_path = args.trd
    eval_data_path = args.evd
    do_eval = args.eval
    checkpoint = args.cp
    load_state_dict = args.lsd
    output_name = args.on

    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=False)
    train_data = ner_data_process(train_data_path)
    for d in train_data.values():
        d['label'] = entity_process(d['entity'])
    train_dataset = UniDataset(tokenizer, train_data, 'text')
    eval_data = ner_data_process(eval_data_path)
    for d in eval_data.values():
        d['label'] = entity_process(d['entity'])
    eval_dataset = UniDataset(tokenizer, eval_data, 'text')
    if load_state_dict:
        model.load_state_dict(torch.load(load_state_dict), strict=False)

    model.to(device)
    training_args = TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        fp16=True,
        save_steps=50000,
        save_total_limit=3,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=epoch,
        log_level='info',
        logging_strategy='epoch',
        logging_dir='logs',
        remove_unused_columns=False,
        seed=42,
        data_seed=0,
        group_by_length=False,
        dataloader_pin_memory=False,
        do_train=True,
        do_eval=do_eval,
        evaluation_strategy='epoch' if do_eval else 'no',
        per_device_eval_batch_size=8,
    )

    def compute_metrics(out_preds):
        preds, labels = out_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = labels.squeeze(1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predictions = [out_filter(p) for p in decoded_preds]
        entity_type = set()
        labels = []
        for i in range(len(predictions)):
            label = decoded_labels[i][1:-1].split(') (')
            entities = {}
            for j in label:
                try:
                    e, w = j.split(', ')[0], j[1:].split(', ')[1]
                    entities[w] = e
                    entity_type.add(e)
                except IndexError:
                    print(j)
            labels.append(entities)
        predict_num = {i: 0 for i in entity_type}
        correct_num = {i: 0 for i in entity_type}
        true_entity = {i: 0 for i in entity_type}
        for entities in labels:
            for e in entities.values():
                true_entity[e] += 1

        for index in range(len(predictions)):
            temp = list(labels[index].keys())
            for entity in predictions[index]:
                k = list(entity.keys())[0]
                if k in temp:
                    try:
                        correct_num[list(entity.values())[0]] += len(list(entity.keys())[0].split(' '))
                        predict_num[list(entity.values())[0]] += len(list(entity.keys())[0].split(' '))
                        true_entity[list(entity.values())[0]] += len(list(entity.keys())[0].split(' ')) - 1
                    except KeyError:
                        continue
        tp, fp, fn = sum(correct_num.values()), sum(predict_num.values()) - sum(correct_num.values()), sum(
            true_entity.values()) - sum(correct_num.values())
        p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
        r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
        f = 0 if p + r == 0 else 2 * p * r / (p + r)
        return {
            'f1-score': f
        }

    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels

    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if do_eval else None,
        compute_metrics=compute_metrics if do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if do_eval else None,
        args=training_args,
        data_collator=collate_fn,
        tokenizer=tokenizer
    )
    trainer.train(resume_from_checkpoint=False)

    save_tuned_parameters(model, os.path.join("output", output_name), device)


if __name__ == "__main__":
    train(parse_train_config(), 'cuda')
