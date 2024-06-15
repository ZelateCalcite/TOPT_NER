import os
import sys

from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments
from utils.data_processor import a_dapt_corpus_process
from utils.trainer import UniDataset, ModifiedTrainer, collate_fn, save_tuned_parameters

checkpoint = 'flan-t5-base'
device = 'cuda'

model = T5ForConditionalGeneration.from_pretrained(checkpoint)
model.to(device)
tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=False)

for name in [
    'ai',
    'literature',
    'music',
    'politics',
    'science'
]:
    generated_corpus = a_dapt_corpus_process(
        './GTOK_corpus/{}.json'.format(name))
    train_dataset = UniDataset(tokenizer, generated_corpus, 'input')

    training_args = TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        fp16=True,
        save_steps=50000,
        save_total_limit=3,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        num_train_epochs=25,
        log_level='info',
        logging_strategy='steps',
        logging_steps=1000,
        logging_dir='logs',
        remove_unused_columns=False,
        seed=42,
        data_seed=0,
        group_by_length=False,
        dataloader_pin_memory=False
    )

    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
        tokenizer=tokenizer
    )

    trainer.train(resume_from_checkpoint=False)
    save_tuned_parameters(model, os.path.join("output", "t5-base-{}-TOPT-e25.pt".format(name)), device)
    model.save_pretrained('output/t5-base-{}-TOPT-e25'.format(name))
    tokenizer.save_pretrained('output/t5-base-{}-TOPT-e25'.format(name))
    print('Saved at output/t5-base-{}-TOPT-e25.pt'.format(name))
