import os
os.environ['TRANSFORMERS_CACHE'] = '/srv/share5/ychen3411/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/srv/share5/ychen3411/huggingface_cache'
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, TFAutoModelForSequenceClassification
import datasets
from datasets import load_dataset, load_metric
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
    }

metric = load_metric("seqeval")


train_ar_dir = "/srv/scratch/ychen3411/project03_ace_event/few-shot-learning-main/ace/ace_oneie_data/ar/train.oneie.json"
train_en_dir = "/srv/scratch/ychen3411/project03_ace_event/few-shot-learning-main/ace/ace_oneie_data/en/train.oneie.json"
dev_ar_dir = "/srv/scratch/ychen3411/project03_ace_event/few-shot-learning-main/ace/ace_oneie_data/ar/dev.oneie.json"
dev_en_dir = "/srv/scratch/ychen3411/project03_ace_event/few-shot-learning-main/ace/ace_oneie_data/en/dev.oneie.json"
test_en_dir = "/srv/scratch/ychen3411/project03_ace_event/few-shot-learning-main/ace/ace_oneie_data/en/test.oneie.json"
test_ar_dir = "/srv/scratch/ychen3411/project03_ace_event/few-shot-learning-main/ace/superlong_ar_fixed/test.ar.oneie.fixed.json"

def data_load(path):
    data = []
    max_length = 256
    """Load data from file."""
    overlength_num = title_num = 0
    ignore_title = False
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            is_title = inst['sent_id'].endswith('-3') \
                and inst['tokens'][-1] != '.' \
                and len(inst['entity_mentions']) == 0
            if ignore_title and is_title:
                title_num += 1
                continue
            if max_length != -1 and inst_len > max_length - 2:
                overlength_num += 1
                continue
            data.append(inst)

    if overlength_num:
        print('Discarded {} overlength instances'.format(overlength_num))
    if title_num:
        print('Discarded {} titles'.format(title_num))
    print('Loaded {} instances from {}'.format(len(data), path))
    return data

def process_ner_labels(examples):

    tokens = examples["tokens"]
    entity_mentions = examples["entity_mentions"]
    labels = [0 for _ in range(len(tokens))]
    for ent in entity_mentions:
        start, end = ent["start"], ent["end"]
        tag = ent["entity_type"]
        for idx in range(start, end):
            if idx == start:
                labels[idx] = entity_label["B-" + tag]
            else:
                labels[idx] = entity_label["I-" + tag]
    examples["ner_tags"] = labels
    return examples

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=256,
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
# import torch
# vocabs = torch.load("../pipelineie/vocabs.pt")
# print(vocabs["vocabs"])

entity_label = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-VEH': 3, 'I-VEH': 4, 'B-GPE': 5, 'I-GPE': 6, 'B-WEA': 7, 'I-WEA': 8, 'B-ORG': 9, 'I-ORG': 10, 'B-LOC': 11, 'I-LOC': 12, 'B-FAC': 13, 'I-FAC': 14}

dataset = load_dataset('json', data_files={'train': [train_en_dir, train_ar_dir], 'dev': dev_ar_dir, 'test_ar': test_ar_dir,
                                           'test_en': dev_en_dir})
tokenizer = AutoTokenizer.from_pretrained("lanwuwei/GigaBERT-v4-Arabic-and-English", do_lower_case=True)
datasets_ner = dataset.map(process_ner_labels)
tokenized_datasets = datasets_ner.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=datasets_ner["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

label2id = entity_label
id2label = {v:k for k,v in entity_label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    "lanwuwei/GigaBERT-v4-Arabic-and-English",
    id2label=id2label,
    label2id=label2id,
)

save_path = "arabic-ner-ace-gigabert/"

args = TrainingArguments(
    save_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5, #2e-5
    num_train_epochs=10,
    weight_decay=0.01,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_total_limit=1,
    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True
)



trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
metric_en = trainer.evaluate(tokenized_datasets['test_en'], metric_key_prefix="test_en")
metric_ar = trainer.evaluate(tokenized_datasets['test_ar'], metric_key_prefix="test_ar")
trainer.log_metrics("test_en", metric_en)
trainer.log_metrics("test_ar", metric_ar)

'''
***** test_en metrics *****
  epoch                      =       10.0
  test_en_f1                 =     0.8882
  test_en_loss               =      0.181
  test_en_precision          =     0.8924
  test_en_recall             =      0.884
  test_en_runtime            = 0:00:01.33
  test_en_samples_per_second =    673.033
  test_en_steps_per_second   =     21.663
***** test_ar metrics *****
  epoch                      =       10.0
  test_ar_f1                 =     0.8939
  test_ar_loss               =     0.2254
  test_ar_precision          =     0.8904
  test_ar_recall             =     0.8975
  test_ar_runtime            = 0:00:00.65
  test_ar_samples_per_second =    489.356
  test_ar_steps_per_second   =     16.769
'''

tf_model = TFAutoModelForSequenceClassification.from_pretrained(save_path, from_pt=True)
tf_model.save_pretrained(save_path)