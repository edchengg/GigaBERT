import os
os.environ['TRANSFORMERS_CACHE'] = '/srv/share5/ychen3411/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/srv/share5/ychen3411/huggingface_cache'
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import datasets
from datasets import load_dataset, load_metric
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import numpy as np

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [id2label[label] for label in labels]
    true_predictions = [id2label[pred] for pred in predictions]

    num_gold_relation = 0
    num_pred_relation = 0
    num_match = 0

    for t, p in zip(true_labels, true_predictions):
        if t != 'O':
            num_gold_relation += 1
        if p != 'O':
            num_pred_relation += 1
        if t != 'O' and p != 'O' and t == p:
            num_match += 1

    precision, recall, f1 = compute_f1(num_pred_relation, num_gold_relation, num_match)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

metric = load_metric("seqeval")


train_ar_dir = "train-ar-relation.json"
train_en_dir = "train-en-relation.json"
dev_ar_dir = "dev-ar-relation.json"
dev_en_dir = "dev-en-relation.json"
test_en_dir = "test-en-relation.json"
test_ar_dir = "test-ar-relation.json"

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

def process_relation_markers(examples):

    tokens = examples["tokens"]
    arg1 = examples['arg1']
    arg2 = examples['arg2']

    arg1_type = arg1['entity_type']
    arg2_type = arg2['entity_type']

    arg1_start = arg1['start']
    arg1_end = arg1['end']
    arg2_start = arg2['start']
    arg2_end = arg2['end']
    new_tokens = []
    for idx, tok in enumerate(tokens):
        if idx == arg1_start:
            new_tokens.append("<{}>".format(arg1_type))

        elif idx == arg2_start:
            new_tokens.append("<{}>".format(arg2_type))

        elif idx == arg1_end:
            new_tokens.append("</{}>".format(arg1_type))

        elif idx == arg2_end:
            new_tokens.append("</{}>".format(arg2_type))
        new_tokens.append(tok)

    examples["tokens"] = new_tokens
    examples["label"] = label2id[examples["label"]]
    return examples

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=256,
    )

    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs

# Physical, Part-whole, Personal-Social, 'ORG-Affiliation, Agent-Artifact, Gen-Affiliation
relation_label = {'ORG-AFF': 1, 'ART': 2, 'PHYS': 3, 'PART-WHOLE': 4, 'GEN-AFF': 5, 'PER-SOC': 6, 'O': 0}
label2id = relation_label
id2label = {v:k for k,v in relation_label.items()}

'''
data format:

{"tokens": ["We", "'", "ll", "have", "a", "couple", "of", "experts", "come", "out", ",", "so", "I", "'", "ll", "withhold", "my", "comments", "until", "then", "."], "arg1": {"id": "CNN_CF_20030303.1900.00-E28-170", "text": "We", "entity_type": "ORG", "mention_type": "PRO", "entity_subtype": "Media", "start": 0, "end": 1}, "arg2": {"id": "CNN_CF_20030303.1900.00-E84-167", "text": "couple", "entity_type": "PER", "mention_type": "PRO", "entity_subtype": "Group", "start": 5, "end": 6}, "label": "O", "sent_id": "CNN_CF_20030303.1900.00-2"}

'''
dataset = load_dataset('json', data_files={'train': [train_ar_dir, train_en_dir], 'dev': dev_ar_dir, 'test_ar': test_ar_dir,
                                           'test_en': dev_en_dir})
tokenizer = AutoTokenizer.from_pretrained("lanwuwei/GigaBERT-v4-Arabic-and-English", do_lower_case=True)
# add new tokens
entity = ['PER', 'VEH', 'GPE', 'WEA', 'ORG', 'LOC', 'FAC']
special_tokens = []
for ent in entity:
    special_tokens.append('<{}>'.format(ent))
    special_tokens.append('</{}>'.format(ent))

tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})


datasets_relation = dataset.map(process_relation_markers)
tokenized_datasets = datasets_relation.map(
    tokenize_function,
    batched=True,
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(
    "lanwuwei/GigaBERT-v4-Arabic-and-English",
    id2label=id2label,
    label2id=label2id,
)
model.resize_token_embeddings(len(tokenizer))
save_path = "arabic-relation-ace-gigabert-enar/"

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

tf_model = TFAutoModelForSequenceClassification.from_pretrained(save_path, from_pt=True)
tf_model.save_pretrained(save_path)


'''
***** test_en metrics *****
  epoch                      =       10.0
  test_en_f1                 =     0.7205
  test_en_loss               =     0.2983
  test_en_precision          =     0.7431
  test_en_recall             =     0.6992
  test_en_runtime            = 0:00:06.26
  test_en_samples_per_second =   1576.247
  test_en_steps_per_second   =     49.343
***** test_ar metrics *****
  epoch                      =       10.0
  test_ar_f1                 =     0.7259
  test_ar_loss               =     0.0935
  test_ar_precision          =     0.7379
  test_ar_recall             =     0.7143
  test_ar_runtime            = 0:00:16.49
  test_ar_samples_per_second =   1137.425
  test_ar_steps_per_second   =     35.577
'''
