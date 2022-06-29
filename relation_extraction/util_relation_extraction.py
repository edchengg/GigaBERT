import json
import os

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

def prepare_relation(data):
    examples = []
    for d in data:
        tokens = d['tokens']
        sent_id = d['sent_id']
        relation_mention = d['relation_mentions']
        relation_map = {}
        for rel in relation_mention:
            relation_type = rel['relation_type']
            ent_1 = rel['arguments'][0]['entity_id']
            ent_2 = rel['arguments'][1]['entity_id']
            relation_map["{}-{}".format(ent_1, ent_2)] = relation_type
            relation_map["{}-{}".format(ent_2, ent_1)] = relation_type
        entity_mention = d['entity_mentions']
        for idx1 in range(len(entity_mention) - 1):
            for idx2 in range(idx1 + 1, len(entity_mention)):
                ent_1 = entity_mention[idx1]
                ent_2 = entity_mention[idx2]
                key = "{}-{}".format(ent_1['id'], ent_2['id'])
                if key in relation_map:
                    ins = {'tokens': tokens,
                     'arg1': ent_1,
                     'arg2': ent_2,
                     'label': relation_map[key]}
                else:
                    ins = {'tokens': tokens,
                           'arg1': ent_1,
                           'arg2': ent_2,
                           'label': 'O'}
                ins['sent_id'] = sent_id
                examples.append(ins)
    return examples

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')


path_dict = {'train-en': train_en_dir, 'dev-en': dev_en_dir, 'test-en': test_en_dir,
             'train-ar': train_ar_dir, 'dev-ar': dev_ar_dir, 'test-ar': test_ar_dir}
for split in ['train', 'dev', 'test']:
    for lang in ['en', 'ar']:
        data = prepare_relation(data_load(path_dict["{}-{}".format(split, lang)]))
        save_json(data, '{}-{}-relation.json'.format(split, lang))







