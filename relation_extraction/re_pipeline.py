from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification

ner_model = AutoModelForTokenClassification.from_pretrained("ychenNLP/arabic-ner-ace")
ner_tokenizer = AutoTokenizer.from_pretrained("ychenNLP/arabic-ner-ace")
ner_pip = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, grouped_entities=True)

re_model = AutoModelForSequenceClassification.from_pretrained("ychenNLP/arabic-relation-extraction")
re_tokenizer = AutoTokenizer.from_pretrained("ychenNLP/arabic-relation-extraction")
re_pip = pipeline("text-classification", model=re_model, tokenizer=re_tokenizer)


def process_ner_output(entity_mention, inputs):
    re_input = []
    for idx1 in range(len(entity_mention) - 1):
        for idx2 in range(idx1 + 1, len(entity_mention)):
            ent_1 = entity_mention[idx1]
            ent_2 = entity_mention[idx2]

            ent_1_type = ent_1['entity_group']
            ent_2_type = ent_2['entity_group']
            ent_1_s = ent_1['start']
            ent_1_e = ent_1['end']
            ent_2_s = ent_2['start']
            ent_2_e = ent_2['end']
            new_re_input = ""
            for c_idx, c in enumerate(inputs):
                if c_idx == ent_1_s:
                    new_re_input += "<{}>".format(ent_1_type)
                elif c_idx == ent_1_e:
                    new_re_input += "</{}>".format(ent_1_type)
                elif c_idx == ent_2_s:
                    new_re_input += "<{}>".format(ent_2_type)
                elif c_idx == ent_2_e:
                    new_re_input += "</{}>".format(ent_2_type)
                new_re_input += c
            re_input.append({"re_input": new_re_input, "arg1": ent_1, "arg2": ent_2, "input": inputs})
    return re_input


def post_process_re_output(re_output, text_input, ner_output):
    final_output = []
    for idx, out in enumerate(re_output):
        if out["label"] != 'O':
            tmp = re_input[idx]
            tmp['relation_type'] = out
            tmp.pop('re_input', None)
            final_output.append(tmp)

    template = {"input": text_input,
                "entity": ner_output,
                "relation": final_output}

    return template


text_input = """ويتزامن ذلك مع اجتماع بايدن مع قادة الدول الأعضاء في الناتو في قمة موسعة في العاصمة الإسبانية، مدريد."""

ner_output = ner_pip(text_input)  # inference NER tags

re_input = process_ner_output(ner_output, text_input)  # prepare a pair of entity and predict relation type

re_output = []
for idx in range(len(re_input)):
    tmp_re_output = re_pip(re_input[idx]["re_input"])  # for each pair of entity, predict relation
    re_output.append(tmp_re_output[0])

re_ner_output = post_process_re_output(re_output, text_input, ner_output)  # post process NER and relation predictions
print("Sentence: ", re_ner_output["input"])
print('====Entity====')
for ent in re_ner_output["entity"]:
    print('{}--{}'.format(ent["word"], ent["entity_group"]))
print('====Relation====')
for rel in re_ner_output["relation"]:
    print('{}--{}:{}'.format(rel['arg1']['word'], rel['arg2']['word'], rel['relation_type']['label']))
'''
Sentence:  ويتزامن ذلك مع اجتماع بايدن مع قادة الدول الأعضاء في الناتو في قمة موسعة في العاصمة الإسبانية، مدريد.
====Entity====
بايدن--PER
قادة--PER
الدول--GPE
الناتو--ORG
العاصمة--GPE
الاسبانية--GPE
مدريد--GPE
====Relation====
قادة--الناتو:ORG-AFF
العاصمة--الاسبانية:PART-WHOLE
'''



