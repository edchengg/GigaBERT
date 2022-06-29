## GigaBERT
This repo contains code and data for downstream tasks in [GigaBERT](https://arxiv.org/pdf/2004.14519.pdf):

	@inproceedings{lan2020gigabert,
	  author     = {Lan, Wuwei and Chen, Yang and Xu, Wei and Ritter, Alan},
  	  title      = {Giga{BERT}: Zero-shot Transfer Learning from {E}nglish to {A}rabic},
  	  booktitle  = {Proceedings of The 2020 Conference on Empirical Methods on Natural Language Processing (EMNLP)},
  	  year       = {2020}
  	} 

## Arabic Relation Extraction System

```python
>>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AuotoModelForSequenceClassification

>>> ner_model = AutoModelForTokenClassification.from_pretrained("ychenNLP/arabic-ner-ace")
>>> ner_tokenizer = AutoTokenizer.from_pretrained("ychenNLP/arabic-ner-ace")
>>> ner_pip = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, grouped_entities=True)

>>> re_model = AutoModelForSequenceClassification.from_pretrained("ychenNLP/arabic-relation-extraction")
>>> re_tokenizer = AutoTokenizer.from_pretrained("ychenNLP/arabic-relation-extraction")
>>> re_pip = pipeline("text-classification", model=re_model, tokenizer=re_tokenizer)

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
    
def post_process_re_output(re_output, re_input, ner_output):
    final_output = []
    for idx, out in enumerate(re_output):
        if out["label"] != 'O':
            tmp = re_input[idx]
            tmp['relation_type'] = out
            tmp.pop('re_input', None)
            final_output.append(tmp)

    template = {"input": re_input["input"],
                "entity": ner_output,
                "relation": final_output}

    return template

>>> input = "Hugging face is a French company in New york."
>>> output = ner_pip(input) # inference NER tags

>>> re_input = process_ner_output(output, input) # prepare a pair of entity and predict relation type

>>> re_output = []
>>> for idx in range(len(re_input)):
>>>     tmp_re_output = re_pip(re_input[idx]["re_input"]) # for each pair of entity, predict relation
>>>     re_output.append(tmp_re_output)



>>> re_ner_output = post_process_re_output(re_output) # post process NER and relation predictions
>>> print("Sentence: ",re_ner_output["input"])
>>> print("Entity: ", re_ner_output["entity"])
>>> print("Relation: ", re_ner_output["relation"])
```

## Funding Acknowledgment
This material is based in part on research sponsored by IARPA via the BETTER program (2019-19051600004).
