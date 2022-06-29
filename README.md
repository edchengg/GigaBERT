## GigaBERT
This repo contains code and data for downstream tasks in [GigaBERT](https://arxiv.org/pdf/2004.14519.pdf):

	@inproceedings{lan2020gigabert,
	  author     = {Lan, Wuwei and Chen, Yang and Xu, Wei and Ritter, Alan},
  	  title      = {Giga{BERT}: Zero-shot Transfer Learning from {E}nglish to {A}rabic},
  	  booktitle  = {Proceedings of The 2020 Conference on Empirical Methods on Natural Language Processing (EMNLP)},
  	  year       = {2020}
  	} 

## Huggingface
- [Arabic Relation Extraction](https://huggingface.co/ychenNLP/arabic-relation-extraction)
- [Arabic NER](https://huggingface.co/ychenNLP/arabic-ner-ace)

## Arabic Relation Extraction Pipeline
- NER --> Relation Extraction

```python
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
ner_model = AutoModelForTokenClassification.from_pretrained("ychenNLP/arabic-ner-ace")
ner_tokenizer = AutoTokenizer.from_pretrained("ychenNLP/arabic-ner-ace")
ner_pip = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, grouped_entities=True)

re_model = AutoModelForSequenceClassification.from_pretrained("ychenNLP/arabic-relation-extraction")
re_tokenizer = AutoTokenizer.from_pretrained("ychenNLP/arabic-relation-extraction")
re_pip = pipeline("text-classification", model=re_model, tokenizer=re_tokenizer)

text_input = """ويتزامن ذلك مع اجتماع بايدن مع قادة الدول الأعضاء في الناتو في قمة موسعة في العاصمة الإسبانية، مدريد."""

ner_output = ner_pip(text_input) # inference NER tags

re_input = process_ner_output(ner_output, text_input) # prepare a pair of entity and predict relation type

re_output = []
for idx in range(len(re_input)):
    tmp_re_output = re_pip(re_input[idx]["re_input"]) # for each pair of entity, predict relation
    re_output.append(tmp_re_output[0])

re_ner_output = post_process_re_output(re_output, text_input, ner_output) # post process NER and relation predictions
print("Sentence: ",re_ner_output["input"])
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
```

## Funding Acknowledgment
This material is based in part on research sponsored by IARPA via the BETTER program (2019-19051600004).
