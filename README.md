# TOPT
code of our paper in EMNLP 2024 main conference.

[Cross-domain NER with Generated Task-Oriented Knowledge: An Empirical Study from Information Density Perspective]([https://arxiv.org/abs/2212.03506](https://aclanthology.org/2024.emnlp-main.95/))

Accepted by The 2024 Conference on Empirical Methods in Natural Language Processing.

# Citation
If you use this code, please cite the paper below:

```
@inproceedings{zhang-etal-2024-cross-domain,
    title = "Cross-domain {NER} with Generated Task-Oriented Knowledge: An Empirical Study from Information Density Perspective",
    author = "Zhang, Zhihao  and
      Lee, Sophia Yat Mei  and
      Wu, Junshuang  and
      Zhang, Dong  and
      Li, Shoushan  and
      Cambria, Erik  and
      Zhou, Guodong",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.95/",
    doi = "10.18653/v1/2024.emnlp-main.95",
    pages = "1595--1609",
    abstract = "Cross-domain Named Entity Recognition (CDNER) is crucial for Knowledge Graph (KG) construction and natural language processing (NLP), enabling learning from source to target domains with limited data. Previous studies often rely on manually collected entity-relevant sentences from the web or attempt to bridge the gap between tokens and entity labels across domains. These approaches are time-consuming and inefficient, as these data are often weakly correlated with the target task and require extensive pre-training.To address these issues, we propose automatically generating task-oriented knowledge (GTOK) using large language models (LLMs), focusing on the reasoning process of entity extraction. Then, we employ task-oriented pre-training (TOPT) to facilitate domain adaptation. Additionally, current cross-domain NER methods often lack explicit explanations for their effectiveness. Therefore, we introduce the concept of information density to better evaluate the model`s effectiveness before performing entity recognition.We conduct systematic experiments and analyses to demonstrate the effectiveness of our proposed approach and the validity of using information density for model evaluation."
}
```
