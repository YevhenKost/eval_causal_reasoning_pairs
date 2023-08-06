## Causal Reasoning Capability of LLMs. Evaluation System

This repository contains code for the evaluation of different  deep learning models for the sentence pair causality detection task.

### Usage
Install the requirements from [requirements.txt](requirements.txt)

All the required settings are located in [settings.py](settings.py). See the docs for each setting field there.

To run MNLI experiment: 
```commandline
python run_mnli.py
```

To run LLM experiment: 
```commandline
python run_llms.py
```

### Dataset
The dataset is [COPA](https://huggingface.co/datasets/pkavumba/balanced-copa/viewer/pkavumba--balanced-copa/train?row=0) dataset with some manual preprocessing and filtering. 
The format example and dataset itself can be found in [copa-processed-ds.json](copa-processed-ds.json)