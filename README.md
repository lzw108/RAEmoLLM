# RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning based on Emotional Information

[RAEmoLLM paper](https://arxiv.org/abs/2406.11093)

## News

📢 *May. 15, 2025* Our RAEmoLLM paper has been accepted by ACL 2025 (main)!

## Introduction

This is the original code of RAEmoLLM, the first framework to address cross-domain misinformation detection using 
in-context learning based on affective information. RAEmoLLM includes three modules. 

(1) In the index construction module, we apply an emotional LLM to obtain affective embeddings from all domains to construct a retrieval database. 

(2) The retrieval module uses this to recommend top K examples (text-label pairs) from source domain data for the target domain contents. 

(3) These examples are adopted as few-shot demonstrations for the inference module to process the target domain content. We evaluate our framework on three misinformation benchmarks.

## Usage

The processed data are in the datasets folder. You can follow the steps to get the retrieval instruction data for LLM inference.
You can change the parameters in .sh file to get different datasets, different affective information and dimensions.

NOTE: If you would like to use the datasets, please make sure to comply with the original data's license or obtain authorization from the author of the original data.

Original data link: [AMTCele](https://aclanthology.org/C18-1287/), [PHEME](https://aclanthology.org/C18-1288/), [COCO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10071453/).

### index construction module

```python
# get affectieve labels by Emollama-chat-7b
bash indexconstruct/get_Emolabel.sh

# get affective embeddings of last hidden layer from Emollama-chat-7b
bash indexconstruct/get_embs.sh
```

### retrieval module

```python
# get retrieval examples for target domains according to different affective information.
bash retrieval/retrieval.sh
```

### inference module

```python
# get retrieval retrieval augmented instructions data for LLMs inference according to different affective information
bash inference/construct_instructs4inference.sh
# LLM inference
bash inference/inference.sh
```

## License

EmoLLMs series are licensed under [MIT]. Please find more details in the [MIT](LICENSE) file.

## Citation

If you use the series of EmoLLMs in your work, please cite our paper:

```bibtex
@article{liu2024raemollm,
  title={RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning based on Emotional Information},
  author={Liu, Zhiwei and Yang, Kailai and Xie, Qianqian and de Kock, Christine and Ananiadou, Sophia and Hovy, Eduard},
  journal={arXiv preprint arXiv:2406.11093},
  year={2024}
}
```

