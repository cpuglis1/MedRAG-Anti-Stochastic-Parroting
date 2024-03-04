# Medical Specific Retrieval Augmented Generation to Combat Stochastic Parroting in Large Language Models
## JHU Master's Independent Study

### Overview
This project is aimed to combat the challenges associated with Stochastic Parroting (or hallucination) in large language models (LLMs). Specifically, in the medical domain, LLMs are not widely accepted or adapted in practice due to models producing unverified or incorrect information. Therefore, I want to explore the Retrieval Augmented Generation as a tool for LLMs as a potential way to reduce parroting that can concurrently verify sources for the information cited. Many models have been fine-tuned to the medical domain, like Medalpaca-13B or MedLLama-13B, and perform reasonably well against standard and publicly available benchmarks, like PubMedQA. Further, Retrieval Augmented LLMs, like Almanac and Clinfo.ai, fare well with medical domain Question Answering. However, there is some room for improvement and additional evaluation for different hallucination benchmarks and with keyword generation/retrieval. For these reasons, I propose the following:
- Build a retrieval augmented LLM workflow using "smaller" LLMs, under 7b parameters, that is able to answer questions while citing sources.
- Develop and embed a knowledge base of PubMed articles in a specific clinical domain.
- Customize retrieval system for the domain using both embedding similarity and lexical search.
- Customize tokenization of prompt terms for best keyword identification.
- Compare model against standard benchmarks like PubMedQA or Med-HALT datasets.

### Goal
Demonstrate a lightweight, proof of concept, workflow that shows improvements over similarly sized models in this space and can assist adaptation of language models in a clinical setting.

### Project Progress
1. Data collected from [Tensorflow/Huggingface PubMed article dataset](https://huggingface.co/datasets/pubmed).
2. 100,000 PubMed abstracts and articles were tokenized using [Stanford's BioMedLM](https://huggingface.co/stanford-crfm/BioMedLM), an autoregressive language model with 2.7B parameters using the standard GPT-2 architecture. The key justification for using this LM was that the tokenizer was custom trained on the PubMed Abstracts as building domain-specific models requires in-domain text to maximize performance on downstream tasks. Code found [here](https://github.com/cpuglis1/MedRAG-Anti-Stochastic-Parroting/blob/main/src/tokenize_text.ipynb).
3. 100,000 PubMed text tokenized abstracts were converted to IDs and embedded in chunks using BioMedLM.
4. All abstracts were added to a [FAISS (Facebook AI Similarity Search) index](https://ai.meta.com/tools/faiss/), which allows developers to quickly search for embeddings of multimedia documents that are similar. The abstracts only index will serve as the benchmark Knowledge-Base. Note, given the size of the data, I cannot embed all 100,000 articles without using compute resources. The current plan is to use the abstracts Knowledge Base for initial testing and eventually compare that to a smaller Knowledge-Base with a few hundred/thousand articles in a medical subdomain (e.g., tumorigenesis). I can always compare domain-specific performance to determine the most suitable results. Code found [here](https://github.com/cpuglis1/MedRAG-Anti-Stochastic-Parroting/blob/main/src/create_faiss.py).
6. Examined retrieval capabilities manually - decent results at a glance. Will dig into the retrieval capabilites and quantify eventually. Test retrieval found [here](https://github.com/cpuglis1/MedRAG-Anti-Stochastic-Parroting/blob/main/src/query_faiss.ipynb).

### Considerations
Primary concern is computational complexity. To navigate this, I plan to:
  1. Use "smaller" base-model LLMs available on HuggingFace.
  2. Fine-tune LLMs using QLoRa and deploying adapters' weight matrices for inference.
  3. Create manageable knowledge base in a specific clinical-domain using FAISS.
  4. Rely on local and cloud-based computing in Jupyter Notebook, Google Colab, and if needed, runpod.io for low-cost computing.
Make sure the project is rooted in mathematical analysis:
  1. By describing the attention mechanism, QLoRa, retrieval mechanisms, tokenization, embeddings with justifications for this domain and to minimize complexity/hallucinations, the paper should be rigorous.

### Novelty and Contribution
1. Unique workflow architecture (small LLMs for question to query generation, summarization/answer).
2. Unique adaptation of established tokenization for clinical RAG workflow.
3. Unique adaptation of established retrieval mechanisms like similarity search, lexical search, and keyword identification.
4. Lightweight proof of concept - Small LLMs with inference can be done with QLoRa adapters.
