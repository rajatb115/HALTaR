# HALTaR: Hallucination-Augmented Lookup for Table Retrieval

In the era of data-driven decision-making, large volumes of knowledge are stored in structured tabular formats across diverse domains. Effectively retrieving relevant tables in response to natural language (NL) queries is a critical prerequisite for open-domain question answering and downstream analytical tasks. However, NL–based table retrieval remains a challenging research problem due to the need to effectively align unstructured NL queries with the rich structural and semantic characteristics of tables. 

Existing approaches largely rely on (i) keyword-based retrieval or (ii) treating tables and NL queries as the same modalities, limiting retrieval effectiveness. We address this challenge with HALTaR, a corpus-augmented table retrieval framework. HALTaR combines a structure-aware lightweight retriever with an LLM-driven query-to-table hallucinator that converts NL queries into structured, intent-centric tables. The hallucinated table is further embedded by a lightweight retriever model, which is then used to retrieve relevant tables. This design enables effective retrieval under schema variability and incomplete queries. Extensive evaluations against public benchmarks – SPIDER, BIRD – demonstrate that HALTaR outperforms strong baselines, making it a practical solution for query-to-table retrieval in real-world settings.

## 1. Directory Overview
The root directory contains four folders: `dataset`, `model`, `output`, and `script`. All folders except `script` are shared via a drive link.  
The directory structure is organized as follows:

```bash
project-root/
├── dataset/
│   ├── bird/
│   └── spider/
│   └── ...
├── model/
│   ├── tablert_base_k3/
│   └── ...
├── output/
├── script/
└── README.md
```

