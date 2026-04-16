# HALTaR: Hallucination-Augmented Lookup for Table Retrieval

In the data-driven era, vast amounts of knowledge are stored in structured tables across domains, making natural language (NL)–based table retrieval a key component for open-domain QA and downstream analytics. However, aligning unstructured NL queries with the structural and semantic properties of tables remains challenging. Prior methods primarily rely on keyword matching or treat queries and tables as homogeneous modalities, which limits retrieval effectiveness.

We propose **HALTaR**, a corpus-augmented table retrieval framework that integrates a structure-aware lightweight retriever with an LLM-based query-to-table hallucinator. The hallucinator transforms NL queries into structured, intent-centric tables, which are then embedded and matched against the corpus. This design improves robustness to schema variability and incomplete queries. Experiments on SPIDER and BIRD show that HALTaR consistently outperforms strong baselines, demonstrating its practical effectiveness for query-to-table retrieval.

## 1. Directory Overview
The root directory contains four folders: `dataset`, `model`, `output`, and `script`. All folders except `script` are shared via a drive link.  
The directory structure is organized as follows:

```bash
project-root/
├── dataset/
│   ├── bird/
│   └── spider/
|   └── ...
├── model/
│   ├── tablert_base_k3/
│   └── ...
├── output/
├── script/
└── README.md
```

