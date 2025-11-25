# CS6101 Programming Assignment 3  
### DeepImpact-Style Retriever + MRF-Based Reranker  
**Indian Institute of Technology Bombay**  
**Course:** Indexing and Retrieving Texts and Graphs (CS6101)  
**Instructor:** Prof. Soumen Chakraborty  

---

## Overview

This repository contains the implementation and report for **Programming Assignment 3** of CS6101 at IIT Bombay.  
The project involves designing a **two-stage neural retrieval pipeline** using BEIR datasets:

---

## Stage 1 — DeepImpact-Style Neural Retriever

Stage 1 implements a lightweight, BERT-based retriever inspired by **DeepImpact**.  
Key components:

- **Frozen BERT encoder** (`bert-base-uncased`)
- **Trainable MLP token-impact head** (predicts non-negative token weights)
- **Softmax pairwise ranking loss**  
- **(query, positive, negative)** triple generation from BEIR Quora dataset  
- **Impact-based posting list** creation  
- Retrieval performed via **token-impact accumulation**  

### Output of Stage 1
- `impact.pt` — trained impact model weights  
- `posting.pkl` — serialized posting list  
- A trained impact scorer usable for first-stage retrieval

---

## Stage 2 — MRF-Based Reranker (Frozen BERT)

Stage 2 reranks the top-100 documents using a **Markov Random Field model** inspired by Metzler & Croft (2005).

Three feature families:

1. **Unigram:** independent term matches  
2. **Ordered pairs:** phrase-level dependencies  
3. **Unordered pairs:** proximity-based term interactions  

Frozen BERT is used to generate:

- Document vector (mean-pooled token embeddings)  
- Query-term vectors + positions  

### Optimization
- Mixture weights **λₜ, λₒ, λᵤ** are tuned via **coordinate ascent**
- Primary objective: **maximize MAP**
- Metrics reported: **MAP**, **NDCG@10**

---

## Datasets Used (BEIR)

- **Quora**  
  - Used for Stage 1 training (triples)  
  - Also used in Stage 2 evaluation (first 100 queries)

- **TREC-COVID**  
  - Included in Stage 1 posting list  
  - Stage 2 only uses first 8 queries (as per PA3 instructions)

---

