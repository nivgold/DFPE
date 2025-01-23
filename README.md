# DFPE: A Diverse Fingerprint Ensemble for Enhancing LLM Performance - Official Implementation

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Paper-ArXiv-green.svg" alt="Paper">
</p>

## Abstract
> Large Language Models (LLMs) have shown remarkable capabilities across various natural language processing tasks but often struggle to excel uniformly in diverse or complex domains. We propose a novel ensemble method - Diverse Fingerprint Ensemble (DFPE), which leverages the complementary strengths of multiple LLMs to achieve more robust performance. Our approach involves: (1) clustering models based on response "fingerprints" patterns, (2) introducing a quantile-based filtering mechanism to remove underperforming models at a per-subject level, and (3) assigning adaptive weights to remaining models based on their subject-wise validation accuracy. In experiments on the Massive Multitask Language Understanding (MMLU) benchmark, DFPE outperforms the best single model by 2.76\% in overall accuracy and 4.30\% in weighted accuracy. This method increase robustness and generalization of LLMs and underscore how model selection, diversity preservation, and performance-driven weighting can effectively address challenging, multi-faceted language understanding tasks.


## Getting Started

### Prerequisites
- Python >= 3.8
- PyTorch >= 2.0
- CUDA-compatible GPU (optional but recommended)


### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/.../dfpe.git
   cd dfpe
2. Install the dependencies
   ```bash
   pip install -r requirements.txt
## Usage

## Repository Stracture
