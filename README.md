# Semantic Similarity Model Evaluation

An evaluation framework for comparing semantic similarity performance across different transformer architectures including sentence transformers, encoder-only models, and decoder models.

## About The Project

This project systematically evaluates various transformer models on semantic similarity tasks using two key datasets: the STS Benchmark and Quora Question Pairs. The evaluation covers three main categories of models:

- **Sentence Transformers**: Fine-tuned models specifically designed for semantic similarity (all-MiniLM-L6-v2, all-mpnet-base-v2, gtr-t5-base)
- **Encoder Models**: BERT-family models adapted for similarity tasks (BERT-Pro, RoBERTa-base, DistilBERT)
- **Decoder Models**: GPT-family models evaluated for similarity understanding (GPT-2, Phi-4-mini, Qwen2, TinyLlama)

The framework evaluates models using multiple metrics including Pearson correlation, Spearman correlation, Mean Absolute Error (MAE), and AUC scores, while also measuring processing time and computational efficiency.

### Key Features

- Comprehensive model comparison across different architectures
- Multiple evaluation metrics and datasets
- Automated result visualization and reporting
- GPU acceleration support
- Memory-efficient batch processing
- Extensible framework for adding new models

### Built With

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [Sentence Transformers](https://www.sbert.net/)
- [Datasets](https://huggingface.co/docs/datasets/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You'll also need CUDA support for GPU acceleration (recommended).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/ajw109/semantic-similarity-research.git
   cd semantic-similarity-research
   ```

2. Install required packages
   ```bash
   pip install transformers sentence-transformers datasets scikit-learn matplotlib pandas scipy numpy
   ```

3. Verify CUDA availability (optional but recommended)
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

## Usage

### Dataset Information

The evaluation uses two key datasets:

- **STS Benchmark**: [PhilipMay/stsb_multi_mt](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt) - Semantic Textual Similarity benchmark with similarity scores from 0-5
- **Quora Question Pairs**: [Kaggle Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset/data) - Binary classification task for identifying duplicate questions

### Running the Evaluations

The project contains three Jupyter notebooks for different experiments:

#### STS Benchmark Evaluation
```bash
jupyter notebook experiment1.ipynb
```
This notebook:
- Automatically loads the STS Benchmark dataset from HuggingFace
- Evaluates models using mean pooling strategy
- Generates performance metrics (Pearson/Spearman correlations, MAE)
- Creates visualization charts comparing different model architectures

#### Quora Question Pairs Evaluation
```bash
jupyter notebook experiment2.ipynb
```
This notebook:
- Loads the Quora Question Pairs dataset
- Computes AUC scores for duplicate question detection
- Measures processing time and efficiency
- Generates comparative visualizations

#### Mean Pooling Strategy Analysis
```bash
jupyter notebook meanpooling.ipynb
```
This notebook:
- Compares different pooling strategies for embeddings
- Analyzes the impact of pooling methods on performance
- Provides detailed analysis of embedding extraction techniques

### Customizing Model Configuration

Within the notebooks, modify the `MODEL_CONFIGS` dictionary to add or remove models:

```python
MODEL_CONFIGS = {
    "your-model": {
        "type": "sentence_transformer",  # or "encoder" or "decoder"
        "model_name": "your-model-name"
    }
}
```

## Evaluation Metrics

### Semantic Similarity (STS Benchmark)
- **Pearson Correlation**: Linear relationship between predicted and ground truth similarity
- **Spearman Correlation**: Monotonic relationship assessment  
- **Mean Absolute Error**: Average prediction error
- **Processing Time**: Model inference speed

### Duplicate Detection (Quora Dataset)
- **AUC Score**: Area under ROC curve for binary classification
- **Mean Cosine Similarity**: Average similarity scores
- **Processing Time**: Inference efficiency

### Performance Rankings

**STS Benchmark (by Pearson correlation):**
1. all-mpnet-base-v2: 0.8806
2. all-MiniLM-L6-v2: 0.8696  
3. gtr-t5-base: 0.8298

**Quora Dataset (by AUC score):**
1. all-mpnet-base-v2: 0.8946
2. gtr-t5-base: 0.8787
3. all-MiniLM-L6-v2: 0.8681

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

AJ Woods - ajwoods@skidmore.edu

Project Link: [https://github.com/ajw109/semantic-similarity-research](https://github.com/ajw109/semantic-similarity-research)

## Acknowledgments

- [SentenceTransformers Library](https://www.sbert.net/) for its pre-trained models
- [Hugging Face](https://huggingface.co/) for transformer models and datasets
- [STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) for its dataset
- [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) for its dataset

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{semantic-similarity-evaluation,
  title={Semantic Similarity Model Evaluation Framework},
  author={AJ Woods},
  year={2025},
  publisher={GitHub},
  url={https://github.com/ajw109/semantic-similarity-research}
}
```
