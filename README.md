# Semantic Similarity Model Evaluation

A comprehensive evaluation framework for comparing semantic similarity performance across different transformer architectures including sentence transformers, encoder-only models, and decoder models.

## About The Project

This project provides a systematic evaluation of various transformer models on semantic similarity tasks using two key datasets: the STS Benchmark and Quora Question Pairs. The evaluation covers three main categories of models:

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

### Running STS Benchmark Evaluation

The main evaluation script tests models on the STS Benchmark dataset:

```bash
python semantic_similarity_evaluation.py
```

This will:
- Load the STS Benchmark dataset
- Evaluate all configured models
- Generate performance metrics
- Save results to CSV
- Create visualization charts

### Running Quora Question Pairs Evaluation

For duplicate question detection evaluation:

```bash
python quora_evaluation.py
```

### Customizing Model Configuration

Modify the `MODEL_CONFIGS` dictionary to add or remove models:

```python
MODEL_CONFIGS = {
    "your-model": {
        "type": "sentence_transformer",  # or "encoder" or "decoder"
        "model_name": "your-model-name"
    }
}
```

### Example Results

The framework generates comprehensive performance comparisons:

**STS Benchmark Results:**
- **Sentence Transformers**: Pearson r: 0.83-0.88, Spearman r: 0.83-0.88
- **Encoder Models**: Pearson r: 0.62-0.66, Spearman r: 0.65-0.68
- **Decoder Models**: Pearson r: 0.23-0.59, Spearman r: 0.22-0.61

**Quora Question Pairs Results:**
- **Best AUC Score**: all-mpnet-base-v2 (0.8946)
- **Fastest Processing**: DistilBERT (1.7s)
- **Most Efficient**: Sentence transformers provide best performance-to-speed ratio

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

## Results and Findings

### Key Insights

1. **Sentence Transformers** consistently outperform general-purpose models on similarity tasks
2. **all-mpnet-base-v2** provides the best overall performance across both datasets
3. **Decoder models** show limited effectiveness for similarity tasks in their base form
4. **Processing speed** varies significantly, with DistilBERT being fastest among competitive models

### Performance Rankings

**STS Benchmark (by Pearson correlation):**
1. all-mpnet-base-v2: 0.8806
2. all-MiniLM-L6-v2: 0.8696  
3. gtr-t5-base: 0.8298

**Quora Dataset (by AUC score):**
1. all-mpnet-base-v2: 0.8946
2. gtr-t5-base: 0.8787
3. all-MiniLM-L6-v2: 0.8681

## Contributing

Contributions are welcome! Here's how you can help:

1. **Add new models** to the evaluation framework
2. **Implement additional datasets** for broader evaluation
3. **Optimize performance** and memory usage
4. **Enhance visualizations** and reporting features
5. **Improve documentation** and examples

### Contributing Steps

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

AJ Woods - ajwoods@skidmore.edu

Project Link: [https://github.com/ajw109/semantic-similarity-research](https://github.com/ajw109/semantic-similarity-research)

## Acknowledgments

- [SentenceTransformers Library](https://www.sbert.net/) for excellent pre-trained models
- [Hugging Face](https://huggingface.co/) for transformer models and datasets
- [STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) for evaluation data
- [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset
- Research community for semantic similarity evaluation methodologies

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
