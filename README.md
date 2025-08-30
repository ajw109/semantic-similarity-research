### semantic-similarity-research

## Overview

The success of ChatGPT has brought unprecedented public attention to large language models (LLMs); and while the performance of these LLMs have been impressive across many tasks, their inner workings remain largely opaque to non-experts. This opacity raises a fundamental question: Do LLMs actually “understand” user input the way humans do? Or are they merely pattern-matching engines trained on vast corpora of data?

This research project investigates whether LLMs possess genuine semantic understanding or rely purely on surface-level statistical patterns learned from massive training datasets.

## Research Question

Our core investigation centers on a critical distinction in language processing:

- **Semantic Understanding**: Grasping the actual meaning and intent behind language
- **Statistical Pattern Matching**: Recognizing and reproducing patterns based on lexical similarities

## Methodology

We designed a comprehensive evaluation framework to test LLMs' ability to:

1. **Distinguish semantically different but lexically similar prompts**
   - Example: Testing if models can differentiate between prompts that use similar words but convey entirely different meanings

2. **Recognize semantically identical but lexically different prompts**
   - Example: Testing if models understand that different phrasings can express the same underlying intent

## Experimental Design

- **Models Tested**: 10 different LLMs
- **Architectural Coverage**: 3 distinct architectural paradigms
- **Evaluation Method**: Controlled comparison of model responses to carefully crafted prompt pairs

## Key Research Implications

This work addresses fundamental questions about:

- The nature of machine "understanding" vs. human comprehension
- The reliability of LLMs for tasks requiring true semantic reasoning
- The gap between impressive performance and genuine language understanding
- The limitations of current evaluation methods for LLMs
