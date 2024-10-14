# WebSector: Multi-Sector Website Classification Framework

This repository contains the code and resources for the WebSector project, a novel approach to multi-sector website classification using the Single Positive Label (SPL) paradigm.

## Dataset

The WebSector Corpus, a large-scale dataset for sector-based website classification, is available on Hugging Face:

[WebSector Corpus](https://huggingface.co/datasets/Shahriar/websector-corpus)

This dataset includes 195,495 websites categorized into ten distinct web sectors, providing a robust foundation for training and evaluating website classification models.

## Models

We provide two pre-trained models, each optimized for different use cases:

1. [WebSector-More-sector](https://huggingface.co/Shahriar/WebSector-Flexible/): Designed for broader multi-sector predictions.
2. [WebSector-Fewer-Sector](https://huggingface.co/Shahriar/WebSector-Conservative/): Focused on primary sector identification with limited multi-sector classifications.

These models represent different modes of the WebSector framework, allowing users to choose based on their specific requirements.

## Pipeline

The WebSector inference pipeline is implemented in `websector_inference/websector_pipeline.py`. This pipeline handles the entire process from input (raw text or URL) to final sector predictions.

To test the pipeline, you can use the Jupyter notebook provided:

`websector_inference/websector_pipeline_test.ipynb`

This notebook demonstrates how to use the WebSector pipeline for both text and URL inputs, showcasing the framework's versatility.

## Usage

To use the WebSector framework:

1. Clone this repository
2. Install the required dependencies (requirements will be provided in the full release)
3. Download the pre-trained models from the Hugging Face links above
4. Use the `websector_pipeline.py` script to make predictions on new websites or text content

Detailed usage instructions and examples will be provided in the full release.

## Citation

During the anonymity period, please refer to this work as "WebSector" in any related research or applications. A formal citation will be provided upon publication.

## Acknowledgments

We thank the anonymous reviewers for their valuable feedback. We also acknowledge the use of public resources that contributed to this research.

---

Note: This README is designed for the anonymity period. It will be updated with more detailed information, including installation instructions, full usage examples, and proper citations, upon the conclusion of the peer review process.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/33719550/0bb6c247-4582-4f08-a0ee-15a267e4f755/WebConf2025_Shahriar_Final-1.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/33719550/df639e12-196c-4340-bf6c-6779670ee463/WebConf2025_Shahriar-6.pdf
