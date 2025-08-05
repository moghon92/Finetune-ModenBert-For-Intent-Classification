# Fine-tuning ModernBERT for Intent Classification

This notebook demonstrates how to fine-tune the ModernBERT model for intent classification using Amazon SageMaker. The process involves preparing the dataset, training the model, and deploying it for inference.

## A Brief Look at ModernBERT

ModernBERT represents a significant advancement in encoder model technology, offering full backwards compatibility while introducing several major architectural improvements over the original BERT. The model comes in two variations:

- ModernBERT Base (149M parameters)
- ModernBERT Large (395M parameters)

### Key Technical Innovations:

    - Rotary Positional Embeddings (RoPE) replace traditional positional encodings, enabling better understanding of word relationships and supporting longer sequences
    - Alternating Attention patterns that combine global and local attention every 3 layers, significantly improving processing efficiency
    - GeGLU layers replace traditional MLP layers, enhancing the original BERT's GeLU activation function
    - Streamlined architecture with removed bias terms for more efficient parameter usage
    - Additional normalization layer after embeddings for improved training stability
    - Advanced sequence packing and unpadding techniques that reduce computational waste
    - Hardware-optimized design that better aligns with modern GPU architectures

The model sets new performance standards in classification, retrieval, and code comprehension tasks, operating 2-4 times faster than previous encoders. This combination of speed and accuracy makes it ideal for high-volume production applications such as LLM routing, where both performance metrics are crucial.

What sets ModernBERT apart is its extensive training on 2 trillion tokens from diverse sources, including web content, programming code, and academic literature. This broad training foundation - significantly more varied than traditional BERT models' Wikipedia-centric approach - enables better understanding of user inputs across multiple domains. The model also features an impressive 8,192 token context length, which is 16 times larger than most existing encoders.

For detailed information about ModernBERT's architecture and development, you can find more documentation on [Hugging Face](https://huggingface.co/blog/modernbert).


## Contents

1. Dataset Preparation
2. Model Fine-tuning
3. Model Deployment
4. Test Predictions
5. Cleanup

## Prerequisites

- An AWS account with SageMaker access
- Python 3.x
- Required libraries: sagemaker, torch, transformers, datasets, sklearn

## Setup

1. Clone this repository
2. Open the notebook in SageMaker Studio or a Jupyter environment
3. Ensure you have the necessary permissions to create SageMaker resources

## Usage

Follow the notebook cells sequentially to:

1. Prepare and analyze the dataset
2. Define training hyperparameters
3. Create a SageMaker estimator and train the model
4. Deploy the trained model to a SageMaker endpoint
5. Make test predictions using the deployed endpoint

## Key Components

- `assets/train.py`: The main training script
- `assets/inference.py`: Script for model inference
- `requirements.txt`: List of required Python packages

## Model

This notebook uses the `answerdotai/ModernBERT-base` model, which is an enhanced version of BERT with improved performance and efficiency.

## Dataset

The dataset used is a travel-related intent classification dataset. It's split into training and testing sets and prepared for the specific task of intent classification.

## Deployment

The model is deployed to a SageMaker endpoint for real-time inference. The notebook demonstrates how to create and interact with this endpoint.

## Cleanup

After you're done experimenting, make sure to delete the SageMaker endpoint to avoid unnecessary charges.

## Note

This notebook is for demonstration purposes. In a production environment, you may need to adjust parameters, add error handling, and implement additional security measures.

For any issues or questions, please open an issue in the repository.
