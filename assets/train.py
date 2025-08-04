import sys
import os
import random
import warnings
import argparse
from random import randrange
import os
import warnings
import random
import torch
import logging


import numpy as np
from sklearn.metrics import f1_score
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Filter out the specific PyTorch warning about scalar tensors
warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.')


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # transformers library seed

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_tokenizer(model_id, model_max_length):
    """Initialize and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = model_max_length  # set max_length for prompts
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset using the provided tokenizer."""
    def tokenize_batch(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")
    
    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

def setup_model(model_id, num_labels, label2id, id2label, dropout_rate=0.2, device="cuda"):
    """Initialize and configure the model with dropout and other optimizations."""    
    #Get model config first
    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        attention_dropout=dropout_rate,          # 20% attention dropout
        embedding_dropout=dropout_rate,          # 20% embedding dropout
        mlp_dropout=dropout_rate,               # 20% MLP dropout
        classifier_dropout=dropout_rate,
        # problem_type="single_label_classification"
    )
    
    # Check if Flash Attention is available
    try:
        from flash_attn import __version__ as flash_attn_version
        logger.info(f"Flash Attention version {flash_attn_version} is installed")
        use_flash_attention = True
    except ImportError:
        logger.error("Flash Attention is not installed. Using standard attention.")
        use_flash_attention = False
    
    # Load model with config    
    # Download the model from huggingface.co/models
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        attn_implementation="flash_attention_2" if use_flash_attention else "eager",
    )
        
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    logger.info("Enabled gradient checkpointing for memory efficiency")

    model = model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params/1e6:.1f}M parameters ({trainable_params/1e6:.1f}M trainable)")
    
    return model

# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    return {"f1": float(score) if score == 1 else score}


def get_training_args(args):
    """Configure training arguments with stronger regularization."""
    return TrainingArguments(
        output_dir=args.model_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=float(args.learning_rate),
        num_train_epochs=args.epochs,
        bf16=True,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard",
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device detected: %s", device)

    # Set random seed for reproducibility
    set_random_seed(42)

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # Setup tokenizer and tokenize datasets
    tokenizer = setup_tokenizer(args.model_name, args.model_max_length)
    tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer)
    tokenized_eval_dataset = tokenize_dataset(test_dataset, tokenizer)

    # Setup label mappings
    unique_labels = list(set(tokenized_train_dataset['labels']))
    num_labels = len(unique_labels)
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    train_dataset = Dataset.from_dict({
        'input_ids': tokenized_train_dataset['input_ids'],
        'attention_mask': tokenized_train_dataset['attention_mask'],
        'labels': [label2id[label] for label in tokenized_train_dataset['labels']]
    })

    test_dataset = Dataset.from_dict({
        'input_ids': tokenized_eval_dataset['input_ids'],
        'attention_mask': tokenized_eval_dataset['attention_mask'],
        'labels': [label2id[label] for label in tokenized_eval_dataset['labels']]
    })

    # Setup model with dropout
    model = setup_model(
        args.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout_rate=args.dropout_rate,
        device=device
    )

    training_args = get_training_args(args)

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.005
        )]
    )

    # Train the model
    logger.info("\nStarting training...")
    trainer.train()
    logger.info("Training completed!")

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=tokenized_eval_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Save the model
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)