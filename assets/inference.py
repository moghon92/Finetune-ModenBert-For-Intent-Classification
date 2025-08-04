import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
os.environ["TOKENIZERS_PARALLELISM"] = "false"

JSON_CONTENT_TYPE = 'application/json'

# Define global variables for model and tokenizer
tokenizer = None
model = None

def model_fn(model_dir):
    """
    Load the model and tokenizer for inference.
    
    Args:
        model_dir (str): The directory where model artifacts are stored
        
    Returns:
        tuple: The loaded model and tokenizer
    """
    global tokenizer, model
    print("model_dir", model_dir)

    # Handle checkpoint selection
    checkpoint_selection = os.environ.get('CHECKPOINT_SELECTION', 'latest')
    model_checkpoint = os.environ.get('MODEL_CHECKPOINT', 'checkpoint-1188')
    
    try:
        # Determine which checkpoint to load
        if checkpoint_selection == 'specific':
            model_path = os.path.join(model_dir, model_checkpoint)
            if not os.path.exists(model_path):
                raise ValueError(f"Specified checkpoint {model_checkpoint} not found")
                
        elif checkpoint_selection == 'latest':
            checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
            if not checkpoints:
                print("No checkpoints found, using main model directory")
                model_path = model_dir
            else:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                model_path = os.path.join(model_dir, latest_checkpoint)
        else:
            model_path = model_dir
            
        print(f"Loading model from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from flash_attn import __version__ as flash_attn_version
        print(f"Flash Attention version {flash_attn_version} is installed")
        use_flash_attention = True
    except Exception:
        print("Flash Attention is not installed. Using standard attention.")
        use_flash_attention = False

        
    attn_implementation = "flash_attention_2" if use_flash_attention else "eager"
    print("Using attention implementation: %s", attn_implementation)
    
    # Load tokenizer and model from the saved model directory
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    tokenizer.model_max_length = 64
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        attn_implementation=attn_implementation,
    )
    label_map = model.config.id2label
    model = model.to(device)
    # Put model in evaluation mode
    model.eval()
    
    return {
        'model': model, 
        'tokenizer': tokenizer, 
        'device': device,
        "label_map": label_map
    }

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    
    Args:
        request_body (str): The request body
        request_content_type (str): The content type of the request
        
    Returns:
        dict: The deserialized request
    """
    if request_content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(request_body)
        print("Input : %s", input_data)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """
    Apply model to the incoming request.
    
    Args:
        input_data (dict): The input data in deserialized format
        model_and_tokenizer (tuple): The model and tokenizer
        
    Returns:
        dict: The prediction result
    """
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    label_map = model_dict["label_map"]

    model = model.to(device)
    
    # Extract text from the input
    text = input_data.get('text', '')
    print("Input parsed successfully: %s", text)

    
    # Tokenize the input text
    inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt").to(device)
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the predicted class and its probability
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    # Return the prediction
    return {
        "predicted_class": label_map[predicted_class],
        "confidence": confidence,
        # "probabilities": probabilities[0].tolist()
    }

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output.
    
    Args:
        prediction (dict): The prediction result
        response_content_type (str): The content type of the response
        
    Returns:
        str: The serialized prediction
    """
    if response_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
