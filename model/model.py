"""
Base model loading and configuration for TinyZero A*PO.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig
)
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def load_base_model(
    model_name: str, 
    device: Optional[torch.device] = None,
    torch_dtype: torch.dtype = torch.bfloat16
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Load base language model and tokenizer.
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-3B")
        device: Device to load model on (None for auto-detection)
        torch_dtype: Data type for model weights
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading model {model_name} on device {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"  # Important for generation
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model configuration
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=None  # We'll handle device placement manually
    )
    
    # Move to device
    model = model.to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def setup_model_for_training(
    model: nn.Module,
    freeze_embeddings: bool = False,
    freeze_layers: Optional[List[int]] = None
) -> nn.Module:
    """
    Configure model for RL training.
    
    Args:
        model: Base language model
        freeze_embeddings: Whether to freeze embedding layers
        freeze_layers: List of layer indices to freeze (None for no freezing)
        
    Returns:
        Configured model
    """
    # Freeze embeddings if requested
    if freeze_embeddings:
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens.requires_grad_(False)
            logger.info("Frozen embedding layer")
    
    # Freeze specific layers if requested
    if freeze_layers is not None:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer_idx in freeze_layers:
                if 0 <= layer_idx < len(model.model.layers):
                    model.model.layers[layer_idx].requires_grad_(False)
                    logger.info(f"Frozen layer {layer_idx}")
    
    # Set all parameters to require gradients (except frozen ones)
    for param in model.parameters():
        if param.requires_grad is None:
            param.requires_grad = True
    
    logger.info(f"Model configured for training. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def prepare_model_inputs(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    max_length: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and prepare batch inputs for the model.
    
    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        device: Device to place tensors on
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenize prompts
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def get_generation_config(
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 200,
    do_sample: bool = True,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None
) -> GenerationConfig:
    """
    Create generation configuration.
    
    Args:
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_new_tokens: Maximum new tokens to generate
        do_sample: Whether to use sampling
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID
        
    Returns:
        GenerationConfig object
    """
    return GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=False,
        output_hidden_states=False
    )


def generate_with_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_config: GenerationConfig,
    return_dict: bool = True
) -> Union[torch.Tensor, Dict]:
    """
    Generate text with log probabilities.
    
    Args:
        model: Language model
        input_ids: Input token IDs
        attention_mask: Attention mask
        generation_config: Generation configuration
        return_dict: Whether to return dictionary with scores
        
    Returns:
        Generated tokens and optionally log probabilities
    """
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=return_dict
        )
    
    if return_dict:
        return {
            "sequences": outputs.sequences,
            "scores": outputs.scores if hasattr(outputs, 'scores') else None
        }
    else:
        return outputs.sequences


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probabilities for given sequences.
    
    Args:
        model: Language model
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: Target token IDs
        
    Returns:
        Log probabilities for each token
    """
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    # Get log probabilities from logits
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Extract log probabilities for target tokens
    batch_size, seq_len = labels.shape
    log_probs_selected = torch.gather(
        log_probs.view(-1, log_probs.size(-1)),
        1,
        labels.view(-1).unsqueeze(1)
    ).view(batch_size, seq_len)
    
    return log_probs_selected


def get_model_info(model: nn.Module) -> Dict:
    """
    Get information about the model.
    
    Args:
        model: Language model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "device": next(model.parameters()).device,
        "dtype": next(model.parameters()).dtype
    }


if __name__ == "__main__":
    # Test model loading
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with a smaller model for local testing
    model_name = "Qwen/Qwen2.5-1.5B"  # Use smaller model for testing
    
    try:
        model, tokenizer = load_base_model(model_name)
        model = setup_model_for_training(model)
        
        # Test generation
        prompts = ["Solve step by step:\n12 * 34 = "]
        inputs = prepare_model_inputs(prompts, tokenizer)
        
        generation_config = get_generation_config(max_new_tokens=50)
        outputs = generate_with_logprobs(
            model, 
            inputs["input_ids"], 
            inputs["attention_mask"], 
            generation_config
        )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        
        # Print model info
        model_info = get_model_info(model)
        print(f"Model info: {model_info}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This is expected if the model is not available locally.")
