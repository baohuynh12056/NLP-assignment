# src/fine_tuning/lora_utils.py
import torch
import numpy as np
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

class ActivationAnalyzer:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.hf_model = model[0].auto_model  # Access core HuggingFace model (e.g., BertModel)
        self.layer_activations = {}
        self.hooks = []

    def _get_hook(self, layer_idx):
        def hook(model, input, output):
            # output of BertLayer is a tuple, first element is hidden_states (batch_size, seq_len, hidden_size)
            hidden_states = output[0].detach()
            
            # Calculate Variance of activation based on the Activation-Guided paper
            # We use the mean of the standard deviation along the hidden_size dimension
            activation_score = hidden_states.std(dim=-1).mean().item()
            
            if layer_idx not in self.layer_activations:
                self.layer_activations[layer_idx] = []
            self.layer_activations[layer_idx].append(activation_score)
        return hook

    def register_hooks(self):
        self.remove_hooks()
        # Iterate through encoder layers (e.g., model.encoder.layer for BERT/BGE architectures)
        layers = self.hf_model.encoder.layer
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._get_hook(idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def analyze(self, dataloader: DataLoader, num_batches: int = 50) -> list:
        """Run forward pass to analyze activations across layers"""
        self.layer_activations = {}
        self.register_hooks()
        
        device = self.hf_model.device
        self.model.eval()
        
        print(f"Analyzing activations over {num_batches} batches...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
                if i >= num_batches: break
                
                # Extract text from InputExample of SentenceTransformers
                texts = [example.texts[0] for example in batch] 
                # Tokenize inputs
                if hasattr(self.model, "preprocess"):
                    features = self.model.preprocess(texts)
                else:
                    features = self.model.tokenize(texts)
                
                features = {k: v.to(device) for k, v in features.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass to trigger hooks
                self.hf_model(**features)
                
        self.remove_hooks()
        
        # Calculate mean score for each layer
        layer_scores = {}
        for layer_idx, scores in self.layer_activations.items():
            layer_scores[layer_idx] = np.mean(scores)
            
        # Sort layers by score descending (highest scoring layer first)
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_layers


def apply_selective_lora(
    model: SentenceTransformer, 
    target_layers: list[int], 
    r: int = 16, 
    lora_alpha: int = 32,
    lora_dropout: float = 0.1
) -> SentenceTransformer:
    """
    Apply LoRA only to specified layers, freezing the rest to save VRAM and prevent catastrophic forgetting.
    """
    hf_model = model[0].auto_model
    
    print(f"Injecting LoRA adapters into layers: {target_layers}")
    
    # BGE uses BERT architecture, attention modules are typically named 'query', 'key', 'value'
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"], # Focus LoRA on Q and V projections
        layers_to_transform=target_layers, # SMART LAYER SELECTION FROM PAPER
        bias="none"
    )
    
    # Wrap HF model with PEFT (Automatically freezes layers without LoRA)
    peft_model = get_peft_model(hf_model, peft_config)
    peft_model.print_trainable_parameters()
    
    # Reassign the PEFT model back into SentenceTransformer pipeline
    model[0].auto_model = peft_model
    return model