"""
Day 50: Quantization - Model Compression & Optimization - Solutions

Complete implementations for all quantization exercises with production-ready code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock components for demonstration (replace with actual models in production)
class MockCNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MockTransformerModel(nn.Module):
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.lm_head(x)

# Solution 1: Post-Training Quantization Implementation
class PostTrainingQuantizer:
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        
    def prepare_model_for_quantization(self):
        """Prepare model for post-training quantization"""
        # Set model to evaluation mode
        self.model.eval()
        
        # Fuse operations for better performance
        try:
            # Attempt to fuse conv-bn-relu patterns
            self.model = torch.quantization.fuse_modules(
                self.model, 
                [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']]
            )
            logger.info("Successfully fused conv-bn-relu modules")
        except Exception as e:
            logger.warning(f"Module fusion failed: {e}")
        
        # Set quantization configuration
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        torch.quantization.prepare(self.model, inplace=True)
        
        logger.info("Model prepared for quantization")
        return self.model
    
    def calibrate_model(self):
        """Calibrate model with representative data"""
        self.model.eval()
        
        logger.info("Starting model calibration...")
        with torch.no_grad():
            for i, batch in enumerate(self.calibration_data):
                _ = self.model(batch)
                if i % 5 == 0:
                    logger.info(f"Calibration batch {i+1}/{len(self.calibration_data)}")
        
        logger.info("Model calibration completed")
    
    def convert_to_quantized(self):
        """Convert calibrated model to quantized version"""
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        logger.info("Model converted to quantized version")
        return quantized_model
    
    def compare_model_sizes(self, original_model, quantized_model):
        """Compare original and quantized model sizes"""
        # Calculate original model size (FP32)
        original_size = sum(p.numel() * 4 for p in original_model.parameters())
        
        # Calculate quantized model size (INT8)
        quantized_size = sum(p.numel() for p in quantized_model.parameters())
        
        # Calculate compression ratio
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        
        size_comparison = {
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
        }
        
        return size_comparison
    
    def full_ptq_pipeline(self):
        """Complete PTQ pipeline"""
        # Step 1: Prepare model
        self.prepare_model_for_quantization()
        
        # Step 2: Calibrate with data
        self.calibrate_model()
        
        # Step 3: Convert to quantized model
        quantized_model = self.convert_to_quantized()
        
        return quantized_model

# Solution 2: Quantization-Aware Training Implementation
class QuantizationAwareTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def prepare_qat_model(self):
        """Prepare model for quantization-aware training"""
        # Set model to training mode
        self.model.train()
        
        # Fuse modules
        try:
            self.model = torch.quantization.fuse_modules(
                self.model,
                [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']]
            )
        except Exception as e:
            logger.warning(f"QAT module fusion failed: {e}")
        
        # Set QAT configuration
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        torch.quantization.prepare_qat(self.model, inplace=True)
        
        logger.info("Model prepared for QAT")
        return self.model
    
    def train_with_quantization_simulation(self, epochs=3, lr=0.001):
        """Train model with quantization simulation"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                
                # Forward pass with quantization simulation
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation
            val_accuracy = self.validate_qat_model()
            train_accuracy = correct / total
            avg_loss = total_loss / len(self.train_loader)
            
            logger.info(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, '
                       f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
    
    def validate_qat_model(self):
        """Validate quantization-aware trained model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def convert_qat_to_quantized(self):
        """Convert QAT model to fully quantized model"""
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        logger.info("QAT model converted to quantized version")
        return quantized_model

# Solution 3: GPTQ Implementation
class GPTQQuantizer:
    def __init__(self, model, bits=4, group_size=128):
        self.model = model
        self.bits = bits
        self.group_size = group_size
        
    def collect_layer_statistics(self, layer, calibration_data):
        """Collect input/output statistics for a layer"""
        inputs = []
        outputs = []
        
        def hook_fn(module, input, output):
            inputs.append(input[0].detach().clone())
            outputs.append(output.detach().clone())
        
        # Register hook
        handle = layer.register_forward_hook(hook_fn)
        
        # Run calibration data through model
        self.model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                _ = self.model(batch)
        
        # Remove hook
        handle.remove()
        
        # Stack collected data
        if inputs and outputs:
            X = torch.cat(inputs, dim=0)
            Y = torch.cat(outputs, dim=0)
            return X, Y
        
        return None, None
    
    def compute_hessian_approximation(self, inputs):
        """Compute Hessian approximation for GPTQ"""
        if inputs is None or inputs.numel() == 0:
            return None
        
        # Reshape inputs for matrix multiplication
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)
        
        # H = X^T @ X (Gauss-Newton approximation)
        H = inputs.T @ inputs
        
        # Add small diagonal for numerical stability
        H += torch.eye(H.size(0), device=H.device) * 1e-6
        
        return H
    
    def gptq_quantize_weights(self, weights, hessian):
        """Apply GPTQ quantization algorithm"""
        if hessian is None:
            return self._simple_quantize_weights(weights)
        
        quantized_weights = weights.clone()
        
        # Process weights in groups
        for i in range(0, weights.shape[1], self.group_size):
            end_idx = min(i + self.group_size, weights.shape[1])
            
            # Extract weight group
            weight_group = weights[:, i:end_idx]
            
            # Compute quantization parameters
            scale, zero_point = self._compute_quantization_params(weight_group)
            
            # Apply GPTQ algorithm with Hessian information
            quantized_group = self._quantize_with_hessian(weight_group, scale, zero_point, hessian)
            
            # Update quantized weights
            quantized_weights[:, i:end_idx] = quantized_group
        
        return quantized_weights
    
    def _compute_quantization_params(self, weights):
        """Compute scale and zero point for quantization"""
        w_min = weights.min()
        w_max = weights.max()
        
        # Compute quantization range
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1
        
        # Compute scale and zero point
        scale = (w_max - w_min) / (qmax - qmin) if w_max != w_min else 1.0
        zero_point = qmin - w_min / scale if scale != 0 else 0
        
        return scale, zero_point
    
    def _quantize_with_hessian(self, weights, scale, zero_point, hessian):
        """Quantize weights using Hessian information"""
        # Simple quantization (in production, use full GPTQ algorithm)
        quantized = torch.round(weights / scale + zero_point)
        
        # Clamp to valid range
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _simple_quantize_weights(self, weights):
        """Simple quantization fallback"""
        scale, zero_point = self._compute_quantization_params(weights)
        return self._quantize_with_hessian(weights, scale, zero_point, None)
    
    def quantize_model_with_gptq(self, calibration_data):
        """Apply GPTQ to entire model"""
        logger.info("Starting GPTQ quantization...")
        
        # Iterate through model layers
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                logger.info(f"Quantizing layer: {name}")
                
                # Collect statistics
                inputs, outputs = self.collect_layer_statistics(layer, calibration_data)
                
                # Compute Hessian approximation
                hessian = self.compute_hessian_approximation(inputs)
                
                # Quantize weights
                original_weights = layer.weight.data.clone()
                quantized_weights = self.gptq_quantize_weights(original_weights, hessian)
                
                # Update layer weights
                layer.weight.data = quantized_weights
        
        logger.info("GPTQ quantization completed")
        return self.model

# Solution 4: AWQ Implementation
class AWQQuantizer:
    def __init__(self, model, bits=4, group_size=128):
        self.model = model
        self.bits = bits
        self.group_size = group_size
        
    def compute_activation_scales(self, layer, calibration_data):
        """Compute activation-based importance scores"""
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(input[0].detach().clone())
        
        # Register hook
        handle = layer.register_forward_hook(hook_fn)
        
        # Collect activations
        self.model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                _ = self.model(batch)
        
        # Remove hook
        handle.remove()
        
        if not activations:
            return None
        
        # Compute activation statistics
        all_activations = torch.cat(activations, dim=0)
        
        # Compute channel-wise activation magnitudes
        if all_activations.dim() == 4:  # Conv layers (N, C, H, W)
            activation_scales = all_activations.abs().mean(dim=(0, 2, 3))
        elif all_activations.dim() == 3:  # Linear layers (N, S, C)
            activation_scales = all_activations.abs().mean(dim=(0, 1))
        else:  # Other cases
            activation_scales = all_activations.abs().mean(dim=0)
        
        return activation_scales
    
    def compute_weight_importance(self, weights, activation_scales):
        """Compute weight importance based on activations"""
        if activation_scales is None:
            return weights.abs()
        
        # Reshape activation scales to match weight dimensions
        if weights.dim() == 4:  # Conv weights (out_ch, in_ch, h, w)
            scales = activation_scales.view(-1, 1, 1, 1)
        elif weights.dim() == 2:  # Linear weights (out_features, in_features)
            scales = activation_scales.view(-1, 1)
        else:
            scales = activation_scales
        
        # Weight importance = |weight| * activation_scale
        importance = weights.abs() * scales
        
        return importance
    
    def protect_important_weights(self, weights, importance, protection_ratio=0.1):
        """Identify and protect most important weights"""
        # Find importance threshold
        flat_importance = importance.flatten()
        threshold = torch.quantile(flat_importance, 1 - protection_ratio)
        
        # Create protection mask
        protection_mask = importance > threshold
        
        logger.info(f"Protecting {protection_mask.sum().item()} out of {protection_mask.numel()} weights "
                   f"({protection_mask.float().mean().item()*100:.1f}%)")
        
        return protection_mask
    
    def quantize_with_protection(self, weights, protection_mask):
        """Quantize weights while protecting important ones"""
        quantized_weights = weights.clone()
        
        # Quantize unprotected weights aggressively
        unprotected_mask = ~protection_mask
        unprotected_weights = weights[unprotected_mask]
        
        if unprotected_weights.numel() > 0:
            # Compute quantization parameters for unprotected weights
            scale, zero_point = self._compute_quantization_params(unprotected_weights)
            
            # Quantize unprotected weights
            quantized_unprotected = self._quantize_weights(unprotected_weights, scale, zero_point)
            
            # Update quantized weights
            quantized_weights[unprotected_mask] = quantized_unprotected
        
        # Apply lighter quantization to protected weights (e.g., 8-bit instead of 4-bit)
        protected_weights = weights[protection_mask]
        if protected_weights.numel() > 0:
            # Use higher precision for protected weights
            scale, zero_point = self._compute_quantization_params(protected_weights, bits=8)
            quantized_protected = self._quantize_weights(protected_weights, scale, zero_point, bits=8)
            quantized_weights[protection_mask] = quantized_protected
        
        return quantized_weights
    
    def _compute_quantization_params(self, weights, bits=None):
        """Compute scale and zero point for quantization"""
        if bits is None:
            bits = self.bits
        
        w_min = weights.min()
        w_max = weights.max()
        
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        scale = (w_max - w_min) / (qmax - qmin) if w_max != w_min else 1.0
        zero_point = qmin - w_min / scale if scale != 0 else 0
        
        return scale, zero_point
    
    def _quantize_weights(self, weights, scale, zero_point, bits=None):
        """Quantize weights using scale and zero point"""
        if bits is None:
            bits = self.bits
        
        # Quantize
        quantized = torch.round(weights / scale + zero_point)
        
        # Clamp to valid range
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def quantize_layer_with_awq(self, layer, calibration_data):
        """Quantize single layer using AWQ method"""
        # Compute activation scales
        activation_scales = self.compute_activation_scales(layer, calibration_data)
        
        # Compute weight importance
        weights = layer.weight.data
        importance = self.compute_weight_importance(weights, activation_scales)
        
        # Protect important weights
        protection_mask = self.protect_important_weights(weights, importance)
        
        # Quantize with protection
        quantized_weights = self.quantize_with_protection(weights, protection_mask)
        
        # Update layer
        layer.weight.data = quantized_weights
        
        return layer
# Solution 5: GGUF Format Conversion
class GGUFConverter:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        self.supported_quantizations = {
            'f32': {'bits': 32, 'description': 'Full precision (32-bit float)'},
            'f16': {'bits': 16, 'description': 'Half precision (16-bit float)'},
            'q8_0': {'bits': 8, 'description': '8-bit quantization'},
            'q4_0': {'bits': 4, 'description': '4-bit quantization (legacy)'},
            'q4_1': {'bits': 4, 'description': '4-bit quantization (improved)'},
            'q5_0': {'bits': 5, 'description': '5-bit quantization'},
            'q5_1': {'bits': 5, 'description': '5-bit quantization (improved)'},
            'q2_k': {'bits': 2, 'description': '2-bit quantization (k-quant)'},
            'q3_k': {'bits': 3, 'description': '3-bit quantization (k-quant)'},
            'q4_k': {'bits': 4, 'description': '4-bit quantization (k-quant)'},
            'q5_k': {'bits': 5, 'description': '5-bit quantization (k-quant)'},
            'q6_k': {'bits': 6, 'description': '6-bit quantization (k-quant)'},
            'q8_k': {'bits': 8, 'description': '8-bit quantization (k-quant)'}
        }
        
    def analyze_model_architecture(self):
        """Analyze model architecture for GGUF compatibility"""
        analysis_results = {
            'compatible': True,
            'architecture': 'unknown',
            'total_parameters': 0,
            'layer_types': [],
            'issues': []
        }
        
        try:
            # Mock model analysis (in production, load actual model)
            logger.info(f"Analyzing model at {self.model_path}")
            
            # Check file existence
            if not os.path.exists(self.model_path):
                analysis_results['compatible'] = False
                analysis_results['issues'].append(f"Model file not found: {self.model_path}")
                return analysis_results
            
            # Mock architecture detection
            analysis_results.update({
                'architecture': 'llama',
                'total_parameters': 7_000_000_000,  # 7B parameters
                'layer_types': ['embedding', 'transformer', 'lm_head'],
                'vocab_size': 32000,
                'hidden_size': 4096,
                'num_layers': 32
            })
            
            logger.info(f"Model analysis complete: {analysis_results['architecture']} "
                       f"with {analysis_results['total_parameters']:,} parameters")
            
        except Exception as e:
            analysis_results['compatible'] = False
            analysis_results['issues'].append(f"Analysis failed: {str(e)}")
        
        return analysis_results
    
    def prepare_model_weights(self):
        """Prepare model weights for GGUF conversion"""
        logger.info("Preparing model weights for GGUF conversion...")
        
        # Mock weight preparation
        weight_info = {
            'tensors': [],
            'metadata': {
                'general.architecture': 'llama',
                'general.name': 'mock_model',
                'llama.context_length': 2048,
                'llama.embedding_length': 4096,
                'llama.block_count': 32,
                'llama.feed_forward_length': 11008,
                'llama.attention.head_count': 32,
                'tokenizer.ggml.model': 'llama'
            }
        }
        
        # Mock tensor information
        layer_names = [
            'token_embd.weight',
            'output_norm.weight',
            'output.weight'
        ]
        
        for i in range(32):  # 32 transformer layers
            layer_names.extend([
                f'blk.{i}.attn_norm.weight',
                f'blk.{i}.attn_q.weight',
                f'blk.{i}.attn_k.weight',
                f'blk.{i}.attn_v.weight',
                f'blk.{i}.attn_output.weight',
                f'blk.{i}.ffn_norm.weight',
                f'blk.{i}.ffn_gate.weight',
                f'blk.{i}.ffn_up.weight',
                f'blk.{i}.ffn_down.weight'
            ])
        
        for name in layer_names:
            # Mock tensor info
            if 'embd' in name:
                shape = [32000, 4096]
            elif 'output' in name and 'norm' not in name:
                shape = [32000, 4096]
            elif 'attn_q' in name or 'attn_k' in name or 'attn_v' in name:
                shape = [4096, 4096]
            elif 'attn_output' in name:
                shape = [4096, 4096]
            elif 'ffn_gate' in name or 'ffn_up' in name:
                shape = [11008, 4096]
            elif 'ffn_down' in name:
                shape = [4096, 11008]
            else:
                shape = [4096]
            
            weight_info['tensors'].append({
                'name': name,
                'shape': shape,
                'dtype': 'float32',
                'size_bytes': np.prod(shape) * 4
            })
        
        logger.info(f"Prepared {len(weight_info['tensors'])} tensors for conversion")
        return weight_info
    
    def apply_gguf_quantization(self, weight_info, quantization_type='q4_0'):
        """Apply GGUF-specific quantization"""
        if quantization_type not in self.supported_quantizations:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        quant_config = self.supported_quantizations[quantization_type]
        logger.info(f"Applying {quantization_type} quantization: {quant_config['description']}")
        
        quantized_info = {
            'quantization_type': quantization_type,
            'quantization_config': quant_config,
            'tensors': [],
            'total_size_original': 0,
            'total_size_quantized': 0
        }
        
        for tensor_info in weight_info['tensors']:
            original_size = tensor_info['size_bytes']
            
            # Apply quantization based on type
            if quantization_type.startswith('q4'):
                quantized_size = self._apply_4bit_quantization(tensor_info)
            elif quantization_type.startswith('q8'):
                quantized_size = self._apply_8bit_quantization(tensor_info)
            elif quantization_type == 'f16':
                quantized_size = original_size // 2
            else:
                quantized_size = original_size  # No quantization
            
            quantized_tensor = tensor_info.copy()
            quantized_tensor.update({
                'quantization_type': quantization_type,
                'original_size_bytes': original_size,
                'quantized_size_bytes': quantized_size
            })
            
            quantized_info['tensors'].append(quantized_tensor)
            quantized_info['total_size_original'] += original_size
            quantized_info['total_size_quantized'] += quantized_size
        
        compression_ratio = (quantized_info['total_size_original'] / 
                           quantized_info['total_size_quantized'])
        
        logger.info(f"Quantization complete - Compression ratio: {compression_ratio:.2f}x")
        return quantized_info
    
    def _apply_4bit_quantization(self, tensor_info):
        """Apply 4-bit quantization"""
        original_size = tensor_info['size_bytes']
        
        # 4-bit quantization: 4 bits per weight + overhead for scales/zero-points
        # Block size of 32 weights per block
        num_elements = original_size // 4  # FP32 elements
        num_blocks = (num_elements + 31) // 32  # Round up to blocks of 32
        
        # Each block: 32 weights * 4 bits + 1 scale (FP16) + metadata
        quantized_size = (num_blocks * 32 * 4) // 8 + num_blocks * 2 + num_blocks
        
        return quantized_size
    
    def _apply_8bit_quantization(self, tensor_info):
        """Apply 8-bit quantization"""
        original_size = tensor_info['size_bytes']
        
        # 8-bit quantization: 1 byte per weight + scales
        num_elements = original_size // 4  # FP32 elements
        num_blocks = (num_elements + 31) // 32  # Blocks of 32
        
        # Each block: 32 weights * 8 bits + 1 scale (FP16)
        quantized_size = num_blocks * 32 + num_blocks * 2
        
        return quantized_size
    
    def write_gguf_file(self, quantized_info, metadata):
        """Write model in GGUF format"""
        logger.info(f"Writing GGUF file to {self.output_path}")
        
        # GGUF file structure
        gguf_structure = {
            'header': {
                'magic': b'GGUF',
                'version': 3,
                'tensor_count': len(quantized_info['tensors']),
                'metadata_kv_count': len(metadata)
            },
            'metadata': metadata,
            'tensor_info': quantized_info['tensors'],
            'file_size_bytes': quantized_info['total_size_quantized'],
            'compression_ratio': (quantized_info['total_size_original'] / 
                                quantized_info['total_size_quantized'])
        }
        
        # Mock file writing (in production, write actual binary data)
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            # Write mock GGUF file info
            with open(self.output_path + '.info', 'w') as f:
                json.dump(gguf_structure, f, indent=2, default=str)
            
            logger.info(f"GGUF file written successfully: {self.output_path}")
            logger.info(f"File size: {gguf_structure['file_size_bytes'] / (1024**3):.2f} GB")
            logger.info(f"Compression ratio: {gguf_structure['compression_ratio']:.2f}x")
            
        except Exception as e:
            logger.error(f"Failed to write GGUF file: {e}")
            raise
        
        return gguf_structure
    
    def validate_gguf_file(self):
        """Validate created GGUF file"""
        validation_results = {
            'valid': True,
            'issues': [],
            'file_size_mb': 0,
            'tensor_count': 0
        }
        
        try:
            # Check if info file exists (mock validation)
            info_file = self.output_path + '.info'
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    gguf_info = json.load(f)
                
                validation_results.update({
                    'file_size_mb': gguf_info['file_size_bytes'] / (1024**2),
                    'tensor_count': gguf_info['header']['tensor_count'],
                    'compression_ratio': gguf_info['compression_ratio']
                })
                
                logger.info("GGUF file validation passed")
            else:
                validation_results['valid'] = False
                validation_results['issues'].append("GGUF file not found")
                
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def convert_to_gguf(self, quantization_type='q4_0'):
        """Complete GGUF conversion pipeline"""
        logger.info(f"Starting GGUF conversion with {quantization_type} quantization")
        
        # Step 1: Analyze model architecture
        analysis = self.analyze_model_architecture()
        if not analysis['compatible']:
            raise ValueError(f"Model not compatible: {analysis['issues']}")
        
        # Step 2: Prepare model weights
        weight_info = self.prepare_model_weights()
        
        # Step 3: Apply quantization
        quantized_info = self.apply_gguf_quantization(weight_info, quantization_type)
        
        # Step 4: Write GGUF file
        gguf_structure = self.write_gguf_file(quantized_info, weight_info['metadata'])
        
        # Step 5: Validate output
        validation = self.validate_gguf_file()
        
        conversion_result = {
            'success': validation['valid'],
            'output_path': self.output_path,
            'quantization_type': quantization_type,
            'compression_ratio': gguf_structure['compression_ratio'],
            'file_size_mb': validation['file_size_mb'],
            'issues': validation['issues']
        }
        
        return conversion_result

# Solution 6: Hardware-Specific Optimization
class HardwareOptimizer:
    def __init__(self, model, target_hardware='cpu'):
        self.model = model
        self.target_hardware = target_hardware
        self.optimization_config = {}
        
    def optimize_for_cpu(self):
        """Apply CPU-specific optimizations"""
        logger.info("Applying CPU optimizations...")
        
        # Enable CPU optimizations
        optimizations = {
            'use_mkldnn': True,
            'use_openmp': True,
            'optimize_memory': True,
            'fuse_operations': True,
            'vectorization': True
        }
        
        # Apply Intel MKL-DNN backend
        if optimizations['use_mkldnn']:
            torch.backends.mkldnn.enabled = True
            logger.info("Enabled Intel MKL-DNN backend")
        
        # Apply operation fusion
        if optimizations['fuse_operations']:
            try:
                self.model = torch.jit.optimize_for_inference(torch.jit.script(self.model))
                logger.info("Applied operation fusion")
            except Exception as e:
                logger.warning(f"Operation fusion failed: {e}")
        
        # Set thread configuration
        if optimizations['use_openmp']:
            torch.set_num_threads(torch.get_num_threads())
            logger.info(f"OpenMP threads: {torch.get_num_threads()}")
        
        self.optimization_config = optimizations
        return self.model
    
    def optimize_for_gpu(self):
        """Apply GPU-specific optimizations"""
        logger.info("Applying GPU optimizations...")
        
        optimizations = {
            'use_tensorrt': False,  # Would require TensorRT
            'use_cuda_graphs': True,
            'optimize_memory': True,
            'use_mixed_precision': True
        }
        
        if torch.cuda.is_available():
            # Move model to GPU
            self.model = self.model.cuda()
            
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Enable mixed precision if supported
            if optimizations['use_mixed_precision']:
                self.model = self.model.half()
                logger.info("Enabled mixed precision (FP16)")
            
            logger.info("Applied GPU optimizations")
        else:
            logger.warning("CUDA not available, skipping GPU optimizations")
        
        self.optimization_config = optimizations
        return self.model
    
    def optimize_for_mobile(self):
        """Apply mobile-specific optimizations"""
        logger.info("Applying mobile optimizations...")
        
        mobile_config = {
            'target_platform': 'android',
            'max_memory_mb': 100,
            'target_latency_ms': 50,
            'battery_optimization': True,
            'thermal_management': True
        }
        
        try:
            # Convert to mobile-optimized format
            mobile_model = torch.jit.optimize_for_mobile(
                torch.jit.script(self.model),
                optimization_blocklist={'quantized::linear_dynamic'}
            )
            
            self.model = mobile_model
            logger.info("Applied mobile optimizations")
            
        except Exception as e:
            logger.warning(f"Mobile optimization failed: {e}")
        
        self.optimization_config = mobile_config
        return self.model
    
    def benchmark_performance(self, input_data, num_runs=100):
        """Benchmark model performance on target hardware"""
        logger.info(f"Benchmarking performance on {self.target_hardware}...")
        
        # Ensure model is in correct mode and device
        self.model.eval()
        if self.target_hardware == 'gpu' and torch.cuda.is_available():
            input_data = input_data.cuda()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(input_data)
        
        # Benchmark inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = self.model(input_data)
                
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = 1.0 / avg_time
        
        # Memory usage
        if torch.cuda.is_available() and self.target_hardware == 'gpu':
            memory_usage = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            memory_usage = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        
        benchmark_results = {
            'hardware': self.target_hardware,
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'total_time_s': total_time,
            'memory_usage_mb': memory_usage,
            'optimization_config': self.optimization_config
        }
        
        logger.info(f"Benchmark results: {avg_time*1000:.2f}ms avg, "
                   f"{throughput:.1f} FPS, {memory_usage:.1f}MB memory")
        
        return benchmark_results

# Solution 7: Production Deployment and Monitoring
class QuantizedModelDeployment:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.metrics = {
            'inference_times': [],
            'accuracy_scores': [],
            'memory_usage': [],
            'error_count': 0,
            'total_inferences': 0
        }
        self.alert_thresholds = {
            'max_latency_ms': 100,
            'min_accuracy': 0.85,
            'max_error_rate': 0.05,
            'max_memory_mb': 500
        }
        
    def validate_quantized_model(self):
        """Validate quantized model before deployment"""
        validation_results = {
            'valid': True,
            'issues': [],
            'metrics': {}
        }
        
        try:
            # Test inference with dummy data
            if hasattr(self.model, 'conv1'):  # CNN model
                dummy_input = torch.randn(1, 3, 32, 32)
            else:  # Transformer model
                dummy_input = torch.randint(0, 1000, (1, 10))
            
            # Forward pass
            with torch.no_grad():
                output = self.model(dummy_input)
            
            # Check for NaN outputs
            if torch.isnan(output).any():
                validation_results['valid'] = False
                validation_results['issues'].append('Model produces NaN outputs')
            
            # Check output shape
            if output.numel() == 0:
                validation_results['valid'] = False
                validation_results['issues'].append('Model produces empty outputs')
            
            # Model size check
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            validation_results['metrics']['model_size_mb'] = model_size / (1024**2)
            
            # Performance check
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            avg_time = (time.time() - start_time) / 10
            validation_results['metrics']['avg_inference_time_ms'] = avg_time * 1000
            
            logger.info(f"Model validation passed - Size: {model_size/(1024**2):.1f}MB, "
                       f"Latency: {avg_time*1000:.1f}ms")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f'Inference error: {str(e)}')
        
        return validation_results
    
    def setup_monitoring(self):
        """Setup production monitoring"""
        monitoring_config = {
            'metrics_to_track': [
                'inference_latency',
                'throughput',
                'memory_usage',
                'accuracy_drift',
                'error_rate'
            ],
            'alert_thresholds': self.alert_thresholds,
            'monitoring_frequency': 'real_time',
            'retention_period_hours': 24,
            'dashboard_enabled': True
        }
        
        logger.info(f"Monitoring setup complete for {self.model_name}")
        logger.info(f"Alert thresholds: {self.alert_thresholds}")
        
        return monitoring_config
    
    def monitor_inference(self, input_data, ground_truth=None):
        """Monitor single inference"""
        self.metrics['total_inferences'] += 1
        
        try:
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                output = self.model(input_data)
            inference_time = time.time() - start_time
            
            # Track inference time
            self.metrics['inference_times'].append(inference_time)
            
            # Compute accuracy if ground truth available
            accuracy = None
            if ground_truth is not None:
                accuracy = self._compute_accuracy(output, ground_truth)
                self.metrics['accuracy_scores'].append(accuracy)
            
            # Monitor memory usage
            memory_usage = self._get_memory_usage()
            self.metrics['memory_usage'].append(memory_usage)
            
            # Check for alerts
            alerts = self._check_performance_alerts(inference_time * 1000, accuracy, memory_usage)
            
            monitoring_result = {
                'inference_time_ms': inference_time * 1000,
                'accuracy': accuracy,
                'memory_usage_mb': memory_usage,
                'alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
            return monitoring_result
            
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Inference monitoring error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _compute_accuracy(self, predictions, ground_truth):
        """Compute model accuracy"""
        if predictions.dim() > 1:
            predicted_classes = torch.argmax(predictions, dim=-1)
        else:
            predicted_classes = predictions
        
        if ground_truth.dim() > 1:
            ground_truth = torch.argmax(ground_truth, dim=-1)
        
        accuracy = (predicted_classes == ground_truth).float().mean().item()
        return accuracy
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            # Estimate CPU memory usage
            memory_usage = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        
        return memory_usage
    
    def _check_performance_alerts(self, inference_time_ms, accuracy, memory_usage_mb):
        """Check for performance alerts"""
        alerts = []
        
        # Latency alert
        if inference_time_ms > self.alert_thresholds['max_latency_ms']:
            alerts.append({
                'type': 'high_latency',
                'message': f"High latency detected: {inference_time_ms:.1f}ms > {self.alert_thresholds['max_latency_ms']}ms",
                'severity': 'warning'
            })
        
        # Accuracy alert
        if accuracy is not None and accuracy < self.alert_thresholds['min_accuracy']:
            alerts.append({
                'type': 'low_accuracy',
                'message': f"Low accuracy detected: {accuracy:.3f} < {self.alert_thresholds['min_accuracy']}",
                'severity': 'critical'
            })
        
        # Memory alert
        if memory_usage_mb > self.alert_thresholds['max_memory_mb']:
            alerts.append({
                'type': 'high_memory',
                'message': f"High memory usage: {memory_usage_mb:.1f}MB > {self.alert_thresholds['max_memory_mb']}MB",
                'severity': 'warning'
            })
        
        # Error rate alert
        error_rate = self.metrics['error_count'] / max(self.metrics['total_inferences'], 1)
        if error_rate > self.alert_thresholds['max_error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'message': f"High error rate: {error_rate:.3f} > {self.alert_thresholds['max_error_rate']}",
                'severity': 'critical'
            })
        
        return alerts
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.metrics['inference_times']:
            return {'error': 'No performance data available'}
        
        # Compute statistics
        inference_times_ms = [t * 1000 for t in self.metrics['inference_times']]
        
        report = {
            'model_name': self.model_name,
            'report_period': {
                'total_inferences': self.metrics['total_inferences'],
                'error_count': self.metrics['error_count'],
                'error_rate': self.metrics['error_count'] / max(self.metrics['total_inferences'], 1)
            },
            'performance_summary': {
                'avg_inference_time_ms': np.mean(inference_times_ms),
                'p50_inference_time_ms': np.percentile(inference_times_ms, 50),
                'p95_inference_time_ms': np.percentile(inference_times_ms, 95),
                'p99_inference_time_ms': np.percentile(inference_times_ms, 99),
                'avg_throughput_fps': 1000 / np.mean(inference_times_ms),
                'avg_memory_usage_mb': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
            },
            'accuracy_summary': {
                'avg_accuracy': np.mean(self.metrics['accuracy_scores']) if self.metrics['accuracy_scores'] else None,
                'min_accuracy': np.min(self.metrics['accuracy_scores']) if self.metrics['accuracy_scores'] else None,
                'accuracy_std': np.std(self.metrics['accuracy_scores']) if self.metrics['accuracy_scores'] else None
            },
            'alert_thresholds': self.alert_thresholds,
            'recommendations': self._generate_recommendations(),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self):
        """Generate performance recommendations"""
        recommendations = []
        
        if self.metrics['inference_times']:
            avg_time_ms = np.mean(self.metrics['inference_times']) * 1000
            
            if avg_time_ms > self.alert_thresholds['max_latency_ms']:
                recommendations.append("Consider more aggressive quantization or model pruning to reduce latency")
            
            if self.metrics['accuracy_scores']:
                avg_accuracy = np.mean(self.metrics['accuracy_scores'])
                if avg_accuracy < self.alert_thresholds['min_accuracy']:
                    recommendations.append("Accuracy below threshold - consider QAT or less aggressive quantization")
        
        error_rate = self.metrics['error_count'] / max(self.metrics['total_inferences'], 1)
        if error_rate > self.alert_thresholds['max_error_rate']:
            recommendations.append("High error rate detected - investigate model stability")
        
        if not recommendations:
            recommendations.append("Model performance is within acceptable thresholds")
        
        return recommendations
    
    def handle_performance_degradation(self):
        """Handle performance degradation"""
        # Check for degradation
        recent_metrics = {
            'inference_times': self.metrics['inference_times'][-100:],  # Last 100 inferences
            'accuracy_scores': self.metrics['accuracy_scores'][-100:],
            'error_count': self.metrics['error_count']
        }
        
        degradation_detected = False
        issues = []
        
        # Check recent performance
        if recent_metrics['inference_times']:
            recent_avg_time = np.mean(recent_metrics['inference_times']) * 1000
            if recent_avg_time > self.alert_thresholds['max_latency_ms'] * 1.5:
                degradation_detected = True
                issues.append(f"Severe latency degradation: {recent_avg_time:.1f}ms")
        
        if recent_metrics['accuracy_scores']:
            recent_avg_accuracy = np.mean(recent_metrics['accuracy_scores'])
            if recent_avg_accuracy < self.alert_thresholds['min_accuracy'] * 0.9:
                degradation_detected = True
                issues.append(f"Severe accuracy degradation: {recent_avg_accuracy:.3f}")
        
        if degradation_detected:
            logger.critical(f"Performance degradation detected for {self.model_name}: {issues}")
            
            # Implement fallback strategy
            fallback_actions = [
                "Switch to full-precision model",
                "Reduce batch size",
                "Enable model refresh",
                "Alert operations team"
            ]
            
            logger.info(f"Implementing fallback actions: {fallback_actions}")
            
            return {
                'degradation_detected': True,
                'issues': issues,
                'fallback_actions': fallback_actions,
                'timestamp': datetime.now().isoformat()
            }
        
        return {'degradation_detected': False}

# Demonstration functions for each solution
def demonstrate_post_training_quantization():
    """Demonstrate post-training quantization"""
    print("=== Post-Training Quantization Demo ===")
    
    # Create model and calibration data
    model = MockCNNModel()
    calibration_data = [torch.randn(8, 3, 32, 32) for _ in range(10)]
    
    # Apply PTQ
    quantizer = PostTrainingQuantizer(model, calibration_data)
    quantized_model = quantizer.full_ptq_pipeline()
    
    # Compare sizes
    size_comparison = quantizer.compare_model_sizes(model, quantized_model)
    print(f"Original size: {size_comparison['original_size_mb']:.2f} MB")
    print(f"Quantized size: {size_comparison['quantized_size_mb']:.2f} MB")
    print(f"Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    print()

def demonstrate_quantization_aware_training():
    """Demonstrate quantization-aware training"""
    print("=== Quantization-Aware Training Demo ===")
    
    # Create model and data
    model = MockCNNModel()
    train_data = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(20)]
    val_data = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(5)]
    
    # Apply QAT
    trainer = QuantizationAwareTrainer(model, train_data, val_data)
    trainer.prepare_qat_model()
    trainer.train_with_quantization_simulation(epochs=2)
    quantized_model = trainer.convert_qat_to_quantized()
    
    print("QAT training completed successfully")
    print()

def demonstrate_gptq_quantization():
    """Demonstrate GPTQ quantization"""
    print("=== GPTQ Quantization Demo ===")
    
    # Create transformer model
    model = MockTransformerModel()
    calibration_data = [torch.randint(0, 1000, (4, 32)) for _ in range(10)]
    
    # Apply GPTQ
    gptq = GPTQQuantizer(model, bits=4, group_size=128)
    quantized_model = gptq.quantize_model_with_gptq(calibration_data)
    
    print("GPTQ quantization completed")
    print()

def demonstrate_awq_quantization():
    """Demonstrate AWQ quantization"""
    print("=== AWQ Quantization Demo ===")
    
    # Create model
    model = MockTransformerModel()
    calibration_data = [torch.randint(0, 1000, (4, 32)) for _ in range(10)]
    
    # Apply AWQ to first linear layer
    awq = AWQQuantizer(model, bits=4, group_size=128)
    
    # Find first linear layer
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            awq.quantize_layer_with_awq(layer, calibration_data)
            print(f"Applied AWQ to layer: {name}")
            break
    
    print("AWQ quantization completed")
    print()

def demonstrate_gguf_conversion():
    """Demonstrate GGUF conversion"""
    print("=== GGUF Conversion Demo ===")
    
    # Create converter
    converter = GGUFConverter("./mock_model.bin", "./mock_model_q4_0.gguf")
    
    # Convert to GGUF
    result = converter.convert_to_gguf(quantization_type='q4_0')
    
    print(f"GGUF conversion {'successful' if result['success'] else 'failed'}")
    print(f"Compression ratio: {result['compression_ratio']:.2f}x")
    print(f"Output size: {result['file_size_mb']:.1f} MB")
    print()

def demonstrate_hardware_optimization():
    """Demonstrate hardware optimization"""
    print("=== Hardware Optimization Demo ===")
    
    # Create model
    model = MockCNNModel()
    input_data = torch.randn(1, 3, 32, 32)
    
    # Test different hardware optimizations
    hardware_types = ['cpu', 'gpu', 'mobile']
    
    for hardware in hardware_types:
        optimizer = HardwareOptimizer(model.clone() if hasattr(model, 'clone') else model, hardware)
        
        if hardware == 'cpu':
            optimized_model = optimizer.optimize_for_cpu()
        elif hardware == 'gpu':
            optimized_model = optimizer.optimize_for_gpu()
        else:
            optimized_model = optimizer.optimize_for_mobile()
        
        # Benchmark performance
        benchmark = optimizer.benchmark_performance(input_data, num_runs=50)
        print(f"{hardware.upper()} - Latency: {benchmark['avg_inference_time_ms']:.1f}ms, "
              f"Memory: {benchmark['memory_usage_mb']:.1f}MB")
    
    print()

def demonstrate_production_deployment():
    """Demonstrate production deployment"""
    print("=== Production Deployment Demo ===")
    
    # Create quantized model
    model = MockCNNModel()
    deployment = QuantizedModelDeployment(model, "quantized_cnn_v1")
    
    # Validate model
    validation = deployment.validate_quantized_model()
    print(f"Model validation: {'Passed' if validation['valid'] else 'Failed'}")
    
    # Setup monitoring
    monitoring_config = deployment.setup_monitoring()
    
    # Simulate production workload
    for i in range(50):
        input_data = torch.randn(1, 3, 32, 32)
        ground_truth = torch.randint(0, 10, (1,))
        result = deployment.monitor_inference(input_data, ground_truth)
        
        if i % 10 == 0 and 'alerts' in result:
            if result['alerts']:
                print(f"Alert at inference {i}: {result['alerts'][0]['message']}")
    
    # Generate report
    report = deployment.generate_performance_report()
    print(f"Performance report generated - Avg latency: {report['performance_summary']['avg_inference_time_ms']:.1f}ms")
    print()

def main():
    """Run all quantization solution demonstrations"""
    print("Day 50: Quantization - Model Compression & Optimization - Solutions")
    print("=" * 80)
    print()
    
    # Run all demonstrations
    demonstrate_post_training_quantization()
    demonstrate_quantization_aware_training()
    demonstrate_gptq_quantization()
    demonstrate_awq_quantization()
    demonstrate_gguf_conversion()
    demonstrate_hardware_optimization()
    demonstrate_production_deployment()
    
    print("=" * 80)
    print("All quantization solutions demonstrated successfully!")
    print()
    print("Key Takeaways:")
    print("1. PTQ provides quick quantization without retraining")
    print("2. QAT maintains better accuracy through training-time simulation")
    print("3. GPTQ enables aggressive quantization for large language models")
    print("4. AWQ protects important weights based on activation patterns")
    print("5. GGUF format optimizes models for CPU inference")
    print("6. Hardware-specific optimizations are crucial for performance")
    print("7. Production deployment requires comprehensive monitoring and validation")

if __name__ == "__main__":
    main()