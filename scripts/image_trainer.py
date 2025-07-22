#!/usr/bin/env python3
"""
Tournament Image Training Script - G.O.D Subnet
Trains SDXL or Flux models for tournament competition
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import toml

# Model types (enum-like)
class ImageModelType:
    SDXL = "sdxl"
    FLUX = "flux"

def extract_dataset(zip_path, extract_dir):
    """Extract dataset zip file and organize images"""
    print(f"Extracting dataset from {zip_path} to {extract_dir}", flush=True)
    
    # Create directories
    img_dir = os.path.join(extract_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    
    # Extract zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Move all images to img/ subdirectory if not already there
    for root, dirs, files in os.walk(extract_dir):
        if root == img_dir:
            continue
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                src = os.path.join(root, file)
                dst = os.path.join(img_dir, file)
                if src != dst and not os.path.exists(dst):
                    shutil.move(src, dst)
    
    print(f"Dataset extracted. Images in: {img_dir}", flush=True)
    return img_dir

def create_training_config(task_id, model_path, model_type, expected_repo_name, img_dir):
    """Create training configuration for Kohya"""
    print("Creating training configuration...", flush=True)
    
    # Output path as per tournament requirements
    output_dir = f"/app/checkpoints/{task_id}/{expected_repo_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type.lower() == ImageModelType.SDXL:
        config = {
            # Model settings
            "pretrained_model_name_or_path": model_path,
            "v2": False,
            "v_parameterization": False,
            
            # Dataset settings  
            "train_data_dir": img_dir,
            "resolution": "1024,1024",
            "enable_bucket": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": 1584,
            "bucket_reso_steps": 64,
            
            # Training parameters
            "train_batch_size": 1,
            "max_train_epochs": 10,
            "max_train_steps": 1000,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "lr_scheduler": "cosine_with_restarts",
            "lr_warmup_steps": 100,
            "optimizer_type": "AdamW8bit",
            
            # LoRA settings
            "network_module": "networks.lora",
            "network_dim": 64,
            "network_alpha": 32,
            "network_dropout": 0.1,
            "network_args": {
                "conv_dim": "32",
                "conv_alpha": "16"
            },
            
            # Output settings
            "output_dir": output_dir,
            "output_name": "trained_model", 
            "save_model_as": "safetensors",
            "save_precision": "fp16",
            "mixed_precision": "fp16",
            
            # Memory optimization
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "xformers": True,
            
            # Logging
            "log_prefix": task_id,
            "logging_dir": f"{output_dir}/logs",
            
            # Other settings
            "seed": 42,
            "clip_skip": 2,
            "max_data_loader_n_workers": 4,
            "persistent_data_loader_workers": True
        }
        
    elif model_type.lower() == ImageModelType.FLUX:
        config = {
            # Model settings
            "pretrained_model_name_or_path": model_path,
            
            # Dataset settings
            "train_data_dir": img_dir,
            "resolution": "512,512",  # Flux typically uses smaller resolution
            "enable_bucket": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": 1024,
            
            # Training parameters - more conservative for Flux
            "train_batch_size": 1,
            "max_train_epochs": 5,
            "max_train_steps": 500,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 50,
            "optimizer_type": "AdamW8bit",
            
            # LoRA settings
            "network_module": "networks.lora",
            "network_dim": 32,
            "network_alpha": 16,
            
            # Output settings
            "output_dir": output_dir,
            "output_name": "trained_model",
            "save_model_as": "safetensors", 
            "save_precision": "fp16",
            "mixed_precision": "fp16",
            
            # Memory optimization
            "cache_latents": True,
            "gradient_checkpointing": True,
            
            # Logging
            "log_prefix": task_id,
            "logging_dir": f"{output_dir}/logs",
            
            # Other settings
            "seed": 42,
            "max_data_loader_n_workers": 2
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Save config
    config_path = f"/dataset/configs/{task_id}.toml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        toml.dump(config, f)
    
    print(f"Config saved to {config_path}", flush=True)
    return config_path, output_dir

def run_training(config_path, model_type):
    """Run the actual training using Kohya scripts"""
    print("=== STARTING TRAINING ===", flush=True)
    
    # Determine training script based on model type
    if model_type.lower() == ImageModelType.SDXL:
        training_script = "/app/sd-scripts/sdxl_train_network.py"
    elif model_type.lower() == ImageModelType.FLUX:
        training_script = "/app/sd-scripts/flux_train_network.py"  
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Run training with accelerate
    cmd = [
        "accelerate", "launch",
        "--dynamo_backend", "no", 
        "--dynamo_mode", "default",
        "--mixed_precision", "fp16",
        "--num_processes", "1",
        "--num_machines", "1", 
        "--num_cpu_threads_per_process", "2",
        training_script,
        "--config_file", config_path
    ]
    
    print(f"Running command: {' '.join(cmd)}", flush=True)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!", flush=True)
        print("STDOUT:", result.stdout, flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}", flush=True)
        print("STDOUT:", e.stdout, flush=True) 
        print("STDERR:", e.stderr, flush=True)
        return False

def main():
    print("=== G.O.D TOURNAMENT IMAGE TRAINER ===", flush=True)
    
    parser = argparse.ArgumentParser(description="Tournament Image Training Script")
    
    # Required arguments as per tournament specification
    parser.add_argument("--task-id", required=True, help="Unique task identifier")
    parser.add_argument("--model", required=True, help="Base model to finetune") 
    parser.add_argument("--dataset-zip", required=True, help="S3 URL to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--expected-repo-name", required=True, help="Expected HuggingFace repository name")
    
    args = parser.parse_args()
    
    print(f"Task ID: {args.task_id}", flush=True)
    print(f"Model: {args.model}", flush=True) 
    print(f"Model Type: {args.model_type}", flush=True)
    print(f"Dataset ZIP: {args.dataset_zip}", flush=True)
    print(f"Expected Repo: {args.expected_repo_name}", flush=True)
    
    # Setup paths according to tournament specification
    model_folder = args.model.replace("/", "--")
    model_path = f"/cache/models/{model_folder}"
    dataset_zip = f"/cache/datasets/{args.task_id}.zip"
    extract_dir = f"/dataset/images/{args.task_id}"
    
    print(f"Model path: {model_path}", flush=True)
    print(f"Dataset zip path: {dataset_zip}", flush=True)
    print(f"Extract dir: {extract_dir}", flush=True)
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"WARNING: Model not found at {model_path}", flush=True)
            print("Available models in /cache/models:", flush=True)
            if os.path.exists("/cache/models"):
                print(os.listdir("/cache/models"), flush=True)
            # Use model name directly if path doesn't exist
            model_path = args.model
        
        # Check if dataset exists
        if not os.path.exists(dataset_zip):
            print(f"ERROR: Dataset zip not found at {dataset_zip}", flush=True)
            return 1
        
        # Extract dataset
        img_dir = extract_dataset(dataset_zip, extract_dir)
        
        # Check if images were extracted
        if not os.path.exists(img_dir) or not os.listdir(img_dir):
            print(f"ERROR: No images found in {img_dir}", flush=True)
            return 1
            
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Found {img_count} images for training", flush=True)
        
        # Create training config
        config_path, output_dir = create_training_config(
            args.task_id, model_path, args.model_type, args.expected_repo_name, img_dir
        )
        
        # Run training
        success = run_training(config_path, args.model_type)
        
        if success:
            print(f"=== TRAINING COMPLETED SUCCESSFULLY ===", flush=True)
            print(f"Model saved to: {output_dir}", flush=True)
            
            # List output files
            if os.path.exists(output_dir):
                print("Output files:", os.listdir(output_dir), flush=True)
            
            return 0
        else:
            print("=== TRAINING FAILED ===", flush=True)
            return 1
            
    except Exception as e:
        print(f"ERROR: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
