import os
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification

def download_pubmedbert_model(model_name, save_dir="./models"):
    """
    Download and save PubMedBERT model locally
    
    Args:
        model_name: Model name, e.g., 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        save_dir: Save directory
    """
    print(f"Starting model download: {model_name}")
    
    model_save_path = os.path.join(save_dir, model_name.replace("/", "_"))
    os.makedirs(model_save_path, exist_ok=True)
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.save_pretrained(model_save_path)
        print(f"Tokenizer saved to: {model_save_path}")
        
        # Download base model (NER head will be added during training)
        print("Downloading model...")
        # Download base model for subsequent initialization during training
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=2  # Temporary setting, will be reconfigured during actual use
        )
        model.save_pretrained(model_save_path)
        print(f"Model saved to: {model_save_path}")
        
        config_info = {
            "model_name": model_name,
            "local_path": model_save_path,
            "download_time": str(pd.Timestamp.now()) if 'pd' in globals() else "unknown"
        }
        
        import json
        with open(os.path.join(model_save_path, "download_info.json"), "w") as f:
            json.dump(config_info, f, indent=2)
            
        print(f" Model download complete! Save path: {model_save_path}")
        return model_save_path
        
    except Exception as e:
        print(f" Download failed: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download PubMedBERT model locally")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        help="Model name to download"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="Model save directory"
    )
    
    args = parser.parse_args()
    
    download_pubmedbert_model(args.model_name, args.save_dir)

if __name__ == "__main__":
    main()