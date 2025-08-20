import os
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification

def download_pubmedbert_model(model_name, save_dir="./models"):
    """
    下载并保存PubMedBERT模型到本地
    
    Args:
        model_name: 模型名称，如 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        save_dir: 保存目录
    """
    print(f"开始下载模型: {model_name}")
    
    # 创建保存目录
    model_save_path = os.path.join(save_dir, model_name.replace("/", "_"))
    os.makedirs(model_save_path, exist_ok=True)
    
    try:
        # 下载tokenizer
        print("正在下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.save_pretrained(model_save_path)
        print(f"Tokenizer已保存到: {model_save_path}")
        
        # 下载模型 (这里先下载基础模型，后面会在训练时添加NER头)
        print("正在下载模型...")
        # 下载基础模型用于后续初始化
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=2  # 临时设置，实际使用时会重新配置
        )
        model.save_pretrained(model_save_path)
        print(f"模型已保存到: {model_save_path}")
        
        # 保存配置信息
        config_info = {
            "model_name": model_name,
            "local_path": model_save_path,
            "download_time": str(pd.Timestamp.now()) if 'pd' in globals() else "unknown"
        }
        
        import json
        with open(os.path.join(model_save_path, "download_info.json"), "w") as f:
            json.dump(config_info, f, indent=2)
            
        print(f"✅ 模型下载完成！保存路径: {model_save_path}")
        return model_save_path
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="下载PubMedBERT模型到本地")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        help="要下载的模型名称"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="模型保存目录"
    )
    
    args = parser.parse_args()
    
    download_pubmedbert_model(args.model_name, args.save_dir)

if __name__ == "__main__":
    main()