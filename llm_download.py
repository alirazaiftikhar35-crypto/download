# ===== Download & Save Hugging Face LLM in 8-bit =====
# pip install transformers torch accelerate bitsandbytes safetensors

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ----------------- USER SETTINGS -----------------
model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Hugging Face model ID
local_model_path = "D:\\projects\\py\\download_weights"                  # Where to save
# --------------------------------------------------

os.makedirs(local_model_path, exist_ok=True)

print(f"Downloading tokenizer for {model_name_or_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

print(f"Downloading model in 8-bit for {model_name_or_path}...")
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     device_map="auto",       
#     torch_dtype=torch.float16, 
#     trust_remote_code=True 
# )
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    dtype="auto"
)


# Save model & tokenizer locally
print(f"Saving model and tokenizer at {local_model_path}...")
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)

print("✅ 8-bit model download & save complete!")