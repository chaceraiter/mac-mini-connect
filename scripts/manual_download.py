from transformers import AutoModelForCausalLM

print("Starting manual download of gpt2-medium...")
try:
    AutoModelForCausalLM.from_pretrained("gpt2-medium")
    print("Download complete and model cached successfully.")
except Exception as e:
    print(f"An error occurred during download: {e}") 