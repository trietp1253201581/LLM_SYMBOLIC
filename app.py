from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

app = FastAPI()

# Load tokenizer từ adapter
fine_tuned_model_path = "trietlm0306/llama2-finetuned-lora-gp"
base_model_path = "NousResearch/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.float16, device_map="auto"
)

# Load LoRA adapter vào base model
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model.eval()

@app.get("/generate")
def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200)
    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
