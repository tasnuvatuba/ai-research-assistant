import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)

# Explicitly use the CPU for this exercise
model = model.to("cpu")
model.eval()

print("Model loaded successfully!")


while True:
    question = input("\nAsk a question (or type 'exit'): ")

    if question.strip().lower() == "exit":
        break

    if not question.strip():
        print("Please enter a question.")
        continue

    messages = [
        {
            "role": "system",
            "content": "You are a helpful research assistant.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0, input_length:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    print("\nModel response:")
    print(answer)
