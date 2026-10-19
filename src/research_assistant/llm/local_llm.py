import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class LocalLLM:
    """Load and run a local Hugging Face language model."""

    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        )

        # Explicitly use the CPU for this project.
        self.model = self.model.to("cpu")
        self.model.eval()

        print("Model loaded successfully!")

    def generate_response(
        self,
        question: str,
        system_prompt: str = "You are a helpful research assistant.",
        max_new_tokens: int = 200,
    ) -> str:
        """Generate a response to a question."""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, input_length:]

        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()