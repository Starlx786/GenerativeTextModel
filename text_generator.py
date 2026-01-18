from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_text(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            no_repeat_ngram_size=2
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    print("=== Generative Text Model ===")
    user_prompt = input("Enter a topic or prompt: ")

    result = generate_text(user_prompt)

    print("\nGenerated Text:\n")
    print(result)
