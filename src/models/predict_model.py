from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "./models/gpt2_output"

model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token


def evaluate(s: str) -> str:
    s = f"User: {s}\nAssistant:"
    # Convert string to array of embedding indices
    encoded_input = tokenizer(s, return_tensors="pt")
    # Pass embedding indices to the model
    output = model.generate(
        encoded_input.input_ids,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=128,
    )
    # Decode resulting embedding indices back to string
    output_str = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Since we did batch, extract the single element of sequence
    output_str = output_str[0]

    # Truncate string to the next "User:" occurrence, if any.
    idx: int = output_str.find("User", 5)
    if idx != -1:
        output_str = output_str[:idx]

    # Truncate "Assistant:" part
    first_idx: int = output_str.find("Assistant: ")
    if first_idx != -1:
        output_str = output_str[first_idx + len("Assistant: "):]
    return output_str


if __name__ == "__main__":
    input_string = input("Enter prompt: ")
    print(evaluate(input_string))
