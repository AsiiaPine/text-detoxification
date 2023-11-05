from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Please, don't ever set this to ./models/gpt2_output,
# it may corrupt the final checkpoint!
OUTPUT_DIR = "gpt2_output"
INITIAL_MODEL = "gpt2"

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(INITIAL_MODEL)
tokenizer = GPT2Tokenizer.from_pretrained(INITIAL_MODEL)

# Load training dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="./data/interim/train_text.txt",
    block_size=512,
)

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
trainer.train()

# Save the fine-tuned model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
