# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from datasets import load_dataset

# # Load the GPT-2 tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# # Set the padding token
# tokenizer.pad_token = tokenizer.eos_token

# # Load the dataset using the datasets library
# dataset = load_dataset('text', data_files='company_data.txt')

# # Tokenize the dataset with padding and truncation
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# # Initialize the data collator
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,
# )

# # Load GPT-2 model
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./gpt2-company-model",
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     save_steps=10_000,
#     save_total_limit=2,
#     logging_steps=200,
# )

# # Create the trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=tokenized_datasets['train'],
# )

# # Train the model
# trainer.train()

# # Save the model
# model.save_pretrained("./gpt2-company-model")
# tokenizer.save_pretrained("./gpt2-company-model")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./gpt2-company-model')  # Replace with your model's path
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-company-model')

# If you added a padding token during training, set it in the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Function to generate text from a prompt
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompt
prompt = "What is the official name of our company according to the company details?"

# Generate and print text
generated_text = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Generated Text: {generated_text}")
