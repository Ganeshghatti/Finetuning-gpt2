# from transformers import GPT2TokenizerFast, Trainer, GPT2LMHeadModel, TrainingArguments
# from datasets import load_dataset

# # Load the tokenizer and model
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token

# # Load the dataset (ensure the json structure is correct for loading)
# raw_datasets = load_dataset("json", data_files="data/torqdesigns.json")
# raw_datasets = raw_datasets["train"]

# # Tokenization function
# def tokenize_function(batch):
#     # Tokenize the combined texts
#     tokenized_input = tokenizer(
#         batch["data"],  # Access the 'data' key if that is where the text is stored
#         truncation=True,
#         padding="max_length",
#         max_length=512
#     )
#     # Set labels to be the same as input_ids
#     tokenized_input["labels"] = tokenized_input["input_ids"]
    
#     return tokenized_input

# # Apply tokenization to the dataset
# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["data"])

# # Convert the dataset to PyTorch tensors
# tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     overwrite_output_dir=True,
#     num_train_epochs=30,
#     per_device_train_batch_size=4,
#     save_steps=10_000,
#     save_total_limit=2,
# )

# # Create the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets,
#     tokenizer=tokenizer,
# )

# # Train the model
# trainer.train()

# # Save the model and tokenizer
# trainer.save_model("./torq_designs")
# tokenizer.save_pretrained("./torq_designs")

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./torq_designs")
tokenizer = GPT2TokenizerFast.from_pretrained("./torq_designs")
tokenizer.pad_token = tokenizer.eos_token

# Ensure the model is in evaluation mode
model.eval()

# Function to generate text
def generate_text(prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.95):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Test the model with a single prompt
prompt = "Expain services provided by Torq Designs"

# Generate and print the output
generated_text = generate_text(prompt, max_length=150)
print(f"Prompt: {prompt}")
print(f"Generated Text: {generated_text}")

