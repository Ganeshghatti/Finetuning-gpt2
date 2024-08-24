from transformers import GPT2TokenizerFast, Trainer, GPT2LMHeadModel, TrainingArguments
from datasets import load_dataset

# Load the tokenizer and model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
raw_datasets = load_dataset("json", data_files="/content/sample_data/the_squirrel_QnA.json")
raw_datasets = raw_datasets["train"]


def tokenize_function(batch):
    combined_texts = []

    # Loop through each example in the batch
    for question, answer in zip(batch["question"], batch["answer"]):
        # Combine the question and answer as a single input
        combined_text = question + " " + answer
        combined_texts.append(combined_text)

    print(combined_texts)
    # Tokenize the combined texts
    tokenized_input = tokenizer(
        combined_texts, truncation=True, padding="max_length", max_length=512
    )

    # Set the labels to be the same as input_ids for language modeling
    tokenized_input["labels"] = tokenized_input["input_ids"].copy()

    return tokenized_input


# Apply tokenization to the dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Convert the dataset to PyTorch tensors
tokenized_datasets.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
# Save the model and tokenizer
trainer.save_model("./trained_gpt2")
tokenizer.save_pretrained("./trained_gpt2")


from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./trained_gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("./trained_gpt2")

# Set the pad token for the tokenizer (important for GPT-2)
tokenizer.pad_token = tokenizer.eos_token
# Define the prompt question
prompt = "List 4 services provided by The Squirrel?"

# Instruction prompt for prompt engineering
instruction = ("You are a chatbot specialized in answering questions related to The Squirrel's services. "
               "If the question is not related to these services, say: 'I am only trained to answer questions "
               "about The Squirrel's services.'")

# Combine instruction and prompt
input_prompt = instruction + "\n\nQuestion: " + prompt + "\nAnswer:"

# Tokenize the prompt
inputs = tokenizer(input_prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs.input_ids, 
    max_length=100,         # Maximum length of generated sequence
    num_return_sequences=1, 
    no_repeat_ngram_size=2, # Prevents repetition of phrases
    do_sample=True,         # Sampling for diverse output
    top_k=50,               # Top-k sampling
    top_p=0.95,             # Nucleus sampling (for diverse responses)
    temperature=0.7         # Controls randomness, lower is more deterministic
)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated response
generated_text