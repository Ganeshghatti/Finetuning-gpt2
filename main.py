from transformers import GPT2TokenizerFast, Trainer, GPT2LMHeadModel, TrainingArguments
from datasets import load_dataset

# Load the tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
raw_datasets = load_dataset("json", data_files="data/the_squirrel_mockdata.json")
raw_datasets = raw_datasets["train"]



def print_one_service_description(dataset):
    # Access the first entry in the dataset
    first_entry = dataset[0]

    # Extract and print one service description
    


def print_company_description(dataset):
    # Access the first entry in the dataset
    first_entry = dataset[0]

    # Extract and print one service description
    print(first_entry["company_name"], first_entry["company_about"])
    if "company_about" in first_entry:
        return tokenizer(
            # first_entry["company_name"],
            first_entry["company_about"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
    else:
        print("No 'company_about' found")


# Print the first service description
token = print_company_description(raw_datasets)
print(token)

# Define training arguments
trainer = Trainer(
    model,
    train_dataset=token,
    tokenizer=tokenizer,
)

trainer.train()