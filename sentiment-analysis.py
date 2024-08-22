from transformers import pipeline

classifier=pipeline("sentiment-analysis")

res=classifier("I am little happy but i am very sad")

print(res)