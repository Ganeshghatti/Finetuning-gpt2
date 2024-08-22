import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('spam_or_not_spam.csv')

# Handle missing values
data.dropna(inplace=True)

# Split the data into features (X) and labels (y)
X = data['email']
y = data['label']

# Convert text data into numerical data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

print(y_train)
# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Function to predict if an email is spam or not
def predict_email_spam(email_text):
    # Transform the input email text into numerical data using the same vectorizer
    email_vectorized = vectorizer.transform([email_text])
    
    # Use the trained model to predict whether the email is spam or not
    prediction = model.predict(email_vectorized)
    
    # Map the prediction to a more interpretable format
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return result

# Test with a sample email
sample_email = """This is a spam email
"""

result = predict_email_spam(sample_email)
print(f"The email is classified as: {result}")
