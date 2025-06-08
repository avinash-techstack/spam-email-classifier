import pickle

# Load the pickle file
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Print the type of the object
print(f"Loaded object type: {type(vectorizer)}")

# If it's a TfidfVectorizer, print vocabulary size
if hasattr(vectorizer, "vocabulary_"):
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# Print some feature names (if applicable)
if hasattr(vectorizer, "get_feature_names_out"):
    print(f"Some feature names: {vectorizer.get_feature_names_out()[:10]}")
