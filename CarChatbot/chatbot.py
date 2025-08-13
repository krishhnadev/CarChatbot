import aiml
import os
import pandas as pd
import nltk
import string
import spacy
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from nltk.sem.logic import LogicParser

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_lg")

kernel = aiml.Kernel()

if os.path.exists("chatbot_brain.brn"):
    kernel.bootstrap(brainFile="chatbot_brain.brn")
else:
    kernel.learn("cars.aiml")
    kernel.saveBrain("chatbot_brain.brn")

qa_df = pd.read_csv('qa_data.csv', names=["Question", "Answer"], skiprows=1)

def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    doc = nlp(" ".join(tokens))
    processed_tokens = [token.lemma_ for token in doc if token.text not in stop_words]
    return ' '.join(processed_tokens)

qa_df['Processed_Question'] = qa_df['Question'].apply(preprocess)
qa_df['Vector'] = qa_df['Processed_Question'].apply(lambda x: nlp(x).vector)

def find_best_match(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = nlp(user_input_processed).vector
    similarity_scores = [cosine_similarity([user_vector], [row])[0][0] for row in qa_df['Vector']]
    best_match_index = similarity_scores.index(max(similarity_scores))
    best_match_score = similarity_scores[best_match_index]
    spacy_scores = [nlp(user_input_processed).similarity(nlp(q)) for q in qa_df['Processed_Question']]
    best_spacy_match_index = spacy_scores.index(max(spacy_scores))
    best_spacy_match_score = spacy_scores[best_spacy_match_index]

    if best_spacy_match_score > 0.75:
        return qa_df.iloc[best_spacy_match_index]['Answer']
    elif best_match_score > 0.6:
        return qa_df.iloc[best_match_index]['Answer']

    fuzzy_match, fuzzy_score = process.extractOne(user_input, qa_df['Question'].tolist())
    if fuzzy_score > 85:
        return qa_df[qa_df['Question'] == fuzzy_match]['Answer'].values[0]
    elif best_match_score > 0.4:
        return "I think I know what you're asking. Do you mean: " + qa_df.iloc[best_match_index]['Question']
    else:
        return "I'm sorry, I don't have an answer for that. Can you ask about something else related to cars?"

read_expr = LogicParser().parse
knowledge_base = set()

def process_logic(user_input):
    try:
        if user_input.lower().startswith("i know that"):
            fact = user_input.replace("I know that ", "").strip()
            if " is " in fact:
                subject, predicate = fact.split(" is ")
                fact = f"{predicate}({subject})"
            expr = read_expr(fact)
            if expr not in knowledge_base:
                knowledge_base.add(expr)
                return f"OK, I will remember that {fact}."
            else:
                return f"I already know that {fact}."
        elif user_input.lower().startswith("check that"):
            fact = user_input.replace("Check that ", "").strip()
            if " is " in fact:
                subject, predicate = fact.split(" is ")
                fact = f"{predicate}({subject})"
            expr = read_expr(fact)
            if expr in knowledge_base:
                return "Correct."
            else:
                return "I don’t know."
    except Exception as e:
        return f"Error in logic processing: {e}"

print("Car Chatbot is ready! Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    aiml_response = kernel.respond(user_input)
    if aiml_response and aiml_response.strip() and "No match for input" not in aiml_response:
        print("Chatbot:", aiml_response)
    elif user_input.lower().startswith("i know that") or user_input.lower().startswith("check that"):
        print("Chatbot:", process_logic(user_input))
    else:
        print("Chatbot:", find_best_match(user_input))
