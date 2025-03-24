import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import joblib
import json

#Function to load and combine datasets from CSV and JSON
def load_and_combine_datasets(file_paths):

    combined_context = ""

    ## for all passed through files paths
    for pattern in file_paths:
        try:
            ## if file is CSV
            if pattern.endswith('.csv'):
                df = pd.read_csv(pattern, on_bad_lines='skip')
                if 'Context' in df.columns and 'Response' in df.columns:
                    context = " ".join(df['Context'].dropna().astype(str).tolist())
                    response = " ".join(df['Response'].dropna().astype(str).tolist())
                    combined_context += f"{context} {response}"

            ## if file is JSON        
            elif pattern.endswith('.json'):
                with open(pattern, 'r', encoding='utf-8') as  file:
                    data = json.load(file)
                    if isinstance(data, list):
                        for entry in data:
                            context = entry.get('Context', '')
                            response = entry.get('Response', '')
                            combined_context += f"{context} {response} "
                    elif isinstance(data, dict):
                        context = data.get('Context', '')
                        response = data.get('Response')
                        combined_context += f"{context} {response} "

        ## Throw an exception if failure to process
        except Exception as e:
            print(f"An error occured while processing {pattern}: {e}")


    return combined_context.strip()


#Chatbot prompt template
template = """
Answer the question below using the provided information

Here is the reference text: {context}

Question: {question}

Answer:
"""

# Intialize the language model and prompt
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

#Function to add emotional tone based on user sentiment
def add_emotional_tone(user_input):

    response = ""

    classifier_mod = joblib.load('BestClassifierModel.pkl')

    blob = TextBlob(user_input, classifier=classifier_mod)

    sentiment = blob.sentiment.polarity

    if sentiment > 0.2 == "pos":
        response += "I'm would gladly help you with that! 😊"
    elif sentiment < -0.2:
        response += "Let's work through this together. 💖 Hopefully this will provide some guidance. "
    else:
        response += "I hope this helps! Let me know if you need anything else. 🙂 \n"
    return response

def response_of_mental_state(user_input):
    response = ""

    classifier_mod = joblib.load('BestClassifierModel.pkl')

    blob = TextBlob(user_input, classifier=classifier_mod)

    pred_class = blob.classify()

    if pred_class == "pos":
        response += "I'm happy to hear you are doing okay! 😊 How can I help you today?"
    elif pred_class == "neg":
        response += "I'm really sorry to hear that. Let's hope for brighter days ahead. Please know that I am here to help you and you are not alone. What can I do to help you today?"
    else:
        response += "Thank you for sharing. Let's hope for brighter days ahead. 🙂 How can I help you today?"
    return response

# Function to detect greetings or inquiries about well-being
def is_greeting_or_how_are_you(user_input):
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "howdy", "hi there"]
    how_are_you = ["how are you", "how's it going", "how are you doing", "how's everything", "how have you been"]

    user_input = user_input.lower()

    if any(greeting in user_input for greeting in greetings):
        return "greeting"
    if any(question in user_input for question in how_are_you):
        return "how_are_you"

    return None

# Main function to handle the conversation
def handle_conversation():

    file_patterns = [
        "Datasets/intents.json",
        "Datasets/train.csv",
    ]

    context = load_and_combine_datasets(file_patterns)

    if not context:
        print("No valid datasets found. Please check the file paths and formats.")
        return

    print("Welcome to the Mental Health AI ChatBot! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Goodbye! I hope you have good day!")
            break

        
        response_type = is_greeting_or_how_are_you(user_input)
        if response_type == "greeting":
            print("Bot: Hi there! 😊 I'm so glad to see you. How can I assist you today?")
            continue
        elif response_type == "how_are_you":
            print("Bot: I'm doing great, thank you for asking! 😄 How are you doing? How is your mental health today?")
            user_input = input("You: ")
            mental_response = response_of_mental_state(user_input)
            print("Bot:", mental_response)
            user_input = input("You: ")

        result = chain.invoke({"context": context, "question": user_input})
        emotional_response = add_emotional_tone(user_input)


        print("Bot:", emotional_response, " ", result)

        context += f"\nUser: {user_input}\nAI: {emotional_response}"

if __name__ == "__main__":
    handle_conversation() 
