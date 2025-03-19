import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from textblob import TextBlob
import json
import glob #file handling pattern

#Function to load and combine datasets from CSV and JSON
def load_and_combine_datasets(file_patterns):
    combined_context = ""
    for pattern in file_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if 'Context' in df.columns and 'Response' in df.columns:
                        context = " ".join(df['Context'].dropna().astype(str).tolist())
                        response = " ".join(df['Response'].dropna().astype(str).tolist())
                        combined_context += f"{context} {response}"
                    elif file_path.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as  file:
                            data = json.load(file)
                            if isinstance(data, list):
                                for entry in data:
                                    context = entry.gey('Context', '')
                                    response = entry.get('Response', '')
                                    combined_context += f"{context} {response} "
                            elif isinstance(data, dict):
                                context = data.get('Context', '')
                                response = data.get('Response')
                                combined_context += f"{context} {response} "
            except Exception as e:
                print(f"An error occured while processing {file_path}: {e}")
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
def add_emotional_tone(response, user_input):
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity

    if sentiment > 0.2:
        response += " I'm so happy to help you with that! 😊"
    elif sentiment < -0.2:
        response += " I'm really sorry to hear that. Let's work through this together. 💖"
    else:
        response += " I hope that helps! Let me know if you need anything else.  🙂"
    
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
        "/Users/srinivasansenthilnathan/Desktop/AI Project/Anxiety Stress Scales/data.csv",
        "/Users/srinivasansenthilnathan/Desktop/AI Project/OSMI Mental Health/data.csv",
        "/Users/srinivasansenthilnathan/Desktop/AI Project/train.csv",
        "/Users/srinivasansenthilnathan/Desktop/AI Project/Mental Health.json"
    ]

    context = load_and_combine_datasets(file_patterns)
    if not context:
        print("No valid datasets found. Please check the file paths and formats.")
        return

    print("Welcome to the AI ChatBot! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Goodbye! Have a great day!")
            break

        response_type = is_greeting_or_how_are_you(user_input)

        if response_type == "greeting":
            print("Bot: Hi there! 😊 I'm so glad to see you. How can I assist you today?")
            continue

        elif response_type == "how_are_you":
            print("Bot: I'm doing great, thank you for asking! 😄 How about you? How can I help you today?")
            continue

        result = chain.invoke({"context": context, "question": user_input})
        emotional_response = add_emotional_tone(result, user_input)
        print("Bot:", emotional_response)

        context += f"\nUser: {user_input}\nAI: {emotional_response}"

if __name__ == "__main__":
    handle_conversation() 

                      
