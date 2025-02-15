import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from textblob import TextBlob

# Load conversation dataset (CSV file) as context
def load_conversation_dataset(file_path):
    try:
        # Load the dataset using pandas (assuming it's a CSV file)
        df = pd.read_csv(file_path)
        
        # Combine 'Context' and 'Response' to form the conversation context
        context = " ".join(df['Context'].dropna().astype(str).tolist()) + " " + " ".join(df['Response'].dropna().astype(str).tolist())
        
        return context
    except FileNotFoundError:
        print("Error: Dataset file not found!")
        return ""

# Define the chatbot prompt template
template = """
Answer the question below using the provided information.

Here is the reference text: {context}

Question: {question}

Answer:
"""

# Initialize the Ollama model
model = OllamaLLM(model="llama3")

# Create the chat prompt
prompt = ChatPromptTemplate.from_template(template)

# Chain the prompt and model together
chain = prompt | model

# Function to generate emotional response based on user sentiment
def add_emotional_tone(response, user_input):
    # Analyze the sentiment of the user input (positive, negative, neutral)
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity  # Get the polarity score

    if sentiment > 0.2:
        # Positive sentiment: Add enthusiasm
        response += " I'm so happy to help you with that! 😊"
    elif sentiment < -0.2:
        # Negative sentiment: Add empathy
        response += " I'm really sorry to hear that. Let's work through this together. 💖"
    else:
        # Neutral sentiment: Keep it friendly but neutral
        response += " I hope that helps! Let me know if you need anything else. 🙂"

    return response

# Function to detect greeting or "How are you?"
def is_greeting_or_how_are_you(user_input):
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "howdy", "hi there"]
    how_are_you = ["how are you", "how's it going", "how are you doing", "how's everything", "how have you been"]
    
    user_input = user_input.lower()
    
    # Check if input is a greeting
    for greeting in greetings:
        if greeting in user_input:
            return "greeting"
    
    # Check if input is asking about "how are you"
    for question in how_are_you:
        if question in user_input:
            return "how_are_you"
    
    return None

# Conversation handling function
def handle_conversation():
    # Load the conversation dataset as context
    context = load_conversation_dataset("/Users/srinivasansenthilnathan/Downloads/train.csv")  # Path to your downloaded CSV file
    if not context:
        return  # Exit if dataset is empty or not found
    
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        
        # Exit condition
        if user_input.lower() == "exit":
            break

        # Check if the input is a greeting or asking "How are you?"
        response_type = is_greeting_or_how_are_you(user_input)
        
        if response_type == "greeting":
            print("Bot: Hi there! 😊 I'm so glad to see you. How can I assist you today?")
            continue  # Skip usual processing and greet the user

        elif response_type == "how_are_you":
            print("Bot: I'm doing great, thank you for asking! 😄 How about you? How can I help you today?")
            continue  # Skip usual processing and respond to "How are you?"

        # Use the chain to invoke the model with the context and the user input
        result = chain.invoke({"context": context, "question": user_input})
        
        # Add emotional response based on sentiment analysis
        emotional_response = add_emotional_tone(result, user_input)
        
        # Print bot's emotional response
        print("Bot:", emotional_response)
        
        # Update context with the current conversation
        context += f"\nUser: {user_input}\nAI: {emotional_response}"  # Append the conversation history

if __name__ == "__main__":
    handle_conversation()
