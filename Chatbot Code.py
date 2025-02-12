import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

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

        # Use the chain to invoke the model with the context and the user input
        result = chain.invoke({"context": context, "question": user_input})
        
        # Print bot's response
        print("Bot:", result)
        
        # Update context with the current conversation
        context += f"\nUser: {user_input}\nAI: {result}"  # Append the conversation history

if __name__ == "__main__":
    handle_conversation()
