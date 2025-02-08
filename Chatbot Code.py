from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Load text file content as context
def load_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print("Error: Text file not found!")
        return ""

# Define the chatbot prompt template
template = """
Answer the question below using the provided information.

Here is the reference text: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = load_text_file("text_for_llama.txt")  # Load context from file
    if not context:
        return  # Exit if file not found
    
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = chain.invoke({"context": context, "question": user_input})
        print("Bot:", result)
        context += f"\nUser: {user_input}\nAI: {result}"  # Update context with conversation history

if __name__ == "__main__":
    handle_conversation()

