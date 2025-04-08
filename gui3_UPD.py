import customtkinter
import tkinterDnD
import tkinter as tk
from tkinter import *
import os
from PIL import Image, ImageTk
import threading
import ChatbotCode_wGUI as CC_utils

customtkinter.set_ctk_parent_class(tkinterDnD.Tk)

customtkinter.set_appearance_mode("light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class ChatApplication(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.gui_setup()

    def gui_setup(self):
        # Configure window
        self.title("Solace.AI")
        self.geometry(f"{1100}x{580}")

        # Configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Load logo image and display in sidebar frame
        logo_path = os.path.join(os.getcwd(), "./Gemini_Generated_Image_xtlzn7xtlzn7xtlz.jpeg")
        logo_img = Image.open(logo_path).resize((200, 100))  # Add the .resize() method with the desired dimensions
        logo_photo = ImageTk.PhotoImage(logo_img)  # Convert PIL.Image to tkinter.PhotoImage
        self.logo_label = tk.Label(self.sidebar_frame, image=logo_photo, bg=self.sidebar_frame["bg"])  # Create a tkinter.Label
        self.logo_label.image = logo_photo  # Keep a reference to the image
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))  # This is the correct position for the logo_label grid statement

        # Create text box and scroll bar
        self.text_frame = Frame(self, bg="white")
        self.text_frame.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.text_frame.grid_propagate(False)  # prevent frame from expanding
        self.text_box = customtkinter.CTkTextbox(self.text_frame, bg_color="white", text_color="#36454F", border_width=0, height=20, width=50, font=customtkinter.CTkFont(size=13))
        self.text_box.pack(side="left", fill="both", expand=True)
        self.scrollbar = customtkinter.CTkScrollbar(self, command=self.text_box.yview)
        self.text_box.configure(yscrollcommand=self.scrollbar.set)

        # Create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text="How Can I Help?")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.entry.bind("<Return>", self.send_message)

        # Define send_button
        self.send_button = customtkinter.CTkButton(self, text="Chat", command=self.send_message)
        self.send_button.grid(row=3, column=3, padx=(0, 20), pady=(20, 20), sticky="nsew")

    def send_message(self, event=None): #event parameter for Enter key binding
        message = self.entry.get()
        self.entry.delete(0, customtkinter.END) #Clear the entry box

        #Add user's message
        self.text_box.insert(customtkinter.END, "You: " + message + "\n")

        #Simulate a response
        genResp = CC_utils.handle_conversation(message)
        response = "Solace: " + str(genResp) + "\n"
        self.text_box.insert(customtkinter.END, response)
        self.text_box.see(customtkinter.END)  # Scroll to the bottom


if __name__ == "__main__":
    app = ChatApplication()
    app.mainloop()