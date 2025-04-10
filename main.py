import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from grammar_check import analyze_grammar

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

def check_sentence():
    input_text = text_input.get("0.0", "end").strip()
    if not input_text:
        messagebox.showwarning("Warning", "Please enter a sentence to check.")
        return

    from grammar_check import analyze_grammar

    result_output.configure(state="normal")
    result_output.delete("0.0", "end")

    result = analyze_grammar(input_text)

    if not result['issues']:
        result_output.insert("0.0", "Your sentence looks good!")
    else:
        result_output.insert("0.0", "⚠ Issues detected:\n\n")
        for issue in result['issues']:
            result_output.insert("end", f"• {issue}\n")

    result_output.configure(state="disabled")



def clear_text():
    text_input.delete("0.0", "end")
    
    # Enable, clear, then disable
    result_output.configure(state="normal")
    result_output.delete("0.0", "end")
    result_output.configure(state="disabled")

def apply_suggestion():
    messagebox.showinfo("SmartSpell", "Suggestion applied!")

app = ctk.CTk()
app.title("SmartSpell")

app.geometry("1000x700")

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
x_position = int((screen_width / 2) - (1000 / 2))  
y_position = int((screen_height / 2) - (700 / 2))  
app.geometry(f"1000x700+{x_position}+{y_position}")

app.resizable(True, True)  

PRIMARY_COLOR = "#355c7d"
SECONDARY_COLOR = "#e6f2ff"
WHITE = "#ffffff"
TEXT_COLOR = "#000000"

def create_label(parent, text, font_size, weight="normal", color=TEXT_COLOR):
    return ctk.CTkLabel(
        parent,
        text=text,
        font=ctk.CTkFont(family="Helvetica", size=font_size, weight=weight),
        text_color=color
    )

main_frame = ctk.CTkFrame(app, corner_radius=0, fg_color=WHITE)
main_frame.pack(fill="both", expand=True)

header_frame = ctk.CTkFrame(main_frame, corner_radius=0, fg_color=PRIMARY_COLOR, height=70)
header_frame.pack(fill="x")
create_label(header_frame, "SmartSpell", 24, "bold", WHITE).pack(side="left", padx=30, pady=15)

content_frame = ctk.CTkFrame(main_frame, corner_radius=0, fg_color=WHITE)
content_frame.pack(fill="both", expand=True, padx=10, pady=25)
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=0)  # Separator
content_frame.grid_columnconfigure(2, weight=1)
content_frame.grid_rowconfigure(1, weight=1)

create_label(content_frame, "Your Text", 16, "bold", PRIMARY_COLOR).grid(row=0, column=0, sticky="w", padx=5, pady=(0, 15))

input_frame = ctk.CTkFrame(content_frame, corner_radius=15, fg_color=SECONDARY_COLOR)
input_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 15))

# Enhanced input text box with improved scrolling
text_input = ctk.CTkTextbox(
    input_frame, 
    font=ctk.CTkFont(family="Helvetica", size=14), 
    corner_radius=10, 
    fg_color=WHITE, 
    text_color=TEXT_COLOR,
    height=400,
    wrap="word"  # Enable word wrapping
)
text_input.pack(fill="both", expand=True, padx=15, pady=15)

separator = ctk.CTkFrame(content_frame, width=2, fg_color=SECONDARY_COLOR)
separator.grid(row=0, column=1, rowspan=2, sticky="ns", pady=2)

create_label(content_frame, "Result", 16, "bold", PRIMARY_COLOR).grid(row=0, column=2, sticky="w", padx=20, pady=(0, 15))

result_frame = ctk.CTkFrame(content_frame, corner_radius=15, fg_color=SECONDARY_COLOR)
result_frame.grid(row=1, column=2, sticky="nsew", padx=(15, 0))

# Enhanced result output with improved scrolling
result_output = ctk.CTkTextbox(
    result_frame, 
    font=ctk.CTkFont(family="Helvetica", size=14), 
    corner_radius=10, 
    fg_color=WHITE, 
    text_color=TEXT_COLOR,
    height=400,
    wrap="word"  # Enable word wrapping
)
result_output.pack(fill="both", expand=True, padx=15, pady=15)
result_output.configure(state="disabled")  # Make read-only after creation

button_container = ctk.CTkFrame(main_frame, corner_radius=0, fg_color=WHITE)
button_container.pack(fill="x", pady=(0, 25))

button_frame = ctk.CTkFrame(button_container, corner_radius=0, fg_color=WHITE)
button_frame.pack(pady=(10, 0))  # This centers the inner frame

buttons = [
    ("Check Text", check_sentence, PRIMARY_COLOR),
    ("Apply Suggestion", apply_suggestion, PRIMARY_COLOR),
    ("Clear", clear_text, "#bbbbbb")
]

for text, command, color in buttons:
    ctk.CTkButton(
        button_frame,
        text=text,
        command=command,
        font=ctk.CTkFont(family="Helvetica", size=14),
        corner_radius=8,
        fg_color=color,
        hover_color="#3d83d9" if color == PRIMARY_COLOR else "#c0c0c0",
        height=45,
        width=150 if text != "Clear" else 100,
        text_color=WHITE
    ).pack(side="left", padx=(0, 15) if text != "Clear" else (0, 0))

app.mainloop()
