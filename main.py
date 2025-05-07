import customtkinter as ctk
from tkinter import messagebox
from engine import correct_text
from context import analyze_context
import re

# Handle spaCy import with version check
try:
    import spacy
    if spacy.__version__ < '3.0.0':
        raise ImportError("SpaCy version too old - please upgrade")
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
except (ImportError, OSError) as e:
    SPACY_AVAILABLE = False
    messagebox.showwarning(
        "Warning", 
        f"spaCy error: {str(e)}\n\n"
        "To install the latest version, run:\n"
        "1. pip uninstall spacy\n"
        "2. pip install spacy\n"
        "3. python -m spacy download en_core_web_sm"
    )

recent_checks = []

def fix_spacing_punctuation(text):
    return re.sub(r'\s+([?.!,])', r'\1', text)

def check_text():
    orig_text = text_input.get("0.0", "end").strip()
    if not orig_text:
        messagebox.showwarning("Warning", "Please enter text.")
        return

    orig_text = fix_spacing_punctuation(orig_text)
    corrected, suggestions = correct_text(orig_text)
    context = analyze_context(orig_text)

    display_corrections(suggestions)
    update_recent_checks(orig_text, corrected)

def apply_suggestions():
    orig_text = text_input.get("0.0", "end").strip()
    corrected, suggestions = correct_text(orig_text)
    
    # Show what changes will be made
    changes = []
    for s in suggestions:
        if s.get("original") and s.get("suggested"):
            changes.append(f"'{s['original']}' → '{s['suggested'][0]}'")
    
    if changes:
        message = "Applied corrections:\n" + "\n".join(changes)
        messagebox.showinfo("Corrections Applied", message)
        
        # Apply the corrections
        text_input.delete("0.0", "end")
        text_input.insert("0.0", corrected)
    else:
        messagebox.showinfo("No Changes", "No corrections were needed.")
    
    # Ensure the corrected text is updated in the recent checks display
    update_recent_checks(orig_text, corrected)

    # Update the text input with the corrected text
    text_input.delete("0.0", "end")
    text_input.insert("0.0", corrected)

def display_corrections(suggestions):
    result_output.configure(state="normal")
    result_output.delete("0.0", "end")
    
    if not suggestions:
        result_output.insert("0.0", "✅ All checks passed!\n")
        result_output.configure(state="disabled")
        return

    # Group suggestions by type
    grammar_issues = [s for s in suggestions if s["type"] == "grammar"]
    spelling_issues = [s for s in suggestions if s["type"] == "spelling"]
    
    # Display grammar issues first
    if grammar_issues:
        result_output.insert("end", "Grammar Issues:\n", "header")
        for s in grammar_issues:
            if s.get("message"):
                # For messages with 'Use X instead of Y' format
                if "Use '" in s["message"] and "' instead of '" in s["message"]:
                    msg = f"• {s['message']}\n"
                else:
                    # For other grammar messages
                    if s.get("original") and s.get("suggested"):
                        msg = f"• Change '{s['original']}' to '{s['suggested'][0]}'\n"
                    else:
                        msg = f"• {s['message']}\n"
                result_output.insert("end", msg, "grammar")
        result_output.insert("end", "\n")
    
    # Then display spelling issues
    if spelling_issues:
        result_output.insert("end", "Spelling Issues:\n", "header")
        for s in spelling_issues:
            suggestions_str = ', '.join(s['suggested']) if s.get('suggested') else 'No suggestions'
            msg = f"• '{s['original']}' → {suggestions_str}\n"
            result_output.insert("end", msg, "spelling")
    
    # Configure tags for different types of text
    result_output.tag_config("header", foreground="#000000")  # Removed 'font' option
    result_output.tag_config("grammar", foreground="#FF0000")  # Red
    result_output.tag_config("spelling", foreground="#FF8C00")  # Orange
    
    result_output.configure(state="disabled")

def update_recent_checks(original, corrected):
    recent_checks.insert(0, {"original": original, "corrected": corrected})
    recent_checks_display.configure(state="normal")
    recent_checks_display.delete("0.0", "end")

    for check in recent_checks[:5]:
        recent_checks_display.insert("end", "Original:\n")
        recent_checks_display.insert("end", f"{check['original']}\n")
        recent_checks_display.insert("end", "Corrected:\n")
        recent_checks_display.insert("end", f"{check['corrected']}\n\n")

    # Removed the 'font' option to avoid scaling incompatibility
    recent_checks_display.configure(state="disabled")

def bind_shortcuts(event=None):
    app.bind('<Control-Return>', lambda e: check_text())
    app.bind('<Control-s>', lambda e: apply_suggestions())
    app.bind('<Control-l>', lambda e: clear_text())

def clear_text():
    text_input.delete("0.0", "end")
    result_output.configure(state="normal")
    result_output.delete("0.0", "end")
    result_output.configure(state="disabled")

def update_char_count(event=None):
    char_count = len(text_input.get("0.0", "end").strip())
    char_counter_label.configure(text=f"Characters: {char_count}")
    
    # Autosave after typing stops for 2 seconds
    if hasattr(update_char_count, 'timer'):
        app.after_cancel(update_char_count.timer)
    update_char_count.timer = app.after(2000, autosave)

def autosave():
    text = text_input.get("0.0", "end").strip()
    if text:
        with open("autosave.txt", "w", encoding="utf-8") as f:
            f.write(text)

# UI Setup
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("SmartSpell")
app.geometry("950x700")

frame = ctk.CTkFrame(app)
frame.pack(fill="both", expand=True, padx=20, pady=20)

title_label = ctk.CTkLabel(frame, text="📝 SmartSpell", font=("Segoe UI", 26, "bold"))
title_label.pack(pady=10)

text_input = ctk.CTkTextbox(frame, height=150, font=("Segoe UI", 15))
text_input.pack(fill="both", expand=True, pady=10)

btn_frame = ctk.CTkFrame(frame)
btn_frame.pack(fill="x", pady=10)

check_btn = ctk.CTkButton(btn_frame, text="Check Text 🔍", command=check_text, fg_color="#5865F2")
check_btn.pack(side="left", padx=10, pady=10)

apply_btn = ctk.CTkButton(btn_frame, text="Apply Corrections ✅", command=apply_suggestions, fg_color="#57F287", text_color="#000000")
apply_btn.pack(side="left", padx=10, pady=10)

# Add clear button
clear_btn = ctk.CTkButton(btn_frame, text="Clear ⌫", command=clear_text, fg_color="#FF0000")
clear_btn.pack(side="left", padx=10, pady=10)

result_output = ctk.CTkTextbox(frame, height=120, state="disabled", font=("Segoe UI", 14))
result_output.pack(fill="both", expand=True, pady=10)

recent_label = ctk.CTkLabel(frame, text="📚 Recent Checks", font=("Segoe UI", 20, "bold"))
recent_label.pack(pady=10)

recent_checks_display = ctk.CTkTextbox(frame, height=130, state="disabled", font=("Segoe UI", 13))
recent_checks_display.pack(fill="both", expand=True, pady=10)

# Add character counter
char_counter_label = ctk.CTkLabel(frame, text="Characters: 0", font=("Segoe UI", 12))
char_counter_label.pack(pady=5)

# Bind events
text_input.bind('<KeyRelease>', update_char_count)
bind_shortcuts()

# Try to load autosaved content
try:
    with open("autosave.txt", "r", encoding="utf-8") as f:
        text_input.insert("0.0", f.read())
        update_char_count()
except FileNotFoundError:
    pass

app.mainloop()