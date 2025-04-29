import customtkinter as ctk
from tkinter import messagebox
from engine import correct_text
from context import analyze_context
import re

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
    grammar_issues = [s for s in suggestions if s["type"] == "grammar"]
    if grammar_issues:
        messagebox.showinfo("Grammar Suggestions", "Please manually correct grammar issues.")
    else:
        text_input.delete("0.0", "end")
        text_input.insert("0.0", corrected)
        messagebox.showinfo("Corrections Applied", "Spelling corrections applied!")

def display_corrections(suggestions):
    result_output.configure(state="normal")
    result_output.delete("0.0", "end")
    
    if not suggestions:
        result_output.insert("0.0", "✅ All checks passed!\n")
    else:
        for s in suggestions:
            tag = s["type"]
            msg = ""
            if tag == "grammar":
                msg = f"Grammar: {s['message']}\n"
                result_output.insert("end", msg, "grammar")
            elif tag == "spelling":
                msg = f"Spelling: {s['original']} → {s['suggested']}\n"
                result_output.insert("end", msg, "spelling")
    
    result_output.tag_config("grammar", foreground="#FF0000")  # Red
    result_output.tag_config("spelling", foreground="#FF8C00")  # Orange
    
    result_output.configure(state="disabled")

def update_recent_checks(original, corrected):
    recent_checks.insert(0, {"original": original, "corrected": corrected})
    recent_checks_display.configure(state="normal")
    recent_checks_display.delete("0.0", "end")

    for check in recent_checks[:5]:
        recent_checks_display.insert("end", "Original:\n", "bold")
        recent_checks_display.insert("end", f"{check['original']}\n")
        recent_checks_display.insert("end", "Corrected:\n", "bold")
        recent_checks_display.insert("end", f"{check['corrected']}\n\n")
    
    recent_checks_display.tag_config("bold", font=("Segoe UI", 13, "bold"))
    recent_checks_display.configure(state="disabled")

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

result_output = ctk.CTkTextbox(frame, height=120, state="disabled", font=("Segoe UI", 14))
result_output.pack(fill="both", expand=True, pady=10)

recent_label = ctk.CTkLabel(frame, text="📚 Recent Checks", font=("Segoe UI", 20, "bold"))
recent_label.pack(pady=10)

recent_checks_display = ctk.CTkTextbox(frame, height=130, state="disabled", font=("Segoe UI", 13))
recent_checks_display.pack(fill="both", expand=True, pady=10)

app.mainloop()
