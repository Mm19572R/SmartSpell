import customtkinter as ctk
from tkinter import messagebox
from engine import correct_text
import re

app = ctk.CTk()
app.title("SmartSpell")
app.geometry("950x700")

text_input = ctk.CTkTextbox(app, height=150, font=("Segoe UI", 15))
text_input.pack(fill="both", expand=True, pady=10)

def highlight_issues(suggestions):
    text_input._textbox.tag_remove("grammar", "1.0", "end")
    text_input._textbox.tag_remove("spelling", "1.0", "end")

    for s in suggestions:
        tag = "grammar" if s["type"] == "grammar" else "spelling"
        target = s.get("original", "").strip()

        if not target:
            continue

        start = "1.0"
        while True:
            start = text_input._textbox.search(target, start, stopindex="end", nocase=True)
            if not start:
                break
            end = f"{start}+{len(target)}c"
            text_input._textbox.tag_add(tag, start, end)
            start = end

    text_input._textbox.tag_configure("grammar", foreground="red")
    text_input._textbox.tag_configure("spelling", foreground="orange")

def display_corrections(suggestions):
    result_output.configure(state="normal")
    result_output.delete("0.0", "end")

    if not suggestions:
        result_output.insert("0.0", "✅ All checks passed!")
    else:
        for s in suggestions:
            category = s["type"].capitalize()
            original = str(s.get("original", "")).strip()
            suggested = s.get("suggested", "").strip()
            if not original or not suggested or original == suggested:
                continue

            result_output._textbox.insert("end", f"{category} issue:\n", category.lower())
            result_output._textbox.insert("end", f"• '{original}' → '{suggested}'\n\n", category.lower())

    result_output.configure(state="disabled")

def apply_suggestions():
    orig_text = text_input.get("0.0", "end").strip()
    corrected, suggestions = correct_text(orig_text)
    text_input.delete("0.0", "end")
    text_input.insert("0.0", corrected)
    messagebox.showinfo("Corrections Applied", "All corrections have been applied.")

def check_text():
    orig_text = text_input.get("0.0", "end").strip()
    if not orig_text:
        messagebox.showwarning("Warning", "Please enter text.")
        return

    corrected, suggestions = correct_text(orig_text)
    highlight_issues(suggestions)
    display_corrections(suggestions)

btn_frame = ctk.CTkFrame(app)
btn_frame.pack(fill="x", pady=10)

check_btn = ctk.CTkButton(btn_frame, text="Check Text", command=check_text, fg_color="#5865F2")
check_btn.pack(side="left", padx=10)

apply_btn = ctk.CTkButton(btn_frame, text="Apply Corrections", command=apply_suggestions, fg_color="#57F287", text_color="#000000")
apply_btn.pack(side="left", padx=10)

result_output = ctk.CTkTextbox(app, height=120, state="disabled", font=("Segoe UI", 14))
result_output.pack(fill="both", expand=True, pady=10)

result_output._textbox.tag_configure("grammar", foreground="red")
result_output._textbox.tag_configure("spelling", foreground="orange")

def autosave():
    text = text_input.get("0.0", "end").strip()
    if text:
        with open("autosave.txt", "w", encoding="utf-8") as f:
            f.write(text)

text_input.bind('<KeyRelease>', lambda e: autosave())

try:
    with open("autosave.txt", "r", encoding="utf-8") as f:
        text_input.insert("0.0", f.read())
except FileNotFoundError:
    pass

app.mainloop()
