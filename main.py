import os
from spellchecker import SpellChecker
from bert_corrector import BertCorrector
import customtkinter as ctk
from tkinter import messagebox

print("Working dir:", os.getcwd())

# initialize BERT + SpellChecker
bert = BertCorrector()

# UI setup
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.title("SmartSpell")
app.geometry("1000x700")
w, h = app.winfo_screenwidth(), app.winfo_screenheight()
app.geometry(f"1000x700+{(w-1000)//2}+{(h-700)//2}")
app.resizable(True, True)

# colors
P, S, W, T = "#355c7d", "#e6f2ff", "#ffffff", "#000000"

# state
fixed_text = ""
suggestions = []

def check_text():
    global fixed_text, suggestions
    txt = text_input.get("0.0", "end").strip()
    if not txt:
        messagebox.showwarning("Warning", "Please enter text to check.")
        return

    result = bert.correct_text(txt)
    fixed_text = result["corrected_text"]
    suggestions = result["corrections"]

    result_output.configure(state="normal")
    result_output.delete("0.0", "end")
    if not suggestions:
        result_output.insert("0.0", "\u2705 Your sentence looks good!")
    else:
        result_output.insert("0.0", "\u26A0 Suggestions Detected:\n\n")
        for s in suggestions:
            result_output.insert("end", f"• {s['original']} → {s['suggested']}\n")
    result_output.configure(state="disabled")

def apply_suggestions():
    if not suggestions:
        messagebox.showinfo("SmartSpell", "No corrections to apply.")
        return
    text_input.delete("0.0", "end")
    text_input.insert("0.0", fixed_text)
    messagebox.showinfo("SmartSpell", "Corrections applied!")

def clear_all():
    global fixed_text, suggestions
    fixed_text = ""
    suggestions = []
    text_input.delete("0.0", "end")
    result_output.configure(state="normal")
    result_output.delete("0.0", "end")
    result_output.configure(state="disabled")

def mklabel(parent, txt, size, bold=False, color=T):
    return ctk.CTkLabel(
        parent,
        text=txt,
        font=ctk.CTkFont(size=size, weight="bold" if bold else "normal"),
        text_color=color
    )

# layout
frame = ctk.CTkFrame(app, fg_color=W)
frame.pack(fill="both", expand=True)

# header
hdr = ctk.CTkFrame(frame, fg_color=P, height=60)
hdr.pack(fill="x")
mklabel(hdr, "SmartSpell", 24, True, W).pack(side="left", padx=20, pady=10)

# content
cf = ctk.CTkFrame(frame, fg_color=W)
cf.pack(fill="both", expand=True, padx=10, pady=20)
cf.grid_columnconfigure(0, weight=1)
cf.grid_columnconfigure(1, weight=0)
cf.grid_columnconfigure(2, weight=1)
cf.grid_rowconfigure(1, weight=1)

mklabel(cf, "Your Text", 16, True, P).grid(row=0, column=0, sticky="w")
inp = ctk.CTkFrame(cf, fg_color=S, corner_radius=10)
inp.grid(row=1, column=0, sticky="nsew", padx=(0,15))
text_input = ctk.CTkTextbox(
    inp, corner_radius=8, fg_color=W, text_color=T,
    font=ctk.CTkFont(size=14)
)
text_input.pack(fill="both", expand=True, padx=10, pady=10)

sep = ctk.CTkFrame(cf, width=2, fg_color=S)
sep.grid(row=0, column=1, rowspan=2, sticky="ns", pady=5)

mklabel(cf, "Result", 16, True, P).grid(row=0, column=2, sticky="w")
outf = ctk.CTkFrame(cf, fg_color=S, corner_radius=10)
outf.grid(row=1, column=2, sticky="nsew", padx=(15,0))
result_output = ctk.CTkTextbox(
    outf, corner_radius=8, fg_color=W, text_color=T,
    font=ctk.CTkFont(size=14)
)
result_output.pack(fill="both", expand=True, padx=10, pady=10)
result_output.configure(state="disabled")

# buttons
btns = ctk.CTkFrame(frame, fg_color=W)
btns.pack(fill="x", pady=(0,20))
for label, cmd in [
    ("Check Text", check_text),
    ("Apply Suggestion", apply_suggestions),
    ("Clear", clear_all)
]:
    clr = P if label != "Clear" else "#bbbbbb"
    ctk.CTkButton(
        btns,
        text=label,
        command=cmd,
        fg_color=clr,
        hover_color="#3d83d9" if clr == P else "#c0c0c0",
        corner_radius=8,
        height=40,
        width=140
    ).pack(side="left", padx=10)

app.mainloop()
