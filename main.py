import os
import requests
import customtkinter as ctk
from tkinter import messagebox

print("Working dir:", os.getcwd())

# public LanguageTool API endpoint
LT_URL = "https://api.languagetool.org/v2/check"

# UI setup
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.title("SmartSpell")
app.geometry("1000x700")
w, h = app.winfo_screenwidth(), app.winfo_screenheight()
app.geometry(f"1000x700+{(w-1000)//2}+{(h-700)//2}")
app.resizable(True, True)

P, S, W, T = "#355c7d", "#e6f2ff", "#ffffff", "#000000"
suggestions = []
corrected_text = ""

def check_text():
    global suggestions, corrected_text

    text = text_input.get("0.0", "end").strip()
    if not text:
        messagebox.showwarning("Warning","Please enter a sentence to check.")
        return

    # call the public LT API
    resp = requests.post(LT_URL, data={"text": text, "language": "en-US"}).json()
    matches = resp.get("matches", [])

    suggestions = []
    # apply first suggestion inline to show corrected text
    corrected_text = text
    for m in matches:
        orig = corrected_text[m["offset"]:m["offset"]+m["length"]]
        if m["replacements"]:
            sug = m["replacements"][0]["value"]
            suggestions.append({"original": orig, "suggested": sug})
            # apply only the first replacement for each match
            corrected_text = (
                corrected_text[:m["offset"]]
                + sug
                + corrected_text[m["offset"]+m["length"]:]
            )

    # display
    result_output.configure(state="normal"); result_output.delete("0.0","end")
    if not suggestions:
        result_output.insert("0.0", "\u2705 Your sentence looks good!")
    else:
        result_output.insert("0.0", "\u26A0 Grammar Suggestions:\n\n")
        for s in suggestions:
            result_output.insert("end", f"• {s['original']} → {s['suggested']}\n")
    result_output.configure(state="disabled")

def apply_suggestion():
    if not suggestions:
        messagebox.showinfo("SmartSpell","No corrections to apply.")
        return
    text_input.delete("0.0","end")
    text_input.insert("0.0", corrected_text)
    messagebox.showinfo("SmartSpell","Grammar fixes applied!")

def clear_all():
    global suggestions, corrected_text
    suggestions = []
    corrected_text = ""
    text_input.delete("0.0","end")
    result_output.configure(state="normal"); result_output.delete("0.0","end")
    result_output.configure(state="disabled")

def mklabel(p,t,sz,b=False,c=T):
    return ctk.CTkLabel(p, text=t, font=ctk.CTkFont(size=sz, weight="bold" if b else "normal"), text_color=c)

# layout (same as before)
frame = ctk.CTkFrame(app, fg_color=W); frame.pack(fill="both",expand=True)
hdr = ctk.CTkFrame(frame, fg_color=P, height=60); hdr.pack(fill="x")
mklabel(hdr,"SmartSpell",24,True,W).pack(side="left",padx=20,pady=10)

cf = ctk.CTkFrame(frame, fg_color=W); cf.pack(fill="both",expand=True,padx=10,pady=20)
cf.grid_columnconfigure(0,weight=1); cf.grid_columnconfigure(1,weight=0); cf.grid_columnconfigure(2,weight=1); cf.grid_rowconfigure(1,weight=1)

mklabel(cf,"Your Text",16,True,P).grid(row=0,column=0,sticky="w")
inp=ctk.CTkFrame(cf,fg_color=S,corner_radius=10); inp.grid(row=1,column=0,sticky="nsew",padx=(0,15))
text_input=ctk.CTkTextbox(inp,corner_radius=8,fg_color=W,text_color=T,font=ctk.CTkFont(size=14)); text_input.pack(fill="both",expand=True,padx=10,pady=10)

sep=ctk.CTkFrame(cf,width=2,fg_color=S); sep.grid(row=0,column=1,rowspan=2,sticky="ns",pady=5)

mklabel(cf,"Result",16,True,P).grid(row=0,column=2,sticky="w")
outf=ctk.CTkFrame(cf,fg_color=S,corner_radius=10); outf.grid(row=1,column=2,sticky="nsew",padx=(15,0))
result_output=ctk.CTkTextbox(outf,corner_radius=8,fg_color=W,text_color=T,font=ctk.CTkFont(size=14)); result_output.pack(fill="both",expand=True,padx=10,pady=10)
result_output.configure(state="disabled")

btns=ctk.CTkFrame(frame,fg_color=W); btns.pack(fill="x",pady=(0,20))
for label,cmd in [("Check Text",check_text),("Apply Suggestion",apply_suggestion),("Clear",clear_all)]:
    clr=P if label!="Clear" else "#bbbbbb"
    ctk.CTkButton(btns,text=label,command=cmd,fg_color=clr,hover_color="#3d83d9" if clr==P else "#c0c0c0",corner_radius=8,height=40,width=140).pack(side="left",padx=10)

app.mainloop()
