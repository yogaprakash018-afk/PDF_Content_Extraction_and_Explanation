import tkinter as tk
from tkinter import filedialog, scrolledtext
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# ------------------------------
# Configure Tesseract
# ------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------------
# Load NLP Models
# ------------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering")

# ------------------------------
# Global text storage
# ------------------------------
document_text = ""

# ------------------------------
# PDF Text Extraction
# ------------------------------
def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

# ------------------------------
# OCR Extraction (for images)
# ------------------------------
def extract_image_text(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)[1]
    text = pytesseract.image_to_string(thresh)
    return text

# ------------------------------
# Summarize Selected Text
# ------------------------------
def summarize_text():
    global document_text
    selected = text_box.selection_get()
    
    if len(selected) < 50:
        result_box.insert(tk.END,"\nSelect larger paragraph.\n")
        return
    
    summary = summarizer(selected, max_length=120, min_length=30, do_sample=False)
    result_box.insert(tk.END,"\nSUMMARY:\n"+summary[0]['summary_text']+"\n")

# ------------------------------
# Explain Paragraph
# ------------------------------
def explain_text():
    selected = text_box.selection_get()
    
    prompt = "Explain in simple terms: " + selected
    
    summary = summarizer(prompt, max_length=150, min_length=40, do_sample=False)
    
    result_box.insert(tk.END,"\nEXPLANATION:\n"+summary[0]['summary_text']+"\n")

# ------------------------------
# Question Answering
# ------------------------------
def ask_question():
    global document_text
    
    question = question_entry.get()
    
    if len(document_text) == 0:
        result_box.insert(tk.END,"\nNo document loaded.\n")
        return
    
    answer = qa_model(question=question, context=document_text)
    
    result_box.insert(tk.END,"\nANSWER:\n"+answer['answer']+"\n")

# ------------------------------
# Load File
# ------------------------------
def load_file():
    global document_text
    
    file_path = filedialog.askopenfilename()
    
    if file_path.endswith(".pdf"):
        document_text = extract_pdf_text(file_path)
    else:
        document_text = extract_image_text(file_path)
    
    text_box.delete(1.0,tk.END)
    text_box.insert(tk.END,document_text)

# ------------------------------
# GUI
# ------------------------------
root = tk.Tk()
root.title("PDF Content Extraction & Explanation System")
root.geometry("900x700")

load_button = tk.Button(root,text="Load PDF/Image",command=load_file)
load_button.pack()

text_box = scrolledtext.ScrolledText(root,height=20)
text_box.pack(fill=tk.BOTH,padx=10,pady=10)

summarize_button = tk.Button(root,text="Summarize Selected Text",command=summarize_text)
summarize_button.pack()

explain_button = tk.Button(root,text="Explain Selected Text",command=explain_text)
explain_button.pack()

question_entry = tk.Entry(root,width=80)
question_entry.pack(pady=10)

ask_button = tk.Button(root,text="Ask Question",command=ask_question)
ask_button.pack()

result_box = scrolledtext.ScrolledText(root,height=10)
result_box.pack(fill=tk.BOTH,padx=10,pady=10)

root.mainloop()
