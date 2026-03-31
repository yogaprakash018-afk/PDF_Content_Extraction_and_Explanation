PDF extraction
Uses: pdfplumber

OCR extraction
Uses: pytesseract + OpenCV

Text explanation / summarization
Uses Transformer NLP models: facebook/bart-large-cnn

Question answering
Uses: BERT QA model through the transformers pipeline.

GUI
Built using: Tkinter

Features:
• Load PDF
• View extracted text
• Select paragraph
• Summarize text
• Explain content
• Ask questions

requirements:
pdfplumber
pytesseract
opencv-python
transformers
torch
pillow
numpy
