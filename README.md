# OCR Receipts Extraction 

This project is a FastAPI-based service for extracting and refining information from receipts. The application leverages various technologies like Tesseract OCR, OpenAI GPT-4, and OpenCV for image processing and text extraction.

## Features
**PDF/Image File Handling:** Upload PDF or image files (JPG, PNG) for text extraction.
**Image Alignment:** Automatically aligns and crops images to standardize them for better OCR performance.
**Text Extraction:** Uses Tesseract OCR to extract text from aligned images.
**Text Refinement:** Refines the extracted text using OpenAI's GPT-4 for improved accuracy.
**File Saving:** Saves the refined text to a .txt file in the extraction folder.


## Requirements
Python 3.7+
FastAPI

**How to run?**

```shell
pip install -r requirements.txt
```

