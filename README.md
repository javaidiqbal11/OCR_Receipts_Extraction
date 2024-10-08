# OCR Receipts Extraction 

This project is a FastAPI-based service for extracting and refining information from receipts. The application leverages various technologies like Tesseract OCR, OpenAI GPT-4, and OpenCV for image processing and text extraction.

## Features
**PDF/Image File Handling:** Upload PDF or image files (JPG, PNG) for text extraction. <br>
**Image Alignment:** Automatically aligns and crops images to standardize them for better OCR performance. <br>
**Text Extraction:** Uses Tesseract OCR to extract text from aligned images. <br>
**Text Refinement:** Refines the extracted text using OpenAI's GPT-4 for improved accuracy. <br>
**File Saving:** Saves the refined text to a .txt file in the extraction folder. <br>


## Requirements
Python 3.7+ <br>
FastAPI <br>
Tesseract 

### The OCR built-in (Tesseract) used for the receipt extraction 

**Setup**
1. Clone the Repository
  ```shell
git clone https://github.com/javaidiqbal11/OCR_Receipts_Extraction.git
cd OCR_Receipts_Extraction
```
2. Install Dependencies
```shell
pip install -r requirements.txt
```
3. Environment Variables
Create a .env file in the root directory with the following contents:
```shell
OPENAI_API_KEY=your-openai-api-key
TESSERACT_CMD_PATH=/path/to/tesseract
```
4. Run the Application
```shell
uvicorn main:app --reload
```
