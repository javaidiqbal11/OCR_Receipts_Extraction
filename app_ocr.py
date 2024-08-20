from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import fitz  # PyMuPDF
import cv2
import pytesseract
import openai
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set up the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise Exception("OPENAI_API_KEY not found. Please set it in your .env file.")
openai.api_key = openai_api_key

# Set the Tesseract command path
tesseract_cmd_path = os.getenv("TESSERACT_CMD_PATH")
if not tesseract_cmd_path:
    raise Exception("TESSERACT_CMD_PATH not found. Please set it in your .env file.")
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

# Create the "extraction" folder if it doesn't exist
extraction_folder = "extraction"
if not os.path.exists(extraction_folder):
    os.makedirs(extraction_folder)

app = FastAPI()

def read_image(file: UploadFile):
    """
    Reads and converts the uploaded file (PDF or image) to a list of PIL images.
    """
    try:
        file_type = file.filename.split('.')[-1].lower()
        
        if file_type == 'pdf':
            pdf_doc = fitz.open(stream=file.file.read(), filetype="pdf")
            images = []
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            return images  # List of PIL images
        
        elif file_type in ['jpg', 'jpeg', 'png']:
            image = Image.open(file.file)
            return [image]  # List of one PIL image

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read image: {e}")

def align_image(image: Image):
    """
    Aligns and standardizes the resolution of the input image.
    """
    try:
        # Convert image to numpy array for OpenCV processing
        image_np = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Use thresholding or edge detection to find contours
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour which will be the document
            largest_contour = max(contours, key=cv2.contourArea)

            # Obtain the bounding box coordinates
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop and align the image
            aligned_image = image_np[y:y+h, x:x+w]

            # Resize to a standard size
            aligned_image = cv2.resize(aligned_image, (1024, 1024))

            return aligned_image
        else:
            raise HTTPException(status_code=400, detail="Failed to align image: No contours found.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to align image: {e}")

def extract_text_from_image(image_np):
    """
    Extracts text from an aligned image using Tesseract OCR.
    """
    try:
        text = pytesseract.image_to_string(image_np, lang='eng')
        return text
    except pytesseract.TesseractNotFoundError as e:
        raise HTTPException(status_code=500, detail="Tesseract OCR not found. Please ensure Tesseract is installed correctly.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from image: {e}")

def process_with_gpt(text):
    """
    Uses GPT-4 to analyze and refine the extracted text.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Specify the model (use "gpt-3.5-turbo" if GPT-4 is not available)
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Extract the necessary information from the following receipt text:\n\n{text}"}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text with GPT-4: {e}")

def save_text_to_file(text, filename):
    """
    Saves the extracted text to a text file within the 'extraction' folder.
    """
    try:
        filepath = os.path.join(extraction_folder, filename)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save text to file: {e}")

@app.post("/extract-receipt-info/")
async def extract_receipt_info(file: UploadFile = File(...)):
    """
    API endpoint to extract receipt information from an uploaded file and save the data to a text file.
    """
    try:
        # Step 1: Read and convert the input file
        images = read_image(file)
        
        extracted_info = []
        base_filename = os.path.splitext(file.filename)[0]  # Get the base filename without extension
        
        all_text = ""  # Store all extracted text to be saved in one go

        for image in images:
            # Step 2: Align the image and standardize the resolution
            aligned_image_np = align_image(image)
            
            # Step 3: Extract text using OCR
            extracted_text = extract_text_from_image(aligned_image_np)
            
            # Step 4: Process text with GPT-4
            refined_info = process_with_gpt(extracted_text)
            
            all_text += refined_info + "\n"  # Append refined text to the overall text
        
        # Save the refined information to a text file (saving all extracted text at once)
        output_filename = f"{base_filename}.txt"  # Save using the base name of the input file
        save_text_to_file(all_text, output_filename)
        
        return {"extracted_info": all_text}
    
    except HTTPException as e:
        # Re-raise HTTP exceptions to return the appropriate response
        raise e
    except Exception as e:
        # Catch-all for any other exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# To run the FastAPI app, use:
# uvicorn filename:app --reload
