import os
import io
import base64
import requests
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import json


# OpenAI API Key
api_key = st.secrets["OPENAI_API_KEY"]

# Define the prompt as a global variable
prompt = """
You are an OCR assistant. Your task is to read the provided document. 
Extract the specified key-value information. Please determine the type of document.  
Extract all the information from the image in JSON format. The image has information in different 
languages. Identify the language and extract the information in that language. Also, extract what 
is indicated by the checkbox. Give a one-liner summary of what the document is about and what 
information it gives. Extract text in all the languages given in the form and write which languages
the form is in. If you cannot extract text in any language, 
mention clearly that you can't extract in this language.
Document types are Invoice, Medical report, birth certificate, death certificate, handwritten document, etc.
For medical document (e.g., prescription, medical report, lab test results, etc.),Generate the output based on FHIR standard.
"""

# Function to extract images from PDF, concatenate them into one image per PDF, and save it
def extract_and_concat_images_from_pdf(pdfpath, outfolder):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    
    doc = fitz.open(pdfpath)  # Open the PDF file

    # Initialize a list to store all images for concatenation
    all_images = []

    for page_number in range(len(doc)):  # Iterate through each page
        page = doc.load_page(page_number)
        image_list = page.get_images(full=True)  # Get a list of all images on the page

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]  # The XREF of the image
            base_image = doc.extract_image(xref)  # Extract the image information
            image_bytes = base_image["image"]  # The image bytes
            image_ext = base_image["ext"]  # The image extension
            image_pil = Image.open(io.BytesIO(image_bytes))

            # Append image to list
            all_images.append(image_pil)

    # Concatenate all images vertically into one image
    combined_image = get_concat_v(*all_images)

    # Save combined image
    combined_image_path = os.path.join(outfolder, "combined_image.jpg")
    combined_image.save(combined_image_path)

    return combined_image_path

def get_concat_v(*ims):
    dst = Image.new('RGB', (ims[0].width, sum(im.height for im in ims)))
    y_offset = 0
    for im in ims:
        dst.paste(im, (0, y_offset))
        y_offset += im.height
    return dst

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to make request to OpenAI API
def question_image(url):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if url.startswith("http://") or url.startswith("https://"):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {"type": "image_url", "image_url": {"url":url, "detail":"high"}},
                    ],
                }
            ],
            "max_tokens": 1000
        })
    else:
        base64_image = encode_image(url)

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}","detail":"high"}},
                    ],
                }
            ],
            "max_tokens": 1000
        })

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error processing image: {response.text}"

# Streamlit frontend
def main():
    st.title("PDF/Image Extraction and OCR")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf", "jpeg","jpg", "png"])
    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            pdfpath = os.path.join("uploaded_files", uploaded_file.name)
            if not os.path.exists("uploaded_files"):
                os.makedirs("uploaded_files")
            with open(pdfpath, "wb") as f: 
                f.write(uploaded_file.getbuffer())
            
            st.write("PDF uploaded successfully!")

            outfolder = "extracted_images"
            combined_image_path = extract_and_concat_images_from_pdf(pdfpath, outfolder)

            st.image(combined_image_path, caption="Combined Image")

            # Perform OCR on the combined image
            result = question_image(combined_image_path)
            st.write("OCR Result:")
            try:
                result_json = json.loads(result)
                st.json(result_json)
            except json.JSONDecodeError:
                st.write(result)
            
            # Process image files
        elif file_type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(uploaded_file)
            image_path = os.path.join("uploaded_files", uploaded_file.name)
            if not os.path.exists("uploaded_files"):
                os.makedirs("uploaded_files")
            image.save(image_path)

            st.image(image_path, caption=os.path.basename(image_path))

            # Perform OCR on the image
            result = question_image(image_path)
            st.write("OCR Result:")
            try:
                result_json = json.loads(result)
                st.json(result_json)
            except json.JSONDecodeError:
                st.write(result)

if __name__ == "__main__":
    main()




