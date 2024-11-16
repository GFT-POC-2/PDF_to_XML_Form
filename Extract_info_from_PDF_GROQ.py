import streamlit as st
import xml.etree.ElementTree as ET
from pdf2image import convert_from_path
import base64
from PIL import Image
import io
import os
import tempfile
from groq import Groq
import google.generativeai as genai
from collections import OrderedDict

# Initialize Groq client with the API key
groq_api_key = "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"  # Replace with your actual API key
client = Groq(api_key=groq_api_key)

# Set up the API key for Gemini
os.environ["API_KEY"] = "AIzaSyCOhsh-JWBd6B006GA0UgdIW6wRcNon7lk"
genai.configure(api_key=os.environ["API_KEY"])

# Function to extract column names from XML
def extract_column_names(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    columns = OrderedDict()  # Use OrderedDict to preserve order
    for elem in root.iter():
        if elem is not root and elem.tag.lower() != "row":  # Skip the root element and any case-insensitive "row"
            if elem.tag not in columns:  # Avoid duplicate entries
                columns[elem.tag] = None
    return list(columns.keys())  # Return the ordered list of column names

# Function to convert PDF to images
def pdf_to_images_in_memory(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf.close()
        images = convert_from_path(temp_pdf.name)
    return images

# Function to encode an image to base64
def encode_image_to_base64(image):
    with io.BytesIO() as img_buffer:
        image.save(img_buffer, format="JPEG")
        return base64.b64encode(img_buffer.getvalue()).decode('utf-8')

# Function to process images with Groq and Llama Vision
def process_images_with_groq(images):
    results = []
    for idx, img in enumerate(images):
        base64_image = encode_image_to_base64(img)
        try:
            # Send the image to Llama Vision
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all the text on this image and nothing else. Do not provide explanation or context."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="llama-3.2-90b-vision-preview",
            )

            # Access the extracted text from the response
            extracted_text = chat_completion.choices[0].message.content  # Adjusted to match the response structure
            results.append({
                "page": idx + 1,
                "text": extracted_text,
            })
        except Exception as e:
            results.append({
                "page": idx + 1,
                "text": f"Error: {e}",
            })
    return results

# Streamlit app
def main():
    st.title("Vision-Language Document Processor")
    st.write("Upload an XML file and a PDF file to extract structured data.")

    uploaded_xml = st.file_uploader("Upload XML File", type=["xml"])
    uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])

    if uploaded_xml and uploaded_pdf:
        try:
            # Step 1: Extract column names from XML
            column_names = extract_column_names(uploaded_xml)

            st.subheader("Extracted Column Names (from XML):")
            st.write(column_names)

            # Step 2: Convert PDF to images
            st.write("\n\n")
            st.write("Processing PDF...")
            images = pdf_to_images_in_memory(uploaded_pdf)

            # Step 3: Extract text from images using Groq and Llama Vision
            st.write("\n\n")
            st.write("Extracting text from PDF pages using Vision-Language Model for Document Understanding...")
            st.write("\n\n")
            extracted_texts = process_images_with_groq(images)

            st.subheader("Extracted Text (from PDF):")
            for result in extracted_texts:
                st.write(f"Page {result['page']} Text:")
                st.text(result['text'])

            # Combine all extracted text
            combined_text = "\n".join([result['text'] for result in extracted_texts])

            # Step 4: Automatically process combined text with Gemini LLM
            if combined_text.strip():
                st.write("\n\n")
                st.write("Using a LLM to extract structured data...")
                model = genai.GenerativeModel("gemini-1.5-flash")
                llm_prompt = f"""
                    From the following extracted text from a PDF:

                    {combined_text}

                    Extract information related to the following fields:

                    {', '.join(column_names)}

                    Output Requirements:
                    - Provide a Python dictionary containing the extracted information.
                    - Use the field names from the list as keys.
                    - If no information is found for a field, use 'None' as the value.
                """
                try:
                    response_llm = model.generate_content(llm_prompt)
                    raw_response = response_llm.text

                    st.subheader("Extracted Structured Data:")
                    st.write(raw_response)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload both XML and PDF files to proceed.")

if __name__ == "__main__":
    main()
