# streamlit run Extract_info_from_PDF_Simplified.py

import streamlit as st
import xml.etree.ElementTree as ET
from PyPDF2 import PdfReader
from collections import OrderedDict
import google.generativeai as genai
import os

# Set up the API key for Gemini
os.environ["API_KEY"] = "AIzaSyCOhsh-JWBd6B006GA0UgdIW6wRcNon7lk"
genai.configure(api_key=os.environ["API_KEY"])


def extract_column_names(xml_file):
    """Extract column names from an XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        columns = OrderedDict()  # Preserve order of elements
        for elem in root.iter():
            if elem is not root and elem.tag.lower() != "row":  # Skip root and "row" elements
                columns.setdefault(elem.tag, None)  # Avoid duplicates
        return list(columns.keys())
    except Exception as e:
        st.error(f"Error processing XML file: {e}")
        return []


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""


def main():
    st.title("PDF to XML Extractor")
    
    # File upload sections
    uploaded_xml = st.file_uploader("Upload XML File", type=["xml"])
    uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])

    if uploaded_xml and uploaded_pdf:
        # Extract column names from the XML
        st.subheader("Extracting column names from XML...")
        column_names = extract_column_names(uploaded_xml)
        if not column_names:
            st.warning("No column names could be extracted. Please check the XML file.")
            return

        st.subheader("Extracted Column Names (from XML):")
        st.write(column_names)

        # Extract text from the PDF
        st.subheader("Extracting text from PDF...")
        combined_text = extract_text_from_pdf(uploaded_pdf)
        if not combined_text.strip():
            st.warning("No text could be extracted from the PDF. Please check the file.")
            return
        else:
            st.write(combined_text)

        # Use LLM to extract structured data
        st.subheader("Processing extracted data with LLM...")
        with st.spinner("Generating structured data using LLM..."):
            try:
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
                response = model.generate_content(llm_prompt)
                structured_data = response.text

                st.subheader("Extracted Structured Data:")
                st.write(structured_data)
            except Exception as e:
                st.error(f"An error occurred while processing with LLM: {e}")
    else:
        st.info("Please upload both XML and PDF files to proceed.")


if __name__ == "__main__":
    main()
