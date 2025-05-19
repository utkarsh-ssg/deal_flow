import os
import pickle
import fitz
from PIL import Image
from openai import OpenAI
import streamlit as st
import base64
import io
import google.generativeai as genai
import json
import pandas as pd
import hashlib


CACHE_DIR = "pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPEN_AI_API_KEY')

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


client = OpenAI(api_key=OPENAI_API_KEY)

def get_cache_path(file_hash):
    return os.path.join(CACHE_DIR, f"{file_hash}.pkl")


def save_to_cache(file_hash, data_dict):
    with open(get_cache_path(file_hash), "wb") as f:
        pickle.dump(data_dict, f)

def load_from_cache(file_hash):
    try:
        with open(get_cache_path(file_hash), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    


def convert_pdf_page_to_image(pdf_bytes, page_num):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img



def process_image(image: Image.Image) -> str:
    if not client.api_key:
        raise ValueError("OpenAI API key not found. Please check your environment variables.")
    
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    prompt = """
    Extract the text content from this image.
    Return all the text content from the image, preserving the structure and relationships.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content


# def process_image_for_title_report(image):
#     if not GOOGLE_API_KEY:
#         st.error("Gemini API key not found. Please check your .env file.")
#         return ""
        
#     model = genai.GenerativeModel("gemini-1.5-flash")
    
#     with io.BytesIO() as output:
#         image.save(output, format="PNG")
#         image_bytes = output.getvalue()
    
#     image_parts = [
#         {
#             "mime_type": "image/png",
#             "data": base64.b64encode(image_bytes).decode('utf-8')
#         }
#     ]
    
#     prompt = """
#     Extract all text content from this land title search report image.
    
#     Focus on accurately capturing:
#     - The complete 'Observation' or 'Observations' section (extremely important)
#     - All property details and legal descriptions
#     - Title information and history
#     - All encumbrances, liens, and mortgages
#     - Any references to supporting documents
#     - Any noted legal issues or restrictions
#     - All dates and monetary values exactly as they appear
    
#     Preserve the original text formatting, paragraph structure, and section organization.
#     Include ALL text visible in the image, maintaining the exact wording.
#     Do not summarize or modify the content in any way.
#     """
    
#     try:
#         response = model.generate_content(
#             [prompt, image_parts[0]],
#             generation_config={"temperature": 0.1}
#         )
#         return response.text
#     except Exception as e:
#         st.error(f"Error processing image with Gemini API: {e}")
#         return ""


   
def extract_structured_data(full_text):
    
    prompt = f"""
        From the following extracted text from TATA Capital Housing Finance documents:
        
        {full_text}
        
        Extract and organize the data into two parts:
        
        PART 1: Extract this table data with these columns aligned by row:
        - Sr. No.
        - Tranche Amount (Rs Cr)
        - Cumulative Disbursement (Rs Cr)
        - Construction % (Europa, Mynsa & Capella)
        - Incremental Collection/Promoters' Contribution (Rs Cr)
        
        PART 2: Extract these as separate bullet point lists that apply to all rows:
        - Pre-Disbursement Conditions: These are the "Pre-Disbursement" conditions for first loan
        - Conditions Precedent: These are the "Takeover Conditions(pre-disbursement and disbursement separetely)" for all other loan except first loan.
        - Conditions Subsequent with Frequency: These are the "Covenants" with both the Covenant and Timeline from the table. Fetch it as "Covenant" : "Timeline".
        
        Return as valid JSON in this exact format:
        {{
        "table_data": [
            {{
            "Sr. No.": 1,
            "Tranche Amount (Rs Cr)": 12.00,
            "Cumulative Disbursement (Rs Cr)": 12.00,
            "Construction % (Europa, Mynsa & Capella) 3 New Towers Proposed"": "",
            "Incremental Collection/Promoters' Contribution Overall Project (Rs Cr)": ""
            }},
            {{
            "Sr. No.": 2,
            "Tranche Amount (Rs Cr)": 5.00,
            "Cumulative Disbursement (Rs Cr)": 17.00,
            "Construction % (Europa, Mynsa & Capella) 3 New Towers Proposed": "10.00%",
            "Incremental Collection/Promoters' Contribution Overall Project (Rs Cr)": 5.00
            }},
            // more rows...
        ],
        "pre_disbursement_conditions": [
            "Condition 1",
            "Condition 2",
            // more conditions...
        ],

        "conditions_precedent": [
            "Condition 1",
            "Condition 2",
            // more conditions...
        ],
        "conditions_subsequent": [
            "Covenant 1 - Timeline: Within X days...",
            "Covenant 2 - Timeline: Quarterly...",
            // more covenants...
        ]
        }}
        
        No explanations, no markdown formatting, just the JSON object.
        """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error extracting structured data with OpenAI API: {e}")
        return ""


    
def extract_structured_summary_report(full_text):
    
    try:
        observation_prompt = f"""
        You are a specialized legal document analyzer focusing on land title search reports.
        
        From the extracted Title Report text below:
        \"\"\"{full_text}\"\"\"
        
        Your ONLY task is to find and extract the complete text from any section labeled 'Observation', 'Observations', 'Title Summary', 'Findings', 'Summary', or 'Report Conclusion'.
        
        Extract this section VERBATIM - do not summarize, paraphrase, or modify the text in any way.
        Include the ENTIRE section including all paragraphs.
        
        If multiple such sections exist, concatenate them in order, separated by line breaks.
        If no such section exists, respond with: "NO_EXPLICIT_OBSERVATION_SECTION_FOUND"
        
        Return ONLY the extracted text without any additional commentary, formatting, or explanation.
        """
        
        observation_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract specific sections from legal documents verbatim, without modification."},
                {"role": "user", "content": observation_prompt}
            ],
            temperature=0.1
        )
        
        extracted_observation = observation_response.choices[0].message.content.strip()
        
        
        if extracted_observation == "NO_EXPLICIT_OBSERVATION_SECTION_FOUND":
            observation_instruction = "No explicit Observation section was found. Provide a brief factual summary of the main findings about the property title status."
        else:
            observation_instruction = f"Use EXACTLY this text for the observation field: \"{extracted_observation}\""
        
        complete_prompt = f"""
        You are a specialized legal document analyzer focusing on land title search reports.
        
        From the extracted Title Report text below:
        \"\"\"{full_text}\"\"\"
        
        Return valid JSON in this exact format:
        {{
            "observation": "...",
            "green_flags": ["..."],
            "yellow_flags": ["..."],
            "red_flags": ["..."],
            "references": ["..."],
            "encumbrances": ["..."]
        }}
        
        FOR THE OBSERVATION FIELD:
        {observation_instruction}
        
        For the other fields:
        - "green_flags": List positive findings that indicate a clear title
        - "yellow_flags": List potential minor issues requiring attention
        - "red_flags": List serious issues that may impede transfer or reduce value
        - "references": Extract all supporting documents, case numbers, deed references, or legal citations
        - "encumbrances": Extract all transaction history, liens, mortgages, easements, covenants, or charges
        
        IMPORTANT: Return ONLY valid JSON without any additional text, explanations, or markdown.
        """
        
        complete_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a specialized legal document analyzer that returns only valid JSON."},
                {"role": "user", "content": complete_prompt}
            ],
            temperature=0.1
        )
        
        json_response = complete_response.choices[0].message.content
        
        try:
            parsed_json = json.loads(json_response)
            return json_response
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'({.*})', json_response, re.DOTALL)
            if json_match:
                potential_json = json_match.group(1)
                try:
                    parsed_json = json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    st.error("Could not parse JSON data from response.")
                    return ""
            else:
                st.error("Could not find JSON data in response.")
                return ""
                
    except Exception as e:
        st.error(f"Error extracting structured data: {e}")
        return ""



def create_excel(data):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            
            table_data = pd.DataFrame(data.get("table_data", []))
            
            pre_disbursement_conditions = data.get("pre_disbursement_conditions", [])
            conditions_precedent = data.get("conditions_precedent", [])
            conditions_subsequent = data.get("conditions_subsequent", [])

            pre_disbursement_conditions_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(pre_disbursement_conditions)])
            conditions_precedent_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(conditions_precedent)])
            conditions_subsequent_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(conditions_subsequent)])

            if not table_data.empty:
                cp_col = []
                cs_col = []

                for i in range(len(table_data)):
                    if i == 0:
                        cp_col.append(pre_disbursement_conditions_text)
                        cs_col.append("")
                    else:
                        cp_col.append(conditions_precedent_text)
                        cs_col.append(conditions_subsequent_text)

                table_data["Conditions Precedent"] = cp_col
                table_data["Conditions Subsequent"] = cs_col

            table_data.to_excel(writer, sheet_name="Extracted Data", index=False)

            conditions_df = pd.DataFrame({
                "Conditions Precedent": pd.Series(conditions_precedent),
                "Conditions Subsequent": pd.Series(conditions_subsequent)
            })
            conditions_df.to_excel(writer, sheet_name="Conditions Detail", index=False)

            workbook = writer.book
            worksheet = writer.sheets["Extracted Data"]

            wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
            worksheet.set_column('F:G', 50, wrap_format)

        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel file: {e}")
        return None


def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def clean_dataframe(df):
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = df.columns.astype(str).str.strip()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(
                lambda x: "" if pd.isna(x)
                else f"{int(x)}" if float(x).is_integer()
                else f"{x:.2f}"
            )
        else:
            df[col] = df[col].fillna("")

    return df