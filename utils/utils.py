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
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import tempfile
import time

from dotenv import load_dotenv

CACHE_DIR = "pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

load_dotenv()

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
        
        
        Extract the full disbursement table data with the columns aligned by row:
        Along with each row associate these following details for each Tranche.

        - Conditions Precedent for Tranche 1: These are the "Pre-Disbursement" conditions and exclusive conditions which needs to be fulfilled before Tranche 1.
        - Conditions Precedent for other Tranche's: These are the conditions that needs to be fulfilled before the respective Tranche disbursement except the first tranche.
        - Conditions Subsequent with Frequency: These are the "Conditions to be satisfied within 30 days from first disbursement".
        
        {{
        "table_data": [
            {{
            "Sr. No.": 1,
            .
            .
            .
            "conditions_precedent": [
                    "Condition 1",
                    "Condition 2",
                    // more conditions...
                ],
            "conditions_subsequent": [
                    "Condition 1,
                    "Condition 2",
                    // more covenants...
                ]
            
            }},
            {{
            "Sr. No.": 2,
            .
            .
            .
            .
            "conditions_precedent": [
                    "Condition 1",
                    "Condition 2",
                    // more conditions...
                ],
            "conditions_subsequent": [
                    "Condition 1,
                    "Condition 2",
                    // more covenants...
                ]
            }}
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
        return ""
    
def extract_structured_lease_data(full_text):
    prompt = f"""
            You are an expert in legal document extraction. Analyze the following lease document text and extract the following sections ONLY if the information is explicitly mentioned in the document. If any information is missing or not clearly stated, leave that field empty. Format the result as a structured JSON.

            \"\"\"{full_text}\"\"\"

            Extract and organize the data into:

            Return ONLY valid JSON with keys matching the field names above. Do not include any markdown formatting or explanatory text.

            JSON Structure:
            {{
                "Agreement Date": "",
                "Lease registered or notarized": "",
                "Lease Start date": "",
                "Lease end date": "",
                "Lessor Name": "",
                "Lessor Address": "",
                "Lessor GST": "",
                "Lessor Aadhar Number": "",
                "Lessor PAN": "",
                "Lessor Legal Entity Type": "",
                "Lessee Name": "",
                "Lessee Address": "",
                "Lessee GST": "",
                "Lessee Aadhar Number": "",
                "Lessee PAN": "",
                "Lessee Legal Entity Type": "",
                "Unit no": "",
                "Address": "",
                "Rent": [],
                "Area(sq ft)":"",
                "Lock In": "",
                "Escalation": "",
                "Additonal Cost": "",
                "Maintenance": "",
                "Property Tax": "",
                "Security Deposit": "",
                "Other Important Clauses": ["",""],
                "Principal risks" : ["",""],
                "Next Escalation Date":"",
                "Risk Score": "",
                "Pending Tenure": "",
                "Pending Lockin":""
            }}

            For Rent, make it as a list of objects with timeline and rent value of each year.
            
            Other Important Clauses should include the following parts(dont give point numbers give full text as an array of strings):
            1. Where lessor needs to pay an early amount of certain period of time.
            2. Anything related to tax/tax deduction/TDS.
            3. Maintenance and stuff related payment.
            4. Anything related to breaching of the clauses.
            5. Anything related to payments.
            6. Scope of Work(full table).
            7. Safeguarding Clauses.
            8. Parking related Clauses.

            For Principal Risks: Assume you are an analyst looking to discount the rentals and give a loan to the owner of this property please identify the top 5 legal and technical risks from lenders perspective.

            Next Escalation Date: You need to calculate based on the available data. Leave empty if not available

            Pending Tenure: You need to calculate based on today's date and tenure end date in months just numerical value.

            Risk Score: Compute the Risk Score (on a scale of 0-100) for a lease under the Rental-Discount Loan framework. The score is based on weighted sub-factors across 5 categories. Each sub-factor is rated from 1 (very low risk) to 5 (high risk). Below are the inputs:

            1. Tenant Quality (20%)
            External credit rating (15%): [Enter rating 1-5]
            Parent/cross-default support (5%): [Enter rating 1-5]
            2. Key Legal Risks in the Lease (25%)
            Early-termination / break rights (10%): [Enter rating 1-5]
            Sub-lease / assignment rights (5%): [Enter rating 1-5]
            Rent-abatement & force-majeure clauses (5%): [Enter rating 1-5]
            Registration / stamping validity (5%): [Enter rating 1-5]
            3. Financial / Structural Ratios (20%)
            DSCR on stressed rent (10%): [Enter rating 1-5]
            Residual lease term ÷ proposed loan tenor (5%): [Enter rating 1-5]
            LTV on discounted rent value (5%): [Enter rating 1-5]
            4. Building Quality & Approvals (20%)
            Occupancy Certificate, fire NOC (7%): [Enter rating 1-5]
            Structural age & stability (7%): [Enter rating 1-5]
            Clear title & zoning compliance (6%): [Enter rating 1-5]
            5. Other Financial / Compliance Risks (15%)
            Property-tax & GST arrears (5%): [Enter rating 1-5]
            Insurance adequacy & lender endorsement (5%): [Enter rating 1-5]
            Regulatory / ESG compliance burden (5%): [Enter rating 1-5]
            Once all the sub-factor ratings are provided, compute the category score as the weighted average of its sub-factors, then compute the overall weighted rating (S), and finally convert it to a Risk Score (R) using:

            S = Σ (Category Weight x Category Rating)
            R = ((S - 1) / 4) x 100
            Return the final Risk Score out of 100.
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
        raw_response = response.choices[0].message.content
        
        clean_json = extract_json_from_response(raw_response)
        
        return clean_json
    except Exception as e:
        print(f"Error in extract_structured_lease_data: {e}")
        return ""

def extract_json_from_response(response_content):
    json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    json_match = re.search(r'```\s*(.*?)\s*```', response_content, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    json_match = re.search(r'(\{.*\})', response_content, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    return response_content.strip()

    
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
            model="gpt-4-turbo",
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
                    return ""
            else:
                return ""
                
    except Exception as e:
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



def parse_date(date_str):
    if not date_str or not isinstance(date_str, str):
        return None
    
    
    date_str_clean = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', date_str.strip())

    
    formats = [
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str_clean, fmt)
        except ValueError:
            continue
    return None

def calculate_next_escalation(lease_start, lease_end):
    today = datetime.today()

    if not isinstance(lease_start, datetime) or not isinstance(lease_end, datetime):
        return "N/A"
    
    if today > lease_end:
        return "N/A"

    
    next_escalation = lease_start + relativedelta(months=+12)

    
    while next_escalation <= today and next_escalation <= lease_end:
        next_escalation += relativedelta(months=+12)

    if next_escalation > lease_end:
        return "N/A"

    return next_escalation.strftime("%d %B %Y")

def calculate_pending_tenure(lease_end):
    today = datetime.today()
    if lease_end is None or today > lease_end:
        return 0
    
    delta = relativedelta(lease_end, today)
    total_months = delta.years * 12 + delta.months
    return total_months



def process_bank_statement(uploaded_file):
    UPLOAD_URL = "https://cartuat.com/api/upload"
    DOWNLOAD_URL = "https://cartuat.com/api/downloadFile"
    AUTH_TOKEN = "API://QFEreQJLUvIWHKSLliicNPOC/MYh9B7dCo95Chz2rT2Sgf9ihi53EpD8LigFS/tw"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.success("File saved temporarily. Uploading...")

    metadata = {
        "password": "",
        "bank": "Other",
        "name": ""
    }

    document_details = [{
        "groupCompany": "",
        "accountNumber": "",
        "accountType": "",
        "internal": False,
        "odCcLimit": "",
        "organizationName": ""
    }]

    files = {
        "file": open(tmp_file_path, "rb"),
        "metadata": (None, json.dumps(metadata), "application/json"),
        "documentDetails": (None, json.dumps(document_details), "application/json"),
    }

    headers = {
        "Accept": "application/json",
        "auth-token": AUTH_TOKEN
    }

    upload_response = requests.post(UPLOAD_URL, files=files, headers=headers)

    if upload_response.status_code == 200:
        st.success("File uploaded successfully!")

        
        try:
            doc_id = upload_response.json().get("docId")
            

            if doc_id:
                time.sleep(10)
                download_headers = {
                    "Accept": "application/json",
                    "auth-token": AUTH_TOKEN,
                    "Content-Type": "text/plain"
                }

                download_response = requests.post(
                    DOWNLOAD_URL,
                    headers=download_headers,
                    data= doc_id
                )

            else:
                st.error("Document ID not found in upload response.")

        except Exception as e:
            st.error("Failed to parse upload response.")
            st.text(str(e))
    else:
        st.error(f"Upload failed. Status code: {upload_response.status_code}")
        st.text(upload_response.text)

    return download_response


def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0