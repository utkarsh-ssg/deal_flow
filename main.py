import streamlit as st
import pandas as pd
import PyPDF2
import io
import os
import tempfile
import base64
import google.generativeai as genai
from PIL import Image
import fitz
import time
from dotenv import load_dotenv
import json
import re
import hashlib
import os
import pickle

CACHE_DIR = "pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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


# Load environment variables
load_dotenv()

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Page setup - Set to light mode
st.set_page_config(page_title="Developer Dashboard", layout="wide")

# Apply consistent light gray theme with targeted fixes for remaining black backgrounds
st.markdown("""
    <style>
    .stApp {
        background-color: #F0F2F5;
        color: #333333;
    }

    /* Hide header/footer */
    header, footer, .css-18e3th9 { visibility: hidden; height: 0px; }

    /* Set text color to dark gray by default for better contrast on gray background */
    .stMarkdown, .stTextInput, .stFileUploader, .stSelectbox, .stButton, .stDataFrame, .stTabs, .stText, .stTextLabel {
        color: #333333 !important;
    }

    /* Fix tab header text */
    .stTabs [data-baseweb="tab"] {
        color: #333333 !important;
    }

    /* Info/warning/error box text - ensure they remain visible against their backgrounds */
    .element-container .stAlert p {
        color: #333333 !important;
    }
    .stAlert.stAlert-info {
        background-color: #E8F0FE !important;
        border-color: #4285F4 !important;
    }
    .stAlert.stAlert-warning {
        background-color: #FFF8E1 !important;
        border-color: #FFA000 !important;
    }
    .stAlert.stAlert-error {
        background-color: #FFEBEE !important;
        border-color: #E53935 !important;
        color: #940000 !important;
    }
    /* Make error text darker for better visibility on pink background */
    .stAlert.stAlert-error p {
        color: #940000 !important;
    }

    /* File uploader styling - FIX FOR BROWSE FILES BUTTON */
    .stFileUploader > label > div {
        color: #333333 !important;
        background-color: #F0F2F5 !important;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 1px dashed #CCCCCC !important;
    }
    /* Specifically target the Browse Files button */
    .stFileUploader [data-testid="stFileUploaderDropzone"] button {
        background-color: #4285F4 !important;
        color: white !important;
    }
    
    /* Button styling - light blue instead of dark */
    .stButton > button {
        background-color: #4285F4;
        color: white !important;
        font-weight: bold;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #3367D6;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stDownloadButton > button:hover {
        background-color: #45a049 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #4285F4;
    }

    /* DataFrame styling - FIX FOR TABLE BACKGROUNDS */
    .dataframe {
        background-color: #FFFFFF !important;
        border-color: #DDDDDD !important;
    }
    .dataframe th {
        background-color: #EAEEF2 !important;
        color: #333333 !important;
        border-color: #DDDDDD !important;
    }
    .dataframe td {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border-color: #DDDDDD !important;
    }
    /* Ensure table cells and borders don't have black colors */
    table, th, td {
        border-color: #DDDDDD !important;
    }
    
    /* Select box styling - FIX FOR MILESTONE DROPDOWN */
    .stSelectbox [data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }
    .stSelectbox [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }
    .stSelectbox [data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    .stSelectbox [data-baseweb="option"] {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }
    .stSelectbox [data-baseweb="option"]:hover {
        background-color: #F0F2F5 !important;
    }
    
    /* Multiselect styling */
    .stMultiSelect [data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }
    .stMultiSelect [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }
    .stMultiSelect [data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    .stMultiSelect [data-baseweb="option"] {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }
    
    /* Fix background color for all tables */
    div[data-testid="stTable"] {
        background-color: #FFFFFF !important;
    }
    div[data-testid="stTable"] table {
        background-color: #FFFFFF !important;
    }
    div[data-testid="stTable"] th {
        background-color: #EAEEF2 !important;
        color: #333333 !important;
        border-color: #DDDDDD !important;
    }
    div[data-testid="stTable"] td {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border-color: #DDDDDD !important;
    }
    
    /* Fix for expanders */
    .streamlit-expanderHeader {
        background-color: #F5F7FA !important;
        color: #333333 !important;
    }
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
    }
    
    /* Fix for checkboxes */
    .stCheckbox {
        color: #333333 !important;
    }
    
    /* Fix for text input */
    .stTextInput [data-baseweb="input"] {
        background-color: #FFFFFF !important;
    }
    
    /* Fix for tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #E8E8E8 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F5F7FA !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
    }
    
    /* Payment Details table fix */
    div:has(> .stSubheader:contains("Payment Details")) + div [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
    }
    
    /* Project Detail Data table fix */
    div:has(> .stSubheader:contains("Project Detail Data")) + div [data-testid="stDataFrame"],
    div:has(> .stSubheader:contains("MIS Data")) + div [data-testid="stDataFrame"],
    div:has(> .stSubheader:contains("COP-MOF Data")) + div [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
    }
    
    /* Fix for Step 4 tables */
    div:has(> .stHeader:contains("Step 4")) [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
    }
    
    /* Override any inline styles with !important */
    [style*="background-color: black"],
    [style*="background-color:#000000"],
    [style*="background: black"] {
        background-color: #FFFFFF !important;
    }
    
    [style*="color: black"],
    [style*="color:#000000"] {
        color: #333333 !important;
    }
[data-testid="stDataFrame"] .ag-root-wrapper,
[data-testid="stDataFrame"] .ag-root,
[data-testid="stDataFrame"] .ag-header,
[data-testid="stDataFrame"] .ag-body-viewport,
[data-testid="stDataFrame"] .ag-center-cols-container,
[data-testid="stDataFrame"] .ag-pinned-left-cols-container,
[data-testid="stDataFrame"] .ag-row,
[data-testid="stDataFrame"] .ag-cell {
    background-color: #FFFFFF !important;
    color: #333333 !important;
}

/* Header background & text */
[data-testid="stDataFrame"] .ag-header-cell,
[data-testid="stDataFrame"] .ag-header-cell-label {
    background-color: #E0E0E0 !important;
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Fix pinned left column (serial numbers) */
[data-testid="stDataFrame"] .ag-pinned-left-cols-container .ag-cell {
    background-color: #F0F0F0 !important;
    color: #000000 !important;
    font-weight: bold !important;
}

/* Zebra striping for rows */
[data-testid="stDataFrame"] .ag-row:nth-child(even) .ag-cell {
    background-color: #F8F9FB !important;
}

/* Hover effect */
[data-testid="stDataFrame"] .ag-row-hover .ag-cell {
    background-color: #D3E3FC !important;
}

/* Scrollbar wrapper fix */
[data-testid="stDataFrame"] .ag-body-horizontal-scroll,
[data-testid="stDataFrame"] .ag-horizontal-left-spacer {
    background-color: #FFFFFF !important;
}

/* Override Streamlit injected styles (dark theme) */
html[data-theme="dark"] [data-testid="stDataFrame"] * {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}
div:has(> .stSubheader:contains("Payment Details")) + div [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border-radius: 10px;
        padding: 10px;
    }

    div:has(> .stSubheader:contains("Payment Details")) + div [data-testid="stDataFrame"] * {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }

    div:has(> .stSubheader:contains("Payment Details")) + div [data-testid="stDataFrame"] thead {
        background-color: #E0E0E0 !important;
        color: black !important;
        font-weight: bold;
    }

    div:has(> .stSubheader:contains("Payment Details")) + div [data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #F5F5F5 !important;
    }

    div:has(> .stSubheader:contains("Payment Details")) + div [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #D3E3FC !important;
        color: black !important;
    }

    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'step_1_data' not in st.session_state:
    st.session_state.step_1_data = None
if 'step_2_data' not in st.session_state:
    st.session_state.step_2_data = None
if 'step_3_data' not in st.session_state:
    st.session_state.step_3_data = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None

# Functions for PDF processing (Step 1)
def convert_pdf_page_to_image(pdf_bytes, page_num):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def process_image_with_gemini(image):
    if not GOOGLE_API_KEY:
        st.error("Gemini API key not found. Please check your .env file.")
        return ""
        
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        image_bytes = output.getvalue()
    
    image_parts = [
        {
            "mime_type": "image/png",
            "data": base64.b64encode(image_bytes).decode('utf-8')
        }
    ]
    
    prompt = """
    Extract the text content from this image of a Housing Finance document.
    Focus on capturing all table data, especially the sections on:
    - Tranche disbursement details
    - Cumulative Disbursement amounts
    - Construction percentages
    - Collection/Promoters' contributions
    - Pre-Disbursement conditions
    - Covenants and their timelines
    
    Return all the text content from the image, preserving the structure and relationships.
    """
    
    try:
        response = model.generate_content(
            [prompt, image_parts[0]],
            generation_config={"temperature": 0.1}
        )
        return response.text
    except Exception as e:
        st.error(f"Error processing image with Gemini API: {e}")
        return ""
    
def extract_structured_data(full_text):
    if not GOOGLE_API_KEY:
        st.error("Gemini API key not found. Please check your .env file.")
        return ""
        
    model = genai.GenerativeModel("gemini-1.5-flash")
    
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
    - Conditions Precedent: These are the "Pre-Disbursement" conditions for all loans
    - Conditions Subsequent with Frequency: These are the "Covenants" with both the Covenant description and Timeline
    
    Return as valid JSON in this exact format:
    {{
      "table_data": [
        {{
          "Sr. No.": 1,
          "Tranche Amount (Rs Cr)": 12.00,
          "Cumulative Disbursement (Rs Cr)": 12.00,
          "Construction % (Europa, Mynsa & Capella)": "",
          "Incremental Collection/Promoters' Contribution (Rs Cr)": ""
        }},
        {{
          "Sr. No.": 2,
          "Tranche Amount (Rs Cr)": 5.00,
          "Cumulative Disbursement (Rs Cr)": 17.00,
          "Construction % (Europa, Mynsa & Capella)": "10.00%",
          "Incremental Collection/Promoters' Contribution (Rs Cr)": 5.00
        }},
        // more rows...
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
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1}
        )
        return response.text
    except Exception as e:
        st.error(f"Error extracting structured data: {e}")
        return ""

def create_excel(data):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            
            table_data = pd.DataFrame(data.get("table_data", []))
            
            conditions_precedent = data.get("conditions_precedent", [])
            conditions_subsequent = data.get("conditions_subsequent", [])
            
            conditions_precedent_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(conditions_precedent)])
            conditions_subsequent_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(conditions_subsequent)])
            
            if not table_data.empty:
                table_data["Conditions Precedent"] = conditions_precedent_text
                table_data["Conditions Subsequent with Frequency"] = conditions_subsequent_text
            
            table_data.to_excel(writer, sheet_name="Extracted Data", index=False)
            
            conditions_df = pd.DataFrame({
                "Conditions Precedent": pd.Series(conditions_precedent),
                "Conditions Subsequent with Frequency": pd.Series(conditions_subsequent)
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

# Functions for Excel processing (Step 2)
def clean_dataframe(df):
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = df.columns.astype(str).str.strip()
    return df

# Navigation functions
def go_to_step(step_number):
    st.session_state.step = step_number

# Main app logic
def main():
    if st.session_state.step == 1:
        step_1()
    elif st.session_state.step == 2:
        step_2()
    elif st.session_state.step == 3:
        step_3()
    elif st.session_state.step == 4:
        step_4()

def step_1():
    st.title('Step 1: Developer Dashboard Snapshot')
    st.write('Upload the Sanction Letter to extract relevant milestones and conditions.')

    uploaded_file = st.file_uploader("Upload sanction letter", type="pdf")

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        file_hash = get_file_hash(pdf_bytes)

        cached_data = load_from_cache(file_hash)
        if cached_data:
            st.success("Loaded from cache. No reprocessing needed")

            st.session_state["file_hash"] = file_hash
            st.session_state["full_text"] = cached_data["full_text"]
            st.session_state["json_data"] = cached_data["json_data"]
            st.session_state["parsed_data"] = cached_data["parsed_data"]
            st.session_state["excel_data"] = cached_data["excel_data"]
            st.session_state.step_1_data = cached_data["parsed_data"]
        else:
            with st.spinner('Processing PDF...'):
                reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                num_pages = len(reader.pages)
                full_text = ""

                progress_bar = st.progress(0)
                for i in range(num_pages):
                    progress_bar.progress((i + 1) / num_pages)
                    image = convert_pdf_page_to_image(pdf_bytes, i)
                    page_text = process_image_with_gemini(image)
                    full_text += f"\n\n--- PAGE {i+1} ---\n\n{page_text}"
                    time.sleep(1)

                json_data = extract_structured_data(full_text)

                try:
                    try:
                        data = json.loads(json_data)
                    except:
                        json_match = re.search(r'(\{.*\})', json_data, re.DOTALL)
                        if json_match:
                            clean_json = json_match.group(1)
                            data = json.loads(clean_json)
                        else:
                            st.error("Could not parse JSON data from response.")
                            st.text(json_data)
                            st.download_button(
                                label="Download raw extracted text",
                                data=full_text,
                                file_name="raw_extracted_text.txt",
                                mime="text/plain"
                            )
                            return

                    excel_data = create_excel(data)

                    save_to_cache(file_hash, {
                        "full_text": full_text,
                        "json_data": json_data,
                        "parsed_data": data,
                        "excel_data": excel_data
                    })

                    st.session_state["file_hash"] = file_hash
                    st.session_state["full_text"] = full_text
                    st.session_state["json_data"] = json_data
                    st.session_state["parsed_data"] = data
                    st.session_state["excel_data"] = excel_data
                    st.session_state.step_1_data = data

                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.text(json_data)
                    return

        
        if "excel_data" in st.session_state:
            st.success("PDF processed successfully!")
            st.download_button(
                label="Download Excel file",
                data=st.session_state["excel_data"],
                file_name="tata_finance_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.subheader("Preview of extracted data:")
            data = st.session_state["parsed_data"]

            if "table_data" in data:
                st.subheader("Table Data:")
                table_df = pd.DataFrame(data["table_data"])
                
                # Custom HTML table with inline styling
                styled_table_html = table_df.to_html(classes='custom-table', index=False)

                st.markdown("""
                    <style>
                    .custom-table {
                        border-collapse: collapse;
                        width: 100%;
                        font-family: Arial, sans-serif;
                    }
                    .custom-table th, .custom-table td {
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                        color: black;
                    }
                    .custom-table th {
                        background-color: #E0E0E0;
                        font-weight: bold;
                    }
                    .custom-table tr:nth-child(even) {
                        background-color: #F5F5F5;
                    }
                    .custom-table tr:hover {
                        background-color: #D3E3FC;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown(styled_table_html, unsafe_allow_html=True)


            if "conditions_precedent" in data:
                st.write("Conditions Precedent:")
                for i, item in enumerate(data["conditions_precedent"]):
                    st.write(f"{i+1}. {item}")

            if "conditions_subsequent" in data:
                st.write("Conditions Subsequent with Frequency:")
                for i, item in enumerate(data["conditions_subsequent"]):
                    st.write(f"{i+1}. {item}")

            st.button("Next", on_click=go_to_step, args=(2,))

def step_2():
    st.title('Step 2: Project Data Comparison')
    st.write('Upload Excel files to compare project data')
    
    # Upload both files
    file1 = st.file_uploader("Upload First Excel File (Earlier Data)", type=["xlsx"], key="file1")
    file2 = st.file_uploader("Upload Second Excel File (Latest Data)", type=["xlsx"], key="file2")
    
    if file1 and file2:
        xls1 = pd.ExcelFile(file1)
        xls2 = pd.ExcelFile(file2)
    
        sheets_to_display = ["Project Detail", "MIS", "COP-MOF"]
        tabs = st.tabs(sheets_to_display)
        
        comparison_results = {}
    
        for tab, sheet in zip(tabs, sheets_to_display):
            with tab:
                st.subheader(f"{sheet} Data")
    
                if sheet in xls2.sheet_names:
                    df2 = clean_dataframe(pd.read_excel(xls2, sheet_name=sheet))
    
                    if sheet == "MIS" and sheet in xls1.sheet_names:
                        df1 = clean_dataframe(pd.read_excel(xls1, sheet_name=sheet))
    
                        if "Sold/Unsold" not in df1.columns or "Sold/Unsold" not in df2.columns:
                            st.error(f"'Sold/Unsold' column missing in one or both '{sheet}' sheets")
                            st.dataframe(df2, use_container_width=True, height=600)
                            comparison_results[sheet] = {"error": "Sold/Unsold column missing", "data": df2}
                            continue
    
                        # Find ID column: prefer "Project ID", fallback to first column
                        possible_id_cols = ["Project ID", "Sl No.", "Sr. No.", "ID"]
                        id_column = next((col for col in possible_id_cols if col in df1.columns and col in df2.columns), df1.columns[0])
    
                        if id_column == df1.columns[0] and id_column not in possible_id_cols:
                            st.warning(f"Using '{id_column}' as identifier for comparison. Please ensure rows match correctly.")
    
                        st.write("**Status Changes Legend:**")
                        st.markdown("🟢 <span style='color:green'>Green</span>: Unsold → Sold &nbsp;&nbsp;&nbsp; 🔴 <span style='color:red'>Red</span>: Sold → Unsold", unsafe_allow_html=True)
    
                        comparison_df = df2.copy()
                        status_map = dict(zip(df1[id_column], df1["Sold/Unsold"].astype(str).str.strip()))
    
                        def highlight_rows(row):
                            current_id = row[id_column]
                            current_status = str(row["Sold/Unsold"]).strip().lower()
                            previous_status = status_map.get(current_id, "").strip().lower()

                            # Status change highlighting
                            if previous_status and current_status != previous_status:
                                if previous_status == "unsold" and current_status == "sold":
                                    return ['background-color: rgba(144, 238, 144, 0.6); color: #00500B'] * len(row)
                                elif previous_status == "sold" and current_status == "unsold":
                                    return ['background-color: rgba(255, 99, 71, 0.6); color: #5C0000'] * len(row)
                            
                            # Static status highlighting (lighter shades)
                            if current_status == "sold":
                                return ['background-color: rgba(220, 255, 220, 0.6); color: #333333'] * len(row)
                            elif current_status == "unsold":
                                return ['background-color: rgba(255, 230, 230, 0.6); color: #333333'] * len(row)

                            # Default: white bg, dark gray text (no black)
                            return ['background-color: #FFFFFF; color: #333333'] * len(row)
    
                        styled_df = comparison_df.style.apply(highlight_rows, axis=1)
                        st.write(styled_df)
                        
                        # Store the analysis results
                        comparison_results[sheet] = {
                            "df1": df1,
                            "df2": df2,
                            "status_changes": {
                                "unsold_to_sold": sum((status_map.get(id_val, "").strip().lower() == "unsold" and 
                                                   row["Sold/Unsold"].strip().lower() == "sold") 
                                                  for id_val, row in comparison_df.iterrows() if id_val in status_map),
                                "sold_to_unsold": sum((status_map.get(id_val, "").strip().lower() == "sold" and 
                                                   row["Sold/Unsold"].strip().lower() == "unsold") 
                                                  for id_val, row in comparison_df.iterrows() if id_val in status_map)
                            }
                        }
                    elif sheet == "COP-MOF" and sheet in xls2.sheet_names:
                        df1 = pd.read_excel(xls1, sheet_name=sheet, header=None)
                        df2 = pd.read_excel(xls2, sheet_name=sheet, header=None)
                        new_header = df2.iloc[2]
                        df2 = df2[3:]
                        df2.columns = new_header
                        df2.reset_index(drop=True, inplace=True)
                        df2.columns = df2.columns.astype(str).str.strip()

                        new_header = df1.iloc[2]
                        df1 = df1[3:]
                        df1.columns = new_header
                        df1.reset_index(drop=True, inplace=True)
                        df1.columns = df1.columns.astype(str).str.strip()

                        
                        
                        styled_df = df2.style.applymap(lambda _: 'background-color: #FFFFFF; color: #333333')
                        st.write(styled_df)
                        comparison_results[sheet] = {
                            "df1": df1,
                            "df2": df2
                        }
    
                    else:
                        styled_df = df2.style.applymap(lambda _: 'background-color: #FFFFFF; color: #333333')
                        st.write(styled_df)
                        comparison_results[sheet] = {"df2": df2}
                else:
                    st.warning(f"'{sheet}' not found in the second file.")
                    comparison_results[sheet] = {"warning": f"'{sheet}' not found in the second file."}
        
        # Store the results in session state
        st.session_state.step_2_data = comparison_results
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Back", on_click=go_to_step, args=(1,))
        with col2:
            st.button("Next", on_click=go_to_step, args=(3,))
    else:
        st.info("Please upload both Excel files to compare and display data.")
        st.button("Back", on_click=go_to_step, args=(1,))

def step_3():
    st.title('Step 3: Disbursement Request Form')
    if st.session_state.step_1_data:
        data = st.session_state.step_1_data

        if "table_data" in data:
            st.subheader("Table Data:")
            table_df = pd.DataFrame(data["table_data"])
            
            # Custom HTML table with inline styling
            styled_table_html = table_df.to_html(classes='custom-table', index=False)

            st.markdown("""
                <style>
                .custom-table {
                    border-collapse: collapse;
                    width: 100%;
                    font-family: Arial, sans-serif;
                }
                .custom-table th, .custom-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                    color: black;
                }
                .custom-table th {
                    background-color: #E0E0E0;
                    font-weight: bold;
                }
                .custom-table tr:nth-child(even) {
                    background-color: #F5F5F5;
                }
                .custom-table tr:hover {
                    background-color: #D3E3FC;
                }
                </style>
            """, unsafe_allow_html=True)

            st.markdown(styled_table_html, unsafe_allow_html=True)


            # Dropdown for milestone selection
            milestone_options = [f"Milestone {i+1}" for i in range(len(table_df))]
            selected_milestone = st.selectbox("Select a Milestone to proceed", ["-- Select --"] + milestone_options)

            if selected_milestone != "-- Select --":
                # st.subheader("Project Details")
                # Try fetching project details from step_2_data
                # step2 = st.session_state.get("step_2_data", {})

                st.subheader("Project Informations")
                st.markdown(f"**Project Name:** ABC Developer")
                st.markdown(f"**Project Location:** Qube Software Park Bellandur")
                st.markdown(f"**Project Description:** Residential Project")
                st.markdown(f"**Requesting Party Details:** Rudra Housing Private Ltd")

                if "conditions_precedent" in data:
                    st.subheader("Conditions Precedent:")
                    selected_cp = []
                    for i, item in enumerate(data["conditions_precedent"]):
                        col1, col2 = st.columns([0.9, 0.1])
                        with col1:
                            st.markdown(f"**{i+1}.** {item}")
                        with col2:
                            checked = st.checkbox("", key=f"cp_{i}", value=False)
                        if checked:
                            selected_cp.append(item)
                    st.session_state["selected_conditions_precedent"] = selected_cp

                if "conditions_subsequent" in data:
                    st.subheader("Conditions Subsequent with Frequency:")
                    selected_cs = []
                    for i, item in enumerate(data["conditions_subsequent"]):
                        col1, col2 = st.columns([0.9, 0.1])
                        with col1:
                            st.markdown(f"**{i+1}.** {item}")
                        with col2:
                            checked = st.checkbox("", key=f"cs_{i}", value=False)
                        if checked:
                            selected_cs.append(item)
                    st.session_state["selected_conditions_subsequent"] = selected_cs


                st.subheader("Loan Informations")
                st.markdown(f"**Loan Number:** ABC11000334")
                st.markdown(f"**Borrower Name:** Rudra Housing Private Ltd")
                st.markdown(f"**Loan Amount:** 30Cr.")

                # ========== SALES INFORMATION ==========
                if "step_2_data" in st.session_state:
                    step2_data = st.session_state["step_2_data"]

                    # Ensure the required sheet and columns exist
                    sales_df = None
                    for sheet_name, sheet_data in step2_data.items():
                        if isinstance(sheet_data, dict) and "df1" in sheet_data:
                            df = sheet_data["df1"]
                            required_cols = [
                                "Flat no", "Tower No"
                            ]
                            if all(col in df.columns for col in required_cols):
                                sales_df = df
                                break

                    if sales_df is not None:
                        st.markdown("<h3 style='color:#003366;'>Sales Information</h3>", unsafe_allow_html=True)

                        selected_flats_by_tower = {}
                        unique_towers = sales_df["Tower No"].dropna().unique()

                        total_sold = 0
                        total_selected = 0

                        for tower in unique_towers:
                            # Styled header for each tower (outside expander)
                            st.markdown(f"<h4 style='color:#2C3E50; margin-bottom: 0;'>Tower: {tower}</h4>", unsafe_allow_html=True)

                            # Applying custom styling to expander header
                            with st.expander(f"Select Flats in Tower {tower}", expanded=True):
                                st.markdown("""
                                    <style>
                                        .streamlit-expanderHeader {
                                            color: #2C3E50;  /* Color for expander header text */
                                            font-weight: bold;  /* Bold text */
                                            font-size: 18px;  /* Adjust font size */
                                        }
                                        .streamlit-expander .streamlit-expanderContent {
                                            color: #333333;  /* Color for content inside expander */
                                        }
                                    </style>
                                """, unsafe_allow_html=True)
                                
                                flats_in_tower = sales_df[sales_df["Tower No"] == tower]["Flat no"].dropna().unique()

                                selected_flats = st.multiselect(
                                    f"Select Flats in Tower {tower}",
                                    flats_in_tower,
                                    key=f"flats_{tower}"
                                )

                                
                                selected_flats_by_tower[tower] = selected_flats
                                
                                # Filter for sold flats in selected
                                selected_df = sales_df[
                                    (sales_df["Tower No"] == tower) &
                                    (sales_df["Flat no"].isin(selected_flats))
                                ]
                                
                                sold_count = selected_df[selected_df["Sold/Unsold"].str.lower() == "sold"].shape[0]
                                selected_count = selected_df.shape[0]

                                total_sold += sold_count
                                total_selected += selected_count

                                st.markdown(f"<div style='margin-top:10px; font-weight:bold;'>Flats Selected: <span style='color:#007ACC'>{selected_count}</span></div>", unsafe_allow_html=True)

                        # Global Summary
                        st.markdown("<hr style='border-top: 2px solid #bbb;'/>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color:#1A5276;'>Total Flats Selected Across All Towers: <span style='color:#2E86C1'>{total_selected}</span></h4>", unsafe_allow_html=True)

                        # Save to session state
                        st.session_state.step_3_data = {
                            "selected_flats_by_tower": selected_flats_by_tower,
                            "total_selected": total_selected,
                            "total_sold": total_sold,
                        }

                    else:
                        st.warning("Sales data not available or missing required columns.")


                # Dummy Payment Details
                st.subheader("Payment Details")
                payment_df = pd.DataFrame([
                    {"Name of the contractor": "Perera", "Amount": "2.3 Cr", "Supporting Document": "25 crs", "Bank Details": "Abc"},
                    {"Name of the contractor": "Lorance", "Amount": "1.7 Cr", "Supporting Document": "25 crs", "Bank Details": "Xzy"},
                    {"Name of the contractor": "Rao", "Amount": "2.5 Cr", "Supporting Document": "25 crs", "Bank Details": "Qwrty"},
                ])
                st.dataframe(payment_df, use_container_width=True)


                # Authorized Signatory Info
                st.markdown("**Authorized Signatory (Print Name)** : John Smith")
                st.markdown("**Signature** : _John Smith_")
                st.markdown("**Date** : 12-Dec-2023")

                st.markdown("---")

                # Lender Info
                st.subheader("Lender Information")
                st.markdown("""
                **Lender Contact Person** : Peter Shaw  
                **Address** : Premier Villa, HSR, Bangalore  
                **Phone number** : 91234569878  
                **Email ID** : petershaw@123.com
                """)

            else:
                st.info("Please select a milestone to continue.")

    else:
        st.warning("No PDF data extracted in Step 1.")


    # Navigation Buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=go_to_step, args=(2,))
    with col2:
        st.button("Next", on_click=go_to_step, args=(4,))

def step_4():
    st.title("Step 4: Final Summary")

    # Step 1 Summary
    step1 = st.session_state.get("step_1_data")
    if step1 and "table_data" in step1:
        st.header("Step 1: PDF Extraction")
        st.subheader("Milestone Table")
        table_df = pd.DataFrame(step1["table_data"])

        # Convert DataFrame to styled HTML table
        styled_table_html = table_df.to_html(classes='custom-table', index=False)

        # Inject custom CSS for styling
        st.markdown("""
            <style>
            .custom-table {
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
                border-radius: 8px;
                overflow: hidden;
            }
            .custom-table th, .custom-table td {
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
                color: #333333;
                background-color: #FFFFFF;
            }
            .custom-table th {
                background-color: #E0E0E0;
                font-weight: bold;
            }
            .custom-table tr:nth-child(even) {
                background-color: #F5F5F5;
            }
            .custom-table tr:hover {
                background-color: #D3E3FC;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display the styled table
        st.markdown(styled_table_html, unsafe_allow_html=True)

        if "conditions_precedent" in step1:
            st.subheader("Selected Conditions Precedent")
            for i, item in enumerate(st.session_state.get("selected_conditions_precedent", [])):
                st.markdown(f"**{i+1}.** {item}")
        if "conditions_subsequent" in step1:
            st.subheader("Selected Conditions Subsequent")
            for i, item in enumerate(st.session_state.get("selected_conditions_subsequent", [])):
                st.markdown(f"**{i+1}.** {item}")
    else:
        st.warning("Step 1 data missing.")

    # Step 2 Summary
    step2 = st.session_state.get("step_2_data")
    if step2:
        st.header("Step 2: Excel Comparison")
        for sheet, data in step2.items():
            if isinstance(data, dict) and sheet == "COP-MOF":
                def render_styled_table(df, title):
                    st.subheader(title)
                    styled_html = df.to_html(classes='custom-table', index=False)
                    st.markdown(styled_html, unsafe_allow_html=True)

                # Add styling once (only needed once in your script)
                st.markdown("""
                    <style>
                    .custom-table {
                        border-collapse: collapse;
                        width: 100%;
                        font-family: Arial, sans-serif;
                        border-radius: 8px;
                        overflow: hidden;
                    }
                    .custom-table th, .custom-table td {
                        border: 1px solid #ddd;
                        padding: 10px;
                        text-align: left;
                        color: #333333;
                        background-color: #FFFFFF;
                    }
                    .custom-table th {
                        background-color: #E0E0E0;
                        font-weight: bold;
                    }
                    .custom-table tr:nth-child(even) {
                        background-color: #F5F5F5;
                    }
                    .custom-table tr:hover {
                        background-color: #D3E3FC;
                    }
                    </style>
                """, unsafe_allow_html=True)

                # Render both tables
                render_styled_table(data["df2"], f"{sheet} - COP-MOF Current")
                render_styled_table(data["df1"], f"{sheet} - COP-MOF Previous")

                df1 = data['df1']
                df2 = data['df2']
                bank_funds = df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "bank funds", "Incurred"].values
                mean_of_finance = df2.loc[df2["PARTICULARS"].str.strip() == "MEANS OF FINANCE", "Incurred"].values
                total_a = df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "total (a)", "Incurred"].values
                cust_adv_2 = df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "customer advance", "Incurred"].values
                cust_adv_1 = df1.loc[df1["PARTICULARS"].str.strip().str.lower() == "customer advance", "Incurred"].values
                promoter_funds_2 = df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "promoter funds", "Incurred"].values
                promoter_funds_1 = df1.loc[df1["PARTICULARS"].str.strip().str.lower() == "promoter funds", "Incurred"].values
                bank_funds_2 = df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "bank funds", "Incurred"].values
                bank_funds_1 = df1.loc[df1["PARTICULARS"].str.strip().str.lower() == "bank funds", "Incurred"].values

                if bank_funds.size > 0:
                    value = float(bank_funds[0]) / 100.0
                    st.markdown(f"""
                        <div style="background-color: #F0F8FF; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; color: #333;">
                            <b>Obligation:</b> ₹{value:.2f} Cr
                        </div>
                    """, unsafe_allow_html=True)

                if mean_of_finance.size > 0 and total_a.size > 0:
                    value2 = float(mean_of_finance[0])
                    value1 = float(total_a[0])
                    value = value2 - value1
                    st.markdown(f"""
                        <div style="background-color: #FFF8E1; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; color: #333;">
                            <b>Balance:</b> ₹{value:.2f} Cr
                        </div>
                    """, unsafe_allow_html=True)

                if cust_adv_2.size > 0 and cust_adv_1.size > 0:
                    value2 = float(cust_adv_2[0])
                    value1 = float(cust_adv_1[0])
                    value = value2 - value1
                    st.markdown(f"""
                        <div style="background-color: #E8F5E9; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; color: #333;">
                            <b>Change in Customer Advance:</b> ₹{value:.2f} Cr
                        </div>
                    """, unsafe_allow_html=True)

                if promoter_funds_2.size > 0 and promoter_funds_1.size > 0:
                    value2 = float(promoter_funds_2[0])
                    value1 = float(promoter_funds_1[0])
                    value = value2 - value1
                    st.markdown(f"""
                        <div style="background-color: #FBE9E7; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; color: #333;">
                            <b>Change in Promoter Funds:</b> ₹{value:.2f} Cr
                        </div>
                    """, unsafe_allow_html=True)

                if bank_funds_2.size > 0 and bank_funds_1.size > 0:
                    value2 = float(bank_funds_2[0])
                    value1 = float(bank_funds_1[0])
                    value = value2 - value1
                    st.markdown(f"""
                        <div style="background-color: #E3F2FD; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; color: #333;">
                            <b>Change in Bank Funds:</b> ₹{value:.2f} Cr
                        </div>
                    """, unsafe_allow_html=True)

    else:
        st.warning("Step 2 data missing.")

    # Step 3 Summary
    sales = st.session_state.get("step_3_data")
    if sales:
        st.header("Step 3: Sales Summary")
        for tower, flats in sales["selected_flats_by_tower"].items():
            st.markdown(f"**Tower {tower} - Selected Flats:** {', '.join(flats)}")
        st.markdown(f"**Total Flats Selected:** {sales.get('total_selected', 0)}")
        st.markdown(f"**Total Flats Sold:** {sales.get('total_sold', 0)}")
    else:
        st.warning("Step 3 sales info missing.")

    st.button("Back", on_click=go_to_step, args=(3,))


# Run the app
if __name__ == "__main__":
    main()