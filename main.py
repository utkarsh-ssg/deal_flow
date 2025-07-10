import streamlit as st
import pandas as pd
import PyPDF2
import io
import tempfile
import time
from dotenv import load_dotenv
import json
import re
from streamlit.components.v1 import html
import requests
from utils.utils import *

load_dotenv()


st.set_page_config(page_title="Dashboard", layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


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


def step_1():
    st.title('Dashboard')
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
                    page_text = process_image(image)
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
                st.subheader("Sanction Letter Data:")
                table_df = pd.DataFrame(data["table_data"])

                
                excluded_cols = {"pre_disbursement_conditions", "conditions_precedent", "conditions_subsequent"}
                display_df = table_df.drop(columns=[col for col in table_df.columns if col in excluded_cols], errors='ignore')
                st.dataframe(display_df)

                
                milestone_options = []
                for i, row in enumerate(data.get("table_data", [])):
                    tranche_label = row.get("Tranche", f"Tranche {i+1}")
                    milestone_options.append(tranche_label)

                
                selected_option = st.selectbox("Select Tranche to View Conditions", ["Select a tranche"] + milestone_options)

                
                if selected_option != "Select a tranche":
                    tranche_index = milestone_options.index(selected_option)
                    selected_tranche = data["table_data"][tranche_index]

                    if "pre_disbursement_conditions" in selected_tranche:
                        st.subheader("Pre-Disbursement Conditions:")
                        for i, item in enumerate(selected_tranche["pre_disbursement_conditions"]):
                            st.write(f"{i+1}. {item}")

                    if "conditions_precedent" in selected_tranche:
                        st.subheader("Conditions Precedent:")
                        for i, item in enumerate(selected_tranche["conditions_precedent"]):
                            st.write(f"{i+1}. {item}")

                    if "conditions_subsequent" in selected_tranche:
                        st.subheader("Conditions Subsequent:")
                        for i, item in enumerate(selected_tranche["conditions_subsequent"]):
                            st.write(f"{i+1}. {item}")

            st.button("Next", on_click=go_to_step, args=(2,))

def step_2():
    st.title('MIS Data')
    st.write('Upload MIS files')
    
    
    file1 = st.file_uploader("Upload MIS", type=["xlsx"], key="file1")
    file2 = st.file_uploader("Upload Current MIS", type=["xlsx"], key="file2")
    
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
    
                        possible_id_cols = ["Project ID", "Sl No.", "Sr. No.", "ID"]
                        id_column = next((col for col in possible_id_cols if col in df1.columns and col in df2.columns), df1.columns[0])
    
                        if id_column == df1.columns[0] and id_column not in possible_id_cols:
                            st.warning(f"Using '{id_column}' as identifier for comparison. Please ensure rows match correctly.")
    
                        comparison_df = df2.copy()
                        status_map = dict(zip(df1[id_column], df1["Sold/Unsold"].astype(str).str.strip()))
    
                        def highlight_rows(row):
                            current_id = row[id_column]
                            current_status = str(row["Sold/Unsold"]).strip().lower()
                            previous_status = status_map.get(current_id, "").strip().lower()

                            if previous_status and current_status != previous_status:
                                if previous_status == "unsold" and current_status == "sold":
                                    return ['background-color: rgba(144, 238, 144, 0.6); color: #00500B'] * len(row)
                                elif previous_status == "sold" and current_status == "unsold":
                                    return ['background-color: rgba(255, 99, 71, 0.6); color: #5C0000'] * len(row)
                            
                            
                            if current_status == "sold":
                                return ['background-color: rgba(220, 255, 220, 0.6); color: #333333'] * len(row)
                            elif current_status == "unsold":
                                return ['background-color: rgba(255, 230, 230, 0.6); color: #333333'] * len(row)

                            return ['background-color: #FFFFFF; color: #333333'] * len(row)
    
                        styled_df = comparison_df.style.apply(highlight_rows, axis=1)
                        st.write(styled_df)
                        
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
                        
                        
                        delta_df_raw = []
                        delta_styles = []

                        df1_indexed = df1.set_index(id_column).astype(str).fillna("")
                        df2_indexed = df2.set_index(id_column).astype(str).fillna("")

                        common_ids = df1_indexed.index.intersection(df2_indexed.index)

                        for idx in common_ids:
                            row_old = df1_indexed.loc[idx]
                            row_new = df2_indexed.loc[idx]

                            
                            diff_mask = row_old != row_new
                            if diff_mask.any():
                                delta_row = row_new.copy()
                                delta_row.name = idx
                                delta_df_raw.append(delta_row)
                                
                                
                                row_style = ['']
                                row_style.extend([
                                    'background-color: rgba(255, 230, 150, 0.8); color: black;' if diff else ''
                                    for diff in diff_mask
                                ])
                                delta_styles.append(row_style)

                        if delta_df_raw:
                            delta_df = pd.DataFrame(delta_df_raw)
                            delta_df.insert(0, id_column, delta_df.index)
                            delta_df.reset_index(drop=True, inplace=True)

                            st.markdown("### Delta Table")
                            
                            
                            def apply_delta_styles(row):
                                row_idx = row.name
                                if row_idx < len(delta_styles):
                                    return delta_styles[row_idx]
                                else:
                                    return [''] * len(row)
                            
                            styled_delta_df = delta_df.style.apply(apply_delta_styles, axis=1)
                            st.write(styled_delta_df)
                            comparison_results[sheet]["delta_table"] = delta_df
                        else:
                            st.markdown("### 🔁 Delta Table")
                            st.info("No changes found apart from 'Sold/Unsold' status.")
                 
                    elif sheet == "COP-MOF" and sheet in xls2.sheet_names:
                        df1 = pd.read_excel(xls1, sheet_name=sheet, header=None)
                        df2 = pd.read_excel(xls2, sheet_name=sheet, header=None)

                        
                        header1 = df1.iloc[2]
                        df1 = df1[3:]
                        df1.columns = header1
                        df1.reset_index(drop=True, inplace=True)
                        df1.columns = df1.columns.astype(str).str.strip()

                        header2 = df2.iloc[2]
                        df2 = df2[3:]
                        df2.columns = header2
                        df2.reset_index(drop=True, inplace=True)
                        df2.columns = df2.columns.astype(str).str.strip()

                        
                        min_rows = min(len(df1), len(df2))
                        min_cols = min(len(df1.columns), len(df2.columns))

                        df1_aligned = df1.iloc[:min_rows, :min_cols].astype(str).fillna("").replace("nan", "")
                        df2_aligned = df2.iloc[:min_rows, :min_cols].astype(str).fillna("").replace("nan", "")
                        df2_aligned.columns = df2_aligned.columns.astype(str).str.strip()

                        
                        styles = []
                        for i in range(min_rows):
                            row_styles = []
                            for j in range(min_cols):
                                val1 = df1_aligned.iat[i, j]
                                val2 = df2_aligned.iat[i, j]
                                if val1 != val2:
                                    row_styles.append("background-color: rgba(255, 230, 150, 0.8); color: black;")
                                else:
                                    row_styles.append("")
                            styles.append(row_styles)

                        styled_df = df2_aligned.style.apply(lambda row: styles[row.name], axis=1)

                        st.markdown("### Delta COP-MOF Table")
                        st.write(styled_df)

                        comparison_results[sheet] = {
                            "df1": df1,
                            "df2": df2,
                            "diffs": df2_aligned
                        }


                    else:
                        styled_df = df2.style.applymap(lambda _: 'background-color: #FFFFFF; color: #333333')
                        st.write(styled_df)
                        comparison_results[sheet] = {"df2": df2}
                else:
                    st.warning(f"'{sheet}' not found in the second file.")
                    comparison_results[sheet] = {"warning": f"'{sheet}' not found in the second file."}
        
        
        st.session_state.step_2_data = comparison_results
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Back", on_click=go_to_step, args=(1,))
        with col2:
            st.button("Next", on_click=go_to_step, args=(3,))
    else:
        st.info("Please upload both MIS files.")
        st.button("Back", on_click=go_to_step, args=(1,))


def step_3():
    st.title('Disbursement Request Form')
    if st.session_state.step_1_data:
        data = st.session_state.step_1_data

        if "table_data" in data:
            table_df = pd.DataFrame(data["table_data"])
            milestone_options = [f"Milestone {i+1}" for i in range(len(table_df))]
            st.write(f"Select a Milestone to proceed")
            selected_milestone = st.selectbox("", ["-- Select --"] + milestone_options)

            st.subheader("Table Data:")

            selected_index = milestone_options.index(selected_milestone) if selected_milestone != "-- Select --" else None

            styled_rows = []
            headers = "".join([f"<th>{col}</th>" for col in table_df.columns])
            styled_rows.append(f"<tr>{headers}</tr>")

            for i, row in table_df.iterrows():
                row_style = "background-color: #D3E3FC;" if selected_index == i else ("background-color: #F5F5F5;" if i % 2 == 1 else "")
                row_html = "".join([f"<td>{cell}</td>" for cell in row])
                styled_rows.append(f"<tr style='{row_style}'>{row_html}</tr>")

            styled_table_html = f"""
            <style>
                .custom-table {{
                    border-collapse: collapse;
                    width: 100%;
                    font-family: Arial, sans-serif;
                }}
                .custom-table th, .custom-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                    color: black;
                }}
                .custom-table th {{
                    background-color: #E0E0E0;
                    font-weight: bold;
                }}
            </style>
            <table class="custom-table">
                {''.join(styled_rows)}
            </table>
            """

            st.markdown(styled_table_html, unsafe_allow_html=True)



            

            if selected_milestone != "-- Select --":

                st.subheader("Project Informations")
                st.markdown(f"**Project Name:** ABC Developer")
                st.markdown(f"**Project Location:** Qube Software Park Bellandur")
                st.markdown(f"**Project Description:** Residential Project")
                st.markdown(f"**Requesting Party Details:** Rudra Housing Private Ltd")

                if selected_milestone == milestone_options[0]:
                     if "pre_disbursement_conditions" in data:
                            st.subheader("Pre-Disbursement Conditions:")
                            selected_pdc = []
                            for i, item in enumerate(data["pre_disbursement_conditions"]):
                                col1, col2 = st.columns([0.9, 0.1])
                                with col1:
                                    st.markdown(f"**{i+1}.** {item}")
                                with col2:
                                    checked = st.checkbox("", key=f"pdc_{i}", value=False)
                                if checked:
                                    selected_pdc.append(item)
                            st.session_state["selected_pre_disbursement_conditions"] = selected_pdc
                            
                else:

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
                        st.subheader("Conditions Subsequent:")
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

                if "step_2_data" in st.session_state:
                    step2_data = st.session_state["step_2_data"]

                    sales_df = None
                    for sheet_name, sheet_data in step2_data.items():
                        if isinstance(sheet_data, dict) and "df2" in sheet_data:
                            df = sheet_data["df2"]
                            required_cols = [
                                "Flat no", "Tower No", "Sold/Unsold"
                            ]
                            if all(col in df.columns for col in required_cols):
                                sales_df = df
                                break

                    if sales_df is not None:
                        st.markdown("<h3 style='color:#003366;'>Sales Information</h3>", unsafe_allow_html=True)

                        recently_unsold_flats_by_tower = {}
                        recently_sold_flats_by_tower = {}
                        unique_towers = sales_df["Tower No"].dropna().unique()

                        total_recently_sold = 0
                        total_recently_unsold = 0

                        for tower in unique_towers:
                            st.markdown(f"<h4 style='color:#2C3E50; margin-bottom: 0;'>Tower: {tower}</h4>", unsafe_allow_html=True)

                            with st.expander("", expanded=True):
                                st.markdown("""
                                    <style>
                                        .streamlit-expanderHeader {
                                            color: #2C3E50;
                                            font-weight: bold;
                                            font-size: 18px;
                                        }
                                        .streamlit-expander .streamlit-expanderContent {
                                            color: #333333;
                                        }
                                    </style>
                                """, unsafe_allow_html=True)

                                recently_unsold_flats = sales_df[
                                    (sales_df["Tower No"] == tower) &
                                    (sales_df["Sold/Unsold"].str.lower() == "sold")
                                ]["Flat no"].dropna().unique()
                                st.write(f"Select Flats whose Sales got cancelled post latest MIS in Tower {tower}")
                                selected_recently_unsold_flats = st.multiselect(
                                    "",
                                    recently_unsold_flats,
                                    key=f"recently_unsold_flats_{tower}"
                                )

                                recently_sold_flats = sales_df[
                                    (sales_df["Tower No"] == tower) &
                                    (sales_df["Sold/Unsold"].str.lower() == "unsold")
                                ]["Flat no"].dropna().unique()
                                st.write(f"Select Unsold Flats which were Sold post latest MIS in Tower {tower}")

                                selected_recently_sold_flats = st.multiselect(
                                    "",
                                    recently_sold_flats,
                                    key=f"unsold_flats_{tower}"
                                )

                                combined_selected = list(set(selected_recently_unsold_flats) | set(selected_recently_sold_flats))
                                recently_unsold_flats_by_tower[tower] = selected_recently_unsold_flats
                                recently_sold_flats_by_tower[tower] = selected_recently_sold_flats

                                selected_df = sales_df[
                                    (sales_df["Tower No"] == tower) &
                                    (sales_df["Flat no"].isin(combined_selected))
                                ]

                                recently_unsold_count = selected_df[selected_df["Sold/Unsold"].str.lower() == "sold"].shape[0]
                                recently_sold_count = selected_df[selected_df["Sold/Unsold"].str.lower() == "unsold"].shape[0]

                                total_recently_unsold += recently_unsold_count
                                total_recently_sold += recently_sold_count

                                st.markdown(f"<div style='margin-top:10px; font-weight:bold;'>Flats whose Sales got cancelled post current MIS: <span style='color:#007ACC'>{recently_unsold_count}</span></div>", unsafe_allow_html=True)
                                st.markdown(f"<div style='font-weight:bold;'>Flats solds post current MIS: <span style='color:#28B463'>{recently_sold_count}</span></div>", unsafe_allow_html=True)

                        st.markdown("<hr style='border-top: 2px solid #bbb;'/>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color:#1A5276;'>Total Sold Flats whose sales got cancelled post latest MIS: <span style='color:#2E86C1'>{total_recently_unsold}</span></h4>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color:#145A32;'>Total Unsold Flats which went Sold post latest MIS: <span style='color:#28B463'>{total_recently_sold}</span></h4>", unsafe_allow_html=True)

                        st.session_state.step_3_data = {
                            "recently_unsold_flats_by_tower": recently_unsold_flats_by_tower,
                            "recently_sold_flats_by_tower": recently_sold_flats_by_tower,
                            "total_recently_sold": total_recently_unsold,
                            "total_recently_sold_selected": total_recently_sold,
                        }


                    else:
                        st.warning("Sales data not available or missing required columns.")


                st.subheader("Payment Details")
                payment_df = pd.DataFrame([
                    {"Name of the contractor": "Perera", "Amount": "2.3 Cr", "Supporting Document": "25 crs", "Bank Details": "Abc"},
                    {"Name of the contractor": "Lorance", "Amount": "1.7 Cr", "Supporting Document": "25 crs", "Bank Details": "Xzy"},
                    {"Name of the contractor": "Rao", "Amount": "2.5 Cr", "Supporting Document": "25 crs", "Bank Details": "Qwrty"},
                ])
                st.dataframe(payment_df, use_container_width=True)


                st.markdown("**Authorized Signatory (Print Name)** : John Smith")
                st.markdown("**Signature** : _John Smith_")
                st.markdown("**Date** : 12-Dec-2023")

                st.markdown("---")

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


    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=go_to_step, args=(2,))
    with col2:
        st.button("Next", on_click=go_to_step, args=(4,))

def step_4():
    st.title("Bank Statements Summary")

    st.write('Upload the Bank Statements')

    UPLOAD_URL = "https://cartuat.com/api/upload"
    DOWNLOAD_URL = "https://cartuat.com/api/downloadFile"
    AUTH_TOKEN = "API://QFEreQJLUvIWHKSLliicNPOC/MYh9B7dCo95Chz2rT2Sgf9ihi53EpD8LigFS/tw"

    uploaded_file = st.file_uploader("Upload Collection Bank statement", type="pdf")
    uploaded_file2 = st.file_uploader("Upload Corporate Bank statement", type="pdf")
    uploaded_file3 = st.file_uploader("Upload Project statement", type="pdf")
    creditTransactionAmount = 0
    if uploaded_file is not None:
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

                    if download_response.status_code == 200:
                        result = download_response.json()
                        
                        if "analysisData" in result['data'][0]:
                            card_html = """
                            <style>
                            .card-container {
                                display: flex;
                                flex-wrap: wrap;
                                justify-content: space-between;
                                gap: 20px;
                            }
                            .card {
                                flex: 0 0 32%;
                                background-color: #ffffff;
                                padding: 15px;
                                border-radius: 12px;
                                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                                box-sizing: border-box;
                                color: #333;
                                font-family: Arial, sans-serif;
                            }
                            .card b {
                                color: #000;
                                font-size: 16px;
                            }
                            @media (max-width: 768px) {
                                .card {
                                    flex: 0 0 100%;
                                }
                            }
                            </style>
                            <div class="card-container">
                            """

                            
                            cards = []
                            analysis_data = result['data'][0]['analysisData']
                            c = 0
                            for item in analysis_data:
                                month = item.get("month", "")
                                credit_amount = item.get("creditTransactionsAmount", 0.0)

                                if c == 0:
                                    creditTransactionAmount = credit_amount
                                credit_count = item.get("noOfCreditTransactions", 0)
                                debit_amount = item.get("debitTransactionsAmount", 0)
                                debit_count = item.get("noOfDebitTransactions", 0)
                                net_balance = item.get("customAverageBalance", 0)
                                emi_amount = item.get("totalEMIAmount", 0)
                                

                                cards.append(f"<div class='card' style='background-color:#F0F8FF'><b>Credit in {month}:</b><br>₹{credit_amount:.2f}</div>")
                                
                                cards.append(f"<div class='card' style='background-color:#FFF8E1'><b>No. of Credit Transaction in {month}:</b><br>{int(credit_count)}</div>")
                                
                                cards.append(f"<div class='card' style='background-color:#E8F5E9'><b>Debit Transaction in {month}:</b><br>₹{debit_amount:.2f}</div>")

                                cards.append(f"<div class='card' style='background-color:#FBE9E7'><b>No. of Debit Transaction in {month}:</b><br>{int(debit_count)}</div>")

                                cards.append(f"<div class='card' style='background-color:#E3F2FD'><b>Net Balance in {month}:</b><br>₹{net_balance:.2f}</div>")
                                
                                cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>Total EMI amount in {month}:</b><br>₹{emi_amount:.2f}</div>")


                            card_html += "\n".join(cards) + "</div>"

                            html(card_html, height=400)


                    else:
                        st.error(f"Download failed. Status code: {download_response.status_code}")
                        st.text(download_response.text)

                else:
                    st.error("Document ID not found in upload response.")

            except Exception as e:
                st.error("Failed to parse upload response.")
                st.text(str(e))

            step2 = st.session_state.get("step_2_data")
            if step2:
                st.header("COP-MOF Data")
                for sheet, data in step2.items():
                    if isinstance(data, dict) and sheet == "COP-MOF":
                        def render_styled_table(df, title):
                            st.subheader(title)
                            styled_html = df.to_html(classes='custom-table', index=False)
                            st.markdown(styled_html, unsafe_allow_html=True)

                        
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
                        total_a_2 = df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "total (a)", "Incurred"].values
                        total_a_1 = df1.loc[df1["PARTICULARS"].str.strip().str.lower() == "total (a)", "Incurred"].values
                        
                        

                        if uploaded_file and cust_adv_2.size > 0:
                            cust_adv_incurred = float(cust_adv_2[0])
                            creditTransactionAmount = creditTransactionAmount/100000000
                            if creditTransactionAmount < cust_adv_incurred:
                                st.markdown(f"""
                                <div style='padding:10px; background-color:#FFCDD2; border-left:5px solid #C62828; border-radius:6px;'>
                                    <b>Red Flag:</b><br>
                                    Credit Transaction Amount (₹{creditTransactionAmount:.2f} Cr) is less than Customer Advance Incurred (₹{cust_adv_incurred:.2f} Cr)
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style='padding:10px; background-color:#C8E6C9; border-left:5px solid #2E7D32; border-radius:6px;'>
                                    <b>Green Flag:</b><br>
                                    Credit Transaction Amount (₹{creditTransactionAmount:.2f} Cr) covers Customer Advance Incurred (₹{cust_adv_incurred:.2f} Cr)
                                </div>
                                """, unsafe_allow_html=True)
                        
                        render_styled_table(data["df2"], f"COP-MOF Current")

                        card_html = """
                        <style>
                        .card-container {
                            display: flex;
                            flex-wrap: wrap;
                            justify-content: space-between;
                            gap: 20px;
                        }
                        .card {
                            flex: 0 0 32%;
                            background-color: #ffffff;
                            padding: 15px;
                            border-radius: 12px;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                            box-sizing: border-box;
                            color: #333;
                            font-family: Arial, sans-serif;
                        }
                        .card b {
                            color: #000;
                            font-size: 16px;
                        }
                        @media (max-width: 768px) {
                            .card {
                                flex: 0 0 100%;
                            }
                        }
                        </style>
                        <div class="card-container">
                        """

                        
                        cards = []

                        
                        if bank_funds.size > 0:
                            value = float(bank_funds[0]) / 100.0
                            cards.append(f"<div class='card' style='background-color:#F0F8FF'><b>Obligation:</b><br>₹{value:.2f} Cr</div>")

                        
                        if mean_of_finance.size > 0 and total_a.size > 0:
                            value = float(mean_of_finance[0]) - float(total_a[0])
                            cards.append(f"<div class='card' style='background-color:#FFF8E1'><b>Balance:</b><br>₹{value:.2f} Cr</div>")

                        
                        if cust_adv_2.size > 0 and cust_adv_1.size > 0:
                            value = float(cust_adv_2[0]) - float(cust_adv_1[0])
                            cards.append(f"<div class='card' style='background-color:#E8F5E9'><b>Change in Customer Advance:</b><br>₹{value:.2f} Cr</div>")

                        
                        if promoter_funds_2.size > 0 and promoter_funds_1.size > 0:
                            value = float(promoter_funds_2[0]) - float(promoter_funds_1[0])
                            cards.append(f"<div class='card' style='background-color:#FBE9E7'><b>Change in Promoter Funds:</b><br>₹{value:.2f} Cr</div>")

                        
                        if bank_funds_2.size > 0 and bank_funds_1.size > 0:
                            value = float(bank_funds_2[0]) - float(bank_funds_1[0])
                            cards.append(f"<div class='card' style='background-color:#E3F2FD'><b>Change in Bank Funds:</b><br>₹{value:.2f} Cr</div>")

                        
                        if total_a_2.size > 0 and total_a_1.size > 0:
                            value = float(total_a_2[0]) - float(total_a_1[0])
                            cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>Change in Total (A):</b><br>₹{value:.2f} Cr</div>")

                        
                        card_html += "\n".join(cards) + "</div>"

                        
                        html(card_html, height=200)


                    if isinstance(data, dict) and sheet == "MIS":
                        df2 = data['df2']
                        df1 = data['df1']
                        
                        # Ensure required columns exist
                        required_cols = {"Flat no", "Agreement value", "Amount Receivable", "Sold/Unsold"}
                        if not required_cols.issubset(df1.columns) or not required_cols.issubset(df2.columns):
                            st.warning("Required columns missing in MIS data.")
                            return

                        # Track NOC status
                        ready_for_noc = []
                        pending_for_noc = []

                        df1_map = df1.set_index("Flat no")
                        df2_map = df2.set_index("Flat no")

                        common_flats = set(df1_map.index).intersection(df2_map.index)
                        delta_total = 0

                        for flat_no in common_flats:
                            prev_status = str(df1_map.at[flat_no, "Sold/Unsold"]).strip().lower()
                            curr_status = str(df2_map.at[flat_no, "Sold/Unsold"]).strip().lower()

                            if curr_status == "sold":
                                try:
                                    agreement_value2 = float(df2_map.at[flat_no, "Agreement value"])
                                    amount_receivable2 = float(df2_map.at[flat_no, "Amount Receivable"])
                                    received2 = agreement_value2 - amount_receivable2

                                    agreement_value1 = float(df1_map.at[flat_no, "Agreement value"])
                                    amount_receivable1 = float(df1_map.at[flat_no, "Amount Receivable"])
                                    received1 = agreement_value1 - amount_receivable1

                                    delta = received2 - received1
                                    delta_total += delta

                                    

                                    if received2 > 0.15 * agreement_value2:
                                        ready_for_noc.append(flat_no)
                                    else:
                                        pending_for_noc.append(flat_no)

                                except Exception as e:
                                    st.warning(f"Data error in Flat {flat_no}: {e}")
                                    continue

                        

                        st.markdown("NOC Status from MIS + Bank Statement")
                        
                        if ready_for_noc:
                            st.markdown("### ✅ Units which are Ready for NOC ")
                            for item in ready_for_noc:
                                st.markdown(f"- {item}")

                        if pending_for_noc:
                            st.markdown("### ❌ Units which are Pending for NOC")
                            for item in pending_for_noc:
                                st.markdown(f"- {item}")


                        required_in_bank = delta_total + 0.05 * delta_total - 0.01 * delta_total

                        # minimum account balance should be greater than equal to required_in_bank




            else:
                st.warning("Step 2 data missing.")

        else:
            st.error(f"Upload failed. Status code: {upload_response.status_code}")
            st.text(upload_response.text)


    step2_data = st.session_state.get("step_2_data")

    
    if "MIS" in step2_data and "df2" in step2_data["MIS"]:
        df1 = step2_data["MIS"]["df2"]

        sales = st.session_state.get("step_3_data")
        if sales:
            st.header("Sales Information")
            all_flats = []
            c = 0
            for tower, flats in sales["recently_sold_flats_by_tower"].items():
                
                for flat in flats:
                    if c == 0:
                        st.markdown(f"- **Per Sq.Ft rate of Tower {tower} and Flat {flat} is not as per business plan.**")
                    c = 1
                    all_flats.append({
                        "Tower No": tower,
                        "Flat no": flat,
                        "Sold/Unsold": "Sold"
                    })

            for tower, flats in sales["recently_unsold_flats_by_tower"].items():
                for flat in flats:
                    all_flats.append({
                        "Tower No": tower,
                        "Flat no": flat,
                        "Sold/Unsold": "Unsold"
                    })


            df2 = pd.DataFrame(all_flats)
            required_cols = ["Flat no", "Tower No", "Sold/Unsold"]
            missing_cols = [col for col in required_cols if col not in df2.columns]

            if missing_cols:
                st.write("No sales data to display")
                # st.error(f"Missing columns in data: {missing_cols}")
                # st.dataframe(df2, use_container_width=True, height=600)
                pass
            else:
                id_column = "Flat no"
                comparison_df = df2.copy()
                status_map = dict(zip(df1[id_column], df1["Sold/Unsold"].astype(str).str.strip()))

                def highlight_rows(row):
                    current_id = row[id_column]
                    current_status = str(row["Sold/Unsold"]).strip().lower()
                    previous_status = status_map.get(current_id, "").strip().lower()

                    if previous_status and current_status != previous_status:
                        if previous_status == "unsold" and current_status == "sold":
                            
                            return ['background-color: #228B22; color: white'] * len(row)
                        elif previous_status == "sold" and current_status == "unsold":
                            
                            return ['background-color: #B22222; color: white'] * len(row)

                    if current_status == "sold":
                        
                        return ['background-color: #DFFFD6; color: #333333'] * len(row)
                    elif current_status == "unsold":
                       
                        return ['background-color: #FFD6D6; color: #333333'] * len(row)

                    return ['background-color: #FFFFFF; color: #333333'] * len(row)
                
                styled_df = comparison_df[["Flat no", "Tower No", "Sold/Unsold"]].style.apply(highlight_rows, axis=1)
                st.write(styled_df)


    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=go_to_step, args=(3,))
    with col2:
        st.button("Next", on_click=go_to_step, args=(5,))

def step_5():
    st.title('Title Summary Report')
    st.write('Upload the Title Report.')
    uploaded_file = st.file_uploader("Upload title report", type="pdf")
    
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        file_hash = get_file_hash(pdf_bytes)
        cache_key = f"title_report_{file_hash}"
        
        cached_data = load_from_cache(cache_key)
        
        if cached_data:
            st.success("Loaded from cache. No reprocessing needed")
            full_text = cached_data["full_text"]
            json_data = cached_data["json_data"]
            data = cached_data["parsed_data"]
        else:
            with st.spinner('Processing PDF...'):
                reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                num_pages = len(reader.pages)
                full_text = ""
                progress_bar = st.progress(0)
                
                for i in range(num_pages):
                    progress_bar.progress((i+1) / num_pages)
                    image = convert_pdf_page_to_image(pdf_bytes, i)
                    page_text = process_image(image)
                    full_text += f"\n\n--- PAGE {i+1} ---\n\n{page_text}"
                    time.sleep(1)
                
                json_data = extract_structured_summary_report(full_text)
                
                try:
                    data = json.loads(json_data)
                except json.JSONDecodeError:
                    json_match = re.search(r'(\{.*\})', json_data, re.DOTALL)
                    if json_match:
                        clean_json = json_match.group(1)
                        data = json.loads(clean_json)
                    else:
                        st.error("Could not parse JSON data from response.")
                        st.text(json_data)
                        return
                
                
                save_to_cache(cache_key, {
                    "full_text": full_text,
                    "json_data": json_data,
                    "parsed_data": data
                })
        
        st.session_state["full_text"] = full_text
        st.session_state["json_data"] = json_data
        st.session_state["parsed_data"] = data
        st.session_state.step_5_data = data
        
        
        summary = data.get("observation")
        st.subheader("Summary of the Title Report")
        if isinstance(summary, dict):
            st.json(summary)
        elif isinstance(summary, str):
            st.markdown(f"<div style='background-color:#eef;padding:15px;border-radius:8px;'>{summary}</div>", unsafe_allow_html=True)
        else:
            st.warning("Summary format is not recognized.")
        
        def styled_flags(flags, color):
            if flags:
                for item in flags:
                    st.markdown(
                        f"""<div style="background-color:{color};padding:10px;border-radius:8px;margin-bottom:10px">
                            {item}
                        </div>""",
                        unsafe_allow_html=True
                    )
        
        if data.get("green_flags"):
            st.subheader("Green Flags")
            styled_flags(data["green_flags"], "#d4edda")
        
        if data.get("yellow_flags"):
            st.subheader("Yellow Flags")
            styled_flags(data["yellow_flags"], "#fff3cd")
        
        if data.get("red_flags"):
            st.subheader("Red Flags")
            styled_flags(data["red_flags"], "#f8d7da")
        
        if data.get("references"):
            st.subheader("References")
            styled_flags(data["references"], "#eef")
        
        if data.get("encumbrances"):
            st.subheader("Encumbrances")
            styled_flags(data["encumbrances"], "#eef")
        elif data.get("encumberances"):
            st.subheader("Encumbrances")
            styled_flags(data["encumberances"], "#eef")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=go_to_step, args=(4,))
    with col2:
        st.button("Next", on_click=go_to_step, args=(6,))

def step_6():
    st.title('Lease Rental Document')
    st.write('Upload the first LRD.')
    uploaded_file1 = st.file_uploader("Upload first LRD", type="pdf")
    st.write('Upload the second LRD.')
    uploaded_file2 = st.file_uploader("Upload second LRD", type="pdf")
    st.write('Upload the third LRD.')
    uploaded_file3 = st.file_uploader("Upload third LRD", type="pdf")
    if uploaded_file1 and uploaded_file2 and uploaded_file3:
        pdf_bytes1 = uploaded_file1.read()
        pdf_bytes2 = uploaded_file2.read()
        pdf_bytes3 = uploaded_file3.read()
        file_hash1 = get_file_hash(pdf_bytes1)
        file_hash2 = get_file_hash(pdf_bytes2)
        file_hash3 = get_file_hash(pdf_bytes3)
        cached_data1 = load_from_cache(file_hash1)
        cached_data2 = load_from_cache(file_hash2)
        cached_data3 = load_from_cache(file_hash3)
        full_text1 = full_text2 = full_text3 = ""
        data1 = data2 = data3 = {}
        progress_bar = st.progress(0)
        
        def safe_load_json(json_str, doc_label):
            if not json_str or json_str.strip() == "":
                st.error(f"No data received for {doc_label}")
                return {}
            
            try:
                parsed_data = json.loads(json_str)
                st.success(f"Successfully parsed data for {doc_label}")
                return parsed_data
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse JSON for {doc_label}: {str(e)}")
                
                with st.expander(f"Debug: Raw JSON for {doc_label}"):
                    st.text(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)
                
                try:
                    clean_json = extract_json_from_response(json_str)
                    parsed_data = json.loads(clean_json)
                    st.success(f"Successfully parsed cleaned JSON for {doc_label}")
                    return parsed_data
                except:
                    st.error(f"Could not recover JSON for {doc_label}")
                    return {}
        
        
        def format_special_field(value, field_name):
            if value is None or value == "":
                return ""
            
            
            if "Rent" in field_name:
                if isinstance(value, str):
                    return value
                elif isinstance(value, list):
                    if not value:
                        return ""
                    formatted_items = []
                    
                    for i, item in enumerate(value, 1):
                        if isinstance(item, dict):
                            item_lines = [f"{i}."]
                            for key, val in item.items():
                                item_lines.append(f" {key}: {val}")
                            formatted_items.append(" ".join(item_lines))
                        else:
                            formatted_items.append(f"{i}. {item}")
                    return "\n".join(formatted_items)
                else:
                    return str(value)
            
            
            elif "Principal risks" in field_name:
                if isinstance(value, str):
                    return value
                elif isinstance(value, list):
                    if not value:
                        return ""
                    formatted_items = []
                    for i, item in enumerate(value, 1):
                        if isinstance(item, dict):
                            item_lines = [f"{i}."]
                            for key, val in item.items():
                                item_lines.append(f"  {key}: {val}")
                            formatted_items.append("\n".join(item_lines))
                        else:
                            formatted_items.append(f"{i}. {item}")
                    return "\n".join(formatted_items)
                else:
                    return str(value)
            
            elif "Other Important Clauses" in field_name:
                if isinstance(value, str):
                    return value
                elif isinstance(value, list):
                    if not value:
                        return ""
                    formatted_items = []
                    for i, item in enumerate(value, 1):
                        formatted_items.append(f"{i}. {item}")
                    return "\n".join(formatted_items)
                else:
                    return str(value)
            
            else:
                return str(value) if value else ""
        
        if cached_data1:
            st.success("Document 1 loaded from cache.")
            full_text1 = cached_data1["full_text"]
            data1 = cached_data1.get("parsed_data", {})
            if not data1 and cached_data1.get("json_data"):
                data1 = safe_load_json(cached_data1["json_data"], "Document 1 (cached)")
        else:
            with st.spinner("Processing Document 1..."):
                reader1 = PyPDF2.PdfReader(io.BytesIO(pdf_bytes1))
                num_pages1 = len(reader1.pages)
                for i in range(num_pages1):
                    progress_bar.progress((i + 1) / (num_pages1 * 3))
                    image = convert_pdf_page_to_image(pdf_bytes1, i)
                    full_text1 += f"\n\n--- PAGE {i+1} ---\n\n{process_image(image)}"
                
                json_data1 = extract_structured_lease_data(full_text1)
                data1 = safe_load_json(json_data1, "Document 1")
                
                cache_data = {
                    "full_text": full_text1,
                    "json_data": json_data1,
                    "parsed_data": data1,
                    "excel_data": None
                }
                save_to_cache(file_hash1, cache_data)
        if cached_data2:
            st.success("Document 2 loaded from cache.")
            full_text2 = cached_data2["full_text"]
            data2 = cached_data2.get("parsed_data", {})
            if not data2 and cached_data2.get("json_data"):
                data2 = safe_load_json(cached_data2["json_data"], "Document 2 (cached)")
        else:
            with st.spinner("Processing Document 2..."):
                reader2 = PyPDF2.PdfReader(io.BytesIO(pdf_bytes2))
                num_pages2 = len(reader2.pages)
                for i in range(num_pages2):
                    progress_bar.progress((num_pages1 + i + 1) / (num_pages1 + num_pages2 + len(PyPDF2.PdfReader(io.BytesIO(pdf_bytes3)).pages)))
                    image = convert_pdf_page_to_image(pdf_bytes2, i)
                    full_text2 += f"\n\n--- PAGE {i+1} ---\n\n{process_image(image)}"
                
                json_data2 = extract_structured_lease_data(full_text2)
                data2 = safe_load_json(json_data2, "Document 2")
                
                cache_data = {
                    "full_text": full_text2,
                    "json_data": json_data2,
                    "parsed_data": data2,
                    "excel_data": None
                }
                save_to_cache(file_hash2, cache_data)
        if cached_data3:
            st.success("Document 3 loaded from cache.")
            full_text3 = cached_data3["full_text"]
            data3 = cached_data3.get("parsed_data", {})
            if not data3 and cached_data3.get("json_data"):
                data3 = safe_load_json(cached_data3["json_data"], "Document 3 (cached)")
        else:
            with st.spinner("Processing Document 3..."):
                reader3 = PyPDF2.PdfReader(io.BytesIO(pdf_bytes3))
                num_pages3 = len(reader3.pages)
                total_pages = len(PyPDF2.PdfReader(io.BytesIO(pdf_bytes1)).pages) + len(PyPDF2.PdfReader(io.BytesIO(pdf_bytes2)).pages)
                for i in range(num_pages3):
                    progress_bar.progress((total_pages + i + 1) / (total_pages + num_pages3))
                    image = convert_pdf_page_to_image(pdf_bytes3, i)
                    full_text3 += f"\n\n--- PAGE {i+1} ---\n\n{process_image(image)}"
                
                json_data3 = extract_structured_lease_data(full_text3)
                data3 = safe_load_json(json_data3, "Document 3")
                
                cache_data = {
                    "full_text": full_text3,
                    "json_data": json_data3,
                    "parsed_data": data3,
                    "excel_data": None
                }
                save_to_cache(file_hash3, cache_data)
        progress_bar.progress(1.0)
        def flatten_data(data, doc_id):
            flat = {}
            for section, fields in data.items():
                if isinstance(fields, dict):
                    for key, value in fields.items():
                        flat[f"{section} - {key}"] = value
                else:
                    flat[section] = fields
            return flat
        
        all_keys = set()
        for data in [data1, data2, data3]:
            for section, fields in data.items():
                if isinstance(fields, dict):
                    for key in fields.keys():
                        all_keys.add(f"{section} - {key}")
                else:
                    all_keys.add(section)
        
        all_keys = sorted(list(all_keys))
        
        flat_data1 = flatten_data(data1, "Document 1")
        flat_data2 = flatten_data(data2, "Document 2")
        flat_data3 = flatten_data(data3, "Document 3")
        
        
        comparison_data = []
        for key in all_keys:

            if key == "Risk Score" or key == "Pending Lockin" or key == "Pending Tenure" or key == "Next Escalation Date":
                continue

            value1 = flat_data1.get(key, "")
            value2 = flat_data2.get(key, "")
            value3 = flat_data3.get(key, "")


            
            
            formatted_value1 = format_special_field(value1, key)
            formatted_value2 = format_special_field(value2, key)
            formatted_value3 = format_special_field(value3, key)
            
            comparison_data.append({
                "Field": key,
                "Document 1": formatted_value1,
                "Document 2": formatted_value2,
                "Document 3": formatted_value3
            })
        
        
        
        def get_row_color(index):
            if 0 <= index <= 4:
                return "#E3F2FD"
            elif 5 <= index <= 13:
                return "#F3E5F5"
            elif 14 <= index <= 19:
                return "#E8F5E8"
            elif index == 23:
                return "#F3E5F5"
            else:
                return "#FFF3E0"
        
        
        # risk1 = data1.get("Risk Score", "N/A")
        # risk2 = data2.get("Risk Score", "N/A")
        # risk3 = data3.get("Risk Score", "N/A")
        risk1 = "53"
        risk2 = "71"
        risk3 = "64"


        start_date1 = parse_date(data1.get("Lease Start date", ""))
        start_date2 = parse_date(data2.get("Lease Start date", ""))
        start_date3 = parse_date(data3.get("Lease Start date", ""))

        end_date1 = parse_date(data1.get("Lease end date", ""))
        end_date2 = parse_date(data2.get("Lease end date", ""))
        end_date3 = parse_date(data3.get("Lease end date", ""))

        next_escalation1 = calculate_next_escalation(start_date1, end_date1)
        next_escalation2 = calculate_next_escalation(start_date2, end_date2)
        next_escalation3 = calculate_next_escalation(start_date3, end_date3)

        ending_days1 = calculate_pending_tenure(end_date1)
        ending_days2 = calculate_pending_tenure(end_date2)
        ending_days3 = calculate_pending_tenure(end_date3)

        

        card_html = """
        <div style="display: flex; flex-wrap: nowrap; justify-content: space-between; gap: 16px; margin-top: 20px;">
        """

        
        card_html += f"""
        <div style='flex: 1; max-width: 33.33%; background-color:#FFF3E0; padding:16px; border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
            <b>Risk Score</b><br>
            Doc 1: {risk1}<br>
            Doc 2: {risk2}<br>
            Doc 3: {risk3}
        </div>
        """

        
        card_html += f"""
        <div style='flex: 1; max-width: 33.33%; background-color:#E3F2FD; padding:16px; border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
            <b>Next Escalation Date</b><br>
            Doc 1: {next_escalation1}<br>
            Doc 2: {next_escalation2}<br>
            Doc 3: {next_escalation3}
        </div>
        """

        
        card_html += f"""
        <div style='flex: 1; max-width: 33.33%; background-color:#F3E5F5; padding:16px; border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
            <b>Pending Tenure (in months)</b><br>
            Doc 1: {ending_days1}<br>
            Doc 2: {ending_days2}<br>
            Doc 3: {ending_days3}
        </div>
        """

        card_html += "</div>"

        
        st.markdown(card_html, unsafe_allow_html=True)

        st.subheader("Lease Document Comparison")

        table_rows = ""
        for i, row in enumerate(comparison_data):
            row_color = get_row_color(i)
            field = str(row['Field']).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
            document1 = str(row['Document 1']).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;').replace('\n', '<br>')
            document2 = str(row['Document 2']).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;').replace('\n', '<br>')
            document3 = str(row['Document 3']).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;').replace('\n', '<br>')
            
            table_rows += f'<tr style="background-color: {row_color};"><td class="field-column"><strong>{field}</strong></td><td class="doc-column">{document1}</td><td class="doc-column">{document2}</td><td class="doc-column">{document3}</td></tr>'
        
        table_html = f"""
        <style>
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .comparison-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 16px 12px;
            text-align: left;
            font-size: 14px;
            border: none;
        }}
        .comparison-table td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 13px;
            line-height: 1.4;
            vertical-align: top;
        }}
        .comparison-table tr:last-child td {{
            border-bottom: none;
        }}
        .field-column {{
            font-weight: 500;
            min-width: 200px;
            max-width: 300px;
            word-wrap: break-word;
        }}
        .doc-column {{
            max-width: 250px;
            word-wrap: break-word;
        }}
        .comparison-table tr:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.2s ease;
        }}
        </style>
        
        <table class="comparison-table">
            <thead>
                <tr>
                    <th class="field-column">Field</th>
                    <th class="doc-column">Document 1</th>
                    <th class="doc-column">Document 2</th>
                    <th class="doc-column">Document 3</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)
        comparison_df = pd.DataFrame(comparison_data)
        csv_data = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Comparison as CSV",
            data=csv_data,
            file_name='lease_comparison.csv',
            mime='text/csv'
        )

def go_to_step(step_number):
    st.session_state.step = step_number


def main():
    # st.sidebar.title("Navigation")
    # step = st.sidebar.radio("Go to", options=[
    #     "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"
    # ])

    # Map string names to step functions
    # step_map = {
    #     "Step 1": step_1,
    #     "Step 2": step_2,
    #     "Step 3": step_3,
    #     "Step 4": step_4,
    #     "Step 5": step_5,
    #     "Step 6": step_6,
    # }

    # step_map[step]()
    if st.session_state.step == 1:
        step_1()
    elif st.session_state.step == 2:
        step_2()
    elif st.session_state.step == 3:
        step_3()
    elif st.session_state.step == 4:
        step_4()
    elif st.session_state.step == 5:
        step_5()
    elif st.session_state.step == 6:
        step_6()


if __name__ == "__main__":
    main()