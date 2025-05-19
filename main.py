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


st.set_page_config(page_title="Developer Dashboard", layout="wide")

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
    st.title('Developer Dashboard')
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

                    with open("output.txt", "w", encoding='utf-8') as file:
                        file.write(full_text)

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

                
                milestone_options = table_df.iloc[:, 0].tolist()
                st.write("Select a Milestone to view related conditions:")
                selected_milestone = st.selectbox("", [""] + milestone_options)

                if selected_milestone:
                    if selected_milestone == milestone_options[0]:
                        if "pre_disbursement_conditions" in data:
                            st.subheader("Pre-Disbursement Conditions:")
                            for i, item in enumerate(data["pre_disbursement_conditions"]):
                                st.write(f"{i+1}. {item}")
                    else:
                        if "conditions_precedent" in data:
                            st.subheader("Conditions Precedent:")
                            for i, item in enumerate(data["conditions_precedent"]):
                                st.write(f"{i+1}. {item}")
                        if "conditions_subsequent" in data:
                            st.subheader("Conditions Subsequent:")
                            for i, item in enumerate(data["conditions_subsequent"]):
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
    
                        # st.write("**Status Changes Legend:**")
                        # st.markdown("🟢 <span style='color:green'>Green</span>: Unsold → Sold &nbsp;&nbsp;&nbsp; 🔴 <span style='color:red'>Red</span>: Sold → Unsold", unsafe_allow_html=True)
    
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

                        
                        
                        styled_df = df2.style.format(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else x).applymap(lambda _: 'background-color: #FFFFFF; color: #333333')
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
    st.title("Bank Statement Summary")

    st.write('Upload the Bank Statement')

    UPLOAD_URL = "https://cartuat.com/api/upload"
    DOWNLOAD_URL = "https://cartuat.com/api/downloadFile"
    AUTH_TOKEN = "API://QFEreQJLUvIWHKSLliicNPOC/MYh9B7dCo95Chz2rT2Sgf9ihi53EpD8LigFS/tw"

    uploaded_file = st.file_uploader("Upload bank statement", type="pdf")

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
                            for item in analysis_data:
                                month = item.get("month", "")
                                credit_amount = item.get("creditTransactionsAmount", 0.0)
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

        else:
            st.error(f"Upload failed. Status code: {upload_response.status_code}")
            st.text(upload_response.text)

    
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

                render_styled_table(data["df2"], f"{sheet} - COP-MOF Current")

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


    else:
        st.warning("Step 2 data missing.")

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
                    progress_bar.progress((i) / num_pages)
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

        
def go_to_step(step_number):
    st.session_state.step = step_number


def main():
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


if __name__ == "__main__":
    main()