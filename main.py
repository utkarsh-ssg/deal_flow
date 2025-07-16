import streamlit as st
import pandas as pd
import PyPDF2
import io
import time
from dotenv import load_dotenv
import json
import re
from streamlit.components.v1 import html
from utils.utils import *
import httpx
import asyncio

load_dotenv()


st.set_page_config(page_title="Asset Management", layout="wide")

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
    st.title('Asset Management')
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
                            
                            st.markdown("### Monthly Changes")
                            
                            def apply_delta_styles(row):
                                row_idx = row.name
                                if row_idx < len(delta_styles):
                                    return delta_styles[row_idx]
                                else:
                                    return [''] * len(row)
                            
                            
                            columns_with_changes = []
                            data_columns = [col for col in delta_df.columns if col != id_column]
                            
                            for col_idx, col in enumerate(data_columns):
                                
                                col_has_changes = False
                                for row_idx in range(len(delta_df)):
                                    if row_idx < len(delta_styles):
                                        
                                        style_idx = col_idx + 1
                                        if style_idx < len(delta_styles[row_idx]) and delta_styles[row_idx][style_idx] != '':
                                            col_has_changes = True
                                            break
                                
                                if col_has_changes:
                                    columns_with_changes.append(col)
                            
                            
                            tab_names = ["All"] + columns_with_changes
                            
                            
                            tabs = st.tabs(tab_names)
                            
                            
                            with tabs[0]:
                                styled_delta_df = delta_df.style.apply(apply_delta_styles, axis=1)
                                st.write(styled_delta_df)
                            
                            
                            for i, col_name in enumerate(columns_with_changes, 1):
                                with tabs[i]:
                                    
                                    filtered_rows = []
                                    col_idx = data_columns.index(col_name)
                                    
                                    for row_idx in range(len(delta_df)):
                                        if row_idx < len(delta_styles):
                                            
                                            style_idx = col_idx + 1
                                            if style_idx < len(delta_styles[row_idx]) and delta_styles[row_idx][style_idx] != '':
                                                filtered_rows.append(row_idx)
                                    
                                    if filtered_rows:
                                        
                                        filtered_df = delta_df.iloc[filtered_rows].copy()
                                        
                                        filtered_df.reset_index(drop=True, inplace=True)
                                        
                                        
                                        def apply_filtered_styles(row):
                                            
                                            row_position = row.name
                                            if row_position < len(filtered_rows):
                                                original_idx = filtered_rows[row_position]
                                                if original_idx < len(delta_styles):
                                                    return delta_styles[original_idx]
                                            return [''] * len(row)
                                        
                                        styled_filtered_df = filtered_df.style.apply(apply_filtered_styles, axis=1)
                                        st.write(styled_filtered_df)
                                    else:
                                        st.info(f"No changes detected in column '{col_name}'")
                            
                            
                            comparison_results[sheet]["delta_table"] = delta_df
                            
                            
                            comparison_results[sheet]["delta_by_column"] = {}
                            for col_name in columns_with_changes:
                                col_idx = data_columns.index(col_name)
                                filtered_rows = []
                                
                                for row_idx in range(len(delta_df)):
                                    if row_idx < len(delta_styles):
                                        style_idx = col_idx + 1
                                        if style_idx < len(delta_styles[row_idx]) and delta_styles[row_idx][style_idx] != '':
                                            filtered_rows.append(row_idx)
                                
                                if filtered_rows:
                                    filtered_data = delta_df.iloc[filtered_rows].copy()
                                    filtered_data.reset_index(drop=True, inplace=True)
                                    comparison_results[sheet]["delta_by_column"][col_name] = filtered_data

                        else:
                            st.markdown("### Monthly Changes")
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
                    
                    if 'MIS' in step2_data and 'delta_by_column' in step2_data['MIS'] and 'Sold/Unsold' in step2_data['MIS']['delta_by_column']:
                        df = step2_data['MIS']['delta_by_column']['Sold/Unsold']
                        
                        
                        sold_units = df[df['Sold/Unsold'] == 'Sold'].copy()
                        
                        if not sold_units.empty:
                            st.subheader("Request NOC for Sold Units")
                            st.write("Select the units for which you want to request NOC:")
                            
                            unit_options = []
                            unit_mapping = {}
                            
                            for idx, row in sold_units.iterrows():
                                tower = row['Tower No']
                                flat_no = row['Flat no']
                                display_text = f"Tower {tower} - Flat {flat_no}"
                                unit_options.append(display_text)
                                unit_mapping[display_text] = {
                                    'sl_no': row['Sl No.'],
                                    'flat_no': flat_no,
                                    'tower_no': tower,
                                    'sold_status': row['Sold/Unsold'],
                                    'agreement_value': row['Agreement value'],
                                    'amount_received': row['Amount Received as on 28-02-2025'],
                                    'amount_receivable': row['Amount Receivable'],
                                    'saleable_area': row['Sealable Area (in sq ft)']
                                }
                            
                            
                            selected_units = st.multiselect(
                                "Select units for NOC request:",
                                options=unit_options,
                                placeholder="Choose units...",
                                help="You can select multiple units"
                            )
                            
                            
                            if selected_units:
                                st.subheader("Selected Units Details")
                                
                                selected_data = []
                                for unit in selected_units:
                                    unit_data = unit_mapping[unit]
                                    selected_data.append(unit_data)
                                
                                
                                selected_df = pd.DataFrame(selected_data)
                                
                                
                                st.dataframe(
                                    selected_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                
                                if st.button("Confirm NOC Request", type="primary"):
                                    if "step_3_data" not in st.session_state or st.session_state["step_3_data"] is None:
                                        st.session_state["step_3_data"] = {}
                                    
                                    st.session_state["step_3_data"]["noc_data"] = {
                                        "selected_units": selected_data
                                    }
                                    
                                    st.success(f"✅ NOC request submitted for {len(selected_data)} units!")
                                    
                                    
                                    
                            
                            
                        
                        else:
                            st.warning("⚠️ No sold units found in the data.")
                

                    # sales_df = None
                    # for sheet_name, sheet_data in step2_data.items():
                    #     if isinstance(sheet_data, dict) and "df2" in sheet_data:
                    #         df = sheet_data["df2"]
                    #         required_cols = [
                    #             "Flat no", "Tower No", "Sold/Unsold"
                    #         ]
                    #         if all(col in df.columns for col in required_cols):
                    #             sales_df = df
                    #             break

                    # if sales_df is not None:
                    #     st.markdown("<h3 style='color:#003366;'>Sales Information</h3>", unsafe_allow_html=True)

                    #     recently_unsold_flats_by_tower = {}
                    #     recently_sold_flats_by_tower = {}
                    #     unique_towers = sales_df["Tower No"].dropna().unique()

                    #     total_recently_sold = 0
                    #     total_recently_unsold = 0

                    #     for tower in unique_towers:
                    #         st.markdown(f"<h4 style='color:#2C3E50; margin-bottom: 0;'>Tower: {tower}</h4>", unsafe_allow_html=True)

                    #         with st.expander("", expanded=True):
                    #             st.markdown("""
                    #                 <style>
                    #                     .streamlit-expanderHeader {
                    #                         color: #2C3E50;
                    #                         font-weight: bold;
                    #                         font-size: 18px;
                    #                     }
                    #                     .streamlit-expander .streamlit-expanderContent {
                    #                         color: #333333;
                    #                     }
                    #                 </style>
                    #             """, unsafe_allow_html=True)

                    #             recently_unsold_flats = sales_df[
                    #                 (sales_df["Tower No"] == tower) &
                    #                 (sales_df["Sold/Unsold"].str.lower() == "sold")
                    #             ]["Flat no"].dropna().unique()
                    #             st.write(f"Select Flats whose Sales got cancelled post latest MIS in Tower {tower}")
                    #             selected_recently_unsold_flats = st.multiselect(
                    #                 "",
                    #                 recently_unsold_flats,
                    #                 key=f"recently_unsold_flats_{tower}"
                    #             )

                    #             recently_sold_flats = sales_df[
                    #                 (sales_df["Tower No"] == tower) &
                    #                 (sales_df["Sold/Unsold"].str.lower() == "unsold")
                    #             ]["Flat no"].dropna().unique()
                    #             st.write(f"Select Unsold Flats which were Sold post latest MIS in Tower {tower}")

                    #             selected_recently_sold_flats = st.multiselect(
                    #                 "",
                    #                 recently_sold_flats,
                    #                 key=f"unsold_flats_{tower}"
                    #             )

                    #             combined_selected = list(set(selected_recently_unsold_flats) | set(selected_recently_sold_flats))
                    #             recently_unsold_flats_by_tower[tower] = selected_recently_unsold_flats
                    #             recently_sold_flats_by_tower[tower] = selected_recently_sold_flats

                    #             selected_df = sales_df[
                    #                 (sales_df["Tower No"] == tower) &
                    #                 (sales_df["Flat no"].isin(combined_selected))
                    #             ]

                    #             recently_unsold_count = selected_df[selected_df["Sold/Unsold"].str.lower() == "sold"].shape[0]
                    #             recently_sold_count = selected_df[selected_df["Sold/Unsold"].str.lower() == "unsold"].shape[0]

                    #             total_recently_unsold += recently_unsold_count
                    #             total_recently_sold += recently_sold_count

                    #             st.markdown(f"<div style='margin-top:10px; font-weight:bold;'>Flats whose Sales got cancelled post current MIS: <span style='color:#007ACC'>{recently_unsold_count}</span></div>", unsafe_allow_html=True)
                    #             st.markdown(f"<div style='font-weight:bold;'>Flats solds post current MIS: <span style='color:#28B463'>{recently_sold_count}</span></div>", unsafe_allow_html=True)

                    #     st.markdown("<hr style='border-top: 2px solid #bbb;'/>", unsafe_allow_html=True)
                    #     st.markdown(f"<h4 style='color:#1A5276;'>Total Sold Flats whose sales got cancelled post latest MIS: <span style='color:#2E86C1'>{total_recently_unsold}</span></h4>", unsafe_allow_html=True)
                    #     st.markdown(f"<h4 style='color:#145A32;'>Total Unsold Flats which went Sold post latest MIS: <span style='color:#28B463'>{total_recently_sold}</span></h4>", unsafe_allow_html=True)

                    #     st.session_state.step_3_data = {
                    #         "recently_unsold_flats_by_tower": recently_unsold_flats_by_tower,
                    #         "recently_sold_flats_by_tower": recently_sold_flats_by_tower,
                    #         "total_recently_sold": total_recently_unsold,
                    #         "total_recently_sold_selected": total_recently_sold,
                    #     }


                    # else:
                    #     st.warning("Sales data not available or missing required columns.")


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

    uploaded_file = st.file_uploader("Upload Collection Bank statement", type="pdf")
    uploaded_file2 = st.file_uploader("Upload Corporate Bank statement", type="pdf")
    # uploaded_file3 = st.file_uploader("Upload Project statement", type="pdf")
    if uploaded_file is not None and uploaded_file2 is not None:
        
        download_response = process_bank_statement(uploaded_file)
        download_response2 = process_bank_statement(uploaded_file2)
        

        if download_response.status_code == 200 and download_response2.status_code == 200:
            result = download_response.json()
            result2 = download_response2.json()
            bajaj_housing_total = 0
            
            if "analysisData" in result['data'][0] and "transactions" in result2['data'][0]:
                
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

                
                transactions = result2['data'][0]['transactions']

                for transaction in transactions:
                    if (transaction.get("type") == "Dr" and "BAJAJ HOUSING F" in transaction.get("narration", "").upper() and "Mar-2025" in transaction.get("monthYear","")):
                        bajaj_housing_total += transaction.get("amount", 0.0)

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

                html(card_html, height=600)

            step2 = st.session_state.get("step_2_data")
            if step2:
                st.header("Latest Data")
                
                
                total_a_2_to_be_incurred = 0
                customer_advance_incurred = 0
                
                
                cop_mof_data = {}
                mis_data = {}
                
                
                for sheet, data in step2.items():
                    if isinstance(data, dict) and sheet == "COP-MOF":
                        
                        df1 = data['df1']
                        df2 = data['df2']
                        
                        
                        cop_mof_data = {
                            'df1': df1,
                            'df2': df2,
                            'bank_funds': df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "bank funds", "Incurred"].values,
                            'mean_of_finance': df2.loc[df2["PARTICULARS"].str.strip() == "MEANS OF FINANCE", "Incurred"].values,
                            'total_a': df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "total (a)", "Incurred"].values,
                            'cust_adv_2': df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "customer advance", "Incurred"].values,
                            'cust_adv_1': df1.loc[df1["PARTICULARS"].str.strip().str.lower() == "customer advance", "Incurred"].values,
                            'promoter_funds_2': df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "promoter funds", "Incurred"].values,
                            'promoter_funds_1': df1.loc[df1["PARTICULARS"].str.strip().str.lower() == "promoter funds", "Incurred"].values,
                            'bank_funds_2': df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "bank funds", "Incurred"].values,
                            'bank_funds_1': df1.loc[df1["PARTICULARS"].str.strip().str.lower() == "bank funds", "Incurred"].values,
                            'total_a_2': df2.loc[df2["PARTICULARS"].str.strip().str.lower() == "total (a)", "Incurred"].values,
                            'total_a_1': df1.loc[df1["PARTICULARS"].str.strip().str.lower() == "total (a)", "Incurred"].values,
                            'total_a_2_to_be_incurred': df1.loc[df1["PARTICULARS"].str.strip().str.lower() == "total (a)", "To be incurred"].values,
                        }
                        
                        
                        total_a_2_to_be_incurred = cop_mof_data['total_a_2_to_be_incurred']
                        customer_advance_incurred = cop_mof_data['cust_adv_2'][0] if cop_mof_data['cust_adv_2'].size > 0 else 0
                        customer_advance_incurred = customer_advance_incurred * 10000000
                        
                    elif isinstance(data, dict) and sheet == "MIS":
                        
                        df2 = data['df2']
                        df1 = data['df1']
                        
                        if "Sold/Unsold" in df1.columns and "Sold/Unsold" in df2.columns:
                            
                            df1_map = df1.set_index("Flat no")
                            df2_map = df2.set_index("Flat no")
                            common_flats = set(df1_map.index).intersection(df2_map.index)

                            sold_prev = df1[df1["Sold/Unsold"].str.lower().str.strip() == "sold"]
                            sold_curr = df2[df2["Sold/Unsold"].str.lower().str.strip() == "sold"]
                            unsold_curr = df2[df2["Sold/Unsold"].str.lower().str.strip() == "unsold"]
                            
                            new_units_sold = set(sold_curr["Flat no"]) - set(sold_prev["Flat no"])
                            df_new_sold = df2[df2["Flat no"].isin(new_units_sold)]
                            
                            
                            flat_bank_match_data = []
                            bank_transactions = result['data'][0].get("transactions", [])
                            total_delta_received = 0
                            
                            def is_match_found(expected_amt, name, transactions):
                                name = name.lower()
                                for txn in transactions:
                                    if txn.get("type") == "Cr" and txn.get("monthYear") == "Mar-2025":
                                        txn_amt = txn.get("amount", 0)
                                        txn_name = txn.get("name", "").lower()
                                        if abs(txn_amt - expected_amt) <= 5000 and name in txn_name:
                                            return True
                                return False
                            
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
                                        total_delta_received += round(delta, 2)
                                        expected_amt = delta * 1.04
                                        
                                        customer_name = df2_map.at[flat_no, "Name of Customer"] if "Name of Customer" in df2_map.columns else ""
                                        tower_no = df2_map.at[flat_no, "Tower No"] if "Tower No" in df2_map.columns else ""
                                        
                                        match_found = is_match_found(expected_amt, customer_name, bank_transactions)
                                        appeared_in_bank_statement = "No transaction check required"
                                        
                                        if match_found:
                                            appeared_in_bank_statement = "✅ Yes"
                                        else:
                                            if delta != 0:
                                                appeared_in_bank_statement = "❌ No"
                                                
                                        flat_bank_match_data.append({
                                            "Flat No": flat_no,
                                            "Tower No": tower_no,
                                            "Customer Name": customer_name,
                                            "Delta Received (₹)": round(delta, 2),
                                            "Expected in Bank (₹)": round(expected_amt, 2),
                                            "Appeared in Bank Statement": appeared_in_bank_statement
                                        })
                                    except Exception as e:
                                        st.warning(f"Error in flat {flat_no}: {e}")
                                        continue
                            
                            
                            mis_data = {
                                'df1': df1,
                                'df2': df2,
                                'df1_map': df1_map,
                                'df2_map': df2_map,
                                'sold_prev': sold_prev,
                                'sold_curr': sold_curr,
                                'unsold_curr': unsold_curr,
                                'new_units_sold': new_units_sold,
                                'df_new_sold': df_new_sold,
                                'flat_bank_match_data': flat_bank_match_data,
                                'total_delta_received': total_delta_received,
                                'common_flats': common_flats,
                                
                                'total_unsold_saleable_area': unsold_curr["Sealable Area (in sq ft)"].apply(safe_float).sum(),
                                'total_sold_units': len(sold_curr),
                                'total_agreement_all_sold': sold_curr["Agreement value"].apply(safe_float).sum(),
                                'total_receivable_all_sold': sold_curr["Amount Receivable"].apply(safe_float).sum(),
                                'num_new_sold': len(df_new_sold),
                                'total_agreement_new_sold': df_new_sold["Agreement value"].apply(safe_float).sum(),
                                'total_receivable_new_sold': df_new_sold["Amount Receivable"].apply(safe_float).sum(),
                            }
                            
                            
                            mis_data['total_received_all_sold'] = mis_data['total_agreement_all_sold'] - mis_data['total_receivable_all_sold']
                            mis_data['pct_received_all_sold'] = (mis_data['total_received_all_sold'] / mis_data['total_agreement_all_sold'] * 100) if mis_data['total_agreement_all_sold'] > 0 else 0.0
                            mis_data['total_received_new_sold'] = mis_data['total_agreement_new_sold'] - mis_data['total_receivable_new_sold']
                            mis_data['pct_received_new_sold'] = (mis_data['total_received_new_sold'] / mis_data['total_agreement_new_sold'] * 100) if mis_data['total_agreement_new_sold'] > 0 else 0.0
                            
                            
                            msi = 27500
                            loan_amount = 300000000
                            overall_unsold_receivable = mis_data['total_unsold_saleable_area'] * msi
                            loan_outstanding_prev = loan_amount
                            loan_outstanding_curr = loan_amount - mis_data['total_received_all_sold']
                            
                            mis_data['cashflow_cover_prev'] = (mis_data['total_receivable_all_sold'] + overall_unsold_receivable) / loan_outstanding_prev
                            mis_data['cashflow_cover_curr'] = (mis_data['total_receivable_all_sold'] + overall_unsold_receivable - total_a_2_to_be_incurred) / loan_outstanding_curr
                            mis_data['security_cover_prev'] = overall_unsold_receivable / loan_outstanding_prev
                            mis_data['security_cover_curr'] = overall_unsold_receivable / loan_outstanding_curr
                            mis_data['expected_swept_amount'] = 0.15 * total_delta_received
                
                
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
                
                
                if cop_mof_data:
                    render_styled_table(cop_mof_data["df2"], "COP-MOF Current")
                    
                    
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
                    
                    
                    if cop_mof_data['bank_funds'].size > 0:
                        value = float(cop_mof_data['bank_funds'][0]) / 100.0
                        cards.append(f"<div class='card' style='background-color:#F0F8FF'><b>Obligation:</b><br>₹{value:.2f} Cr</div>")
                    
                    if cop_mof_data['mean_of_finance'].size > 0 and cop_mof_data['total_a'].size > 0:
                        value = float(cop_mof_data['mean_of_finance'][0]) - float(cop_mof_data['total_a'][0])
                        cards.append(f"<div class='card' style='background-color:#FFF8E1'><b>Balance:</b><br>₹{value:.2f} Cr</div>")
                    
                    if cop_mof_data['cust_adv_2'].size > 0 and cop_mof_data['cust_adv_1'].size > 0:
                        value = float(cop_mof_data['cust_adv_2'][0]) - float(cop_mof_data['cust_adv_1'][0])
                        cards.append(f"<div class='card' style='background-color:#E8F5E9'><b>Change in Customer Advance:</b><br>₹{value:.2f} Cr</div>")
                    
                    if cop_mof_data['promoter_funds_2'].size > 0 and cop_mof_data['promoter_funds_1'].size > 0:
                        value = float(cop_mof_data['promoter_funds_2'][0]) - float(cop_mof_data['promoter_funds_1'][0])
                        cards.append(f"<div class='card' style='background-color:#FBE9E7'><b>Change in Promoter Funds:</b><br>₹{value:.2f} Cr</div>")
                    
                    if cop_mof_data['bank_funds_2'].size > 0 and cop_mof_data['bank_funds_1'].size > 0:
                        value = float(cop_mof_data['bank_funds_2'][0]) - float(cop_mof_data['bank_funds_1'][0])
                        cards.append(f"<div class='card' style='background-color:#E3F2FD'><b>Change in Bank Funds:</b><br>₹{value:.2f} Cr</div>")
                    
                    if cop_mof_data['total_a_2'].size > 0 and cop_mof_data['total_a_1'].size > 0:
                        value = float(cop_mof_data['total_a_2'][0]) - float(cop_mof_data['total_a_1'][0])
                        cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>Change in Total (A):</b><br>₹{value:.2f} Cr</div>")
                    
                    card_html += "\n".join(cards) + "</div>"
                    html(card_html, height=200)
                
                
                if mis_data:
                    
                    if mis_data['flat_bank_match_data']:
                        st.subheader("Matching Received Amount with Bank Statement")
                        result_df = pd.DataFrame(mis_data['flat_bank_match_data'])
                        st.dataframe(result_df, use_container_width=True)
                    else:
                        st.info("No matching sold flats found for delta-bank check.")
                    
                    
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
                    
                    
                    cards.append(f"<div class='card' style='background-color:#F0F8FF'><b>Total units sold:</b><br>{mis_data['total_sold_units']}</div>")
                    cards.append(f"<div class='card' style='background-color:#FFF8E1'><b>Recently sold units:</b><br>{mis_data['num_new_sold']}</div>")
                    cards.append(f"<div class='card' style='background-color:#E8F5E9'><b>Amount Received from recently sold units:</b><br>₹{mis_data['total_received_new_sold']:,.2f}</div>")
                    cards.append(f"<div class='card' style='background-color:#FBE9E7'><b>% Received from recently sold units:</b><br>{mis_data['pct_received_new_sold']:.2f}%</div>")
                    cards.append(f"<div class='card' style='background-color:#E3F2FD'><b>Amount Received from all sold units:</b><br>₹{mis_data['total_received_all_sold']:,.2f}</div>")
                    cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>% Received from All Sold Units:</b><br>{mis_data['pct_received_all_sold']:.2f}%</div>")
                    cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>Feb Cash Flow Cover*:</b><br>{mis_data['cashflow_cover_prev']:.2f}</div>")
                    cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>Current Cash Flow Cover*:</b><br>{mis_data['cashflow_cover_curr']:.2f}</div>")
                    cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>Feb Security Cover**:</b><br>{mis_data['security_cover_prev']:.2f}</div>")
                    cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>Current Security Cover**:</b><br>{mis_data['security_cover_curr']:.2f}</div>")
                    cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>Expected Amount to be debited to Financial Institution:</b><br>₹ {mis_data['expected_swept_amount']:.2f}</div>")
                    cards.append(f"<div class='card' style='background-color:#FFF3E0'><b>Actual Amount Debited this month to Financial Institution from 30% account:</b><br>₹ {bajaj_housing_total:.2f}</div>")
                    
                    card_html += "\n".join(cards) + "</div>"
                    html(card_html, height=420)
                    
                    
                    expected_swept_amount = mis_data['expected_swept_amount']

                    st.markdown(
                        """
                        <div style="
                            background-color: #f9f9f9;
                            border-left: 4px solid #007ACC;
                            padding: 1rem;
                            margin-top: 1rem;
                            margin-bottom: 1rem;
                            border-radius: 6px;
                            font-size: 16px;
                        ">
                            <strong>*Cash flow cover</strong> = (Sold balance receivable + Unsold Receivable at MSP) / Loan outstanding  
                            <br><br>
                            <strong>**Security Cover</strong> = Unsold units at MSP / Loan outstanding
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    
                    
                    if bajaj_housing_total < expected_swept_amount:
                        st.markdown(f"""
                            <div style='padding:10px; background-color:#FFCDD2; border-left:5px solid #C62828; border-radius:6px;'>
                                <b>Red Flag:</b><br>
                                Swept Amount (₹{bajaj_housing_total:.2f}) is less than Expected Amount Swept (₹{expected_swept_amount:.2f})
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='padding:10px; background-color:#C8E6C9; border-left:5px solid #2E7D32; border-radius:6px;'>
                            <b>Green Flag:</b><br>
                            Swept Amount (₹{bajaj_housing_total:.2f}) covers Expected Amount Swept (₹{expected_swept_amount:.2f})
                        </div>
                        """, unsafe_allow_html=True)
                    
                    
                    average_expected = 1000000
                    if bajaj_housing_total < average_expected:
                        st.markdown(f"""
                            <div style='padding:10px; background-color:#FFCDD2; border-left:5px solid #C62828; border-radius:6px;'>
                                <b>Red Flag:</b><br>
                                Swept Amount (₹{expected_swept_amount:.2f}) is less than Average Swept (₹{average_expected:.2f})
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='padding:10px; background-color:#C8E6C9; border-left:5px solid #2E7D32; border-radius:6px;'>
                            <b>Green Flag:</b><br>
                            Swept Amount (₹{expected_swept_amount:.2f}) covers Average Swept (₹{average_expected:.2f})
                        </div>
                        """, unsafe_allow_html=True)
                    
                    
                    if customer_advance_incurred != mis_data['total_received_all_sold']:
                        st.markdown(f"""
                            <div style='padding:10px; background-color:#FFCDD2; border-left:5px solid #C62828; border-radius:6px;'>
                                <b>Red Flag:</b><br>
                                Customer Advance Incurred (₹{customer_advance_incurred:.2f}) is not equal to the Total Received Amount from Sales (₹{mis_data['total_received_all_sold']:.2f})
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='padding:10px; background-color:#C8E6C9; border-left:5px solid #2E7D32; border-radius:6px;'>
                            <b>Green Flag:</b><br>
                            Customer Advance Incurred (₹{customer_advance_incurred:.2f}) equals to the Total Received Amount from Sales (₹{mis_data['total_received_all_sold']:.2f})
                        </div>
                        """, unsafe_allow_html=True)
            


            else:
                st.warning("Step 2 data missing.")  


            if "step_3_data" in st.session_state and "noc_data" in st.session_state["step_3_data"] and "selected_units" in st.session_state["step_3_data"]['noc_data']:
                noc_data = st.session_state["step_3_data"]['noc_data']['selected_units']
                
                
                ready_for_noc = []
                not_ready_for_noc = []
                
                for unit in noc_data:
                    agreement_value = safe_float(unit.get('agreement_value'))
                    amount_received = safe_float(unit.get('amount_received'))
                    
                    threshold = 27500 * safe_float(unit.get('saleable_area'))
                    
                    
                    unit_with_status = unit.copy()
                    unit_with_status['threshold_amount'] = threshold
                    unit_with_status['percentage_received'] = (amount_received / agreement_value * 100) if agreement_value > 0 else 0
                    
                    
                    if agreement_value >= threshold:
                        ready_for_noc.append(unit_with_status)
                    else:
                        not_ready_for_noc.append(unit_with_status)
                
                
                if ready_for_noc:
                    st.subheader("Units Ready for NOC")
                    
                    ready_df = pd.DataFrame(ready_for_noc)
                    
                    numeric_columns = ['agreement_value', 'amount_received', 'amount_receivable', 'percentage_received', 'saleable_area']
                    for col in numeric_columns:
                        ready_df[col] = ready_df[col].apply(safe_float)

                    ready_df['agreement_value_per_sq_ft'] = ready_df['agreement_value']/ready_df['saleable_area']
                    
                    display_columns = {
                        'tower_no': 'Tower',
                        'flat_no': 'Flat No',
                        'agreement_value': 'Agreement Value',
                        'amount_received': 'Amount Received',
                        'amount_receivable': 'Amount Receivable',
                        # 'percentage_received': 'Percentage Received (%)',
                        'saleable_area': 'Saleable Area (sq ft)',
                        'agreement_value_per_sq_ft': 'Agreement Value ₹ per sq ft'
                    }
                    
                    
                    ready_display = ready_df[list(display_columns.keys())].rename(columns=display_columns)
                    
                    ready_display['Agreement Value'] = ready_display['Agreement Value'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "₹0")
                    ready_display['Amount Received'] = ready_display['Amount Received'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "₹0")
                    ready_display['Amount Receivable'] = ready_display['Amount Receivable'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "₹0")
                    # ready_display['Percentage Received (%)'] = ready_display['Percentage Received (%)'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
                    ready_display['MSP (₹ per sq ft)'] = "27500"
                    # ready_display['Saleable Area'] = ready_display['Saleable Area (sq ft)']
                    st.dataframe(ready_display, use_container_width=True, hide_index=True)

                if not_ready_for_noc:
                    st.subheader("Units Not Ready for NOC")


                    not_ready_df = pd.DataFrame(not_ready_for_noc)

                    
                    numeric_columns = ['agreement_value', 'amount_received', 'threshold_amount', 'percentage_received', 'saleable_area']
                    for col in numeric_columns:
                        not_ready_df[col] = not_ready_df[col].apply(safe_float)
                    
                    
                    not_ready_df['shortfall'] = not_ready_df['threshold_amount'] - not_ready_df['amount_received']
                    not_ready_df['agreement_value_per_sq_ft'] = not_ready_df['agreement_value']/not_ready_df['saleable_area']
                    
                    display_columns = {
                        'tower_no': 'Tower',
                        'flat_no': 'Flat No',
                        'agreement_value': 'Agreement Value',
                        'amount_received': 'Amount Received',
                        'threshold_amount': 'Required Amount',
                        # 'percentage_received': 'Percentage Received (%)',
                        'shortfall': 'Shortfall',
                        'saleable_area': 'Saleable Area (sq ft)',
                        'agreement_value_per_sq_ft': 'Agreement Value ₹ per sq ft'
                    }
                    
                    not_ready_display = not_ready_df[list(display_columns.keys())].rename(columns=display_columns)
                    
                    
                    not_ready_display['Agreement Value'] = not_ready_display['Agreement Value'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "₹0")
                    not_ready_display['Amount Received'] = not_ready_display['Amount Received'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "₹0")
                    not_ready_display['Required Amount'] = not_ready_display['Required Amount'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "₹0")
                    # not_ready_display['Percentage Received (%)'] = not_ready_display['Percentage Received (%)'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
                    not_ready_display['Shortfall'] = not_ready_display['Shortfall'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "₹0")
                    not_ready_display['MSP (₹ per sq ft)'] = "27500"
                    # not_ready_display['Saleable Area'] = not_ready_display['Saleable Area (sq ft)']
                    not_ready_display['Comment'] = 'MSP condition not met. Need to request for Deviation'
                    st.dataframe(not_ready_display, use_container_width=True, hide_index=True)
                    
                    
                st.session_state.step_4_data = {
                    "collection_bank_statement": result['data'][0]['analysisData'],
                    "corporate_bank_statement": result2['data'][0]['transactions'],
                    "units_ready_for_noc": ready_for_noc,
                    "units_not_ready_for_noc": not_ready_for_noc
                }
                st.subheader("NOC Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("Total Units", len(noc_data))

                with col2:
                    st.write("Ready for NOC", len(ready_for_noc))

                with col3:
                    st.write("Not Ready for NOC", len(not_ready_for_noc))
            else:
                st.warning("No NOC data found. Please select units first.")
            
            chatbot_interface()
        else:
            st.error(f"Download failed. Status code: {download_response.status_code}")
            st.text(download_response.text)

        


    step2_data = st.session_state.get("step_2_data")

    
    # if "MIS" in step2_data and "df2" in step2_data["MIS"]:
    #     df1 = step2_data["MIS"]["df2"]

    #     sales = st.session_state.get("step_3_data")
    #     if sales:
    #         st.header("Sales Information")
    #         all_flats = []
    #         c = 0
    #         for tower, flats in sales["recently_sold_flats_by_tower"].items():
                
    #             for flat in flats:
    #                 if c == 0:
    #                     st.markdown(f"- **Per Sq.Ft rate of Tower {tower} and Flat {flat} is not as per business plan.**")
    #                 c = 1
    #                 all_flats.append({
    #                     "Tower No": tower,
    #                     "Flat no": flat,
    #                     "Sold/Unsold": "Sold"
    #                 })

    #         for tower, flats in sales["recently_unsold_flats_by_tower"].items():
    #             for flat in flats:
    #                 all_flats.append({
    #                     "Tower No": tower,
    #                     "Flat no": flat,
    #                     "Sold/Unsold": "Unsold"
    #                 })


    #         df2 = pd.DataFrame(all_flats)
    #         required_cols = ["Flat no", "Tower No", "Sold/Unsold"]
    #         missing_cols = [col for col in required_cols if col not in df2.columns]

    #         if missing_cols:
    #             st.write("No sales data to display")
    #             # st.error(f"Missing columns in data: {missing_cols}")
    #             # st.dataframe(df2, use_container_width=True, height=600)
    #             pass
    #         else:
    #             id_column = "Flat no"
    #             comparison_df = df2.copy()
    #             status_map = dict(zip(df1[id_column], df1["Sold/Unsold"].astype(str).str.strip()))

    #             def highlight_rows(row):
    #                 current_id = row[id_column]
    #                 current_status = str(row["Sold/Unsold"]).strip().lower()
    #                 previous_status = status_map.get(current_id, "").strip().lower()

    #                 if previous_status and current_status != previous_status:
    #                     if previous_status == "unsold" and current_status == "sold":
                            
    #                         return ['background-color: #228B22; color: white'] * len(row)
    #                     elif previous_status == "sold" and current_status == "unsold":
                            
    #                         return ['background-color: #B22222; color: white'] * len(row)

    #                 if current_status == "sold":
                        
    #                     return ['background-color: #DFFFD6; color: #333333'] * len(row)
    #                 elif current_status == "unsold":
                       
    #                     return ['background-color: #FFD6D6; color: #333333'] * len(row)

    #                 return ['background-color: #FFFFFF; color: #333333'] * len(row)
                
    #             styled_df = comparison_df[["Flat no", "Tower No", "Sold/Unsold"]].style.apply(highlight_rows, axis=1)
    #             st.write(styled_df)
    

    

    

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


def create_context_from_session():
    
    context_parts = []
    session_data = dict(st.session_state)
    print("Session Data", session_data)
    
    context_parts.append("Here is a summary of the available data for analysis:")

    if "step_1_data" in session_data:
        print("Session Data 1",session_data['step_1_data'])
        context_parts.append(f"Sanction letter details {session_data['step_1_data']}")
    
    if 'step_4_data' in session_data:
        context_parts.append("\n=== BANK STATEMENT DATA ===")
        bank_data = session_data['step_4_data']
        context_parts.append(f"Collection Bank Statement Details are: {bank_data['collection_bank_statement']}")
        context_parts.append(f"Corporate Bank Statement Details are: {bank_data['corporate_bank_statement']}")
        context_parts.append(f"Details about units where are ready for noc: {bank_data['units_ready_for_noc']}")
        context_parts.append(f"Details about units where are not ready for noc: {bank_data['units_not_ready_for_noc']}")
        if isinstance(bank_data, dict):
            transaction_count = len(bank_data.get('transactions', []))
            context_parts.append(f"A bank statement has been processed with {transaction_count} transactions.")
    
    
    if 'step_2_data' in session_data:
        context_parts.append("\n=== FINANCIAL DATA (COP-MOF & MIS) ===")
        step2_data = session_data['step_2_data']
        
        if 'COP-MOF' in step2_data:
            context_parts.append("--- COP-MOF (Cost of Project & Means of Finance) Analysis ---")
            cop_data = step2_data['COP-MOF']
            if isinstance(cop_data, dict) and 'df2' in cop_data and isinstance(cop_data['df2'], pd.DataFrame):
                df2 = cop_data['df2']
                context_parts.append(f"Financial statements contain {len(df2)} line items.")
                
                for _, row in df2.head(3).iterrows():
                    context_parts.append(f"  - {row.get('PARTICULARS', 'N/A')}: {row.get('Incurred', 'N/A')}")
        
        if 'MIS' in step2_data:
            context_parts.append("\n--- MIS (Sales) Analysis ---")
            mis_data = step2_data['MIS']
            if isinstance(mis_data, dict) and 'df2' in mis_data and isinstance(mis_data['df2'], pd.DataFrame):
                df2 = mis_data['df2']
                
                if 'Sold/Unsold' in df2.columns:
                    sold_units = df2[df2['Sold/Unsold'].str.lower().str.strip() == 'sold']
                    unsold_units = df2[df2['Sold/Unsold'].str.lower().str.strip() == 'unsold']
                    
                    context_parts.append(f"Total sold units: {len(sold_units)}")
                    context_parts.append(f"Total unsold units: {len(unsold_units)}")
                    
                    if len(sold_units) > 0 and 'Agreement value' in sold_units.columns and 'Amount Receivable' in sold_units.columns:
                        total_agreement = pd.to_numeric(sold_units['Agreement value'], errors='coerce').sum()
                        total_receivable = pd.to_numeric(sold_units['Amount Receivable'], errors='coerce').sum()
                        context_parts.append(f"Total agreement value of sold units: ₹{total_agreement:,.2f}")
                        context_parts.append(f"Total amount receivable from sold units: ₹{total_receivable:,.2f}")
    
    # Section for NOC & Sales Data
    if 'step_3_data' in session_data:
        context_parts.append("\n=== NOC & RECENT SALES DATA ===")
        step3_data = session_data['step_3_data']
        
        if 'noc_data' in step3_data and isinstance(step3_data.get('noc_data'), dict) and 'selected_units' in step3_data['noc_data']:
            noc_units = step3_data['noc_data']['selected_units']
            context_parts.append(f"NOC Analysis: {len(noc_units)} units were selected for analysis.")
            
            # ready_for_noc = 0
            # not_ready_for_noc = 0
            
            # for unit in noc_units:
            #     try:
            #         agreement_value = float(unit.get('agreement_value', 0))
            #         saleable_area = float(unit.get('saleable_area', 0))
            #         # Define a clear threshold; avoid magic numbers
            #         THRESHOLD_PER_SQFT = 27500
            #         threshold_value = THRESHOLD_PER_SQFT * saleable_area
                    
            #         if agreement_value >= threshold_value:
            #             ready_for_noc += 1
            #         else:
            #             not_ready_for_noc += 1
            #     except (ValueError, TypeError):
            #         not_ready_for_noc += 1 # Count as not ready if data is invalid
            
            # context_parts.append(f"Units ready for NOC: {ready_for_noc}")
            # context_parts.append(f"Units not ready for NOC: {not_ready_for_noc}")
        
        if 'recently_sold_flats_by_tower' in step3_data and isinstance(step3_data['recently_sold_flats_by_tower'], dict):
            context_parts.append("\n--- Recent Sales by Tower ---")
            for tower, flats in step3_data['recently_sold_flats_by_tower'].items():
                context_parts.append(f"Tower {tower}: {len(flats)} units recently sold.")
    
    return "\n".join(context_parts)


async def get_ai_response(user_message, context):
    """
    Sends the user message and session context to the Gemini API and returns the response.
    """
    prompt = f"""
    You are an expert financial data analysis assistant for a real estate company.
    Your role is to answer questions based ONLY on the data provided from the user's session.
    Do not make up information or answer questions outside of the provided context.
    If the answer cannot be found in the provided data, state that clearly and politely.
    Basically what we are doing is we upload a previous MIS excel sheet (February) and a
    new MIS excel sheet (March) we basically do a comparitive analysis for real estate units.
    Also we do comparitive analysis with bank statements.
    Recently sold units means units which were sold in new MIS but were unsold in previous MIS.
    Same logic for all other stuff where it mentions recently.
    Remember Received amount  = Agreement Value - Receivable.
    Here is the summary of the user's session data:
    ---
    {context}
    ---

    User's question: "{user_message}"

    Based on the data above, please provide a concise and helpful answer.
    """
    
    try:
        # Prepare the payload for the Gemini API
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }]
        }
        
        
        api_key = os.getenv('GEMINI_API_KEY')
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        # Make the asynchronous API call using httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=45.0)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            result = response.json()

        # Safely extract the text from the API response
        if (result.get("candidates") and 
            result["candidates"][0].get("content") and 
            result["candidates"][0]["content"].get("parts")):
            text = result["candidates"][0]["content"]["parts"][0].get("text")
            return text or "I received a response, but it was empty."
        else:
            # Log the unexpected response for debugging purposes
            print("Unexpected API response structure:", result)
            return "Sorry, I received an unexpected response from the AI. Please try again."

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return f"Sorry, I encountered an error communicating with the AI service (HTTP {e.response.status_code}). Please try again later."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "Sorry, an unexpected error occurred. Please check the logs and try again."

def chatbot_interface():
    """
    Sets up the Streamlit user interface for the chatbot.
    """
    st.subheader("Data Analysis Chatbot")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
   
    # Generate the context from the current session state
    context = create_context_from_session()

    # For debugging: display the context that will be sent to the AI
    with st.expander("View AI Context"):
        st.text(context)
    
    # Container to display the chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            is_user = message["role"] == "user"
            # Apply different styles for user and assistant messages
            style = (
                "background-color: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 10px; margin-left: 20%;" 
                if is_user 
                else "background-color: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 10px; margin-right: 20%;"
            )
            name = "You" if is_user else "Assistant"
            st.markdown(f'<div style="{style}"><strong>{name}:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                
    # Input area for the user
    user_input = st.chat_input("Ask a question about your data...")
    
    if user_input:
        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show a spinner while waiting for the AI response
        with st.spinner("Assistant is thinking..."):
            # Run the async API call
            ai_response = asyncio.run(get_ai_response(user_input, context))
        
        # Add AI response to history
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        # Rerun the app to display the new messages
        st.rerun()


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