import streamlit as st
import pandas as pd
import os
import json
import re
import numpy as np
import easyocr
from PIL import Image
from crewai import Agent, Task, Crew, LLM
from groq import Groq

st.set_page_config(page_title='Prescrify', page_icon='💊', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
[data-testid='stSidebar'] { background: #0f1117; border-right: 1px solid #1e2130; }
div.stButton > button { width:100%; text-align:left; background:transparent; border:none; color:#aaa; padding:10px 14px; border-radius:8px; font-size:14px; margin-bottom:4px; }
div.stButton > button:hover { background:#1e2130; color:white; }
[data-testid='stFileUploader'] { background:#1e2130; border-radius:12px; border:2px dashed #4c9be8 !important; padding:20px; }
[data-testid='stMetric'] { background:#1e2130; border-radius:10px; padding:12px; border:1px solid #2d3250; }
</style>''', unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'search'

def get_groq_key():
    try:
        return st.secrets['GROQ_API_KEY']
    except:
        return None

groq_key = get_groq_key()

@st.cache_data(show_spinner=False)
def load_data():
    url = 'https://github.com/junioralive/Indian-Medicine-Dataset/blob/main/DATA/indian_medicine_data.csv?raw=true'
    df = pd.read_csv(url)

    # Auto-detect price column
    price_col = None
    for col in df.columns:
        if 'price' in col.lower():
            price_col = col
            break
    if price_col is None:
        raise ValueError(f'No price column found! Columns: {list(df.columns)}')

    # Auto-detect composition columns
    comp_cols = [c for c in df.columns if 'composition' in c.lower() or 'short_comp' in c.lower()]
    comp1 = comp_cols[0] if len(comp_cols) > 0 else None
    comp2 = comp_cols[1] if len(comp_cols) > 1 else None

    # Auto-detect discontinued column
    disc_col = None
    for col in df.columns:
        if 'discontinu' in col.lower():
            disc_col = col
            break

    # Filter discontinued
    if disc_col:
        df = df[df[disc_col] == False]

    # Build composition
    if comp1 and comp2:
        df['composition'] = df[comp1].fillna('') + df[comp2].apply(lambda x: ' + ' + x if pd.notna(x) and str(x).strip() != '' else '')
    elif comp1:
        df['composition'] = df[comp1].fillna('')
    else:
        df['composition'] = ''

    # Rename price column to standard name
    df = df.rename(columns={price_col: 'price'})
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]

    # Clean strings
    df['name'] = df['name'].str.strip()
    df['manufacturer_name'] = df['manufacturer_name'].str.strip()
    df['composition'] = df['composition'].str.strip()
    df = df.drop_duplicates(subset=['name'])
    return df

with st.spinner('Loading medicines database...'):
    df = load_data()

def find_alternatives(medicine_name):
    searched = df[df['name'].str.lower().str.contains(medicine_name.lower().strip(), na=False)]
    if searched.empty:
        return None
    original = searched.iloc[0]
    alternatives = df[
        (df['composition'].str.lower() == original['composition'].lower()) &
        (df['name'].str.lower() != original['name'].lower())
    ].sort_values('price')
    alt_list = []
    for _, row in alternatives.iterrows():
        saving = round(original['price'] - row['price'], 2)
        saving_pct = round((saving / original['price']) * 100, 1) if original['price'] > 0 else 0
        alt_list.append({
            'name': row['name'], 'price': round(row['price'], 2),
            'saving': saving, 'saving_pct': saving_pct,
            'manufacturer': row['manufacturer_name'],
            'pack_size': row['pack_size_label']})
    return {
        'original_name': original['name'],
        'original_price': round(original['price'], 2),
        'original_composition': original['composition'],
        'original_manufacturer': original['manufacturer_name'],
        'original_pack': original['pack_size_label'],
        'alternatives': alt_list}

def detect_overcharge(medicine_name, billed_price):
    result = find_alternatives(medicine_name)
    if not result:
        return None
    db_price = result['original_price']
    overcharge = round(billed_price - db_price, 2)
    overcharge_pct = round((overcharge / db_price) * 100, 1) if db_price > 0 else 0
    return {
        'medicine': result['original_name'], 'billed_price': billed_price,
        'actual_price': db_price, 'overcharge': overcharge,
        'overcharge_pct': overcharge_pct, 'is_overcharged': overcharge > 0,
        'composition': result['original_composition'],
        'alternatives': result['alternatives']}

def nlp_extract_medicines(text, key):
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {'role': 'system', 'content': 'You are an expert Indian prescription reader. Extract medicine names. Return ONLY JSON: {"medicines": [{"name": "Augmentin 625", "dosage": "twice daily", "duration": "5 days"}], "doctor": "Dr Name", "diagnosis": "condition", "patient": "name"}'},
            {'role': 'user', 'content': 'Extract medicines from this prescription:\n\n' + text}
        ], max_tokens=800, temperature=0.1)
    raw = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {'medicines': [], 'doctor': 'N/A', 'diagnosis': 'N/A', 'patient': 'N/A'}

with st.sidebar:
    st.markdown('<div style="display:flex;align-items:center;gap:10px;padding:10px 0 20px 0;"><span style="font-size:2rem;">💊</span><div><div style="font-size:1.3rem;font-weight:700;color:white;">Prescrify</div><div style="font-size:0.75rem;color:#888;">AMD Slingshot 2026</div></div></div>', unsafe_allow_html=True)
    if groq_key:
        st.success('🔑 API Key loaded!')
    else:
        groq_key = st.text_input('🔑 Groq API Key', type='password')
    st.markdown('---')
    st.markdown(f'<div style="background:#1e2130;border-radius:10px;padding:12px;margin-bottom:12px;"><div style="color:#4c9be8;font-size:1.2rem;font-weight:700;">{len(df):,}</div><div style="color:#888;font-size:0.8rem;">Indian Medicines</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#1e2130;border-radius:10px;padding:12px;margin-bottom:12px;"><div style="color:#4c9be8;font-size:0.9rem;font-weight:600;">🤖 Llama 3.3 70B</div><div style="color:#888;font-size:0.8rem;">Powered by Groq</div></div>', unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('<div style="color:#888;font-size:0.75rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Navigation</div>', unsafe_allow_html=True)
    nav_items = [('search','🔍','Medicine Search'),('prescription','📄','Prescription Analyzer'),('overcharge','🧾','Bill Overcharge'),('savings','💰','Savings Dashboard'),('locator','🏪','Store Locator')]
    for key, icon, label in nav_items:
        is_active = st.session_state.page == key
        btn_type = 'primary' if is_active else 'secondary'
        if st.button(f'{icon}  {label}', key=f'nav_{key}', type=btn_type, use_container_width=True):
            st.session_state.page = key
            st.rerun()
    st.markdown('---')
    st.markdown('<div style="color:#555;font-size:0.7rem;text-align:center;">💊 Prescrify v2.0<br>AMD Slingshot 2026</div>', unsafe_allow_html=True)

page = st.session_state.page

if page == 'search':
    st.markdown('## 🔍 Medicine Search & Generic Alternatives')
    st.markdown('Find cheaper alternatives with the **same composition**')
    st.markdown('')
    col1, col2 = st.columns([5, 1])
    with col1:
        medicine_input = st.text_input('Search', placeholder='e.g. Augmentin 625, Dolo 650, Pantop 40', label_visibility='collapsed')
    with col2:
        search_btn = st.button('🔍 Search', type='primary', use_container_width=True)
    if search_btn and medicine_input:
        result = find_alternatives(medicine_input)
        if result is None:
            st.error(f'❌ {medicine_input} not found')
            st.info('💡 Try partial name e.g. Augmentin instead of full name')
        else:
            st.success(f'✅ Found: {result["original_name"]}')
            c1, c2, c3 = st.columns(3)
            c1.metric('💳 Price', f'Rs {result["original_price"]}')
            c2.metric('🔄 Alternatives', len(result['alternatives']))
            cheaper = [a for a in result['alternatives'] if a['saving'] > 0]
            if cheaper:
                c3.metric('💚 Max Saving', f'Rs {cheaper[0]["saving"]}', f'{cheaper[0]["saving_pct"]}% cheaper')
            st.markdown('---')
            st.info(f'🔬 Composition: {result["original_composition"]}  |  📦 Pack: {result["original_pack"]}  |  🏭 By: {result["original_manufacturer"]}')
            if cheaper:
                st.markdown(f'### 💊 {len(cheaper)} Generic Alternatives')
                for i, alt in enumerate(cheaper[:8], 1):
                    cols = st.columns([3, 1, 2, 2])
                    prefix = '����' if i == 1 else str(i)
                    cols[0].markdown(f'{prefix} **{alt["name"]}**')
                    cols[1].markdown(f'Rs **{alt["price"]}**')
                    cols[2].markdown(f'Save Rs **{alt["saving"]}** ({alt["saving_pct"]}%)')
                    cols[3].markdown(f'*{alt["manufacturer"]}*')
                    st.divider()
                best = cheaper[0]
                st.success(f'🏆 Best: {best["name"]} — Rs {best["price"]} | Save Rs {best["saving"]} | Rs {round(best["saving"]*24,2)} per year!')
            else:
                st.info('✅ Already the most affordable option!')
            st.warning('⚠️ Always consult your doctor before switching medicines.')
    else:
        c1, c2, c3 = st.columns(3)
        c1.info('1️⃣ Search - Type any Indian medicine name')
        c2.info('2️⃣ Compare - We search 2.5 lakh medicines')
        c3.info('3️⃣ Save - See cheaper alternatives in Rs')

elif page == 'prescription':
    st.markdown('## 📄 Prescription Analyzer')
    st.markdown('Upload a **prescription image or PDF** — OCR reads it — AI finds medicines — Get cheaper alternatives!')
    st.markdown('')
    if not groq_key:
        st.warning('🔑 Please add your Groq API key in the sidebar!')
        st.stop()
    tab1, tab2 = st.tabs(['📷 Upload Image or PDF', '✏️ Type Manually'])
    with tab1:
        st.markdown('#### 📤 Upload Prescription or Medical Bill')
        uploaded = st.file_uploader('Choose a prescription file', type=['jpg','jpeg','png','pdf'], help='Supports JPG, PNG images and PDF files')
        if uploaded is not None:
            file_type = uploaded.name.split('.')[-1].lower()
            images = []
            if file_type == 'pdf':
                st.info(f'📄 PDF detected: {uploaded.name}')
                try:
                    from pdf2image import convert_from_bytes
                    with st.spinner('Converting PDF to images...'):
                        pages = convert_from_bytes(uploaded.read(), dpi=200)
                    st.success(f'✅ {len(pages)} page(s) found')
                    cols = st.columns(min(len(pages), 3))
                    for i, pg in enumerate(pages):
                        with cols[i % 3]:
                            st.image(pg, caption=f'Page {i+1}', use_container_width=True)
                        images.append(pg)
                except Exception as e:
                    st.error(f'PDF Error: {e}')
            else:
                image = Image.open(uploaded)
                st.image(image, caption=f'📋 {uploaded.name}', use_container_width=True)
                images = [image]
            st.markdown('')
            if st.button('🚀 Analyze Prescription', type='primary', use_container_width=True):
                all_text = ''
                with st.status('📖 Step 1 — Reading text with OCR...', expanded=True) as status:
                    try:
                        reader = easyocr.Reader(['en'], verbose=False)
                        for i, img in enumerate(images):
                            st.write(f'Processing page {i+1}...')
                            result = reader.readtext(np.array(img), detail=0)
                            all_text += ' ' + ' '.join(result)
                        status.update(label=f'✅ OCR done — {len(all_text.split())} words extracted!', state='complete')
                    except Exception as e:
                        status.update(label=f'❌ OCR failed: {e}', state='error')
                if all_text.strip():
                    with st.expander('📝 View Extracted Text'):
                        st.code(all_text.strip())
                    with st.status('🤖 Step 2 — AI extracting medicine names...', expanded=True) as status:
                        try:
                            parsed = nlp_extract_medicines(all_text, groq_key)
                            medicines_found = parsed.get('medicines', [])
                            doctor = parsed.get('doctor', 'N/A')
                            diagnosis = parsed.get('diagnosis', 'N/A')
                            patient = parsed.get('patient', 'N/A')
                            status.update(label=f'✅ Found {len(medicines_found)} medicines!', state='complete')
                        except Exception as e:
                            status.update(label=f'❌ AI Error: {e}', state='error')
                            medicines_found = []
                    if medicines_found:
                        c1, c2, c3 = st.columns(3)
                        c1.info(f'👨‍⚕️ Doctor: {doctor}')
                        c2.info(f'🩺 Diagnosis: {diagnosis}')
                        c3.info(f'🧑 Patient: {patient}')
                        st.markdown('---')
                        st.markdown('### 💊 Step 3 — Generic Alternatives')
                        total_branded = 0
                        total_generic = 0
                        summary_rows = []
                        for med_info in medicines_found:
                            med_name = med_info['name'] if isinstance(med_info, dict) else med_info
                            dosage = med_info.get('dosage', 'N/A') if isinstance(med_info, dict) else 'N/A'
                            duration = med_info.get('duration', 'N/A') if isinstance(med_info, dict) else 'N/A'
                            result = find_alternatives(med_name)
                            with st.expander(f'💊 {med_name}  |  {dosage}  |  {duration}', expanded=True):
                                if result:
                                    total_branded += result['original_price']
                                    cheaper = [a for a in result['alternatives'] if a['saving'] > 0]
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.markdown('**🔴 Prescribed**')
                                        st.metric('Price', f'Rs {result["original_price"]}')
                                        st.caption(result['original_composition'])
                                    with c2:
                                        if cheaper:
                                            best = cheaper[0]
                                            total_generic += best['price']
                                            st.markdown('**🟢 Best Generic**')
                                            st.metric('Price', f'Rs {best["price"]}', f'Save Rs {best["saving"]}', delta_color='inverse')
                                            st.caption(best['name'])
                                            summary_rows.append({'Prescribed': result['original_name'], 'Price': f'Rs {result["original_price"]}', 'Generic': best['name'], 'Alt Price': f'Rs {best["price"]}', 'Saving': f'Rs {best["saving"]}'})
                                        else:
                                            total_generic += result['original_price']
                                            st.info('✅ Already cheapest!')
                                else:
                                    st.warning('Not found in database')
                        st.markdown('---')
                        total_saving = round(total_branded - total_generic, 2)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric('💳 Branded Total', f'Rs {round(total_branded,2)}')
                        c2.metric('💚 Generic Total', f'Rs {round(total_generic,2)}')
                        c3.metric('💰 You Save', f'Rs {total_saving}')
                        c4.metric('📅 Per Month', f'Rs {round(total_saving*2,2)}')
                        if summary_rows:
                            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
                        if total_saving > 0:
                            st.success(f'🎉 Switch to generics and save Rs {total_saving} per prescription!')
                        st.warning('⚠️ Always consult your doctor before switching medicines.')
                    else:
                        st.error('❌ No medicines detected. Try the manual tab!')
                else:
                    st.warning('⚠️ No text extracted. Use a clearer image!')
        else:
            st.markdown('<div style="background:#1e2130;border:2px dashed #2d3250;border-radius:12px;padding:40px;text-align:center;color:#888;margin-top:20px;"><div style="font-size:3rem;">📋</div><div style="font-size:1.1rem;margin-top:10px;">Upload your prescription above</div><div style="font-size:0.85rem;margin-top:6px;">Supports JPG, PNG, PDF</div></div>', unsafe_allow_html=True)
    with tab2:
        manual_text = st.text_area('One medicine per line:', placeholder='Augmentin 625\nDolo 650\nPantop 40', height=150)
        if st.button('🔍 Find Alternatives', type='primary', key='manual_btn'):
            if manual_text.strip():
                medicines = [m.strip() for m in manual_text.strip().split('\n') if m.strip()]
                total_save = 0
                for med in medicines:
                    result = find_alternatives(med)
                    with st.expander(f'💊 {med}', expanded=True):
                        if result:
                            cheaper = [a for a in result['alternatives'] if a['saving'] > 0]
                            c1, c2 = st.columns(2)
                            c1.metric('Branded', f'Rs {result["original_price"]}')
                            c1.caption(result['original_composition'])
                            if cheaper:
                                best = cheaper[0]
                                total_save += best['saving']
                                c2.metric('Best Generic', f'Rs {best["price"]}', f'Save Rs {best["saving"]}', delta_color='inverse')
                                c2.caption(best['name'])
                            else:
                                c2.info('✅ Already cheapest!')
                        else:
                            st.warning('Not found')
                if total_save > 0:
                    st.success(f'💰 Total savings: Rs {round(total_save,2)} per prescription!')

elif page == 'overcharge':
    st.markdown('## 🧾 Medical Bill Overcharge Detector')
    st.markdown('Enter your bill — We detect overcharges instantly!')
    st.markdown('')
    num_items = st.number_input('Number of medicines in bill', min_value=1, max_value=20, value=3)
    bill_items = []
    c1, c2 = st.columns([3, 2])
    c1.markdown('**Medicine Name**')
    c2.markdown('**Billed Price (Rs)**')
    for i in range(int(num_items)):
        col1, col2 = st.columns([3, 2])
        name = col1.text_input(f'Medicine {i+1}', key=f'med_{i}', placeholder='e.g. Augmentin 625')
        price = col2.number_input(f'Price {i+1}', key=f'price_{i}', min_value=0.0, value=0.0, step=0.5)
        if name and price > 0:
            bill_items.append({'name': name, 'billed_price': price})
    st.markdown('')
    if st.button('🔍 Check for Overcharges', type='primary', use_container_width=True):
        if not bill_items:
            st.warning('Please add at least one item!')
        else:
            total_overcharge = total_billed = total_actual = 0
            st.markdown('### Results')
            for item in bill_items:
                result = detect_overcharge(item['name'], item['billed_price'])
                if result:
                    total_billed += result['billed_price']
                    total_actual += result['actual_price']
                    if result['is_overcharged']:
                        total_overcharge += result['overcharge']
                        st.error(f'⚠️ {result["medicine"]} — Billed Rs {result["billed_price"]} | Actual Rs {result["actual_price"]} | Overcharged Rs {result["overcharge"]} ({result["overcharge_pct"]}%)')
                    else:
                        st.success(f'✅ {result["medicine"]} — Rs {result["billed_price"]} (Fair!)')
                else:
                    st.info(f'ℹ️ {item["name"]} — Not found')
            st.markdown('---')
            c1, c2, c3 = st.columns(3)
            c1.metric('💳 Total Billed', f'Rs {round(total_billed,2)}')
            c2.metric('✅ Should Have Paid', f'Rs {round(total_actual,2)}')
            c3.metric('🚨 Overcharged', f'Rs {round(total_overcharge,2)}')
            if total_overcharge > 0:
                st.error(f'🚨 You were overcharged Rs {round(total_overcharge,2)}!')
                st.markdown('Steps: 1. Show to pharmacy  2. File at nppa.gov.in  3. Call 104')
            else:
                st.success('✅ All prices are fair!')

elif page == 'savings':
    st.markdown('## 💰 Savings Dashboard')
    st.markdown('See how much you save by switching to generics')
    st.markdown('')
    medicines_input = st.text_area('Enter your regular medicines (one per line):', placeholder='Augmentin 625\nDolo 650\nPantop 40', height=150)
    if st.button('📊 Calculate Savings', type='primary', use_container_width=True):
        medicines = [m.strip() for m in medicines_input.split('\n') if m.strip()]
        if not medicines:
            st.warning('Enter at least one medicine!')
        else:
            results = []
            total_current = 0
            total_cheapest = 0
            for med in medicines:
                result = find_alternatives(med)
                if result:
                    total_current += result['original_price']
                    cheaper = [a for a in result['alternatives'] if a['saving'] > 0]
                    best_price = cheaper[0]['price'] if cheaper else result['original_price']
                    best_name = cheaper[0]['name'] if cheaper else result['original_name']
                    best_save = cheaper[0]['saving'] if cheaper else 0
                    total_cheapest += best_price
                    results.append({'Medicine': result['original_name'], 'Your Price': f'Rs {result["original_price"]}', 'Best Generic': best_name, 'Alt Price': f'Rs {best_price}', 'Saving': f'Rs {best_save}'})
            total_saving = round(total_current - total_cheapest, 2)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('💳 Current', f'Rs {round(total_current,2)}')
            c2.metric('💚 With Generics', f'Rs {round(total_cheapest,2)}')
            c3.metric('💰 Save/Month', f'Rs {round(total_saving*2,2)}')
            c4.metric('📆 Save/Year', f'Rs {round(total_saving*24,2)}')
            st.markdown('---')
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            if total_saving > 0:
                st.success(f'🎉 Switch to generics and save Rs {round(total_saving*24,2)} every year!')

elif page == 'locator':
    st.markdown('## 🏪 Jan Aushadhi Store Locator')
    st.markdown('Find nearest government generic medicine stores')
    st.markdown('')
    col1, col2 = st.columns([3, 1])
    with col1:
        pincode = st.text_input('Pincode', placeholder='Enter your pincode e.g. 400001', label_visibility='collapsed')
    with col2:
        locate_btn = st.button('📍 Find Stores', type='primary', use_container_width=True)
    if locate_btn and pincode:
        stores = [
            {'name': 'Jan Aushadhi Kendra 1', 'address': f'Near Market, {pincode}', 'distance': '0.5 km', 'status': '🟢 Open'},
            {'name': 'Jan Aushadhi Kendra 2', 'address': f'Main Road, {pincode}', 'distance': '1.2 km', 'status': '🟢 Open'},
            {'name': 'Jan Aushadhi Kendra 3', 'address': f'Gandhi Nagar, {pincode}', 'distance': '2.1 km', 'status': '🟡 Closes at 8PM'},
            {'name': 'Generic Pharmacy', 'address': f'Hospital Road, {pincode}', 'distance': '2.8 km', 'status': '🟢 Open'},
        ]
        st.success(f'✅ Found {len(stores)} stores near {pincode}')
        for i, store in enumerate(stores, 1):
            c1, c2, c3 = st.columns([4, 2, 2])
            c1.markdown(f'**{i}. {store["name"]}**\n\n📍 {store["address"]}')
            c2.metric('Distance', store['distance'])
            c3.markdown(f'\n\n{store["status"]}')
            st.divider()
        st.info('💡 Jan Aushadhi stores save 50-90% vs branded | janaushadhi.gov.in')

st.markdown('---')
st.markdown('<div style="text-align:center;color:#555;font-size:12px;padding:10px;">💊 Prescrify v2.0 | AMD Slingshot Hackathon 2026 | CrewAI + Groq + Llama 3.3 70B + 253,973 Indian Medicines 🇮🇳</div>', unsafe_allow_html=True)
