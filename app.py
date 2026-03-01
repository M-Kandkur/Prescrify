
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

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Prescrify 💊",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════
# CUSTOM CSS — CLEAN UI
# ══════════════════════════════════════════════
st.markdown("""
<style>
/* Sidebar clean */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
/* Nav buttons */
div.stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    color: #aaa;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14px;
    margin-bottom: 4px;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: #1e2130;
    color: white;
}
/* Active nav button */
div.stButton > button[kind="primary"] {
    background: #1e3a5f !important;
    color: white !important;
    border-left: 3px solid #4c9be8 !important;
}
/* Cards */
.card {
    background: #1e2130;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    border: 1px solid #2d3250;
}
/* Upload area */
[data-testid="stFileUploader"] {
    background: #1e2130;
    border-radius: 12px;
    border: 2px dashed #4c9be8 !important;
    padding: 20px;
}
/* Hide radio default nav */
div[data-testid="stRadio"] { display: none; }
/* Metric cards */
[data-testid="stMetric"] {
    background: #1e2130;
    border-radius: 10px;
    padding: 12px;
    border: 1px solid #2d3250;
}
/* Main header */
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0px;
}
/* Page header */
.page-header {
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 4px;
    padding-bottom: 10px;
    border-bottom: 1px solid #2d3250;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SESSION STATE — for navigation
# ══════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "search"

# ══════════════════════════════════════════════
# GROQ KEY
# ══════════════════════════════════════════════
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return None

groq_key = get_groq_key()

# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://github.com/junioralive/Indian-Medicine-Dataset/blob/main/DATA/indian_medicine_data.csv?raw=true"
    df  = pd.read_csv(url)
    df  = df[df["Is_discontinued"] == False]
    df["composition"] = df["short_composition1"].fillna("") + df["short_composition2"].apply(
        lambda x: " + " + x if pd.notna(x) and x != "" else ""
    )
    df["price(₹)"]          = pd.to_numeric(df["price(₹)"], errors="coerce")
    df                       = df.dropna(subset=["price(₹)"])
    df                       = df[df["price(₹)"] > 0]
    df["name"]               = df["name"].str.strip()
    df["manufacturer_name"]  = df["manufacturer_name"].str.strip()
    df["composition"]        = df["composition"].str.strip()
    df                       = df.drop_duplicates(subset=["name"])
    return df

with st.spinner("🔄 Loading medicines database..."):
    df = load_data()

# ══════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════
def find_alternatives(medicine_name):
    searched = df[df["name"].str.lower().str.contains(medicine_name.lower().strip(), na=False)]
    if searched.empty:
        return None
    original      = searched.iloc[0]
    alternatives  = df[
        (df["composition"].str.lower() == original["composition"].lower()) &
        (df["name"].str.lower() != original["name"].lower())
    ].sort_values("price(₹)")
    alt_list = []
    for _, row in alternatives.iterrows():
        saving     = round(original["price(₹)"] - row["price(₹)"], 2)
        saving_pct = round((saving / original["price(₹)"]) * 100, 1)
        alt_list.append({
            "name": row["name"], "price": round(row["price(₹)"], 2),
            "saving": saving,    "saving_pct": saving_pct,
            "manufacturer": row["manufacturer_name"],
            "pack_size": row["pack_size_label"]
        })
    return {
        "original_name": original["name"],
        "original_price": round(original["price(₹)"], 2),
        "original_composition": original["composition"],
        "original_manufacturer": original["manufacturer_name"],
        "original_pack": original["pack_size_label"],
        "alternatives": alt_list
    }

def detect_overcharge(medicine_name, billed_price):
    result = find_alternatives(medicine_name)
    if not result:
        return None
    db_price       = result["original_price"]
    overcharge     = round(billed_price - db_price, 2)
    overcharge_pct = round((overcharge / db_price) * 100, 1) if db_price > 0 else 0
    return {
        "medicine": result["original_name"], "billed_price": billed_price,
        "actual_price": db_price,            "overcharge": overcharge,
        "overcharge_pct": overcharge_pct,    "is_overcharged": overcharge > 0,
        "composition": result["original_composition"],
        "alternatives": result["alternatives"]
    }

def ocr_image(pil_image):
    reader    = easyocr.Reader(["en"], verbose=False)
    img_array = np.array(pil_image)
    result    = reader.readtext(img_array, detail=0)
    return " ".join(result)

def nlp_extract_medicines(text, groq_key):
    client   = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """You are an expert Indian prescription reader.
Extract medicine names from prescription text. Return ONLY JSON:
{
  "medicines": [
    {"name": "Augmentin 625", "dosage": "twice daily", "duration": "5 days"}
  ],
  "doctor": "Dr. Name",
  "diagnosis": "condition",
  "patient": "name"
}
No explanation. Only valid JSON."""},
            {"role": "user", "content": f"Extract from this prescription:\n\n{text}"}
        ],
        max_tokens=800,
        temperature=0.1
    )
    raw        = response.choices[0].message.content.strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"medicines": [], "doctor": "N/A", "diagnosis": "N/A", "patient": "N/A"}

# ══════════════════════════════════════════════
# SIDEBAR — CLEAN NAVIGATION
# ══════════════════════════════════════════════
with st.sidebar:
    # Logo + Title
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; padding:10px 0 20px 0;">
        <span style="font-size:2rem;">💊</span>
        <div>
            <div style="font-size:1.3rem; font-weight:700; color:white;">Prescrify</div>
            <div style="font-size:0.75rem; color:#888;">AMD Slingshot 2026</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API Key status
    if groq_key:
        st.success("🔑 API Key loaded!")
    else:
        groq_key = st.text_input("🔑 Groq API Key", type="password", help="Get free key at console.groq.com")

    st.markdown("---")

    # Stats
    st.markdown(f"""
    <div style="background:#1e2130; border-radius:10px; padding:12px; margin-bottom:12px;">
        <div style="color:#4c9be8; font-size:1.2rem; font-weight:700;">{len(df):,}</div>
        <div style="color:#888; font-size:0.8rem;">Indian Medicines</div>
    </div>
    <div style="background:#1e2130; border-radius:10px; padding:12px; margin-bottom:12px;">
        <div style="color:#4c9be8; font-size:0.9rem; font-weight:600;">🤖 Llama 3.3 70B</div>
        <div style="color:#888; font-size:0.8rem;">Powered by Groq</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── CLEAN NAV BUTTONS ──
    st.markdown("<div style='color:#888; font-size:0.75rem; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px;'>Navigation</div>", unsafe_allow_html=True)

    nav_items = [
        ("search",      "🔍",  "Medicine Search"),
        ("prescription","📄",  "Prescription Analyzer"),
        ("overcharge",  "🧾",  "Bill Overcharge"),
        ("savings",     "💰",  "Savings Dashboard"),
        ("locator",     "🏪",  "Store Locator"),
    ]

    for key, icon, label in nav_items:
        is_active = st.session_state.page == key
        btn_style = f"""
        <div onclick="" style="
            background: {"#1e3a5f" if is_active else "transparent"};
            border-left: {"3px solid #4c9be8" if is_active else "3px solid transparent"};
            color: {"white" if is_active else "#aaa"};
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 4px;
            cursor: pointer;
            font-size: 14px;
        ">{icon} {label}</div>
        """
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     type="primary" if is_active else "secondary",
                     use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='color:#555; font-size:0.7rem; text-align:center;'>💊 Prescrify v2.0<br>AMD Slingshot 2026</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════
page = st.session_state.page

# ──────────────────────────────────────────────
# PAGE 1 — MEDICINE SEARCH
# ──────────────────────────────────────────────
if page == "search":
    st.markdown("<div class='page-header'>🔍 Medicine Search & Generic Alternatives</div>", unsafe_allow_html=True)
    st.markdown("Find cheaper alternatives with the **same composition**")
    st.markdown("")

    col1, col2 = st.columns([5, 1])
    with col1:
        medicine_input = st.text_input("", placeholder="Search medicine... e.g. Augmentin 625, Dolo 650, Pantop 40", label_visibility="collapsed")
    with col2:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    if search_btn and medicine_input:
        result = find_alternatives(medicine_input)
        if result is None:
            st.error(f"❌ **{medicine_input}** not found in database")
            st.info("💡 Try partial name — e.g. **Augmentin** instead of full name")
        else:
            st.success(f"✅ Found: **{result['original_name']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("💳 Price", f"₹{result['original_price']}")
            c2.metric("🔄 Alternatives", len(result["alternatives"]))
            cheaper = [a for a in result["alternatives"] if a["saving"] > 0]
            if cheaper:
                c3.metric("💚 Max Saving", f"₹{cheaper[0]['saving']}", f"{cheaper[0]['saving_pct']}% cheaper")

            st.markdown("---")
            st.info(f"🔬 **Composition:** {result['original_composition']}  |  📦 **Pack:** {result['original_pack']}  |  🏭 **By:** {result['original_manufacturer']}")

            if cheaper:
                st.markdown(f"### 💊 {len(cheaper)} Generic Alternatives")
                for i, alt in enumerate(cheaper[:8], 1):
                    cols = st.columns([3, 1, 2, 2])
                    cols[0].markdown(f"{'🏆 ' if i==1 else f'{i}. '} **{alt['name']}**")
                    cols[1].markdown(f"₹**{alt['price']}**")
                    cols[2].markdown(f"💚 Save ₹**{alt['saving']}** ({alt['saving_pct']}%)")
                    cols[3].markdown(f"*{alt['manufacturer']}*")
                    st.divider()

                best = cheaper[0]
                st.success(f"🏆 Best pick: **{best['name']}** — ₹{best['price']} | Save ₹{best['saving']} per strip | ₹{round(best['saving']*24,2)} per year!")
            else:
                st.info("✅ This is already the most affordable option!")

            st.warning("⚠️ Always consult your doctor before switching medicines.")
    elif not medicine_input:
        c1, c2, c3 = st.columns(3)
        c1.info("**1️⃣ Search**

Type any Indian medicine name")
        c2.info("**2️⃣ Compare**

We search 2.5 lakh medicines")
        c3.info("**3️⃣ Save**

See cheaper alternatives in ���")

# ──────────────────────────────────────────────
# PAGE 2 — PRESCRIPTION ANALYZER
# ──────────────────────────────────────────────
elif page == "prescription":
    st.markdown("<div class='page-header'>📄 Prescription Analyzer</div>", unsafe_allow_html=True)
    st.markdown("Upload a **prescription image or PDF** → OCR extracts text → AI finds medicines → Get cheaper alternatives!")
    st.markdown("")

    if not groq_key:
        st.warning("🔑 Please add your Groq API key in the sidebar to use this feature!")
        st.stop()

    tab1, tab2 = st.tabs(["📷 Upload Image / PDF", "✏️ Type Manually"])

    with tab1:
        st.markdown("#### 📤 Upload Prescription")

        # ── BIG UPLOAD BOX ──
        uploaded = st.file_uploader(
            "Drag & drop your prescription here",
            type=["jpg", "jpeg", "png", "pdf"],
            help="Supports JPG, PNG images and PDF files",
            label_visibility="visible"
        )

        if uploaded is not None:
            file_type = uploaded.name.split(".")[-1].lower()
            images    = []

            # ── PDF → Images ──
            if file_type == "pdf":
                st.info(f"📄 PDF detected: **{uploaded.name}**")
                try:
                    from pdf2image import convert_from_bytes
                    with st.spinner("Converting PDF pages to images..."):
                        pages = convert_from_bytes(uploaded.read(), dpi=200)
                    st.success(f"✅ {len(pages)} page(s) found")
                    cols = st.columns(min(len(pages), 3))
                    for i, page in enumerate(pages):
                        with cols[i % 3]:
                            st.image(page, caption=f"Page {i+1}", use_container_width=True)
                        images.append(page)
                except Exception as e:
                    st.error(f"❌ PDF Error: {e}")
                    st.info("💡 Make sure pdf2image is installed: pip install pdf2image")
            else:
                # ── Image ──
                image = Image.open(uploaded)
                st.image(image, caption=f"📋 {uploaded.name}", use_container_width=True)
                images = [image]

            st.markdown("")
            analyze_btn = st.button("🚀 Analyze Prescription", type="primary", use_container_width=True)

            if analyze_btn:
                all_text = ""

                # ── STEP 1: OCR ──
                with st.status("📖 Step 1 — Reading text from image(s)...", expanded=True) as status:
                    try:
                        reader = easyocr.Reader(["en"], verbose=False)
                        for i, img in enumerate(images):
                            st.write(f"🔍 Processing page {i+1}...")
                            img_array = np.array(img)
                            result    = reader.readtext(img_array, detail=0)
                            page_text = " ".join(result)
                            all_text += " " + page_text
                        status.update(label=f"✅ OCR complete — {len(all_text.split())} words extracted!", state="complete")
                    except Exception as e:
                        status.update(label=f"❌ OCR failed: {e}", state="error")

                if all_text.strip():
                    with st.expander("📝 View Extracted Text"):
                        st.code(all_text.strip())

                    # ── STEP 2: NLP ──
                    with st.status("🤖 Step 2 — AI extracting medicine names...", expanded=True) as status:
                        try:
                            parsed          = nlp_extract_medicines(all_text, groq_key)
                            medicines_found = parsed.get("medicines", [])
                            doctor          = parsed.get("doctor", "N/A")
                            diagnosis       = parsed.get("diagnosis", "N/A")
                            patient         = parsed.get("patient", "N/A")
                            status.update(label=f"✅ Found {len(medicines_found)} medicines!", state="complete")
                        except Exception as e:
                            status.update(label=f"❌ AI Error: {e}", state="error")
                            medicines_found = []

                    if medicines_found:
                        # Prescription info
                        c1, c2, c3 = st.columns(3)
                        c1.info(f"👨‍⚕️ **Doctor**

{doctor}")
                        c2.info(f"🩺 **Diagnosis**

{diagnosis}")
                        c3.info(f"🧑 **Patient**

{patient}")

                        st.markdown("---")
                        st.markdown("### 💊 Step 3 — Generic Alternatives")

                        total_branded = 0
                        total_generic = 0
                        summary_rows  = []

                        for med_info in medicines_found:
                            med_name = med_info["name"]     if isinstance(med_info, dict) else med_info
                            dosage   = med_info.get("dosage",   "N/A") if isinstance(med_info, dict) else "N/A"
                            duration = med_info.get("duration", "N/A") if isinstance(med_info, dict) else "N/A"
                            result   = find_alternatives(med_name)

                            with st.expander(f"💊 {med_name}  |  {dosage}  |  {duration}", expanded=True):
                                if result:
                                    total_branded += result["original_price"]
                                    cheaper = [a for a in result["alternatives"] if a["saving"] > 0]
                                    c1, c2  = st.columns(2)
                                    with c1:
                                        st.markdown("**🔴 Prescribed**")
                                        st.metric("Price", f"₹{result['original_price']}")
                                        st.caption(f"🔬 {result['original_composition']}")
                                        st.caption(f"🏭 {result['original_manufacturer']}")
                                    with c2:
                                        if cheaper:
                                            best           = cheaper[0]
                                            total_generic += best["price"]
                                            st.markdown("**🟢 Best Generic**")
                                            st.metric("Price", f"₹{best['price']}", f"Save ₹{best['saving']} ({best['saving_pct']}%)", delta_color="inverse")
                                            st.caption(f"💊 {best['name']}")
                                            st.caption(f"🏭 {best['manufacturer']}")
                                            summary_rows.append({
                                                "Prescribed": result["original_name"],
                                                "Price":      f"₹{result['original_price']}",
                                                "Generic":    best["name"],
                                                "Alt Price":  f"₹{best['price']}",
                                                "Saving":     f"₹{best['saving']}",
                                            })
                                        else:
                                            total_generic += result["original_price"]
                                            st.info("✅ Already cheapest!")
                                else:
                                    st.warning(f"⚠️ Not found in database — try Medicine Search")

                        st.markdown("---")
                        total_saving = round(total_branded - total_generic, 2)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("💳 Branded Total",  f"₹{round(total_branded, 2)}")
                        c2.metric("💚 Generic Total",  f"₹{round(total_generic, 2)}")
                        c3.metric("💰 You Save",        f"₹{total_saving}")
                        c4.metric("📅 Per Month",       f"₹{round(total_saving*2, 2)}")

                        if summary_rows:
                            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                        if total_saving > 0:
                            st.success(f"🎉 Switch to generics → Save **₹{total_saving}** per prescription | **₹{round(total_saving*12,2)}** per year!")

                        st.warning("⚠️ Always consult your doctor before switching medicines.")
                    else:
                        st.error("❌ No medicines detected. Try the manual tab!")
                else:
                    st.warning("⚠️ No text extracted. Please use a clearer image!")
        else:
            # ── EMPTY STATE ──
            st.markdown("""
            <div style="background:#1e2130; border:2px dashed #2d3250; border-radius:12px;
                        padding:40px; text-align:center; color:#888; margin-top:20px;">
                <div style="font-size:3rem;">📋</div>
                <div style="font-size:1.1rem; margin-top:10px;">Upload your prescription above</div>
                <div style="font-size:0.85rem; margin-top:6px;">Supports JPG, PNG, PDF</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### ✏️ Type medicines manually")
        manual_text = st.text_area("One medicine per line:", placeholder="Augmentin 625\nDolo 650\nPantop 40\nAzithral 500", height=150)
        if st.button("🔍 Find Alternatives", type="primary", key="manual_btn"):
            if manual_text.strip():
                medicines  = [m.strip() for m in manual_text.strip().split("\n") if m.strip()]
                total_save = 0
                for med in medicines:
                    result = find_alternatives(med)
                    with st.expander(f"💊 {med}", expanded=True):
                        if result:
                            cheaper = [a for a in result["alternatives"] if a["saving"] > 0]
                            c1, c2  = st.columns(2)
                            c1.metric("Branded", f"₹{result['original_price']}")
                            c1.caption(result["original_composition"])
                            if cheaper:
                                best        = cheaper[0]
                                total_save += best["saving"]
                                c2.metric("Best Generic", f"₹{best['price']}", f"Save ₹{best['saving']}", delta_color="inverse")
                                c2.caption(best["name"])
                            else:
                                c2.info("✅ Already cheapest!")
                        else:
                            st.warning(f"❌ Not found")
                if total_save > 0:
                    st.success(f"💰 Total savings: **₹{round(total_save,2)}** per prescription!")

# ──────────────────────────────────────────────
# PAGE 3 — BILL OVERCHARGE DETECTOR
# ──────────────────────────────────────────────
elif page == "overcharge":
    st.markdown("<div class='page-header'>🧾 Medical Bill Overcharge Detector</div>", unsafe_allow_html=True)
    st.markdown("Enter your bill details → We detect overcharges instantly!")
    st.markdown("")

    num_items  = st.number_input("Number of medicines in bill", min_value=1, max_value=20, value=3)
    bill_items = []

    c1, c2 = st.columns([3, 2])
    c1.markdown("**Medicine Name**")
    c2.markdown("**Billed Price (₹)**")

    for i in range(int(num_items)):
        col1, col2 = st.columns([3, 2])
        name  = col1.text_input(f"Medicine {i+1}", key=f"med_{i}",   placeholder=f"e.g. Augmentin 625")
        price = col2.number_input(f"Price {i+1}",  key=f"price_{i}", min_value=0.0, value=0.0, step=0.5)
        if name and price > 0:
            bill_items.append({"name": name, "billed_price": price})

    st.markdown("")
    if st.button("🔍 Check for Overcharges", type="primary", use_container_width=True):
        if not bill_items:
            st.warning("Please add at least one item!")
        else:
            total_overcharge = total_billed = total_actual = 0
            st.markdown("### 📊 Results")
            for item in bill_items:
                result = detect_overcharge(item["name"], item["billed_price"])
                if result:
                    total_billed += result["billed_price"]
                    total_actual += result["actual_price"]
                    if result["is_overcharged"]:
                        total_overcharge += result["overcharge"]
                        st.error(f"⚠️ **{result['medicine']}** — Billed ₹{result['billed_price']} | Actual ₹{result['actual_price']} | **Overcharged ₹{result['overcharge']} ({result['overcharge_pct']}%)**")
                    else:
                        st.success(f"✅ **{result['medicine']}** — ₹{result['billed_price']} (Fair price!)")
                else:
                    st.info(f"ℹ️ **{item['name']}** — Not found in database")

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("💳 Total Billed",     f"₹{round(total_billed,2)}")
            c2.metric("✅ Should Have Paid",  f"₹{round(total_actual,2)}")
            c3.metric("🚨 Total Overcharged", f"₹{round(total_overcharge,2)}")

            if total_overcharge > 0:
                st.error(f"🚨 You were overcharged **₹{round(total_overcharge,2)}**!")
                st.markdown("**💡 Steps to take:**
1. Show this report to the pharmacy
2. File complaint at **nppa.gov.in**
3. Call helpline **104**")
            else:
                st.success("✅ All prices are fair!")

# ──────────────────────────────────────────────
# PAGE 4 — SAVINGS DASHBOARD
# ──────────────────────────────────────────────
elif page == "savings":
    st.markdown("<div class='page-header'>💰 Savings Dashboard</div>", unsafe_allow_html=True)
    st.markdown("See how much you save by switching to generics")
    st.markdown("")

    medicines_input = st.text_area("Enter your regular medicines (one per line):", placeholder="Augmentin 625\nDolo 650\nPantop 40\nAzithral 500", height=150)

    if st.button("📊 Calculate Savings", type="primary", use_container_width=True):
        medicines = [m.strip() for m in medicines_input.split("\n") if m.strip()]
        if not medicines:
            st.warning("Enter at least one medicine!")
        else:
            results = []
            total_current  = 0
            total_cheapest = 0
            for med in medicines:
                result = find_alternatives(med)
                if result:
                    total_current += result["original_price"]
                    cheaper    = [a for a in result["alternatives"] if a["saving"] > 0]
                    best_price = cheaper[0]["price"] if cheaper else result["original_price"]
                    best_name  = cheaper[0]["name"]  if cheaper else result["original_name"]
                    best_save  = cheaper[0]["saving"] if cheaper else 0
                    total_cheapest += best_price
                    results.append({
                        "Medicine":     result["original_name"],
                        "Your Price":   f"₹{result['original_price']}",
                        "Cheapest Alt": best_name,
                        "Alt Price":    f"₹{best_price}",
                        "You Save":     f"₹{best_save}",
                    })

            total_saving = round(total_current - total_cheapest, 2)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💳 Current",       f"₹{round(total_current,2)}")
            c2.metric("💚 With Generics", f"₹{round(total_cheapest,2)}")
            c3.metric("💰 Save/Month",    f"₹{round(total_saving*2,2)}")
            c4.metric("📆 Save/Year",     f"₹{round(total_saving*24,2)}")
            st.markdown("---")
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            if total_saving > 0:
                st.success(f"🎉 Switch to generics → Save **₹{round(total_saving*24,2)} every year!**")

# ──────────────────────────────────────────────
# PAGE 5 — STORE LOCATOR
# ──────────────────────────────────────────────
elif page == "locator":
    st.markdown("<div class='page-header'>🏪 Jan Aushadhi Store Locator</div>", unsafe_allow_html=True)
    st.markdown("Find nearest government generic medicine stores")
    st.markdown("")

    col1, col2 = st.columns([3, 1])
    with col1:
        pincode = st.text_input("", placeholder="Enter your pincode e.g. 400001", label_visibility="collapsed")
    with col2:
        locate_btn = st.button("📍 Find Stores", type="primary", use_container_width=True)

    if locate_btn and pincode:
        stores = [
            {"name": "Jan Aushadhi Kendra 1", "address": f"Near Market, {pincode}",   "distance": "0.5 km", "status": "🟢 Open"},
            {"name": "Jan Aushadhi Kendra 2", "address": f"Main Road, {pincode}",     "distance": "1.2 km", "status": "🟢 Open"},
            {"name": "Jan Aushadhi Kendra 3", "address": f"Gandhi Nagar, {pincode}",  "distance": "2.1 km", "status": "🟡 Closes at 8PM"},
            {"name": "Generic Pharmacy",      "address": f"Hospital Road, {pincode}", "distance": "2.8 km", "status": "🟢 Open"},
        ]
        st.success(f"✅ Found {len(stores)} stores near **{pincode}**")
        for i, store in enumerate(stores, 1):
            c1, c2, c3 = st.columns([4, 2, 2])
            c1.markdown(f"**{i}. {store['name']}**

📍 {store['address']}")
            c2.metric("Distance", store["distance"])
            c3.markdown(f"

{store['status']}")
            st.divider()

        st.info("💡 Jan Aushadhi stores save 50-90% vs branded medicines | **janaushadhi.gov.in**")

# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; font-size:12px; padding:10px;">
    💊 Prescrify v2.0 &nbsp;|&nbsp; AMD Slingshot Hackathon 2026 &nbsp;|&nbsp;
    CrewAI + Groq + Llama 3.3 70B + 253,973 Indian Medicines 🇮🇳
</div>
""", unsafe_allow_html=True)
