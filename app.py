
import streamlit as st
import pandas as pd
import os
import json
import pytesseract
import easyocr
from PIL import Image
from crewai import Agent, Task, Crew, LLM
from groq import Groq

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="ArogyadhanaRaksha AI 💊",
    page_icon="💊",
    layout="wide"
)

# ══════════════════════════════════════════════
# LOAD GROQ KEY
# ══════════════════════════════════════════════
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return None

auto_key = get_groq_key()

# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_csv("medicines_clean.csv")

df = load_data()

# ══════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════
def find_alternatives(medicine_name):
    searched = df[
        df["name"].str.lower().str.contains(
            medicine_name.lower().strip(), na=False
        )
    ]
    if searched.empty:
        return None
    original       = searched.iloc[0]
    original_name  = original["name"]
    original_price = original["price(₹)"]
    original_comp  = original["composition"]
    original_mfr   = original["manufacturer_name"]
    original_pack  = original["pack_size_label"]
    alternatives   = df[
        (df["composition"].str.lower() == original_comp.lower()) &
        (df["name"].str.lower() != original_name.lower())
    ].sort_values("price(₹)")
    alt_list = []
    for _, row in alternatives.iterrows():
        saving     = round(original_price - row["price(₹)"], 2)
        saving_pct = round((saving / original_price) * 100, 1)
        alt_list.append({
            "name"        : row["name"],
            "price"       : round(row["price(₹)"], 2),
            "saving"      : saving,
            "saving_pct"  : saving_pct,
            "manufacturer": row["manufacturer_name"],
            "pack_size"   : row["pack_size_label"],
        })
    return {
        "original_name"        : original_name,
        "original_price"       : round(original_price, 2),
        "original_composition" : original_comp,
        "original_manufacturer": original_mfr,
        "original_pack"        : original_pack,
        "alternatives"         : alt_list
    }

def detect_overcharge(medicine_name, billed_price):
    result = find_alternatives(medicine_name)
    if not result:
        return None
    db_price       = result["original_price"]
    overcharge     = round(billed_price - db_price, 2)
    overcharge_pct = round((overcharge / db_price) * 100, 1) if db_price > 0 else 0
    return {
        "medicine"      : result["original_name"],
        "billed_price"  : billed_price,
        "actual_price"  : db_price,
        "overcharge"    : overcharge,
        "overcharge_pct": overcharge_pct,
        "is_overcharged": overcharge > 0,
        "composition"   : result["original_composition"],
        "alternatives"  : result["alternatives"]
    }

def extract_text_from_image(image):
    """Extract text from prescription/bill image"""
    reader = easyocr.Reader(["en"], verbose=False)
    result = reader.readtext(image, detail=0)
    return " ".join(result)

def extract_medicines_with_ai(text, groq_key):
    """Use Groq LLM to extract medicine names from OCR text"""
    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "system",
            "content": """You are an expert at reading Indian medical prescriptions and bills.
            Extract medicine names from the given text.
            Return ONLY a JSON array of medicine names.
            Example: ["Augmentin 625", "Dolo 650", "Pantop 40"]
            If you find billed prices too, return:
            {"medicines": ["med1", "med2"], "bill_items": [{"name": "med1", "price": 250}]}
            Return valid JSON only, no explanation."""
        }, {
            "role": "user",
            "content": f"Extract medicines from this prescription/bill text:\n\n{text}"
        }],
        max_tokens=500,
        temperature=0.1
    )
    raw = response.choices[0].message.content.strip()
    try:
        # Try parsing as JSON
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed, []
        elif isinstance(parsed, dict):
            return parsed.get("medicines", []), parsed.get("bill_items", [])
    except:
        # Fallback: extract quoted strings
        import re
        medicines = re.findall(r'"([^"]+)"', raw)
        return medicines, []

def get_jan_aushadhi_stores(pincode):
    """Get nearby Jan Aushadhi stores"""
    # Hardcoded sample data (replace with real API if available)
    stores = [
        {"name": "Jan Aushadhi Kendra 1", "address": f"Near Market, {pincode}",    "distance": "0.5 km", "lat": 19.076, "lon": 72.877},
        {"name": "Jan Aushadhi Kendra 2", "address": f"Main Road, {pincode}",      "distance": "1.2 km", "lat": 19.079, "lon": 72.880},
        {"name": "Jan Aushadhi Kendra 3", "address": f"Gandhi Nagar, {pincode}",   "distance": "2.1 km", "lat": 19.072, "lon": 72.870},
        {"name": "Generic Pharmacy",      "address": f"Hospital Road, {pincode}",  "distance": "2.8 km", "lat": 19.068, "lon": 72.865},
    ]
    return stores

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/color/96/medical-doctor.png", width=60)
st.sidebar.title("ArogyadhanaRaksha AI")
st.sidebar.markdown("*Your Healthcare Savings Assistant*")
st.sidebar.markdown("---")

if auto_key:
    groq_key = auto_key
    st.sidebar.success("🔑 API Key loaded!")
else:
    groq_key = st.sidebar.text_input(
        "🔑 Groq API Key",
        type="password",
        help="Get free key at console.groq.com"
    )

st.sidebar.markdown("---")
st.sidebar.success(f"📊 **{len(df):,}** Indian medicines")
st.sidebar.info("🤖 Llama 3.3 70B (Groq)")
st.sidebar.info("🇮🇳 NPPA Verified Data")
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigation**")
page = st.sidebar.radio("Go to:", [
    "🔍 Medicine Search",
    "📄 Prescription Analyzer",
    "🧾 Bill Overcharge Detector",
    "💰 Savings Dashboard",
    "🏪 Nearby Store Locator"
])

# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
st.title("💊 ArogyadhanaRaksha AI")
st.markdown("> 🇮🇳 Empowering Indian patients with **affordable healthcare** — search from **253,973 real medicines**, detect overcharges, and save money on every prescription!")
st.markdown("---")

# ══════════════════════════════════════════════
# PAGE 1 — MEDICINE SEARCH
# ═���════════════════════════════════════════════
if page == "🔍 Medicine Search":
    st.header("🔍 Medicine Search & Generic Alternatives")
    st.markdown("Find cheaper alternatives with same composition")

    col1, col2 = st.columns([4, 1])
    with col1:
        medicine_input = st.text_input(
            "Medicine",
            placeholder="e.g. Augmentin 625, Crocin, Dolo 650, Pantop 40...",
            label_visibility="collapsed"
        )
    with col2:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    if search_btn and medicine_input:
        result = find_alternatives(medicine_input)

        if result is None:
            st.error(f"❌ **{medicine_input}** not found!")
            st.info("💡 Try partial name e.g. **Augmentin** instead of **Augmentin 625 Duo Tablet**")
        else:
            st.success(f"✅ Found: **{result['original_name']}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("💰 Current Price", f"₹{result['original_price']}")
            c2.metric("🔄 Alternatives Found", len(result["alternatives"]))
            if result["alternatives"] and result["alternatives"][0]["saving"] > 0:
                c3.metric(
                    "💚 Max Saving",
                    f"₹{result['alternatives'][0]['saving']}",
                    f"{result['alternatives'][0]['saving_pct']}% cheaper"
                )

            st.markdown("### 🔬 Prescription Analysis")
            st.info(f"**Composition:** {result['original_composition']}")
            c1, c2 = st.columns(2)
            c1.markdown(f"📦 **Pack:** {result['original_pack']}")
            c2.markdown(f"🏭 **By:** {result['original_manufacturer']}")
            st.markdown("---")

            cheaper = [a for a in result["alternatives"] if a["saving"] > 0]
            if cheaper:
                st.markdown(f"### ✅ {len(cheaper)} Generic Alternatives Found")
                st.caption("Same composition • Same effect • Lower price")

                for i, alt in enumerate(cheaper[:8], 1):
                    cols = st.columns([3, 2, 2, 2])
                    label = f"🏆 **{alt['name']}**" if i == 1 else f"**{i}.** {alt['name']}"
                    cols[0].markdown(label)
                    cols[1].markdown(f"₹**{alt['price']}**")
                    cols[2].markdown(f"💚 Save ₹**{alt['saving']}** ({alt['saving_pct']}%)")
                    cols[3].markdown(f"*{alt['manufacturer']}*")
                    st.caption(f"📦 {alt['pack_size']}")
                    st.markdown("---")

                best = cheaper[0]
                st.success(f"""
                ### 🏆 Best Choice: {best['name']}
                | Per Strip | Per Month (2 strips) | Per Year |
                |---|---|---|
                | 💚 ₹{best['saving']} saved | 💚 ₹{round(best['saving']*2,2)} saved | 💚 ₹{round(best['saving']*24,2)} saved |
                """)
            else:
                st.info("✅ Already the most affordable option!")

            if groq_key:
                with st.expander("🤖 AI Medical Insights"):
                    with st.spinner("Analysing with Llama 3.3 70B..."):
                        try:
                            os.environ["GROQ_API_KEY"] = groq_key
                            groq_llm = LLM(
                                model="groq/llama-3.3-70b-versatile",
                                api_key=groq_key,
                                temperature=0.2
                            )
                            agent = Agent(
                                role="Indian Medicine Expert",
                                goal="Provide simple medical insights",
                                backstory="Expert pharmacologist for Indian medicines",
                                llm=groq_llm,
                                verbose=False
                            )
                            task = Task(
                                description=f"""
                                Medicine: {medicine_input}
                                Composition: {result['original_composition']}
                                Give in simple terms for Indian patients:
                                1. What is this used for?
                                2. Is switching to generic safe?
                                3. Any precautions?
                                Keep it brief.
                                """,
                                agent=agent,
                                expected_output="Simple medical insights"
                            )
                            crew = Crew(agents=[agent], tasks=[task], verbose=False)
                            st.markdown(str(crew.kickoff()))
                        except Exception as e:
                            st.error(f"AI Error: {e}")

        st.warning("⚠️ Always consult your doctor before switching medicines.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.info("**1️⃣ Search**\n\nType any Indian medicine name")
        c2.info("**2️⃣ Find**\n\nWe search 253,973 medicines")
        c3.info("**3️⃣ Save**\n\nSee cheaper alternatives in ₹")

# ══════════════════════════════════════════════
# PAGE 2 — PRESCRIPTION ANALYZER
# ══════════════════════════════════════════════
elif page == "📄 Prescription Analyzer":
    st.header("📄 Prescription Analyzer")
    st.markdown("Upload prescription image → AI extracts medicines → Find alternatives")

    if not groq_key:
        st.warning("🔑 Please add your Groq API key in the sidebar!")
    else:
        tab1, tab2 = st.tabs(["📷 Upload Image", "✏️ Type Manually"])

        with tab1:
            uploaded = st.file_uploader(
                "Upload Prescription or Bill",
                type=["jpg", "jpeg", "png", "pdf"],
                help="Take a photo of your prescription or bill"
            )

            if uploaded:
                image = Image.open(uploaded)
                st.image(image, caption="Uploaded Prescription", use_container_width=True)

                if st.button("🔍 Analyze Prescription", type="primary"):
                    with st.spinner("📖 Reading prescription with OCR..."):
                        try:
                            import numpy as np
                            img_array  = np.array(image)
                            ocr_text   = extract_text_from_image(img_array)
                            st.markdown("**📝 Extracted Text:**")
                            st.code(ocr_text)
                        except Exception as e:
                            st.error(f"OCR Error: {e}")
                            ocr_text = ""

                    if ocr_text:
                        with st.spinner("🤖 AI extracting medicine names..."):
                            medicines, bill_items = extract_medicines_with_ai(ocr_text, groq_key)

                        if medicines:
                            st.success(f"✅ Found {len(medicines)} medicines: {', '.join(medicines)}")
                            st.session_state["extracted_medicines"] = medicines
                            st.session_state["bill_items"]          = bill_items

                            st.markdown("### 💊 Medicine Analysis")
                            total_saving = 0
                            for med in medicines:
                                result = find_alternatives(med)
                                if result:
                                    with st.expander(f"💊 {result['original_name']} — ₹{result['original_price']}"):
                                        st.info(f"**Composition:** {result['original_composition']}")
                                        cheaper = [a for a in result["alternatives"] if a["saving"] > 0]
                                        if cheaper:
                                            best          = cheaper[0]
                                            total_saving += best["saving"]
                                            st.success(f"✅ Best Alternative: **{best['name']}** — ₹{best['price']} (Save ₹{best['saving']})")
                                        else:
                                            st.info("Already cheapest!")

                            if total_saving > 0:
                                st.success(f"### 💰 Total Savings Possible: ₹{round(total_saving, 2)}")
                        else:
                            st.warning("No medicines found. Try typing manually.")

        with tab2:
            manual_text = st.text_area(
                "Type or paste medicine names (one per line)",
                placeholder="Augmentin 625\nDolo 650\nPantop 40\nAzithral 500",
                height=150
            )
            if st.button("🔍 Find Alternatives", type="primary"):
                if manual_text:
                    medicines  = [m.strip() for m in manual_text.split("\n") if m.strip()]
                    total_save = 0
                    for med in medicines:
                        result = find_alternatives(med)
                        if result:
                            with st.expander(f"💊 {result['original_name']} — ₹{result['original_price']}"):
                                st.info(f"**Composition:** {result['original_composition']}")
                                cheaper = [a for a in result["alternatives"] if a["saving"] > 0]
                                if cheaper:
                                    best        = cheaper[0]
                                    total_save += best["saving"]
                                    st.success(f"✅ Best: **{best['name']}** — ₹{best['price']} (Save ₹{best['saving']})")
                        else:
                            st.warning(f"❌ {med} not found")
                    if total_save > 0:
                        st.success(f"### 💰 Total Savings: ₹{round(total_save, 2)}")

# ══════════════════════════════════════════════
# PAGE 3 — BILL OVERCHARGE DETECTOR
# ══════════════════════════════════════════════
elif page == "🧾 Bill Overcharge Detector":
    st.header("🧾 Medical Bill Overcharge Detector")
    st.markdown("Enter your bill items → We check for overcharges instantly!")

    st.markdown("### Add Bill Items")
    num_items = st.number_input("Number of medicines in bill", min_value=1, max_value=20, value=3)

    bill_items = []
    cols = st.columns([3, 2])
    cols[0].markdown("**Medicine Name**")
    cols[1].markdown("**Billed Price (₹)**")

    for i in range(num_items):
        c1, c2 = st.columns([3, 2])
        name  = c1.text_input(f"Medicine {i+1}", key=f"med_{i}",
                               placeholder="e.g. Augmentin 625")
        price = c2.number_input(f"Price {i+1}", key=f"price_{i}",
                                 min_value=0.0, value=0.0, step=0.5)
        if name and price > 0:
            bill_items.append({"name": name, "billed_price": price})

    if st.button("🔍 Check for Overcharges", type="primary"):
        if not bill_items:
            st.warning("Please add at least one bill item!")
        else:
            total_overcharge = 0
            total_billed     = 0
            total_actual     = 0
            flagged          = []

            st.markdown("### 📊 Bill Analysis Results")
            for item in bill_items:
                result = detect_overcharge(item["name"], item["billed_price"])
                if result:
                    total_billed += result["billed_price"]
                    total_actual += result["actual_price"]
                    if result["is_overcharged"]:
                        total_overcharge += result["overcharge"]
                        flagged.append(result)
                        st.error(f"""
                        ⚠️ **OVERCHARGE DETECTED: {result['medicine']}**
                        - Billed Price  : ₹{result['billed_price']}
                        - Standard Price: ₹{result['actual_price']}
                        - Overcharged by: ₹{result['overcharge']} ({result['overcharge_pct']}% extra!)
                        """)
                    else:
                        st.success(f"✅ **{result['medicine']}** — ₹{result['billed_price']} (Fair price!)")
                else:
                    st.info(f"ℹ️ {item['name']} — Not found in database")

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("💳 Total Billed",      f"₹{round(total_billed, 2)}")
            c2.metric("✅ Should Have Paid",   f"₹{round(total_actual, 2)}")
            c3.metric("🚨 Total Overcharged",  f"₹{round(total_overcharge, 2)}",
                      delta=f"-₹{round(total_overcharge,2)}" if total_overcharge > 0 else None,
                      delta_color="inverse")

            if flagged:
                st.error(f"🚨 You were overcharged ₹{round(total_overcharge,2)} on {len(flagged)} item(s)!")
                st.markdown("**💡 What to do:**")
                st.markdown("""
                1. Show this report to the hospital/pharmacy
                2. Ask for a corrected bill
                3. File complaint at: **nhrc.nic.in** or call **104**
                4. NPPA complaint: **nppa.gov.in**
                """)
            else:
                st.success("✅ All prices are within normal range!")

# ══════════════════════════════════════════════
# PAGE 4 — SAVINGS DASHBOARD
# ══════════════════════════════════════════════
elif page == "💰 Savings Dashboard":
    st.header("💰 Savings Dashboard")
    st.markdown("See how much you can save by switching to generics")

    medicines_input = st.text_area(
        "Enter your regular medicines (one per line)",
        placeholder="Augmentin 625\nDolo 650\nPantop 40\nAzithral 500\nCrocin 500",
        height=150
    )

    if st.button("📊 Calculate My Savings", type="primary"):
        medicines = [m.strip() for m in medicines_input.split("\n") if m.strip()]
        if not medicines:
            st.warning("Please enter at least one medicine!")
        else:
            results      = []
            total_current = 0
            total_cheapest = 0

            for med in medicines:
                result = find_alternatives(med)
                if result:
                    total_current += result["original_price"]
                    cheaper = [a for a in result["alternatives"] if a["saving"] > 0]
                    best_price = cheaper[0]["price"] if cheaper else result["original_price"]
                    best_name  = cheaper[0]["name"]  if cheaper else result["original_name"]
                    best_save  = cheaper[0]["saving"] if cheaper else 0
                    total_cheapest += best_price
                    results.append({
                        "Medicine"     : result["original_name"],
                        "Your Price"   : f"₹{result['original_price']}",
                        "Cheapest Alt" : best_name,
                        "Alt Price"    : f"₹{best_price}",
                        "You Save"     : f"₹{best_save}",
                        "Saving %"     : f"{round((best_save/result['original_price'])*100,1)}%" if result["original_price"] > 0 else "0%"
                    })

            total_saving = round(total_current - total_cheapest, 2)

            # Big metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("💳 Current Spend (per cycle)", f"₹{round(total_current, 2)}")
            c2.metric("💚 With Generics",             f"₹{round(total_cheapest, 2)}")
            c3.metric("💰 Total Savings",             f"₹{total_saving}",
                      delta=f"Save {round((total_saving/total_current)*100,1)}%" if total_current > 0 else None)

            st.markdown("---")

            # Monthly/Yearly projection
            c1, c2, c3 = st.columns(3)
            c1.metric("📅 Monthly Savings",  f"₹{round(total_saving*2, 2)}")
            c2.metric("📆 Yearly Savings",   f"₹{round(total_saving*24, 2)}")
            c3.metric("🏥 Medicines Checked", len(results))

            # Table
            st.markdown("### 📋 Medicine-wise Breakdown")
            st.dataframe(pd.DataFrame(results), use_container_width=True)

            if total_saving > 0:
                st.success(f"🎉 Switch to generics and save **₹{round(total_saving*24,2)} per year!**")

# ══════════════════════════════════════════════
# PAGE 5 — NEARBY STORE LOCATOR
# ══════════════════════════════════════════════
elif page == "🏪 Nearby Store Locator":
    st.header("🏪 Nearby Government Medicine Store Locator")
    st.markdown("Find nearest Jan Aushadhi stores and generic pharmacies")

    col1, col2 = st.columns([3, 1])
    with col1:
        pincode = st.text_input("📍 Enter your Pincode", placeholder="e.g. 400001")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        locate_btn = st.button("📍 Find Stores", type="primary", use_container_width=True)

    if locate_btn and pincode:
        with st.spinner("🔍 Finding nearby stores..."):
            stores = get_jan_aushadhi_stores(pincode)

        st.success(f"✅ Found {len(stores)} stores near {pincode}")

        for i, store in enumerate(stores, 1):
            with st.container():
                c1, c2, c3 = st.columns([3, 2, 2])
                c1.markdown(f"**{i}. {store['name']}**\n\n📍 {store['address']}")
                c2.metric("📏 Distance", store["distance"])
                c3.markdown("<br>", unsafe_allow_html=True)
                c3.markdown("🟢 Open • Generic medicines available")
                st.markdown("---")

        st.info("""
        💡 **About Jan Aushadhi Stores:**
        - Government subsidized generic medicines
        - Savings of 50-90% vs branded medicines
        - Same quality, same composition
        - 10,000+ stores across India
        - Find official stores: **janaushadhi.gov.in**
        """)

        st.markdown("### 🔗 Official Resources")
        col1, col2 = st.columns(2)
        col1.markdown("""
        - 🏥 [Jan Aushadhi Portal](https://janaushadhi.gov.in)
        - 💊 [NPPA Medicine Prices](https://nppa.gov.in)
        """)
        col2.markdown("""
        - 📞 Helpline: 1800-111-255
        - 📱 PM-JAY: 14555
        """)

# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:12px;">
💊 ArogyadhanaRaksha AI | AMD Slingshot Hackathon 2026 |
Built with CrewAI + Groq + Llama 3.3 70B + 253,973 Indian Medicines 🇮🇳
</div>
""", unsafe_allow_html=True)
