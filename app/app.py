import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# ===================== SETTINGS =====================
st.set_page_config("House Price Prediction", "🏠", layout="centered")

# ===================== MODEL LOAD =====================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "Model")

@st.cache_resource
def load_models():
      try:
        lr = pickle.load(open(os.path.join(MODEL_DIR, "linear_model.pkl"), "rb"))
        rf = pickle.load(open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb"))
        dt = pickle.load(open(os.path.join(MODEL_DIR, "dt_model.pkl"), "rb"))
        gb = pickle.load(open(os.path.join(MODEL_DIR, "gb_model.pkl"), "rb"))
        xgb = pickle.load(open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb"))
        encoder = pickle.load(open(os.path.join(MODEL_DIR, "location_encoder.pkl"), "rb"))

        return lr, rf, dt, gb, xgb, encoder

      except Exception as e:
        st.error(f"🚨 Error loading models: {e}")
        return None, None, None, None, None, None


lr_model, rf_model, dt_model, gb_model, xgb_model, encoder = load_models()

if None in [lr_model, rf_model, dt_model, gb_model, xgb_model, encoder]:
    st.stop()
  # ===================== HELPER FUNCTIONS =====================
def get_location_code(location, encoder):
    if location in encoder.classes_:
        return encoder.transform([location])[0]
    else:
        return int(len(encoder.classes_) / 2)

def format_inr(amount):
    amount = int(amount)
    s = str(amount)
    if len(s) <= 3:
        return f"₹ {s}"
    last3 = s[-3:]
    rest = s[:-3][::-1]
    groups = [rest[i:i+2] for i in range(0, len(rest), 2)]
    s = ",".join(groups)[::-1] + "," + last3
    return f"₹ {s}"



def price_category(price):
    if price < 5000000:
        return "🟢 Affordable Area"
    elif price < 9000000:
        return "🟡 Mid-Range Area"
    else:
        return "🔴 Premium Area"

def get_recommendation(price):
        """
        Returns zone name and confidence percentage based on price
        """
        if price < 5000000:
            return "🟢 Low Budget Zone", 85
        elif price < 9000000:
            return "🟡 Mid Budget Zone", 75
        else:
            return "🔴 Premium Zone", 90


# ================== MAP DATA : COMPLETE GUJARAT ==================
AREA_MAP = {

    # ===== Ahmedabad =====
    "Ahmedabad - SG Highway": {"lat": 23.0736, "lon": 72.5185, "rate": 6500},
    "Ahmedabad - Bopal": {"lat": 23.0396, "lon": 72.4660, "rate": 6000},
    "Ahmedabad - Gota": {"lat": 23.0984, "lon": 72.5407, "rate": 6200},
    "Ahmedabad - Maninagar": {"lat": 22.9971, "lon": 72.5996, "rate": 5800},

    # ===== Surat =====
    "Surat - Vesu": {"lat": 21.1411, "lon": 72.7702, "rate": 5800},
    "Surat - Adajan": {"lat": 21.1950, "lon": 72.7944, "rate": 5500},
    "Surat - Varachha": {"lat": 21.2255, "lon": 72.8847, "rate": 5200},

    # ===== Vadodara =====
    "Vadodara - Alkapuri": {"lat": 22.3092, "lon": 73.1796, "rate": 5600},
    "Vadodara - Gotri": {"lat": 22.3084, "lon": 73.1410, "rate": 5400},

    # ===== Rajkot =====
    "Rajkot - Kalawad Road": {"lat": 22.2926, "lon": 70.7606, "rate": 5200},
    "Rajkot - University Road": {"lat": 22.2882, "lon": 70.7855, "rate": 5400},

    # ===== Gandhinagar =====
    "Gandhinagar - Sector 21": {"lat": 23.2156, "lon": 72.6369, "rate": 5200},
    "Gandhinagar - Infocity": {"lat": 23.1958, "lon": 72.6333, "rate": 5600},

    # ===== Bhavnagar =====
    "Bhavnagar - Hill Drive": {"lat": 21.7645, "lon": 72.1519, "rate": 4600},

    # ===== Jamnagar =====
    "Jamnagar - Patel Colony": {"lat": 22.4707, "lon": 70.0577, "rate": 4500},

    # ===== Junagadh =====
    "Junagadh - Joshipura": {"lat": 21.5306, "lon": 70.4579, "rate": 4200},

    # ===== Anand / Nadiad =====
    "Anand - Vallabh Vidyanagar": {"lat": 22.5525, "lon": 72.9234, "rate": 4800},
    "Nadiad - College Road": {"lat": 22.6938, "lon": 72.8616, "rate": 4700},

    # ===== Mehsana =====
    "Mehsana - Modhera Road": {"lat": 23.6030, "lon": 72.3940, "rate": 4100},

    # ===== Bharuch =====
    "Bharuch - Zadeshwar": {"lat": 21.7051, "lon": 73.0016, "rate": 4400},

    # ===== Navsari =====
    "Navsari - Jalalpore": {"lat": 20.9467, "lon": 72.9289, "rate": 4600},

    # ===== Morbi =====
    "Morbi - Ravapar Road": {"lat": 22.8173, "lon": 70.8377, "rate": 3900},

    # ===== Porbandar =====
    "Porbandar - Chhaya": {"lat": 21.6417, "lon": 69.6293, "rate": 4200},
    "Porbandar - Chopati":{"lat": 21.628322, "lon":69.6129944, "rate": 4500},
    
    # ===== Gandhidham =====
    "Gandhidham - Adipur": {"lat": 23.0733, "lon": 70.1337, "rate": 4800},

    # ===== Valsad / Vapi =====
    "Valsad - Tithal Road": {"lat": 20.5992, "lon": 72.9342, "rate": 5200},
    "Vapi - GIDC": {"lat": 20.3715, "lon": 72.9040, "rate": 5600},

    # ===== Palanpur =====
    "Palanpur - Abu Road": {"lat": 24.1715, "lon": 72.4340, "rate": 4100},

    # ===== Godhra =====
    "Godhra - Kalal Darwaja": {"lat": 22.7755, "lon": 73.6143, "rate": 3900},

    # ===== Surendranagar =====
    "Surendranagar - Wadhwan": {"lat": 22.7211, "lon": 71.6726, "rate": 4300},

    # ===== Dahod =====
    "Dahod - Station Road": {"lat": 22.8316, "lon": 74.2556, "rate": 3600},

    # ===== Amreli =====
    "Amreli - Lathi Road": {"lat": 21.6032, "lon": 71.2182, "rate": 4000},

    # ===== Patan =====
    "Patan - Siddhpur Road": {"lat": 23.8500, "lon": 72.1250, "rate": 4100},

    # ===== Bhuj =====
    "Bhuj - Jubilee Ground": {"lat": 23.2419, "lon": 69.6669, "rate": 4500},

    # ===== Dwarka =====
    "Dwarka - Gomti Ghat": {"lat": 22.2442, "lon": 68.9685, "rate": 4700},

    # ===== Mahuva =====
    "Mahuva - Talaja Road": {"lat": 21.0922, "lon": 71.7706, "rate": 4200},

    # ===== Botad =====
    "Botad - Paliyad Road": {"lat": 22.1704, "lon": 71.6685, "rate": 4300},

    # ===== Veraval =====
    "Veraval - Somnath Road": {"lat": 20.9159, "lon": 70.3629, "rate": 4600},

    # ===== Gondal =====
    "Gondal - Bhojrajpara": {"lat": 21.9619, "lon": 70.7923, "rate": 4400},

    # ===== Jetpur / Navagadh =====
    "Jetpur - Navagadh": {"lat": 21.7548, "lon": 70.6230, "rate": 4300},

    # ===== Sidhpur =====
    "Sidhpur - Patan Road": {"lat": 23.9190, "lon": 72.3735, "rate": 4000},
}
# ================================================================


# ===================== UI =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at top,#020617,#000); color:#e5e7eb; }
header, footer { display: none; }
.header { border:2px solid #22d3ee; border-radius:22px; padding:35px; margin-bottom:25px;
box-shadow:0 0 35px rgba(34,211,238,.4); text-align: center; }
label { color:white !important; font-weight:600; }
input, select { background:#020617 !important; color:white !important;
border:2px solid #22d3ee !important; border-radius:12px !important; }
button { border-radius:14px !important; font-weight:700 !important;
background:transparent !important; color:#22d3ee !important;
border:2px solid #22d3ee !important; width: 100%; }
button:hover { background:rgba(34,211,238,.1) !important; transform:scale(1.02); }
.result-box { margin-top:25px; padding:30px; border-radius:20px;
background:linear-gradient(135deg,#022c22,#064e3b);
border:2px solid #10b981; box-shadow:0 0 35px rgba(16,185,129,.6); }
.conf { margin-top:20px; padding:26px; border-radius:20px;
border:2px solid #22d3ee; background:#020617;
box-shadow:0 0 30px rgba(34,211,238,.35); }
.progress { height:12px; background:#020617; border-radius:999px;
border:1px solid #22d3ee; overflow:hidden; margin: 10px 0; }
.progress span { height:100%; display:block;
background:linear-gradient(90deg,#22d3ee,#10b981); }
.badge { display:inline-block; margin-top:10px; padding:6px 14px;
border-radius:999px; border:1px solid #38bdf8;
color:#7dd3fc; font-size:12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="header"><h1>🏠 House Price Prediction</h1></div>', unsafe_allow_html=True)

# ===================== MAP =====================
st.markdown("### 🗺️ Select Location from Map")

m = folium.Map(
    location=[22.2587, 71.1924],
    zoom_start=7,
    tiles="OpenStreetMap"
)

for city, info in AREA_MAP.items():
    folium.Marker(
        location=[info["lat"], info["lon"]],
        popup=city,
        tooltip=city,
        icon=folium.Icon(color="blue", icon="home", prefix="fa")
    ).add_to(m)

map_data = st_folium(m, height=350,width=700)

selected_city = list(AREA_MAP.keys())[0]

if map_data and map_data.get("last_object_clicked"):
    lat = map_data["last_object_clicked"]["lat"]
    lon = map_data["last_object_clicked"]["lng"]

    selected_city = min(
        AREA_MAP.keys(),
        key=lambda c: abs(AREA_MAP[c]["lat"] - lat) + abs(AREA_MAP[c]["lon"] - lon)
    )

    st.success(f"📍 Selected Area: {selected_city}")


locations = list(AREA_MAP.keys())
default_index = locations.index(selected_city) if selected_city in locations else 0

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("📍 Location", locations, index=default_index)
    area = st.number_input("📐 Area (sqft)", min_value=100, value=1500)
with col2:
    bhk = st.selectbox("🛏 BHK", [1,2,3,4,5], index=2)
    bath = st.selectbox("🚿 Bathrooms", [1,2,3,4,5], index=1)

# ===================== PREDICTION FUNCTION =====================
def predict_price(loc, area, bhk, bath):

    loc_code = get_location_code(loc, encoder)
    X = np.array([[area, bhk, bath, loc_code]])

    # Individual Predictions
    p_lr = lr_model.predict(X)[0]
    p_rf = rf_model.predict(X)[0]
    p_dt = dt_model.predict(X)[0]
    p_gb = gb_model.predict(X)[0]
    p_xgb = xgb_model.predict(X)[0]

    # Advanced Hybrid (Weighted 5 Model)
    hybrid_log = (
        0.2 * p_lr +
        0.25 * p_rf +
        0.15 * p_dt +
        0.2 * p_gb +
        0.2 * p_xgb
    )

    price = int(np.expm1(hybrid_log))

    # Location Rate Adjustment
    if loc in AREA_MAP:
        price = int(price * (AREA_MAP[loc]["rate"] / 5000))

    low = int(price * 0.9)
    high = int(price * 1.1)

    zone, conf = get_recommendation(price)

    return price, low, high, zone, conf


# ===================== SINGLE PREDICTION =====================
if st.button("🚀 Calculate Property Price", use_container_width=True):
    price, low, high, zone, conf = predict_price(location, area, bhk, bath)
    st.markdown(f"""
    <div class="result-box">
        <h2 style="color:white; margin:0;">{format_inr(price)}</h2>
        <p style="color:#10b981; margin:0;">{zone} | Confidence: {conf}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Price Range Graph
    st.markdown("### 📈 Price Prediction Analysis")
    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    labels = ["Min", "Predicted", "Max"]
    values = [low, price, high]
    bars = ax.bar(labels, values, color=['#1e293b','#22d3ee','#1e293b'], edgecolor="#22d3ee", linewidth=1)
    for bar,val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, val, format_inr(val), ha='center', va='bottom', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # ================= AREA TREND GRAPH =================
    st.markdown("### 📊 Area vs Price Trend")

    areas = np.linspace(area * 0.7, area * 1.3, 6).astype(int)
    trend_prices = []

    for a in areas:
        X_t = np.array([[a, bhk, bath, get_location_code(location, encoder)]])

        p_lr = lr_model.predict(X_t)[0]
        p_rf = rf_model.predict(X_t)[0]
        p_dt = dt_model.predict(X_t)[0]
        p_gb = gb_model.predict(X_t)[0]
        p_xgb = xgb_model.predict(X_t)[0]

        hybrid_t = (
            0.2 * p_lr +
            0.25 * p_rf +
            0.15 * p_dt +
            0.2 * p_gb +
            0.2 * p_xgb
        )

        p_t = int(np.expm1(hybrid_t))

        if location in AREA_MAP:
            p_t = int(p_t * (AREA_MAP[location]["rate"] / 5000))

        trend_prices.append(p_t)   # VERY IMPORTANT LINE

    # Ensure both arrays same length
    trend_prices = trend_prices[:len(areas)]

    fig2, ax2 = plt.subplots(figsize=(8,4))
    fig2.patch.set_facecolor("#020617")
    ax2.set_facecolor("#020617")

    ax2.plot(areas, trend_prices, marker="o", linewidth=2)
    ax2.tick_params(colors='white')
    ax2.set_xlabel("Area (sqft)", color="white")
    ax2.set_ylabel("Price (INR)", color="white")

    st.pyplot(fig2)

price_filter = st.selectbox(
    "🎯 Select Price Category",
    ["Affordable (Low Price)", "Mid-Range", "Premium"]
)

if st.button("📌 Show All Gujarat Recommendations", use_container_width=True):

    gujarat_locations = list(AREA_MAP.keys())
    result_data = []

    # ---------------- GET ALL DATA ----------------
    for loc in gujarat_locations:
        price, low, high, zone, conf = predict_price(loc, area, bhk, bath)
        tag = price_category(price)
        result_data.append((loc, price, low, high, tag))

    # ---------------- FILTER LOGIC ----------------
    result_data = sorted(result_data, key=lambda x: x[1])  # sort by price ascending
    if price_filter == "Affordable (Low Price)":
        result_data = result_data[:len(result_data)//3]  # lowest third
    elif price_filter == "Mid-Range":
        result_data = result_data[len(result_data)//3 : 2*len(result_data)//3]  # middle third
    else:  # Premium
        result_data = result_data[2*len(result_data)//3:]  # highest third

    # ---------------- DISPLAY RESULTS ----------------
    for loc, price, low, high, tag in result_data:
        st.markdown(f"""
        <div class="result-box">
            <h3 style="color:#22d3ee; margin:0;">{loc}</h3>
            <h2 style="color:white; margin:0;">{format_inr(price)}</h2>
            <p style="color:#38bdf8; margin:0;">{tag}</p>
            <p style="color:#10b981; margin:0;">
                Price Range: {format_inr(low)} – {format_inr(high)}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- BAR CHART ----------------
    TOP_N = 10
    labels = [x[0] for x in result_data[:TOP_N]]
    values = [x[1] for x in result_data[:TOP_N]]

    st.markdown("### 📊 Gujarat Property Price Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")

    bars = ax.barh(labels, values, color="#22d3ee")

    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.01,
            bar.get_y() + bar.get_height()/2,
            format_inr(val),
            va="center",
            color="white"
        )

    ax.set_xlabel("Property Price (₹)", color="white")
    ax.tick_params(colors="white")
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)
