import streamlit as st
import scipy.io
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
# ---------- PROFESSIONAL HEADER ----------
col1, col2 = st.columns([1.2, 6])

with col1:
    st.image("endurance_logo.png", width=170)  # Logo file

with col2:
    st.markdown("""
    <h1 style="margin-bottom:0; color:#0A3D62; font-family:sans-serif;">RULion – Battery RUL Platform</h1>
    <h4 style="margin-top:0; color:gray; font-family:sans-serif;">
    Endurance Technologies Hackathon | Predictive Maintenance Dashboard
    </h4>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:2px solid #0A3D62;'>", unsafe_allow_html=True)


st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #F1F3F6;
    padding: 20px;
}
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: #0B3C5D;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="RULion | Endurance Technologies",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #0B3C5D;
    text-align: center;
}
.sub-title {
    font-size: 16px;
    color: #555;
    text-align: center;
    margin-bottom: 20px;
}
.divider {
    height: 4px;
    background-color: #0B3C5D;
    border-radius: 5px;
    margin-bottom: 25px;
}
</style>

<div class="main-title">🔋 RULion – Battery Remaining Useful Life Platform</div>
<div class="sub-title">
Endurance Technologies Hackathon | Data-Driven Predictive Maintenance
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🔧 Controls")
data_folder = "data"

# Load all .mat battery files
battery_files = []
for root, dirs, files in os.walk(data_folder):
    for f in files:
        if f.endswith(".mat"):
            battery_files.append(os.path.join(root, f))
battery_files.sort()

dropdown_dict = {os.path.relpath(f, data_folder): f for f in battery_files}
selected_name = st.sidebar.selectbox("Select Battery File:", list(dropdown_dict.keys()))
selected_file = dropdown_dict[selected_name]

future_cycles_num = st.sidebar.slider("Predict Future Cycles", min_value=10, max_value=100, value=50)

# -----------------------------
# Load battery helper
# -----------------------------
def load_battery(file_path):
    mat = scipy.io.loadmat(file_path)
    key = list(mat.keys())[-1]
    battery = mat[key]
    cycles = battery[0][0]['cycle'][0]
    capacity = []
    for c in cycles:
        if c['type'][0] == 'discharge':
            capacity.append(c['data'][0][0]['Capacity'][0][0])
    cycle_numbers = list(range(1, len(capacity)+1))
    return cycle_numbers, capacity

# -----------------------------
# Compute RUL helper
# -----------------------------
def compute_rul(capacity):
    """
    Compute Remaining Useful Life (RUL) as number of cycles left until capacity < 80% of initial
    """
    failure_capacity = 0.8 * capacity[0]
    rul = []
    for i, c in enumerate(capacity):
        remaining = 0
        for future_c in capacity[i:]:
            if future_c > failure_capacity:
                remaining += 1
        rul.append(remaining)
    return rul

# -----------------------------
# Load all batteries
# -----------------------------
all_data = {}
all_rul_data = {}

for file in battery_files:
    cycle_numbers, capacity = load_battery(file)
    rul = compute_rul(capacity)
    all_data[file] = (cycle_numbers, capacity)
    all_rul_data[file] = (cycle_numbers, rul)

# -----------------------------
# Selected battery
# -----------------------------
cycle_numbers, capacity = all_data[selected_file]
rul = all_rul_data[selected_file][1]
failure_capacity = 0.8 * capacity[0]

# Linear Regression for prediction using full RUL
X = np.array(cycle_numbers).reshape(-1,1)
y = np.array(rul)
model = LinearRegression()
model.fit(X, y)

future_cycles = np.array(range(cycle_numbers[-1]+1, cycle_numbers[-1]+1+future_cycles_num)).reshape(-1,1)
predicted_rul = model.predict(future_cycles)
predicted_rul = [max(0, r) for r in predicted_rul]

# -----------------------------
# KPIs / Metrics
# -----------------------------
st.markdown("""
<style>
.kpi-card {
    background-color: #F8F9FA;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}
.kpi-title {
    font-size: 16px;
    color: #555;
}
.kpi-value {
    font-size: 36px;
    font-weight: 700;
    color: #0B3C5D;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📊 Dashboard Overview")

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🔋 Total Batteries</div>
        <div class="kpi-value">{len(battery_files)}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">⏳ Current RUL (Cycles)</div>
        <div class="kpi-value">{rul[-1]}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📈 Predicted RUL (+{future_cycles_num} cycles)</div>
        <div class="kpi-value">{int(predicted_rul[0])}</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Battery Health Status
# -----------------------------
current_rul = rul[-1]
if current_rul > 50:
    status = "Healthy"
    color = "green"
elif 20 < current_rul <= 50:
    status = "Near Failure"
    color = "orange"
else:
    status = "Critical"
    color = "red"

st.markdown("""
<style>
.status-box {
    padding: 15px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

if current_rul > 50:
    status = "🟢 HEALTHY"
    bg_color = "#D4EDDA"
    text_color = "#155724"
elif 20 < current_rul <= 50:
    status = "🟠 NEAR FAILURE"
    bg_color = "#FFF3CD"
    text_color = "#856404"
else:
    status = "🔴 CRITICAL"
    bg_color = "#F8D7DA"
    text_color = "#721C24"

st.markdown(
    f"""
    <div class="status-box" style="background-color:{bg_color}; color:{text_color};">
        Battery Health Status: {status}
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📊 Multi-Battery View", "🔹 Individual Battery View"])

# -----------------------------
# Multi-Battery Tab
# -----------------------------
with tab1:

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("All Batteries Capacity Curves")
    fig = go.Figure()
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    for i, file in enumerate(battery_files):
        cycle_nums, cap = all_data[file]
        fig.add_trace(go.Scatter(x=cycle_nums, y=cap, mode='lines+markers',
                                 name=os.path.relpath(file, data_folder),
                                 line=dict(color=colors[i%len(colors)]),
                                 hovertemplate='Cycle: %{x}<br>Capacity: %{y:.3f}'))
    fig.update_layout(title="Multi-Battery Capacity Degradation", xaxis_title="Cycle Number", yaxis_title="Capacity (Ah)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Individual Battery Tab
# -----------------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"{selected_name} Capacity")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=cycle_numbers, y=capacity, mode='lines+markers',
                                  name='Capacity', line=dict(color='blue'),
                                  hovertemplate='Cycle: %{x}<br>Capacity: %{y:.3f}'))
        fig1.add_trace(go.Scatter(x=cycle_numbers, y=[failure_capacity]*len(cycle_numbers), mode='lines',
                                  name='Failure Threshold (80%)', line=dict(color='red', dash='dash')))
        fig1.update_layout(title="Capacity Curve", xaxis_title="Cycle Number", yaxis_title="Capacity (Ah)")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"{selected_name} RUL Prediction")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cycle_numbers, y=rul, mode='lines+markers',
                                  name='Current RUL', line=dict(color='green'),
                                  hovertemplate='Cycle: %{x}<br>RUL: %{y:.0f}'))
        fig2.add_trace(go.Scatter(x=list(range(cycle_numbers[-1]+1, cycle_numbers[-1]+1+future_cycles_num)), y=predicted_rul,
                                  mode='lines+markers', name='Predicted RUL',
                                  line=dict(color='red', dash='dash'),
                                  hovertemplate='Cycle: %{x}<br>Predicted RUL: %{y:.0f}'))
        fig2.update_layout(title="RUL Prediction", xaxis_title="Cycle Number", yaxis_title="RUL (cycles)")
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Multi-Battery Summary Table
# -----------------------------
st.markdown("---")

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("⚡ Multi-Battery Health Summary")
summary = []
for file in battery_files:
    last_rul = all_rul_data[file][1][-1]
    summary.append([os.path.relpath(file, data_folder), last_rul])
summary_df = pd.DataFrame(summary, columns=["Battery", "Current RUL"])

def color_health(val):
    if val < 20:
        color = 'background-color: red'
    elif 20 <= val < 50:
        color = 'background-color: orange'
    else:
        color = 'background-color: lightgreen'
    return color

st.dataframe(summary_df.style.applymap(color_health, subset=['Current RUL']))

# -----------------------------
# Export Button
# -----------------------------
export_data = []
for file in battery_files:
    cycle_nums, rul_vals = all_rul_data[file]
    future_cycles_file = np.array(range(cycle_nums[-1]+1, cycle_nums[-1]+1+future_cycles_num)).reshape(-1,1)
    predicted_rul_file = LinearRegression().fit(
        np.array(cycle_nums).reshape(-1,1),
        np.array(rul_vals)
    ).predict(future_cycles_file)
    predicted_rul_file = [max(0,r) for r in predicted_rul_file]
    export_data.extend(list(zip(
        [os.path.relpath(file, data_folder)]*len(cycle_nums + list(range(cycle_nums[-1]+1, cycle_nums[-1]+1+future_cycles_num))),
        cycle_nums + list(range(cycle_nums[-1]+1, cycle_nums[-1]+1+future_cycles_num)),
        rul_vals + predicted_rul_file
    )))

export_df = pd.DataFrame(export_data, columns=["Battery", "Cycle", "RUL"])
st.download_button(
    label=f"📥 Download All Batteries RUL CSV",
    data=export_df.to_csv(index=False),
    file_name="All_Batteries_RUL.csv",
    mime="text/csv"
)
st.success("✅ All batteries RUL data ready for download")

