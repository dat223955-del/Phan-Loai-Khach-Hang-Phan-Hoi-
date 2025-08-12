import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Cấu hình giao diện
st.set_page_config(page_title="Dự đoán phản hồi Marketing", layout="centered")
st.title("📊 Dự đoán phản hồi chiến dịch Marketing")

# Đường dẫn model và scaler
model_path = "model.pkl"
scaler_path = "scaler.pkl"
features_path = "features_used.pkl"

# Load model, scaler, features
if not all(os.path.exists(path) for path in [model_path, scaler_path, features_path]):
    st.error("❌ Không tìm thấy một trong các file: `model.pkl`, `scaler.pkl`, `features_used.pkl`.")
    st.stop()

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(features_path)
except Exception as e:
    st.error(f"❌ Lỗi khi tải file: {e}")
    st.stop()

# Nhập thông tin người dùng
st.markdown("Nhập thông tin khách hàng để dự đoán khả năng phản hồi chiến dịch:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        year_birth = st.number_input("🗓️ Năm sinh", 1940, 2025, 1980)
        income = st.number_input("💰 Thu nhập hàng năm (USD)", 0.0, 1_000_000.0, 40000.0)
        kidhome = st.number_input("👶 Số trẻ nhỏ trong nhà", 0, 5, 0)
        teenhome = st.number_input("🧒 Số trẻ vị thành niên trong nhà", 0, 5, 1)
        recency = st.number_input("📅 Số ngày kể từ lần tương tác gần nhất", 0, 100, 10)
        mnt_wines = st.number_input("🍷 Chi tiêu cho rượu vang (USD)", 0, 10000, 200)
        mnt_fruits = st.number_input("🍎 Chi tiêu cho trái cây (USD)", 0, 10000, 50)
        mnt_meat = st.number_input("🥩 Chi tiêu cho thịt (USD)", 0, 10000, 150)
        mnt_fish = st.number_input("🐟 Chi tiêu cho cá (USD)", 0, 10000, 100)
        mnt_sweets = st.number_input("🍬 Chi tiêu cho đồ ngọt (USD)", 0, 10000, 30)
        mnt_gold = st.number_input("🥇 Chi tiêu cho sản phẩm vàng (USD)", 0, 10000, 50)
        education = st.selectbox("🎓 Trình độ học vấn", ["Tốt nghiệp Đại học", "Tiến sĩ", "Thạc sĩ", "Chu kỳ 2", "Cơ bản"])
        marital_status = st.selectbox("💍 Tình trạng hôn nhân", ["Độc thân", "Đã kết hôn", "Sống chung", "Ly hôn", "Góa"])

    with col2:
        num_deals = st.number_input("🛍️ Số lần mua hàng qua khuyến mãi", 0, 50, 2)
        num_web = st.number_input("🌐 Số lần mua qua website", 0, 50, 5)
        num_catalog = st.number_input("📦 Số lần mua qua catalog", 0, 50, 1)
        num_store = st.number_input("🏬 Số lần mua tại cửa hàng", 0, 50, 6)
        num_visits = st.number_input("📈 Số lần truy cập website mỗi tháng", 0, 50, 4)
        cmp1 = st.selectbox("📩 Phản hồi chiến dịch 1?", ["Không", "Có"]) == "Có"
        cmp2 = st.selectbox("📩 Phản hồi chiến dịch 2?", ["Không", "Có"]) == "Có"
        cmp3 = st.selectbox("📩 Phản hồi chiến dịch 3?", ["Không", "Có"]) == "Có"
        cmp4 = st.selectbox("📩 Phản hồi chiến dịch 4?", ["Không", "Có"]) == "Có"
        cmp5 = st.selectbox("📩 Phản hồi chiến dịch 5?", ["Không", "Có"]) == "Có"
        complain = st.selectbox("⚠️ Có từng khiếu nại?", ["Không", "Có"]) == "Có"
        z_cost = st.number_input("💸 Chi phí tiếp cận khách hàng", 0, 1000, 3)
        z_revenue = st.number_input("💸 Doanh thu từ khách hàng", 0, 1000, 11)

        submitted = st.form_submit_button("🔍 Dự đoán")

# Mapping label
edu_map_vi = {
    "Tốt nghiệp Đại học": 0,
    "Tiến sĩ": 1,
    "Thạc sĩ": 2,
    "Chu kỳ 2": 3,
    "Cơ bản": 4
}
marital_map_vi = {
    "Góa": 0,
    "Ly hôn": 1,
    "Sống chung": 2,
    "Đã kết hôn": 3,
    "Độc thân": 4
}

# Dự đoán
if submitted:
    try:
        input_dict = {
            "Year_Birth": year_birth,
            "Education": edu_map_vi[education],
            "Marital_Status": marital_map_vi[marital_status],
            "Income": income,
            "Kidhome": kidhome,
            "Teenhome": teenhome,
            "Recency": recency,
            "MntWines": mnt_wines,
            "MntFruits": mnt_fruits,
            "MntMeatProducts": mnt_meat,
            "MntFishProducts": mnt_fish,
            "MntSweetProducts": mnt_sweets,
            "MntGoldProds": mnt_gold,
            "NumDealsPurchases": num_deals,
            "NumWebPurchases": num_web,
            "NumCatalogPurchases": num_catalog,
            "NumStorePurchases": num_store,
            "NumWebVisitsMonth": num_visits,
            "AcceptedCmp1": int(cmp1),
            "AcceptedCmp2": int(cmp2),
            "AcceptedCmp3": int(cmp3),
            "AcceptedCmp4": int(cmp4),
            "AcceptedCmp5": int(cmp5),
            "Complain": int(complain),
            "Z_CostContact": z_cost,
            "Z_Revenue": z_revenue
        }

        # Chuyển thành DataFrame và chọn đúng cột
        df_input = pd.DataFrame([input_dict])
        df_selected = df_input[selected_features]  # Quan trọng: đúng tên và thứ tự

        # Scale
        input_scaled = scaler.transform(df_selected)

        # Dự đoán
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

        # Hiển thị kết quả
        st.subheader("🧠 Kết quả dự đoán:")
        if prediction == 1:
            st.success("✅ Người này **CÓ khả năng phản hồi** chiến dịch marketing.")
        else:
            st.warning("❌ Người này **KHÔNG có khả năng phản hồi** chiến dịch.")

        if prob is not None:
            st.info(f"🔢 Xác suất phản hồi: **{prob:.2%}**")

    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình dự đoán: {e}")

      

#cd "C:\Users\DAT HANDSOME\OneDrive\Máy tính\DOAN1"
#python -m streamlit run DoAN.py



