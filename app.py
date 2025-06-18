import streamlit as st
import pandas as pd
import joblib
from streamlit_folium import st_folium
import folium
from math import radians, cos, sin, sqrt, atan2

# 모델과 데이터 불러오기
model = joblib.load("pet_rf_model_gps.pkl")
df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")

# 위도, 경도 DMS → DD 변환 함수
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
df["Lon_dd"] = df["Lon"].apply(dms_to_dd)

# 거리 계산 함수 (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# UI
st.title("🌳 보행자 열쾌적성 예측 시스템")
st.markdown("지도에서 위치를 클릭하면 SVF, GVI, BVI 기반 PET를 예측합니다.")

# 지도 중심 위치 설정
m = folium.Map(location=[35.2322, 129.084], zoom_start=16)

# 클릭 이벤트
click_data = st_folium(m, width=700, height=500)

# 클릭 처리
if click_data and click_data.get("last_clicked"):
    lat = click_data["last_clicked"]["lat"]
    lon = click_data["last_clicked"]["lng"]
    st.success(f"클릭한 위치: 위도 {lat:.5f}, 경도 {lon:.5f}")

    # 가장 가까운 측정 위치 찾기
    df["dist"] = df.apply(lambda row: haversine(lat, lon, row["Lat_dd"], row["Lon_dd"]), axis=1)
    nearest = df.loc[df["dist"].idxmin()]

    st.markdown("### 📍 가장 가까운 측정 지점 정보")
    st.write(nearest[["측정위치", "SVF", "GVI", "BVI", "AirTemperature", "Humidity", "WindSpeed", "PET"]])

    # 예측
    X_pred = nearest[["SVF", "GVI", "BVI", "AirTemperature", "Humidity", "WindSpeed"]].values.reshape(1, -1)
    predicted_pet = model.predict(X_pred)[0]

    st.markdown(f"### 🔥 예측된 PET: **{predicted_pet:.2f}°C**")
else:
    st.info("지도를 클릭해 주세요.")