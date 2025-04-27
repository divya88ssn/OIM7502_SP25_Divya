# Install if not already:
# pip install streamlit pandas numpy scikit-learn matplotlib seaborn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- Section 1: Load and Prepare Data ---

# Load Zillow datasets
zhvi_df = pd.read_csv("fin_proj/data/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
zori_df = pd.read_csv("fin_proj/data/Metro_zori_uc_sfrcondomfr_sm_month.csv")

# Constants
d = 0.20       # Down payment
r = 0.065 / 12 # Monthly interest
i = 0.004      # Insurance rate
t = 0.0125     # Property tax rate
n = 360        # Loan term months

# Melt datasets
zhvi_long = zhvi_df.melt(
    id_vars=['RegionID', 'RegionName', 'RegionType', 'StateName', 'SizeRank'],
    var_name='Date',
    value_name='ZHVI'
)
zori_long = zori_df.melt(
    id_vars=['RegionID', 'RegionName', 'RegionType', 'StateName', 'SizeRank'],
    var_name='Date',
    value_name='ZORI'
)

zhvi_long.dropna(subset=['ZHVI'], inplace=True)
zori_long.dropna(subset=['ZORI'], inplace=True)
zhvi_long['Date'] = pd.to_datetime(zhvi_long['Date'])
zori_long['Date'] = pd.to_datetime(zori_long['Date'])

# Mortgage and Income Calculations
zhvi_long['Mortgage'] = zhvi_long['ZHVI'] * (1 - d) * r * (1 + r) ** n / ((1 + r) ** n - 1)
zhvi_long['TotalMonthlyPayment'] = zhvi_long['Mortgage'] + zhvi_long['ZHVI'] * ((i + t + 0.005) / 12)
zhvi_long['IncomeNeededToAffordHome'] = 12 * zhvi_long['TotalMonthlyPayment'] / 0.3
zori_long['IncomeNeededToAffordRent'] = 12 * zori_long['ZORI'] / 0.3

# YoY % Change
zhvi_long = zhvi_long.sort_values(['RegionName', 'Date'])
zori_long = zori_long.sort_values(['RegionName', 'Date'])
zhvi_long['ZHVI_YoY'] = zhvi_long.groupby('RegionName')['ZHVI'].pct_change(periods=12) * 100
zori_long['ZORI_YoY'] = zori_long.groupby('RegionName')['ZORI'].pct_change(periods=12) * 100

# Latest snapshot
latest_date = zhvi_long['Date'].max()
latest_home = zhvi_long[zhvi_long['Date'] == latest_date][['RegionName', 'TotalMonthlyPayment', 'IncomeNeededToAffordHome']]
latest_rent = zori_long[zori_long['Date'] == latest_date][['RegionName', 'IncomeNeededToAffordRent']]
region_summary = pd.merge(latest_home, latest_rent, on='RegionName', how='inner')

# --- Section 2: Clustering and Anomaly Detection ---

# Affordability clustering
avg_features = region_summary[['IncomeNeededToAffordHome', 'IncomeNeededToAffordRent']].dropna()
kmeans_avg = KMeans(n_clusters=3, random_state=42)
region_summary['Cluster_Avg'] = kmeans_avg.fit_predict(avg_features)

# Volatility clustering
zhvi_vol = zhvi_long.groupby('RegionName')['ZHVI_YoY'].std().reset_index(name='Volatility_ZHVI')
zori_vol = zori_long.groupby('RegionName')['ZORI_YoY'].std().reset_index(name='Volatility_ZORI')
volatility = pd.merge(zhvi_vol, zori_vol, on='RegionName', how='inner')
vol_clean = volatility.dropna()
vol_features = vol_clean[['Volatility_ZHVI', 'Volatility_ZORI']]
kmeans_vol = KMeans(n_clusters=3, random_state=42)
vol_clean['Cluster_Vol'] = kmeans_vol.fit_predict(vol_features)
volatility = pd.merge(volatility[['RegionName']], vol_clean[['RegionName', 'Cluster_Vol']], on='RegionName', how='left')
region_summary = pd.merge(region_summary, volatility, on='RegionName', how='left')

# Mark anomalies manually
region_summary['Is_Anomaly'] = 0
region_summary.loc[
    (region_summary['IncomeNeededToAffordHome'] > 400000) | (region_summary['Cluster_Vol'] == 2),
    'Is_Anomaly'
] = 1

# --- Section 3: Streamlit Layout ---

st.set_page_config(page_title="🏡 US Metro Housing Explorer", layout="wide")
st.title("🏡 US Metro Housing Affordability Explorer")
st.markdown(f"**Data snapshot:** {latest_date.strftime('%B %Y')}")

# --- Section 4: Single Metro Explorer ---

st.header("📍 Explore a Single Metro")

selected_region = st.selectbox("Select Metro:", region_summary['RegionName'].sort_values())

if selected_region:
    region_data = region_summary[region_summary['RegionName'] == selected_region].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Monthly Payment (Mortgage)", f"${region_data['TotalMonthlyPayment']:,.0f}")
    with col2:
        st.metric("Income Needed to Afford Buying", f"${region_data['IncomeNeededToAffordHome']:,.0f}")
    with col3:
        st.metric("Income Needed to Afford Renting", f"${region_data['IncomeNeededToAffordRent']:,.0f}")

    st.subheader("Affordability Cluster")
    if region_data['Cluster_Avg'] == 2:
        st.success("🟢 Lower cost metro (more affordable).")
    elif region_data['Cluster_Avg'] == 1:
        st.error("🔴 Higher tier metro (less affordable).")
    elif region_data['Cluster_Avg'] == 0:
        st.warning("🟡 Mid-tier cost metro.")

    st.subheader("Volatility Cluster")
    if region_data['Cluster_Vol'] == 0:
        st.success("🟢 Stable price changes.")
    elif region_data['Cluster_Vol'] == 1:
        st.warning("🟡 Moderate volatility.")
    elif region_data['Cluster_Vol'] == 2:
        st.error("🔴 High volatility metro.")

    st.subheader("Anomaly Status")
    if region_data['Is_Anomaly'] == 1:
        st.error("⚠️ Flagged as Anomaly!")
    else:
        st.success("✅ No anomaly detected.")

    # --- Section 5: YoY Trendlines for Single Metro ---

    st.header("📈 Year-over-Year (YoY) Trends for Selected Metro")

    temp_zhvi = zhvi_long[zhvi_long['RegionName'] == selected_region]
    temp_zori = zori_long[zori_long['RegionName'] == selected_region]

    fig2, ax2 = plt.subplots(figsize=(20, 8))
    ax2.plot(temp_zhvi['Date'], temp_zhvi['ZHVI_YoY'], label='Home Value Index (ZHVI) YoY %', color='blue')
    ax2.plot(temp_zori['Date'], temp_zori['ZORI_YoY'], label='Rental Index (ZORI) YoY %', color='green')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title(f"{selected_region} - YoY % Change in Home Value and Rent Indices")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('YoY % Change')
    ax2.legend()
    st.pyplot(fig2)

# --- Section 6: Affordability Distribution Across Metros ---

st.header("📊 Affordability Distribution Across Metros")

affordability_long = pd.DataFrame({
    'AffordabilityType': ['Buy'] * len(region_summary) + ['Rent'] * len(region_summary),
    'IncomeNeeded': pd.concat([
        region_summary['IncomeNeededToAffordHome'],
        region_summary['IncomeNeededToAffordRent']
    ], ignore_index=True)
})

fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.boxplot(x='AffordabilityType', y='IncomeNeeded', data=affordability_long, ax=ax1)
ax1.set_title('Income Needed to Afford Buying vs Renting')
st.pyplot(fig1)

# --- Section 7: Compare Two Metros Side-by-Side (Separate) ---

st.header("🏙️ Compare Two Metros Independently")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📍 Metro A")
    metro_a = st.selectbox("Select Metro A:", region_summary['RegionName'].sort_values(), key="metro_a")

    if metro_a:
        metro_a_data = region_summary[region_summary['RegionName'] == metro_a].iloc[0]

        st.metric("Monthly Payment (Mortgage)", f"${metro_a_data['TotalMonthlyPayment']:,.0f}")
        st.metric("Income Needed to Afford Buying", f"${metro_a_data['IncomeNeededToAffordHome']:,.0f}")
        st.metric("Income Needed to Afford Renting", f"${metro_a_data['IncomeNeededToAffordRent']:,.0f}")

        st.markdown("#### Affordability Cluster")
        if metro_a_data['Cluster_Avg'] == 0:
            st.success("🟢 Lower cost metro (more affordable).")
        elif metro_a_data['Cluster_Avg'] == 1:
            st.error("🔴 Higher tier metro (less affordable).")
        elif metro_a_data['Cluster_Avg'] == 2:
            st.warning("🟡 Mid-tier cost metro.")

        st.markdown("#### Volatility Cluster")
        if metro_a_data['Cluster_Vol'] == 0:
            st.success("🟢 Stable price changes.")
        elif metro_a_data['Cluster_Vol'] == 1:
            st.warning("🟡 Moderate volatility.")
        elif metro_a_data['Cluster_Vol'] == 2:
            st.error("🔴 High volatility metro.")

        st.markdown("#### Anomaly Status")
        if metro_a_data['Is_Anomaly'] == 1:
            st.error("⚠️ Flagged as Anomaly!")
        else:
            st.success("✅ No anomaly detected.")

with col_b:
    st.subheader("📍 Metro B")
    metro_b = st.selectbox("Select Metro B:", region_summary['RegionName'].sort_values(), key="metro_b")

    if metro_b:
        metro_b_data = region_summary[region_summary['RegionName'] == metro_b].iloc[0]

        st.metric("Monthly Payment (Mortgage)", f"${metro_b_data['TotalMonthlyPayment']:,.0f}")
        st.metric("Income Needed to Afford Buying", f"${metro_b_data['IncomeNeededToAffordHome']:,.0f}")
        st.metric("Income Needed to Afford Renting", f"${metro_b_data['IncomeNeededToAffordRent']:,.0f}")

        st.markdown("#### Affordability Cluster")
        if metro_b_data['Cluster_Avg'] == 0:
            st.success("🟢 Lower cost metro (more affordable).")
        elif metro_b_data['Cluster_Avg'] == 1:
            st.error("🔴 Higher tier metro (less affordable).")
        elif metro_b_data['Cluster_Avg'] == 2:
            st.warning("🟡 Mid-tier cost metro.")

        st.markdown("#### Volatility Cluster")
        if metro_b_data['Cluster_Vol'] == 0:
            st.success("🟢 Stable price changes.")
        elif metro_b_data['Cluster_Vol'] == 1:
            st.warning("🟡 Moderate volatility.")
        elif metro_b_data['Cluster_Vol'] == 2:
            st.error("🔴 High volatility metro.")

        st.markdown("#### Anomaly Status")
        if metro_b_data['Is_Anomaly'] == 1:
            st.error("⚠️ Flagged as Anomaly!")
        else:
            st.success("✅ No anomaly detected.")

st.caption("Built with Zillow datasets | Powered by Streamlit 🚀")

