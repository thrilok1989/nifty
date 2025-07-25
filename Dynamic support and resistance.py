import streamlit as st from streamlit_autorefresh import st_autorefresh import requests import pandas as pd import numpy as np from datetime import datetime import math from scipy.stats import norm from pytz import timezone import plotly.graph_objects as go

=== Streamlit Config ===

st.set_page_config(page_title="Nifty Options Analyzer", layout="wide") st_autorefresh(interval=300000, key="datarefresh")  # Refresh every 5 min

=== Sidebar: Dynamic Sensitivity Tuning ===

st.sidebar.title("🔧 Zone Sensitivity Settings") zone_threshold = st.sidebar.slider("Support/Resistance Strength Multiplier", 1.0, 2.0, 1.2, 0.1) range_pts = st.sidebar.slider("Strike Range Around Spot", 50, 300, 100, 10) vol_threshold = st.sidebar.slider("Minimum Volume Threshold", 500, 10000, 2000, 500)

=== Telegram Config ===

TELEGRAM_BOT_TOKEN = "<your_token_here>" TELEGRAM_CHAT_ID = "<your_chat_id_here>"

def send_telegram_message(message): url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" data = {"chat_id": TELEGRAM_CHAT_ID, "text": message} try: response = requests.post(url, data=data) if response.status_code != 200: st.warning("⚠️ Telegram message failed.") except Exception as e: st.error(f"❌ Telegram error: {e}")

=== Option Greeks Calculation ===

def calculate_greeks(option_type, S, K, T, r, sigma): d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T)) d2 = d1 - sigma * math.sqrt(T) delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1) gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T)) vega = S * norm.pdf(d1) * math.sqrt(T) / 100 theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2) if option_type == 'CE' else - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365 rho = (K * T * math.exp(-r * T) * norm.cdf(d2) if option_type == 'CE' else -K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100 return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

=== Bias and Zone Utilities ===

def delta_volume_bias(price, volume, chg_oi): try: if price > 0 and volume > 0 and chg_oi > 0: return "Long Buildup" elif price < 0 and volume > 0 and chg_oi > 0: return "Short Buildup" elif price < 0 and volume > 0 and chg_oi < 0: return "Long Unwinding" elif price > 0 and volume > 0 and chg_oi < 0: return "Short Covering" else: return "Neutral" except: return "Neutral"

def detect_nearby_zones(df, spot): nearby_df = df[df['strikePrice'].between(spot - range_pts, spot + range_pts)].copy()

nearby_df['ZoneTag'] = nearby_df.apply(
    lambda row:
        'Support' if row['openInterest_PE'] > zone_threshold * row['openInterest_CE'] and row['totalTradedVolume_PE'] > vol_threshold else
        'Resistance' if row['openInterest_CE'] > zone_threshold * row['openInterest_PE'] and row['totalTradedVolume_CE'] > vol_threshold else
        'Neutral', axis=1)

support_zones = group_by_continuity(nearby_df[nearby_df['ZoneTag'] == 'Support']['strikePrice'].tolist())
resistance_zones = group_by_continuity(nearby_df[nearby_df['ZoneTag'] == 'Resistance']['strikePrice'].tolist())
return support_zones, resistance_zones

def group_by_continuity(strike_list): strike_list.sort() zones = [] if not strike_list: return zones current = [strike_list[0]] for i in range(1, len(strike_list)): if strike_list[i] - strike_list[i - 1] <= 50: current.append(strike_list[i]) else: if len(current) >= 2: zones.append((min(current), max(current))) current = [strike_list[i]] if len(current) >= 2: zones.append((min(current), max(current))) return zones

def is_in_zone(price, zones): return any(low <= price <= high for (low, high) in zones)

def plot_oi_zones(df, support_zones, resistance_zones): fig = go.Figure() fig.add_trace(go.Bar(x=df['strikePrice'], y=df['openInterest_CE'], name='Call OI', marker_color='red')) fig.add_trace(go.Bar(x=df['strikePrice'], y=df['openInterest_PE'], name='Put OI', marker_color='green'))

for low, high in support_zones:
    fig.add_vrect(x0=low, x1=high, fillcolor="lightgreen", opacity=0.3, layer="below", line_width=0)

for low, high in resistance_zones:
    fig.add_vrect(x0=low, x1=high, fillcolor="lightcoral", opacity=0.3, layer="below", line_width=0)

fig.update_layout(title="OI Zones with Support & Resistance", barmode='group')
st.plotly_chart(fig, use_container_width=True)

def main(): try: now = datetime.now(timezone("Asia/Kolkata")) session = requests.Session() session.headers.update({"User-Agent": "Mozilla/5.0"}) session.get("https://www.nseindia.com", timeout=5) url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY" response = session.get(url, timeout=10) data = response.json()

records = data['records']['data']
    expiry = data['records']['expiryDates'][0]
    underlying = data['records']['underlyingValue']

    today = datetime.today()
    expiry_date = datetime.strptime(expiry, "%d-%b-%Y")
    T = max((expiry_date - today).days, 1) / 365
    r = 0.06

    calls, puts = [], []

    for item in records:
        if 'CE' in item and item['CE']['expiryDate'] == expiry:
            ce = item['CE']
            if ce['impliedVolatility'] > 0:
                ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                                   calculate_greeks('CE', underlying, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100))))
            calls.append(ce)

        if 'PE' in item and item['PE']['expiryDate'] == expiry:
            pe = item['PE']
            if pe['impliedVolatility'] > 0:
                pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                                   calculate_greeks('PE', underlying, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100))))
            puts.append(pe)

    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

    st.markdown(f"### 📍 Spot Price: {underlying}")

    support_zones, resistance_zones = detect_nearby_zones(df, underlying)

    zone_msg_parts = []
    if support_zones:
        support_str = ', '.join([f"{low}-{high}" for low, high in support_zones])
        zone_msg_parts.append(f"📉 Support Zone(s): {support_str}")
        st.success(f"📉 Support Zone(s): {support_str}")

    if resistance_zones:
        resist_str = ', '.join([f"{low}-{high}" for low, high in resistance_zones])
        zone_msg_parts.append(f"📈 Resistance Zone(s): {resist_str}")
        st.warning(f"📈 Resistance Zone(s): {resist_str}")

    zone_summary_text = "\n".join(zone_msg_parts)
    if zone_summary_text:
        send_telegram_message(zone_summary_text)

    plot_oi_zones(df, support_zones, resistance_zones)

except Exception as e:
    st.error(f"❌ Error: {e}")
    send_telegram_message(f"❌ Error: {str(e)}")

if name == 'main': main()

