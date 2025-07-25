import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone
import plotly.graph_objects as go

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=300000, key="datarefresh")  # Refresh every 5 min

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

def calculate_greeks(option_type, S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
             if option_type == 'CE' else - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) +
             r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    rho = (K * T * math.exp(-r * T) * norm.cdf(d2)
           if option_type == 'CE' else -K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
    return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

def group_by_continuity(strike_list):
    strike_list.sort()
    zones = []
    if not strike_list:
        return zones
    current = [strike_list[0]]
    for i in range(1, len(strike_list)):
        if strike_list[i] - strike_list[i - 1] <= 50:
            current.append(strike_list[i])
        else:
            if len(current) >= 2:
                zones.append((min(current), max(current)))
            current = [strike_list[i]]
    if len(current) >= 2:
        zones.append((min(current), max(current)))
    return zones

def detect_nearby_zones(df, spot, threshold=1.2, range_pts=100, vol_threshold=2000):
    nearby_df = df[df['strikePrice'].between(spot - range_pts, spot + range_pts)].copy()
    nearby_df['ZoneTag'] = nearby_df.apply(
        lambda row: 'Support' if row['openInterest_PE'] > threshold * row['openInterest_CE']
                                 and row['totalTradedVolume_PE'] > vol_threshold else
                    'Resistance' if row['openInterest_CE'] > threshold * row['openInterest_PE']
                                   and row['totalTradedVolume_CE'] > vol_threshold else
                    'Neutral', axis=1)
    support_zones = group_by_continuity(nearby_df[nearby_df['ZoneTag'] == 'Support']['strikePrice'].tolist())
    resistance_zones = group_by_continuity(nearby_df[nearby_df['ZoneTag'] == 'Resistance']['strikePrice'].tolist())
    return support_zones, resistance_zones

def plot_oi_zones(df, support_zones, resistance_zones):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['strikePrice'], y=df['openInterest_CE'], name='Call OI', marker_color='red'))
    fig.add_trace(go.Bar(x=df['strikePrice'], y=df['openInterest_PE'], name='Put OI', marker_color='green'))
    for low, high in support_zones:
        fig.add_vrect(x0=low, x1=high, fillcolor="lightgreen", opacity=0.3, layer="below", line_width=0)
    for low, high in resistance_zones:
        fig.add_vrect(x0=low, x1=high, fillcolor="lightcoral", opacity=0.3, layer="below", line_width=0)
    fig.update_layout(title="OI Zones with Support & Resistance", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Gamma_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
}

def delta_volume_bias(price, volume, chg_oi):
    if price > 0 and volume > 0 and chg_oi > 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi > 0:
        return "Bearish"
    elif price > 0 and volume > 0 and chg_oi < 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi < 0:
        return "Bearish"
    else:
        return "Neutral"

def final_verdict(score):
    if score >= 4:
        return "Strong Bullish"
    elif score >= 2:
        return "Bullish"
    elif score <= -4:
        return "Strong Bearish"
    elif score <= -2:
        return "Bearish"
    else:
        return "Neutral"

def is_spot_within_zone(spot, zones):
    for low, high in zones:
        if low <= spot <= high:
            return (low, high)
    return None

def main():
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=5)
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        response = session.get(url, timeout=10)
        data = response.json()

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

        support_zones, resistance_zones = detect_nearby_zones(df, underlying)

        st.markdown(f"### 📍 Spot Price: {underlying}")

        msg_parts = []
        if support_zones:
            support_str = ', '.join([f"{low}-{high}" for low, high in support_zones])
            st.success(f"📉 Support Zone(s): {support_str}")
            msg_parts.append(f"📉 Support Zone(s): {support_str}")
        if resistance_zones:
            resist_str = ', '.join([f"{low}-{high}" for low, high in resistance_zones])
            st.warning(f"📈 Resistance Zone(s): {resist_str}")
            msg_parts.append(f"📈 Resistance Zone(s): {resist_str}")
        if msg_parts:
            send_telegram_message("\n".join(msg_parts))

        plot_oi_zones(df, support_zones, resistance_zones)

        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')

        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            if abs(row['strikePrice'] - atm_strike) > 100:
                continue

            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
                "Gamma_Bias": "Bullish" if row['Gamma_CE'] < row['Gamma_PE'] else "Bearish",
                "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
                "BidQty_Bias": "Bearish" if row['bidQty_PE'] > row['bidQty_CE'] else "Bullish",
                "IV_Bias": "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish",
                "DVP_Bias": delta_volume_bias(
                    row['lastPrice_CE'] - row['lastPrice_PE'],
                    row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                    row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
                )
            }

            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)

            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

        df_summary = pd.DataFrame(bias_results)
        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"

        st.success(f"🧠 Market View: **{market_view}**")
        st.dataframe(df_summary)

        # Signal logic
        signal_sent = False
        for row in bias_results:
            strike = row['Strike']
            zone_range = is_spot_within_zone(underlying, support_zones if "Bullish" in row["Verdict"] else resistance_zones)
            if zone_range is None:
                continue

            if row['Verdict'] not in ["Bullish", "Strong Bullish", "Bearish", "Strong Bearish"]:
                continue

            option_type = 'CE' if "Bullish" in row['Verdict'] else 'PE'
            ltp = df.loc[df['strikePrice'] == strike, f'lastPrice_{option_type}'].values[0]
            iv = df.loc[df['strikePrice'] == strike, f'impliedVolatility_{option_type}'].values[0]
            target = round(ltp * (1 + iv / 100), 2)
            stop_loss = round(ltp * 0.8, 2)

            signal_msg = f"📍 Spot: {underlying}\n🔹 {'CALL' if option_type == 'CE' else 'PUT'} Entry Zone {zone_range[0]}-{zone_range[1]}\nStrike: {strike} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}\nBias Score: {row['BiasScore']} ({row['Verdict']})"
            send_telegram_message(signal_msg)
            st.info(signal_msg)
            signal_sent = True
            break

        if not signal_sent:
            send_telegram_message(f"📍 Spot: {underlying}\nNo Signal Triggered — Out of Zone or Weak Bias")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        send_telegram_message(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()