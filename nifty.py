import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone

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
             if option_type == 'CE' else
             - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    rho = (K * T * math.exp(-r * T) * norm.cdf(d2) if option_type == 'CE'
           else -K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
    return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

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

def sudden_liquidity_spike(row):
    ce_spike = row['changeinOpenInterest_CE'] > 1.5 * row['openInterest_CE'] and row['totalTradedVolume_CE'] > 1500
    pe_spike = row['changeinOpenInterest_PE'] > 1.5 * row['openInterest_PE'] and row['totalTradedVolume_PE'] > 1500
    return ce_spike or pe_spike

weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Gamma_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
}

def determine_level(row):
    if row['openInterest_PE'] > 1.5 * row['openInterest_CE'] and row['changeinOpenInterest_PE'] > row['changeinOpenInterest_CE']:
        return "Support"
    elif row['openInterest_CE'] > 1.5 * row['openInterest_PE'] and row['changeinOpenInterest_CE'] > row['changeinOpenInterest_PE']:
        return "Resistance"
    else:
        return "Neutral"

def is_in_zone(spot, strike, level):
    if level == "Support":
        return strike - 20 <= spot <= strike + 10
    elif level == "Resistance":
        return strike - 10 <= spot <= strike + 20
    return False

def analyze():
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:00", "%H:%M").time()
        market_end = datetime.strptime("15:40", "%H:%M").time()

        if 'last_closed_alert' not in st.session_state:
            st.session_state.last_closed_alert = None

        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market is closed. Script will resume during trading hours.")
            if (
                st.session_state.last_closed_alert is None or
                (now - st.session_state.last_closed_alert).seconds > 3600
            ):
                send_telegram_message("⏳ Market is closed. Script will resume during trading hours (Mon–Fri 9:00–15:40).")
                st.session_state.last_closed_alert = now
            return

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

        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            if abs(row['strikePrice'] - atm_strike) > 100:
                continue

            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
                "Gamma_Bias": "Bullish" if row['Gamma_CE'] < row['Gamma_PE'] else "Bearish",
                "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
                "BidQty_Bias": "Bearish" if row['bidQty_PE'] < row['bidQty_CE'] else "Bullish",
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

            if sudden_liquidity_spike(row):
                send_telegram_message(
                    f"⚡ Sudden Liquidity Spike!\nStrike: {row['strikePrice']}\nCE OI Chg: {row['changeinOpenInterest_CE']} | PE OI Chg: {row['changeinOpenInterest_PE']}\nVol CE: {row['totalTradedVolume_CE']} | PE: {row['totalTradedVolume_PE']}"
                )

        df_summary = pd.DataFrame(bias_results)
        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"

        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False

        for row in bias_results:
            if is_in_zone(underlying, row['Strike'], row['Level']) and row['Level'] == "Support" and total_score >= 4 and "Bullish" in market_view:
                option_type = 'CE'
            elif is_in_zone(underlying, row['Strike'], row['Level']) and row['Level'] == "Resistance" and total_score <= -4 and "Bearish" in market_view:
                option_type = 'PE'
            elif row['Level'] == "Neutral" and total_score <= -4 and "Bearish" in market_view:
                option_type = 'PE'
            elif row['Level'] == "Neutral" and total_score >= 4 and "Bullish" in market_view:
                option_type = 'CE'
            else:
                continue

            ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
            iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
            target = round(ltp * (1 + iv / 100), 2)
            stop_loss = round(ltp * 0.8, 2)

            atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based{' near S/R' if row['Level'] != 'Neutral' else ''})"
            suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"
            send_telegram_message(
                f"📍 Spot: {underlying}\n🔹 {atm_signal}\n{suggested_trade}\nBias Score (ATM ±2): {total_score} ({market_view})\nLevel: {atm_row['Level']}\nBiases: ChgOI: {atm_row['ChgOI_Bias']}, Volume: {atm_row['Volume_Bias']}, Gamma: {atm_row['Gamma_Bias']}, AskQty: {atm_row['AskQty_Bias']}, BidQty: {atm_row['BidQty_Bias']}, IV: {atm_row['IV_Bias']}, DVP: {atm_row['DVP_Bias']}"
            )
            signal_sent = True
            break

        if not signal_sent:
            send_telegram_message(
                f"📍 Spot: {underlying}\n{market_view}\nNo Signal — Market is {market_view}\nBias Score: {total_score} ({market_view})\nLevel: {atm_row['Level']}\nBiases: ChgOI: {atm_row['ChgOI_Bias']}, Volume: {atm_row['Volume_Bias']}, Gamma: {atm_row['Gamma_Bias']}, AskQty: {atm_row['AskQty_Bias']}, BidQty: {atm_row['BidQty_Bias']}, IV: {atm_row['IV_Bias']}, DVP: {atm_row['DVP_Bias']}"
            )

        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.success(f"🧠 Market View: **{market_view}**")
        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")
        st.dataframe(df_summary)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        send_telegram_message(f"❌ Error: {str(e)}")

analyze()
