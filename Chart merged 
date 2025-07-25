import streamlit as st from streamlit_autorefresh import st_autorefresh import requests import pandas as pd import numpy as np from datetime import datetime import math from scipy.stats import norm from pytz import timezone import plotly.graph_objects as go

=== Streamlit Config ===

st.set_page_config(page_title="Nifty Options Analyzer", layout="wide") st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 min

=== Telegram Config ===

TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU" TELEGRAM_CHAT_ID = "5704496584"

def send_telegram_message(message): url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" data = {"chat_id": TELEGRAM_CHAT_ID, "text": message} try: response = requests.post(url, data=data) if response.status_code != 200: st.warning("⚠️ Telegram message failed.") except Exception as e: st.error(f"❌ Telegram error: {e}")

def calculate_greeks(option_type, S, K, T, r, sigma): d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T)) d2 = d1 - sigma * math.sqrt(T) delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1) gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T)) vega = S * norm.pdf(d1) * math.sqrt(T) / 100 theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2) if option_type == 'CE' else - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365 rho = (K * T * math.exp(-r * T) * norm.cdf(d2) if option_type == 'CE' else -K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100 return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

def group_by_continuity(strike_list): strike_list.sort() zones = [] if not strike_list: return zones current = [strike_list[0]] for i in range(1, len(strike_list)): if strike_list[i] - strike_list[i - 1] <= 50: current.append(strike_list[i]) else: if len(current) >= 2: zones.append((min(current), max(current))) current = [strike_list[i]] if len(current) >= 2: zones.append((min(current), max(current))) return zones

def detect_nearby_zones(df, spot, threshold=1.2, range_pts=100, vol_threshold=2000): nearby_df = df[df['strikePrice'].between(spot - range_pts, spot + range_pts)].copy() nearby_df['ZoneTag'] = nearby_df.apply( lambda row: 'Support' if row['openInterest_PE'] > threshold * row['openInterest_CE'] and row['totalTradedVolume_PE'] > vol_threshold else 'Resistance' if row['openInterest_CE'] > threshold * row['openInterest_PE'] and row['totalTradedVolume_CE'] > vol_threshold else 'Neutral', axis=1) support_zones = group_by_continuity(nearby_df[nearby_df['ZoneTag'] == 'Support']['strikePrice'].tolist()) resistance_zones = group_by_continuity(nearby_df[nearby_df['ZoneTag'] == 'Resistance']['strikePrice'].tolist()) return support_zones, resistance_zones

def is_spot_within_zone(spot, zones): for low, high in zones: if low <= spot <= high: return (low, high) return None

=== Live Chart Data State ===

if 'price_history' not in st.session_state: st.session_state.price_history = []

if 'buy_sell_history' not in st.session_state: st.session_state.buy_sell_history = []

def update_price_history(now, price): st.session_state.price_history.append((now, price)) st.session_state.price_history = st.session_state.price_history[-30:]

def predict_next_move(history, market_view, support_zones, resistance_zones, spot): if len(history) < 2: return "Neutral", "gray" prev = history[-2][1] curr = history[-1][1] momentum = "UP" if curr > prev else "DOWN" if curr < prev else "NEUTRAL" if market_view.startswith("Strong"): market_bias = market_view.split(" ")[1].upper() else: market_bias = market_view.upper()

if is_spot_within_zone(spot, support_zones) and market_bias == "BULLISH":
    st.session_state.buy_sell_history.append((history[-1][0], history[-1][1], 'BUY'))
    send_telegram_message(f"🟢 BUY Entry Signal @ {round(history[-1][1], 2)}")
    return "UP", "green"
if is_spot_within_zone(spot, resistance_zones) and market_bias == "BEARISH":
    st.session_state.buy_sell_history.append((history[-1][0], history[-1][1], 'SELL'))
    send_telegram_message(f"🔴 SELL Entry Signal @ {round(history[-1][1], 2)}")
    return "DOWN", "red"
return momentum, "blue" if momentum == "UP" else "red" if momentum == "DOWN" else "gray"

def plot_prediction_chart(price_history, prediction, color): times = [x[0] for x in price_history] prices = [x[1] for x in price_history] fig = go.Figure() fig.add_trace(go.Scatter(x=times, y=prices, mode='lines+markers', name='Spot', line=dict(color='blue'))) for ts, pr, signal in st.session_state.buy_sell_history: fig.add_trace(go.Scatter(x=[ts], y=[pr], mode='markers+text', name=signal, text=[signal], textposition="top center", marker=dict(color='green' if signal == 'BUY' else 'red', size=10))) fig.add_annotation(x=times[-1], y=prices[-1], text=f"Next Move: {prediction}", showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor=color, font=dict(color="white")) fig.update_layout(title="📊 Predicted Next Move", xaxis_title="Time", yaxis_title="Spot", height=400) st.plotly_chart(fig, use_container_width=True)

=== Main Function ===

def main(): try: ...  # Your existing main function logic remains unchanged

# Update price history & show prediction
    update_price_history(now, underlying)
    prediction, color = predict_next_move(st.session_state.price_history, market_view, support_zones, resistance_zones, underlying)
    plot_prediction_chart(st.session_state.price_history, prediction, color)

except Exception as e:
    st.error(f"❌ Error: {e}")
    send_telegram_message(f"❌ Error: {str(e)}")

if name == "main": main()

