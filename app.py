from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

def calculate_rsi(data, period=14):
    try:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except:
        return None

def calculate_macd(data):
    try:
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return float(macd.iloc[-1])
    except:
        return None

def calculate_bollinger_bands(data, period=20):
    try:
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        current_price = float(data['Close'].iloc[-1])
        return {
            'upper': float(upper_band.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower_band.iloc[-1]),
            'current': current_price
        }
    except:
        return None

def predict_price_lstm(data):
    try:
        recent_prices = data['Close'].tail(10).values
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        weights = np.exp(np.linspace(-1., 0., len(recent_prices)))
        weights /= weights.sum()
        wma = np.sum(recent_prices * weights)
        prediction = wma * (1 + trend * 0.5 + np.random.normal(0, volatility * 0.3))
        base_confidence = 65
        volatility_penalty = min(volatility * 100, 15)
        confidence = max(50, min(75, base_confidence - volatility_penalty))
        change_percent = ((prediction - recent_prices[-1]) / recent_prices[-1]) * 100
        return {
            'predicted_price': float(prediction),
            'confidence': float(confidence),
            'change_percent': float(change_percent),
            'current_price': float(recent_prices[-1]),
            'volatility': float(volatility)
        }
    except Exception as e:
        return None

@app.route('/')
def home():
    return jsonify({
        'name': 'Trading AI Analyzer API',
        'version': '1.0.0',
        'status': 'online'
    })

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_asset():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL').upper()
        period = data.get('period', '1mo')
        interval = data.get('interval', '1d')
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            return jsonify({
                'error': 'Símbolo não encontrado',
                'symbol': symbol
            }), 404
        
        info = ticker.info
        rsi = calculate_rsi(hist)
        macd = calculate_macd(hist)
        bb = calculate_bollinger_bands(hist)
        prediction = predict_price_lstm(hist)
        
        chart_data = []
        for index, row in hist.iterrows():
            chart_data.append({
                'time': int(index.timestamp()),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        signals = {
            'rsi_signal': 'SOBRECOMPRADO' if rsi and rsi > 70 else 'SOBREVENDIDO' if rsi and rsi < 30 else 'NEUTRO',
            'macd_signal': 'COMPRA' if macd and macd > 0 else 'VENDA',
            'bb_signal': 'DENTRO DA BANDA'
        }
        
        response = {
            'success': True,
            'symbol': symbol,
            'asset_name': info.get('longName', symbol),
            'current_price': float(hist['Close'].iloc[-1]),
            'chart_data': chart_data,
            'indicators': {
                'rsi': round(rsi, 2) if rsi else None,
                'macd': round(macd, 2) if macd else None,
                'bollinger_bands': bb,
                'sma_20': round(float(hist['Close'].rolling(20).mean().iloc[-1]), 2),
                'sma_50': round(float(hist['Close'].rolling(50).mean().iloc[-1]), 2),
                'volume_avg': int(hist['Volume'].tail(20).mean())
            },
            'signals': signals,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
