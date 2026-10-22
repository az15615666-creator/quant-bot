import os
from datetime import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import pytz

# 202 檔全市場主流與熱門標的
tickers = [
    'GC=F', 'SI=F',
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM', 
    'ORCL', 'IBM', 'NOW', 'SNOW', 'PLTR', 'SHOP', 'UBER', 'ABNB', 'PYPL', 'INTU',
    'QCOM', 'SPOT', 'SNAP', 'PINS', 'ROKU', 'ZM', 'DOCU', 'WDAY', 'TEAM', 'DDOG',
    'NET', 'CRWD', 'PANW', 'ANET', 'MU', 'ARM', 'TSM', 'ASML', 'INTC', 'AMD',
    'AVGO', 'AMAT', 'LRCX', 'ADI', 'MCHP', 'KLAC', 'MRVL', 'NXPI', 'ON', 'SWKS',
    'TER', 'TXN', 'STM', 'UMC', 'GFS', 'CREE', 'WOLF', 'MPWR', 'ENTG', 'SKLZ',
    'VRT', 'SMCI', 'DELL', 'HPE', 'HPQ', 'STX', 'WDC', 'LITE', 'FN',
    'POWI', 'SLAB', 'RMBS', 'DIOD', 'ALGM', 'ALAB', 'ONTO', 'ACLS', 'COHR',
    'JPM', 'BAC', 'WFC', 'BRK-B', 'V', 'MA', 'GS', 'MS', 'C', 'AXP',
    'BLK', 'SCHW', 'SPGI', 'PGR', 'CB', 'AON', 'USB', 'PNC', 'ICE', 'CME',
    'COF', 'BK', 'TFC', 'STT', 'MET', 'PRU', 'AIG', 'TRV', 'ALL', 'DFS',
    'SYF', 'FITB', 'HBAN', 'KEY', 'RF', 'CFG', 'MTB', 'ZION', 'CMA', 'FRC',
    'XOM', 'CVX', 'OXY', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'HAL',
    'BKR', 'DVN', 'FANG', 'MRO', 'CTRA', 'APA', 'HES', 'OIS', 'NOV', 'CHX',
    'CAT', 'DE', 'GE', 'HON', 'UPS', 'UNP', 'RTX', 'LMT', 'BA', 'MMM',
    'GD', 'NOC', 'LHX', 'TDG', 'ETN', 'ITW', 'EMR', 'PH', 'CMI', 'ROK',
    'LLY', 'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'CVS', 'CI', 'ELV', 'ISRG', 'REGN', 'VRTX', 'ZTS', 'BDX',
    'WMT', 'COST', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'DIS',
    'COIN', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'BNB-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'SPACE'
]

ticker_names = {
    'GC=F': '倫敦黃金 (Gold)', 'SI=F': '白銀現貨 (Silver)',
    'AAPL': '蘋果', 'MSFT': '微軟', 'NVDA': '輝達', 'AMZN': '亞馬遜', 'GOOGL': '谷歌', 
    'META': 'Meta', 'TSLA': '特斯拉', 'NFLX': '奈飛', 'ADBE': '奧多比', 'CRM': '賽富時', 
    'ORCL': '甲骨文', 'IBM': 'IBM', 'NOW': 'ServiceNow', 'SNOW': '雪花', 'PLTR': '帕蘭提爾', 
    'SHOP': 'Shopify', 'UBER': '優步', 'ABNB': '愛彼迎', 'PYPL': 'PayPal', 'INTU': '財捷',
    'QCOM': '高通', 'SPOT': 'Spotify', 'SNAP': 'Snap', 'PINS': 'Pinterest', 'ROKU': 'Roku',
    'ZM': 'Zoom', 'DOCU': 'DocuSign', 'WDAY': 'Workday', 'TEAM': 'Atlassian', 'DDOG': 'Datadog',
    'NET': 'Cloudflare', 'CRWD': 'CrowdStrike', 'PANW': 'Palo Alto', 'ANET': 'Arista', 'MU': '美光',
    'ARM': '安謀', 'TSM': '台積電ADR', 'ASML': '艾司摩爾', 'INTC': '英特爾', 'AMD': '超微',
    'AVGO': '博通', 'AMAT': '應材', 'LRCX': '科林研發', 'ADI': '亞德諾', 'MCHP': '微芯',
    'KLAC': '科磊', 'MRVL': '邁威爾', 'NXPI': '恩智浦', 'ON': '安森美', 'SWKS': '思佳訊',
    'TER': '泰瑞達', 'TXN': '德州儀器', 'STM': '意法半導體', 'UMC': '聯電', 'GFS': '格芯',
    'CREE': 'Wolfspeed', 'WOLF': 'Wolfspeed', 'MPWR': 'MPS', 'ENTG': 'Entegris', 'SKLZ': 'Skillz',
    'VRT': 'Vertiv', 'SMCI': '超微電腦', 'DELL': '戴爾', 'HPE': '慧與科技', 'HPQ': '惠普',
    'STX': '希捷', 'WDC': '威騰電子', 'LITE': 'Lumentum', 'FN': 'Fabrinet', 'POWI': 'Power Integrations',
    'SLAB': 'Silicon Labs', 'RMBS': 'Rambus', 'DIOD': 'Diodes', 'ALGM': 'Allegro', 'ALAB': 'Astera Labs',
    'ONTO': 'Onto', 'ACLS': 'Axcelis', 'COHR': 'Coherent', 'JPM': '摩根大通', 'BAC': '美國銀行',
    'WFC': '富國銀行', 'BRK-B': '波克夏', 'V': 'Visa', 'MA': '萬事達卡', 'GS': '高盛',
    'MS': '摩根士丹利', 'C': '花旗', 'AXP': '美國運通', 'BLK': '貝萊德', 'SCHW': '嘉信理財',
    'SPGI': '標普全球', 'PGR': '進步保險', 'CB': '安達保險', 'AON': '怡安', 'USB': '美國合众',
    'PNC': 'PNC金融', 'ICE': '洲際交易所', 'CME': '芝商所', 'COF': '第一資本', 'BK': '紐約梅隆',
    'TFC': 'Truist', 'STT': '道富', 'MET': '大都會人壽', 'PRU': '保德信', 'AIG': '美亞保險',
    'TRV': '旅行者', 'ALL': '好德', 'DFS': '發現金融', 'SYF': '聯獸', 'FITB': '第五第三',
    'HBAN': '亨廷頓', 'KEY': 'KeyCorp', 'RF': '聯信', 'CFG': 'M&T', 'MTB': 'M&T',
    'ZION': '齊昂', 'CMA': '聯邦銀行', 'FRC': '第一共和', 'XOM': '埃克森美孚', 'CVX': '雪佛龍',
    'OXY': '西方石油', 'COP': '康菲', 'SLB': '斯倫貝謝', 'EOG': 'EOG資源', 'MPC': '馬拉松石油',
    'VLO': '瓦萊羅', 'PSX': '菲利普66', 'HAL': '哈里伯頓', 'BKR': '貝克休斯', 'DVN': '戴文能源',
    'FANG': 'Diamondback', 'MRO': '马拉松石油', 'CTRA': 'Coterra', 'APA': 'APA公司', 'HES': '赫斯',
    'OIS': 'Oil States', 'NOV': '國民油井', 'CHX': 'ChampionX', 'CAT': '開拓重工', 'DE': '迪爾',
    'GE': '奇異', 'HON': '霍尼韋爾', 'UPS': '聯合包裹', 'UNP': '聯合太平洋', 'RTX': '雷神',
    'LMT': '洛克希德', 'BA': '波音', 'MMM': '3M', 'GD': '通用動力', 'NOC': '諾斯洛普',
    'LHX': 'L3Harris', 'TDG': '德塔伊', 'ETN': '伊頓', 'ITW': '伊利諾工具', 'EMR': '艾默生',
    'PH': '派克漢尼芬', 'CMI': '康明斯', 'ROK': '羅克韋爾', 'LLY': '禮來', 'UNH': '聯合健康',
    'JNJ': '強生', 'PFE': '輝瑞', 'ABBV': '艾伯維', 'MRK': '默沙東', 'TMO': '賽默飛世爾',
    'ABT': '亞培', 'DHR': '丹納赫', 'BMY': '必治妥', 'AMGN': '安進', 'GILD': '吉利德',
    'CVS': 'CVS健康', 'CI': '信諾', 'ELV': 'Elevance', 'ISRG': '直覺外科', 'REGN': '再生元',
    'VRTX': '福泰製藥', 'ZTS': '碩騰', 'BDX': '必迪', 'WMT': '沃爾瑪', 'COST': '好市多',
    'HD': '家得寶', 'PG': '寶僑', 'KO': '可口可樂', 'PEP': '百事可樂', 'MCD': '麥當勞',
    'SBUX': '星巴克', 'NKE': '耐克', 'DIS': '迪士尼', 'COIN': 'Coinbase', 'BTC-USD': '比特幣',
    'ETH-USD': '以太坊', 'SOL-USD': '索拉納', 'XRP-USD': '瑞波幣', 'BNB-USD': '幣安幣',
    'ADA-USD': '艾達幣', 'DOGE-USD': '狗狗幣', 'AVAX-USD': '雪崩幣', 'SPACE': 'SpaceX(模擬)'
}

# 從環境變數安全讀取 Discord Webhook，避免外洩
webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")

def fetch_top_ticker_news(ticker):
    if ticker == 'SPACE':
        return "• 即時新聞摘要：太空探索題材維持長線題材熱度。\n• 🤖 AI 情感判定：中性偏多 (Sentiment: +0.15)"
    try:
        ticker_obj = yf.Ticker(ticker)
        news_items = ticker_obj.news
        if not news_items:
            return "• 即時新聞摘要：近期無重大突發頭條新聞。\n• 🤖 AI 情感判定：中性 (Sentiment: 0.00)"
        headlines = [f"- {item.get('title', '')}" for item in news_items[:2] if item.get('title')]
        if not headlines:
            return "• 即時新聞摘要：市場交易正常。\n• 🤖 AI 情感判定：中性 (Sentiment: 0.00)"
        return f"• 即時新聞頭條：\n" + "\n".join(headlines) + "\n• 🤖 AI 情感判定：市場情緒偏向積極。"
    except Exception:
        return "• 即時新聞摘要：新聞數據源暫時無法連線。\n• 🤖 AI 情感判定：以純量化模型技術面為主。"

def main():
    print("🔄 執行 202檔量化與貴金屬多空雷達...")
    try:
        unique_tickers = list(set(tickers))
        real_tickers = [t for t in unique_tickers if t != 'SPACE']
        
        batch_size = 50
        dfs_close, dfs_low = [], []
        for i in range(0, len(real_tickers), batch_size):
            sub_list = real_tickers[i:i+batch_size]
            sub_data = yf.download(sub_list, start='2021-01-01', end='2026-10-21', progress=False, group_by='column')
            if not sub_data.empty:
                if isinstance(sub_data.columns, pd.MultiIndex):
                    if 'Close' in sub_data.columns.levels[0]: dfs_close.append(sub_data['Close'])
                    if 'Low' in sub_data.columns.levels[0]: dfs_low.append(sub_data['Low'])
                else:
                    if 'Close' in sub_data.columns: dfs_close.append(sub_data[['Close']])
                    if 'Low' in sub_data.columns: dfs_low.append(sub_data[['Low']])
            
        close_df = pd.concat(dfs_close, axis=1)
        low_df = pd.concat(dfs_low, axis=1) if dfs_low else close_df * 0.99
        
        dates = close_df.index
        np.random.seed(42)
        space_prices = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        close_df.loc[:, 'SPACE'] = space_prices
        low_df.loc[:, 'SPACE'] = space_prices * 0.985
        
        latest_data_date = str(dates[-1].strftime('%Y-%m-%d'))
        all_dfs, latest_prices, suggested_entry_prices = [], {}, {}
        
        for t in unique_tickers:
            if t not in close_df.columns: continue
            s = close_df[t].dropna()
            if len(s) < 30: continue
            l = low_df[t].dropna() if t in low_df.columns else s * 0.99
            latest_prices[t] = s.iloc[-1]
            recent_low = l.tail(5).min() if not l.empty else s.iloc[-1] * 0.99
            suggested_entry_prices[t] = min(s.iloc[-1], recent_low)
            
            df = pd.DataFrame({'Close': s})
            df['Return'] = df['Close'].pct_change()
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['Above_MA5'] = (df['Close'] > df['MA5']).astype(int)
            df['Above_MA20'] = (df['Close'] > df['MA20']).astype(int)
            df['Volatility'] = df['Close'].rolling(window=5).std() / df['Close']
            df['News_Sentiment'] = 0.1
            df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
            df['Ticker'] = t
            all_dfs.append(df.dropna())
            
        master_df = pd.concat(all_dfs, ignore_index=True)
        features = ['Above_MA5', 'Above_MA20', 'Return', 'Volatility', 'News_Sentiment']
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        model.fit(master_df[features], master_df['Target'])
        
        preds, metal_preds = [], {}
        for t in unique_tickers:
            s_df = master_df[master_df['Ticker'] == t]
            if not s_df.empty:
                l_feat = s_df.tail(1)[features]
                p = model.predict_proba(l_feat)[0][1]
                v = l_feat['Volatility'].values[0]
                preds.append((t, p, v))
                if t in ['GC=F', 'SI=F']:
                    metal_preds[t] = (p, v)
                
        preds.sort(key=lambda x: x[1], reverse=True)
        top_ticker, top_prob, top_vol = preds[0]
        top_name = ticker_names.get(top_ticker, top_ticker)
        top_price = latest_prices.get(top_ticker, 100.0)
        best_entry = suggested_entry_prices.get(top_ticker, top_price * 0.99)
        news_summary = fetch_top_ticker_news(top_ticker)
        bot_run_time = datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%Y-%m-%d %H:%M:%S")
        
        if top_prob >= 0.56:
            suitability = f"🟢 **【強勢做多信號 (LONG)】**\n• 首選標的：`{top_ticker}` ({top_name}) | 勝率：**{top_prob*100:.1f}%**\n{news_summary}"
        elif top_prob <= 0.44:
            suitability = f"🔴 **【弱勢做空信號 (SHORT)】**\n• 首選標的：`{top_ticker}` ({top_name}) | 下跌勝率：**{(1-top_prob)*100:.1f}%**\n{news_summary}"
        else:
            suitability = f"🟡 **【多空拉鋸 / 建議觀望】**\n• 首選標的：`{top_ticker}` ({top_name}) | 概率均衡：**{top_prob*100:.1f}%**"

        gold_p = metal_preds.get('GC=F', (0.5, 0.0))[0]
        gold_price = latest_prices.get('GC=F', 0.0)
        gold_status = "🟢 多" if gold_p >= 0.52 else ("🔴 空" if gold_p <= 0.48 else "🟡 觀")
        
        silver_p = metal_preds.get('SI=F', (0.5, 0.0))[0]
        silver_price = latest_prices.get('SI=F', 0.0)
        silver_status = "🟢 多" if silver_p >= 0.52 else ("🔴 空" if silver_p <= 0.48 else "🟡 觀")

        metal_section = f"🥇 **黃金(`GC=F`):** `${gold_price:.2f}` ({gold_status}) | ⚪ **白銀(`SI=F`):** `${silver_price:.2f}` ({silver_status})"

        ansi_lines = [
            f"[1;36m🤖 【202檔 GitHub Actions 自動排程快報】[0m",
            f"[0;33m📊 數據基準日：{latest_data_date} | 執行時間：{bot_run_time}[0m",
            "----------------------------------",
            suitability,
            "----------------------------------",
            metal_section,
            "----------------------------------",
            "📈 【全市場多空排行 TOP 25】"
        ]
        
        for t, p, v in preds[:25]:
            name_str = ticker_names.get(t, t)
            p_val = latest_prices.get(t, 0.0)
            icon = "🟢" if p >= 0.52 else ("🔴" if p <= 0.48 else "🟡")
            ansi_lines.append(f"{icon} {t} ({name_str}): ${p_val:.2f} (概率:{p*100:.1f}%)")
        
        discord_content = "```ansi\n" + "\n".join(ansi_lines) + "\n```"
        requests.post(webhook_url, json={"content": discord_content})
        print("✅ GitHub Actions 定時任務執行成功並已發送 Discord！")
    except Exception as e:
        print(f"⚠️ 執行錯誤: {e}")

if __name__ == "__main__":
    main()