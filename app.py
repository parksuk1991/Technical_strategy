# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Mean Reversion + Diversification (Slots)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Mean Reversion Strategy + Diversification (Slots 적용)")
st.markdown("---")

# 기본 추천 자산 리스트
DEFAULT_ASSETS = ["QQQ", "SPY", "EWY", "IEMG", "IDEV", "ACWI"]
DEFAULT_ASSETS_STR = ", ".join(DEFAULT_ASSETS)

# ---------------------------
# 사이드바 설정
# ---------------------------
with st.sidebar:
    st.header("⚙️ 설정")

    mode = st.radio("Mode", options=["Single Asset", "Diversified Portfolio"], index=1)

    st.markdown("### Ticker 입력 방식")
    if mode == "Single Asset":
        st.markdown("원하시는 티커를 입력하세요 (예: QQQ). 기본값은 QQQ입니다.")
        user_ticker = st.text_input("Ticker", value="QQQ").strip().upper()
        tickers = [user_ticker] if user_ticker else []
    else:
        st.markdown("콤마(,)로 구분하여 여러 티커를 입력하세요. 예: QQQ, SPY, EWY")
        tickers_text = st.text_input("Tickers (comma-separated)", value=DEFAULT_ASSETS_STR)
        tickers = [t.strip().upper() for t in (tickers_text or "").split(",") if t.strip()]

    start_date = st.date_input(
        "시작 날짜",
        value=datetime(1999, 3, 10),
        min_value=datetime(1999, 3, 10),
        max_value=datetime.now()
    )
    end_date = st.date_input(
        "종료 날짜",
        value=datetime.now(),
        min_value=datetime(1999, 3, 10),
        max_value=datetime.now()
    )

    st.markdown("---")
    st.subheader("Diversification / Slots settings")

    # 포트폴리오 자금(유동성 체크에 사용)
    portfolio_size = st.number_input("Portfolio Size (USD, 슬롯 유동성 체크 용도)", min_value=1000.0, value=1_000_000.0, step=1000.0, help="슬롯당 배분금액(포트폴리오/5)을 계산하여 ADV 대비 5% 조건을 검사하기 위해 사용합니다.")

    # 슬롯 관련 설정
    use_slots = st.checkbox("Use Slots (5 slots) for entry allocation", value=True)
    slots = st.number_input("Number of slots", min_value=1, max_value=10, value=5)
    liquidity_lookback_days = st.number_input("Liquidity lookback (days, ~영업일)", min_value=21, max_value=252, value=63, help="3개월 대략 63 영업일")
    adv_cap_pct = st.number_input("Max allocation per trade vs ADV (%)", min_value=0.1, max_value=100.0, value=5.0) / 100.0

    st.markdown("---")
    st.subheader("Diversification method (포트폴리오 가중치 방식)")
    div_method = st.selectbox("Diversification method",
                              options=["Slots (priority)", "Equal Weight", "Inverse Volatility"],
                              index=0,
                              help="Slots: 슬롯 기반 (선호). Equal/InverseVol: 클래식 포트폴리오 가중")

    rebalance_freq = st.selectbox("Rebalance Frequency", options=["Monthly", "Quarterly"], index=0)
    vol_lookback = st.number_input("Volatility lookback days (for Inverse Vol)", min_value=21, max_value=252, value=63)

    st.markdown("---")
    st.markdown("### 전략 규칙 적용 (요약)")
    st.markdown("""
    1. Rolling mean of (High - Low) over last 25 days.
    2. IBS = (Close - Low) / (High - Low).
    3. Lower band = 10-day rolling High - 2.5 × HL_avg_25.
    4. Long when Close < Lower band AND IBS < 0.3.
    5. Exit when Close > yesterday's High.
    6. Also exit when Close < 200-SMA.
    7. Only consider stocks with Price > $10 and traded in all sessions in last 3 months.
    8. If more than N slots candidates, choose by ATR (14) normalized — prioritize high ATR.
    9. Limit per-trade allocation vs median ADV (3 months): slot_allocation <= adv_cap_pct * median_ADV.
    """)
    st.markdown("---")
    st.caption("BM = Buy & Hold of the same asset (or portfolio) using the same weight allocation as applied to the strategy.")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_data(ticker: str, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)  # auto_adjust False to keep volume * price correct
    except Exception:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Avg_25'] = df['HL_Range'].rolling(window=25, min_periods=1).mean()
    denom = (df['High'] - df['Low']).replace(0, np.nan)
    df['IBS'] = (df['Close'] - df['Low']) / denom
    df['Rolling_High_10'] = df['High'].rolling(window=10, min_periods=1).max()
    df['Lower_Band'] = df['Rolling_High_10'] - (2.5 * df['HL_Avg_25'])
    df['Prev_High'] = df['High'].shift(1)
    # SMA 200 (영문 규칙에 맞게 200일)
    df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    df['Entry_Condition'] = (df['Close'] < df['Lower_Band']) & (df['IBS'] < 0.3)
    df['Exit_By_PrevHigh'] = df['Close'] > df['Prev_High']
    df['Exit_By_SMA200'] = df['Close'] < df['SMA_200']
    df['Exit_Condition'] = df['Exit_By_PrevHigh'] | df['Exit_By_SMA200']
    # ATR-ish for volatility ranking (여기서는 HL_Range의 14일 평균을 사용)
    df['ATR_14'] = df['HL_Range'].rolling(window=14, min_periods=1).mean()
    # Liquidity metrics (ADV, traded all sessions last N days 계산은 build_portfolio에서 사용)
    df['Dollar_Vol'] = df['Volume'] * df['Close']
    return df

def calculate_strategy_returns_from_positions(df: pd.DataFrame):
    """
    df must contain 'Position' column where Position is 1 when long, 0 otherwise.
    Calculate Strategy_Daily_Return = Position.shift(1) * close_pct_change  (same convention as original)
    Also compute BuyHold_Daily_Return
    """
    df = df.copy()
    df['BuyHold_Daily_Return'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Daily_Return'] = df['Position'].shift(1).fillna(0) * df['Close'].pct_change().fillna(0)
    df['Strategy_Cumulative'] = (1 + df['Strategy_Daily_Return']).cumprod().fillna(method='ffill').fillna(1)
    df['BuyHold_Cumulative'] = (1 + df['BuyHold_Daily_Return']).cumprod().fillna(method='ffill').fillna(1)
    return df

def calculate_metrics(returns: pd.Series):
    returns = returns.dropna()
    if len(returns) == 0:
        return {'Total Return': 0, 'CAGR': 0, 'Volatility': 0, 'Sharpe Ratio': 0,
                'Sortino Ratio': 0, 'Max Drawdown': 0, 'Calmar Ratio': 0, 'Win Rate': 0}
    total_return = (1 + returns).prod() - 1
    n_days = len(returns)
    n_years = n_days / 252 if n_days > 0 else 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol != 0 else 0
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino = (returns.mean() * 252) / downside_std if (not np.isnan(downside_std) and downside_std != 0) else 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    return {'Total Return': total_return, 'CAGR': cagr, 'Volatility': vol,
            'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino, 'Max Drawdown': max_dd,
            'Calmar Ratio': calmar, 'Win Rate': win_rate}

# ---------------------------
# Portfolio assembly with slots / liquidity / filtering
# ---------------------------
def prepare_ticker_series(ticker, start, end):
    df = download_data(ticker, start, end)
    if df is None or df.empty:
        return None
    df = calculate_indicators(df)
    return df

def compute_slot_weights(per_asset_dfs: dict, all_index: pd.DatetimeIndex, slots:int, portfolio_size:float, liquidity_lookback_days:int, adv_cap_pct:float):
    """
    날짜별로 슬롯 기반으로 최대 slots개의 신규 진입 후보를 선정하여 각각 weight = 1/slots 할당.
    - Entry 후보: per_asset_dfs[t]['Entry_Condition'] == True on that date
    - 필터: price > 10, price > SMA_200, traded all sessions in last liquidity_lookback_days, allocated_shares <= adv_cap_pct * median_ADV_shares
    - 후보가 slots 초과하면 ATR_14 (또는 normalized ATR)로 내림차순 정렬하여 상위 slots 선택
    반환: daily_weights DataFrame (index=all_index, columns=assets), 값은 해당 날의 슬롯 기준 weight (예: 0.2) 또는 0
    Note: 이 방식은 '그날 새로 진입하는 포지션'에 대해서 slots를 적용합니다. 이미 열린 과거 포지션을 엄격히 전역적으로 트래킹하여 동시 보유수를 5로 제한하려면 별도의 시뮬레이션 루프가 필요합니다.
    """
    assets = list(per_asset_dfs.keys())
    # initialize
    weights = pd.DataFrame(0.0, index=all_index, columns=assets)

    # Precompute liquidity metrics per asset (median ADV in shares over lookback window and traded_all flag)
    for t, df in per_asset_dfs.items():
        if df is None or df.empty:
            continue
        # ensure df index is datetime and sorted
        df = df.sort_index()
        # median ADV in shares over lookback window (rolling median of Volume)
        df['ADV_rolling_median_shares'] = df['Volume'].rolling(window=liquidity_lookback_days, min_periods=1).median()
        df['ADV_rolling_median_dollar'] = (df['Volume'] * df['Close']).rolling(window=liquidity_lookback_days, min_periods=1).median()
        # traded all sessions in lookback window
        df['Traded_All_Last_N'] = df['Volume'].rolling(window=liquidity_lookback_days, min_periods=1).apply(lambda x: np.all(x > 0)).astype(bool)
        per_asset_dfs[t] = df  # update

    # For each date, select candidates
    slot_allocation = portfolio_size / slots if slots > 0 else portfolio_size
    for current_date in all_index:
        candidates = []
        for t, df in per_asset_dfs.items():
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            # Entry condition true today?
            if not (row.get('Entry_Condition', False)):
                continue
            # price > 10
            if pd.isna(row['Close']) or row['Close'] <= 10:
                continue
            # price > SMA_200
            if pd.isna(row['SMA_200']) or row['Close'] <= row['SMA_200']:
                continue
            # traded all sessions in last N days
            if not row.get('Traded_All_Last_N', False):
                continue
            # ADV median shares at date
            adv_median_shares = row.get('ADV_rolling_median_shares', np.nan)
            if pd.isna(adv_median_shares) or adv_median_shares <= 0:
                continue
            # allocated_shares for slot allocation
            allocated_shares = slot_allocation / row['Close'] if row['Close'] > 0 else np.inf
            # check allocated_shares <= adv_cap_pct * adv_median_shares
            if allocated_shares > adv_cap_pct * adv_median_shares:
                # liquidity constraint violated
                continue
            # candidate with volatility score (ATR_14)
            vol_score = row.get('ATR_14', 0.0)
            candidates.append((t, vol_score))

        if len(candidates) == 0:
            continue

        # Sort descending by vol_score and pick top slots
        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in candidates_sorted[:slots]]

        for sel in selected:
            if sel in weights.columns:
                weights.at[current_date, sel] = 1.0 / slots

    # forward-fill weights to daily (so weights persist until next rebalance/selection)
    # But because we want weights only when position opened by slot selection, and not to auto-fill indefinitely,
    # we choose to forward-fill so portfolio uses the most recent selection until next selection day.
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)
    return weights

def compute_monthly_weights(price_returns_df: pd.DataFrame, method: str, lookback_days: int, freq: str):
    # keep existing equal weight / inverse vol logic (used when not using slots)
    if price_returns_df.shape[1] == 0:
        return pd.DataFrame(index=price_returns_df.index)
    if method == "Equal Weight":
        n = price_returns_df.shape[1]
        w = pd.DataFrame(1.0 / n, index=price_returns_df.resample(freq).mean().index, columns=price_returns_df.columns)
        daily_weights = w.reindex(price_returns_df.index, method='ffill').fillna(method='ffill').fillna(1.0 / price_returns_df.shape[1])
        return daily_weights
    elif method == "Inverse Volatility":
        vol = price_returns_df.rolling(window=lookback_days, min_periods=10).std()
        vol_at_reb = vol.resample(freq).last()
        inv_vol = 1.0 / vol_at_reb
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0)
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1.0 / price_returns_df.shape[1])
        daily_weights = weights.reindex(price_returns_df.index, method='ffill').fillna(method='ffill')
        daily_weights = daily_weights.fillna(1.0 / price_returns_df.shape[1])
        return daily_weights
    else:
        raise ValueError("Unknown method")

def build_portfolio(assets_list, start, end, div_method, rebalance_freq, vol_lookback, use_slots_flag, slots, portfolio_size, liquidity_lookback_days, adv_cap_pct):
    per_asset_dfs = {}
    for t in assets_list:
        df = prepare_ticker_series(t, start, end)
        if df is None or df.empty:
            st.warning(f"{t}: No data or failed to download.")
            continue
        per_asset_dfs[t] = df

    if len(per_asset_dfs) == 0:
        return {'per_asset_dfs': {}, 'portfolio_df': pd.DataFrame(), 'daily_weights': pd.DataFrame()}

    # build unified index (business days covered by any asset)
    all_index = sorted(set().union(*[set(df.index) for df in per_asset_dfs.values()]))
    all_index = pd.DatetimeIndex(sorted(all_index))

    # Compute per-asset position series using their own Entry/Exit rules (BUT we will apply slot-based gating on returns)
    per_asset_positions = {}
    per_asset_strategy_daily = {}
    per_asset_bh_daily = {}
    for t, df in per_asset_dfs.items():
        tmp = df.reindex(all_index).copy()
        # naive local simulation: create Position series with simple logic (enter when Entry_Condition true, exit when Exit_Condition true)
        # This per-asset Position is independent; later slot weights will gate which of these per-asset returns are actually used in the portfolio.
        tmp['Position'] = 0
        position = 0
        for i in range(len(tmp)):
            if pd.isna(tmp['Entry_Condition'].iloc[i]):
                tmp['Position'].iloc[i] = position
                continue
            if position == 0 and tmp['Entry_Condition'].iloc[i]:
                position = 1
            elif position == 1 and tmp['Exit_Condition'].iloc[i]:
                position = 0
            tmp['Position'].iloc[i] = position
        tmp = calculate_strategy_returns_from_positions(tmp)
        per_asset_positions[t] = tmp['Position']
        per_asset_strategy_daily[t] = tmp['Strategy_Daily_Return']
        per_asset_bh_daily[t] = tmp['BuyHold_Daily_Return']
        # ensure per_asset_dfs updated with indicators aligned to all_index
        per_asset_dfs[t] = tmp  # now tmp contains indicators + Position + returns

    strat_returns = pd.DataFrame(per_asset_strategy_daily, index=all_index).fillna(0.0)
    bh_returns = pd.DataFrame(per_asset_bh_daily, index=all_index).fillna(0.0)

    # Decide on weights:
    if use_slots_flag and div_method == "Slots (priority)":
        # compute slot weights (daily)
        daily_weights = compute_slot_weights(per_asset_dfs, all_index, slots, portfolio_size, liquidity_lookback_days, adv_cap_pct)
    else:
        # fallback to classic monthly/quarterly weighting
        freq_map = {'Monthly': 'M', 'Quarterly': 'Q'}
        freq = freq_map.get(rebalance_freq, 'M')
        method_map = {"Equal Weight": "Equal Weight", "Inverse Volatility": "Inverse Volatility"}
        method = method_map.get(div_method, "Equal Weight")
        daily_weights = compute_monthly_weights(bh_returns.fillna(0), method, vol_lookback, freq)

    # Align daily_weights
    if daily_weights is None or daily_weights.empty:
        # default equal weight across available assets but we keep sums possibly <1 (cash = remainder)
        n = max(1, len(per_asset_dfs))
        daily_weights = pd.DataFrame(1.0 / n, index=all_index, columns=strat_returns.columns)
    daily_weights = daily_weights.reindex(all_index).fillna(0.0)
    # Important: If daily_weights sum < 1, cash remains uninvested (0 return)

    # Portfolio returns are weighted sum of per-asset strategy returns and BH returns using same daily_weights
    portfolio_strategy_return = (daily_weights * strat_returns).sum(axis=1)
    portfolio_bh_return = (daily_weights * bh_returns).sum(axis=1)

    portfolio_df = pd.DataFrame(index=all_index)
    portfolio_df['Strategy_Daily_Return'] = portfolio_strategy_return.fillna(0.0)
    portfolio_df['BuyHold_Daily_Return'] = portfolio_bh_return.fillna(0.0)
    portfolio_df['Strategy_Cumulative'] = (1 + portfolio_df['Strategy_Daily_Return']).cumprod().fillna(method='ffill').fillna(1)
    portfolio_df['BuyHold_Cumulative'] = (1 + portfolio_df['BuyHold_Daily_Return']).cumprod().fillna(method='ffill').fillna(1)

    return {
        'per_asset_dfs': per_asset_dfs,
        'portfolio_df': portfolio_df,
        'daily_weights': daily_weights
    }

# ---------------------------
# Main
# ---------------------------
try:
    if not tickers:
        st.error("하나 이상의 유효한 티커를 입력하세요.")
        st.stop()

    if mode == "Single Asset":
        asset = tickers[0]
        with st.spinner(f'{asset} 데이터 다운로드 중...'):
            qdf = download_data(asset, start_date, end_date)

        if qdf is None or qdf.empty:
            st.error(f"{asset}의 데이터를 불러올 수 없습니다. 티커를 확인하세요.")
            st.stop()

        qdf = calculate_indicators(qdf)
        # single-asset generate_signals style (local only)
        qdf['Position'] = 0
        position = 0
        for i in range(len(qdf)):
            if pd.isna(qdf['Entry_Condition'].iloc[i]):
                qdf['Position'].iloc[i] = position
                continue
            if position == 0 and qdf['Entry_Condition'].iloc[i]:
                position = 1
            elif position == 1 and qdf['Exit_Condition'].iloc[i]:
                position = 0
            qdf['Position'].iloc[i] = position

        qdf = calculate_strategy_returns_from_positions(qdf)

        latest = qdf.iloc[-1]
        st.header(f"🔔 현재 시그널 - {asset}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("날짜", qdf.index[-1].strftime('%Y-%m-%d'))
        with c2:
            st.metric(f"{asset} 종가", f"${latest['Close']:.2f}")
        with c3:
            current_pos = int(latest['Position'])
            st.metric("현재 포지션", "✅ 보유 중" if current_pos == 1 else "⏸️ 대기 중")
        with c4:
            if latest['Entry_Condition'] and current_pos == 0:
                st.markdown("### 🟢 진입 신호")
            elif latest['Exit_By_SMA200'] and current_pos == 1:
                st.markdown("### 🔴 청산 신호 (SMA_200)")
            elif latest['Exit_By_PrevHigh'] and current_pos == 1:
                st.markdown("### 🔴 청산 신호 (Prev High)")
            else:
                st.markdown("### ⚪ 신호 없음")

        qdf_clean = qdf.dropna(subset=['Strategy_Daily_Return', 'BuyHold_Daily_Return'])
        strat_metrics = calculate_metrics(qdf_clean['Strategy_Daily_Return'])
        bm_metrics = calculate_metrics(qdf_clean['BuyHold_Daily_Return'])

        st.header("📈 성과 지표 (Strategy vs BM)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("전략 누적수익률", f"{(qdf['Strategy_Cumulative'].iloc[-1]-1)*100:.2f}%")
            st.metric("전략 CAGR", f"{strat_metrics['CAGR']*100:.2f}%")
        with c2:
            st.metric("BM 누적수익률", f"{(qdf['BuyHold_Cumulative'].iloc[-1]-1)*100:.2f}%")
            st.metric("BM CAGR", f"{bm_metrics['CAGR']*100:.2f}%")
        with c3:
            st.metric("전략 샤프 비율", f"{strat_metrics['Sharpe Ratio']:.2f}")
            st.metric("전략 최대낙폭", f"{strat_metrics['Max Drawdown']*100:.2f}%")
        with c4:
            st.metric("총 데이터 길이 (일)", len(qdf))

        st.header("📊 차트")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qdf.index, y=qdf['Strategy_Cumulative'], name='Strategy', line=dict(color='#2E86AB')))
        fig.add_trace(go.Scatter(x=qdf.index, y=qdf['BuyHold_Cumulative'], name='BM', line=dict(color='#A23B72')))
        fig.update_layout(title=f'{asset} 누적 수익률 (Strategy vs BM)', xaxis_title='날짜', yaxis_title='누적 수익률', template='plotly_white', height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Portfolio mode
        assets_list = tickers
        with st.spinner('다중 자산 데이터 다운로드 및 처리 중...'):
            result = build_portfolio(assets_list, start_date, end_date, div_method, rebalance_freq, vol_lookback,
                                     use_slots_flag=(use_slots and div_method.startswith("Slots")), slots=slots,
                                     portfolio_size=portfolio_size, liquidity_lookback_days=liquidity_lookback_days,
                                     adv_cap_pct=adv_cap_pct)

        per_asset_dfs = result['per_asset_dfs']
        portfolio_df = result['portfolio_df']
        daily_weights = result['daily_weights']

        st.header("📊 Diversified Portfolio (Slots 기반 집계)")

        # Weights snapshot (most recent)
        try:
            if daily_weights is None or daily_weights.empty:
                st.info("가중치가 계산되지 않았습니다.")
            else:
                last_weights = daily_weights.ffill().iloc[-1].astype(float).fillna(0.0)
                last_weights_df = last_weights.rename_axis('Asset').reset_index(name='Weight')
                st.subheader("Weights snapshot (most recent selection / rebalance)")
                st.bar_chart(last_weights_df.set_index('Asset'))
        except Exception as e_w:
            st.error("가중치 시각화 중 오류가 발생했습니다.")
            st.exception(e_w)

        if portfolio_df is None or portfolio_df.empty:
            st.error("포트폴리오 데이터가 비어있습니다.")
            st.stop()

        latest = portfolio_df.iloc[-1]
        st.metric("날짜", portfolio_df.index[-1].strftime('%Y-%m-%d'))
        st.metric("포트폴리오 전략 누적수익률", f"{(portfolio_df['Strategy_Cumulative'].iloc[-1]-1)*100:.2f}%")
        st.metric("포트폴리오 BM 누적수익률", f"{(portfolio_df['BuyHold_Cumulative'].iloc[-1]-1)*100:.2f}%")

        strat_metrics = calculate_metrics(portfolio_df['Strategy_Daily_Return'])
        bm_metrics = calculate_metrics(portfolio_df['BuyHold_Daily_Return'])
        st.header("📈 성과 지표 (Portfolio Strategy vs BM)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("전략 누적수익률", f"{strat_metrics['Total Return']*100:.2f}%")
            st.metric("전략 CAGR", f"{strat_metrics['CAGR']*100:.2f}%")
        with c2:
            st.metric("BM 누적수익률", f"{bm_metrics['Total Return']*100:.2f}%")
            st.metric("BM CAGR", f"{bm_metrics['CAGR']*100:.2f}%")
        with c3:
            st.metric("전략 샤프 비율", f"{strat_metrics['Sharpe Ratio']:.2f}")
            st.metric("전략 최대낙폭", f"{strat_metrics['Max Drawdown']*100:.2f}%")
        with c4:
            st.metric("데이터 길이 (일)", len(portfolio_df))

        st.header("📊 차트 (Portfolio)")
        tab1, tab2, tab3 = st.tabs(["누적 수익률", "낙폭", "개별 자산 기여"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df['Strategy_Cumulative'], name='Strategy', line=dict(color='#2E86AB')))
            fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df['BuyHold_Cumulative'], name='BM', line=dict(color='#A23B72')))
            fig.update_layout(title=f'Portfolio 누적 수익률 ({div_method} / slots={slots})', template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            strat_cum = portfolio_df['Strategy_Cumulative']
            running_max = strat_cum.expanding().max()
            drawdown = (strat_cum - running_max) / running_max * 100
            bh_cum = portfolio_df['BuyHold_Cumulative']
            bh_max = bh_cum.expanding().max()
            bh_draw = (bh_cum - bh_max) / bh_max * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=portfolio_df.index, y=drawdown, name='Strategy Drawdown', fill='tozeroy', line=dict(color='#EF553B')))
            fig2.add_trace(go.Scatter(x=portfolio_df.index, y=bh_draw, name='BM Drawdown', fill='tozeroy', line=dict(color='#FFA500')))
            fig2.update_layout(title='Drawdown', xaxis_title='날짜', yaxis_title='낙폭 (%)', template='plotly_white', height=500)
            st.plotly_chart(fig2, use_container_width=True)
        with tab3:
            contributions = {}
            for t, df in per_asset_dfs.items():
                tmp = df.reindex(portfolio_df.index).copy()
                tmp['Strat_Cum'] = (1 + tmp.get('Strategy_Daily_Return', 0)).cumprod().fillna(method='ffill').fillna(1)
                contributions[t] = tmp['Strat_Cum']
            if len(contributions) > 0:
                contrib_df = pd.DataFrame(contributions)
                st.line_chart(contrib_df)
            else:
                st.info("개별 자산 기여 데이터가 없습니다.")

        st.markdown("---")
        st.subheader("최근 슬롯 기반 선택 샘플 (최근 50일)")
        if daily_weights is not None and not daily_weights.empty:
            last_n = daily_weights.tail(50)
            # 각 날짜별 선택된 자산 리스트 출력
            selections = []
            for idx, row in last_n.iterrows():
                selected = [col for col, val in row.items() if val > 0]
                selections.append({'Date': idx, 'Selected': ", ".join(selected) if selected else '-'})
            sel_df = pd.DataFrame(selections).set_index('Date')
            st.dataframe(sel_df, use_container_width=True)
        else:
            st.info("슬롯 기반 선택 데이터가 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.exception(e)
