import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="Mean Reversion + Diversification Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Mean Reversion Strategy + Diversification")
st.markdown("---")

# 기본 자산 리스트
ASSETS = ["QQQ", "SPY", "EWY", "IEMG", "IDEV", "ACWI"]

# 사이드바: 모드(단일자산 vs 포트폴리오), 자산선택, 기간, 다각화 옵션
with st.sidebar:
    st.header("⚙️ 설정")
    mode = st.radio("Mode", options=["Single Asset", "Diversified Portfolio"], index=0)
    if mode == "Single Asset":
        asset = st.selectbox("자산 선택", options=ASSETS, index=0)
    else:
        asset = "PORTFOLIO"
        st.markdown("Portfolio assets:")
        st.write(", ".join(ASSETS))
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
    st.subheader("Diversification settings")
    if mode == "Diversified Portfolio":
        div_method = st.selectbox("Diversification method",
                                  options=["Equal Weight", "Inverse Volatility"],
                                  help="Equal Weight: simple average weights. Inverse Volatility: 1/σ normalized based on historical volatility.")
        rebalance_freq = st.selectbox("Rebalance Frequency", options=["Monthly", "Quarterly"], index=0)
        vol_lookback = st.number_input("Volatility lookback days (for Inverse Vol)", min_value=21, max_value=252, value=63)
    else:
        div_method = None
        rebalance_freq = None
        vol_lookback = None

    st.markdown("---")
    st.markdown("### 전략 규칙 적용 (요약)")
    st.markdown("""
    1. Rolling mean of (High - Low) over last 25 days.
    2. IBS = (Close - Low) / (High - Low).
    3. Lower band = 10-day rolling High - 2.5 × HL_avg_25.
    4. Long when Close < Lower band AND IBS < 0.3.
    5. Exit when Close > yesterday's High.
    6. Also exit when Close < 300-SMA.
    """)
    st.markdown("---")
    st.caption("BM = Buy & Hold of the same asset (or portfolio). Shown as 'BM' in charts/metrics.")

# ---------------------------
# Helpers: download, indicators, signals, returns, metrics
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_data(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
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
    df['SMA_300'] = df['Close'].rolling(window=300, min_periods=1).mean()
    df['Entry_Condition'] = (df['Close'] < df['Lower_Band']) & (df['IBS'] < 0.3)
    df['Exit_By_PrevHigh'] = df['Close'] > df['Prev_High']
    df['Exit_By_SMA300'] = df['Close'] < df['SMA_300']
    df['Exit_Condition'] = df['Exit_By_PrevHigh'] | df['Exit_By_SMA300']
    return df

def generate_signals(df: pd.DataFrame):
    df = df.copy()
    df['Signal'] = 0
    df['Position'] = 0
    df['Trade_Number'] = 0
    df['Exit_Reason'] = ""

    position = 0
    trade_number = 0
    trades = []
    entry_price = np.nan
    entry_date = None

    for i in range(len(df)):
        if pd.isna(df['Entry_Condition'].iloc[i]):
            # can't evaluate yet
            continue

        if position == 0 and df['Entry_Condition'].iloc[i]:
            position = 1
            entry_price = df['Close'].iloc[i]
            entry_date = df.index[i]
            trade_number += 1
            df.iat[i, df.columns.get_loc('Signal')] = 1
            df.iat[i, df.columns.get_loc('Trade_Number')] = trade_number

        elif position == 1:
            df.iat[i, df.columns.get_loc('Trade_Number')] = trade_number
            if df['Exit_Condition'].iloc[i]:
                position = 0
                exit_price = df['Close'].iloc[i]
                exit_date = df.index[i]
                df.iat[i, df.columns.get_loc('Signal')] = -1

                reasons = []
                if df['Exit_By_PrevHigh'].iloc[i]:
                    reasons.append('Prev_High')
                if df['Exit_By_SMA300'].iloc[i]:
                    reasons.append('SMA_300')
                reason_text = "|".join(reasons) if reasons else "Exit"
                df.iat[i, df.columns.get_loc('Exit_Reason')] = reason_text

                pnl = (exit_price - entry_price) / entry_price if entry_price and entry_price != 0 else np.nan
                trades.append({
                    'Trade_Number': trade_number,
                    'Entry_Date': entry_date,
                    'Exit_Date': exit_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Return': pnl,
                    'Days': (exit_date - entry_date).days if entry_date is not None else None,
                    'Exit_Reason': reason_text
                })

        df.iat[i, df.columns.get_loc('Position')] = position

    trades_df = pd.DataFrame(trades)
    return df, trades_df

def calculate_returns(df: pd.DataFrame):
    df = df.copy()
    df['Strategy_Daily_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['BuyHold_Daily_Return'] = df['Close'].pct_change()
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
# Portfolio assembly utilities
# ---------------------------
def prepare_ticker_series(ticker, start, end):
    df = download_data(ticker, start, end)
    if df is None or len(df) == 0:
        return None, None
    df = calculate_indicators(df)
    df, trades = generate_signals(df)
    df = calculate_returns(df)
    return df, trades

def compute_monthly_weights(price_returns_df: pd.DataFrame, method: str, lookback_days: int, freq: str):
    # price_returns_df: DataFrame columns=tickers, index=daily dates, values=buy&hold daily returns
    # method: "Equal Weight" or "Inverse Volatility"
    # freq: 'M' or 'Q'
    if method == "Equal Weight":
        n = price_returns_df.shape[1]
        w = pd.DataFrame(1.0 / n, index=price_returns_df.resample(freq).mean().index, columns=price_returns_df.columns)
        # expand to daily by forward fill from resample index start
        daily_weights = w.reindex(price_returns_df.index, method='ffill').fillna(method='ffill').fillna(1.0 / price_returns_df.shape[1])
        return daily_weights
    elif method == "Inverse Volatility":
        # compute rolling lookback vol per asset at each rebalance date (end of period)
        # use historical buy&hold returns (price_returns_df)
        vol = price_returns_df.rolling(window=lookback_days, min_periods=10).std()
        # take vol at rebalance dates (end of period)
        vol_at_reb = vol.resample(freq).last()
        inv_vol = 1.0 / vol_at_reb
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0)
        # normalize
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1.0 / price_returns_df.shape[1])
        # forward fill to daily index
        daily_weights = weights.reindex(price_returns_df.index, method='ffill').fillna(method='ffill')
        # If still NaN (start), fill equal
        daily_weights = daily_weights.fillna(1.0 / price_returns_df.shape[1])
        return daily_weights
    else:
        raise ValueError("Unknown method")

def build_portfolio(assets_list, start, end, div_method, rebalance_freq, vol_lookback):
    # prepare per-ticker dfs and trades
    per_asset_dfs = {}
    per_asset_trades = {}
    for t in assets_list:
        df, trades = prepare_ticker_series(t, start, end)
        if df is None:
            st.warning(f"{t}: No data.")
            continue
        per_asset_dfs[t] = df
        per_asset_trades[t] = trades

    # align indices
    all_index = sorted(set().union(*[set(df.index) for df in per_asset_dfs.values()]))
    # build DataFrames of daily strategy and buyhold returns
    strat_returns = pd.DataFrame(index=all_index)
    bh_returns = pd.DataFrame(index=all_index)
    for t, df in per_asset_dfs.items():
        tmp = df.reindex(all_index).copy()
        strat_returns[t] = tmp['Strategy_Daily_Return']
        bh_returns[t] = tmp['BuyHold_Daily_Return']

    # compute weights
    freq_map = {'Monthly': 'M', 'Quarterly': 'Q'}
    freq = freq_map.get(rebalance_freq, 'M')
    daily_weights = compute_monthly_weights(bh_returns.fillna(0), div_method, vol_lookback, freq)

    # compute portfolio returns at each day (weighted sum across assets)
    # If asset has NaN returns on that day, treat as 0 return for that asset (cash)
    strat_returns_filled = strat_returns.fillna(0)
    bh_returns_filled = bh_returns.fillna(0)
    # align weights index and fill missing days
    daily_weights = daily_weights.reindex(strat_returns_filled.index, method='ffill').fillna(1.0 / max(1, len(per_asset_dfs)))
    portfolio_strategy_return = (daily_weights * strat_returns_filled).sum(axis=1)
    portfolio_bh_return = (daily_weights * bh_returns_filled).sum(axis=1)

    # Build portfolio-level cumulative series and metrics
    portfolio_df = pd.DataFrame(index=strat_returns_filled.index)
    portfolio_df['Strategy_Daily_Return'] = portfolio_strategy_return
    portfolio_df['BuyHold_Daily_Return'] = portfolio_bh_return
    portfolio_df['Strategy_Cumulative'] = (1 + portfolio_df['Strategy_Daily_Return']).cumprod().fillna(method='ffill').fillna(1)
    portfolio_df['BuyHold_Cumulative'] = (1 + portfolio_df['BuyHold_Daily_Return']).cumprod().fillna(method='ffill').fillna(1)

    return {
        'per_asset_dfs': per_asset_dfs,
        'per_asset_trades': per_asset_trades,
        'portfolio_df': portfolio_df,
        'daily_weights': daily_weights
    }

# ---------------------------
# Main: Single asset or Portfolio
# ---------------------------
try:
    if mode == "Single Asset":
        with st.spinner(f'{asset} 데이터 다운로드 중...'):
            qdf = download_data(asset, start_date, end_date)

        if qdf is None or len(qdf) == 0:
            st.error("데이터를 불러올 수 없습니다. 날짜 범위를 확인하거나 네트워크 문제를 점검하세요.")
            st.stop()

        with st.spinner('지표 및 시그널 계산 중...'):
            qdf = calculate_indicators(qdf)
            qdf, trades_df = generate_signals(qdf)
            qdf = calculate_returns(qdf)

        # 화면 표시 (single)
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
            elif latest['Exit_By_SMA300'] and current_pos == 1:
                st.markdown("### 🔴 청산 신호 (SMA_300)")
            elif latest['Exit_By_PrevHigh'] and current_pos == 1:
                st.markdown("### 🔴 청산 신호 (Prev High)")
            else:
                st.markdown("### ⚪ 신호 없음")

        with st.expander("📊 현재 지표 상세"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("IBS", f"{latest.get('IBS', np.nan):.3f}")
                st.metric("하단밴드", f"${latest.get('Lower_Band', np.nan):.2f}")
            with col2:
                st.metric("전일 고가", f"${latest.get('Prev_High', np.nan):.2f}")
                st.metric("10일 최고가", f"${latest.get('Rolling_High_10', np.nan):.2f}")
            with col3:
                st.metric("300-SMA", f"${latest.get('SMA_300', np.nan):.2f}")
                st.markdown(f"**종가 < 하단밴드:** {'✅' if latest.get('Close', np.nan) < latest.get('Lower_Band', np.nan) else '❌'}")
                st.markdown(f"**IBS < 0.3:** {'✅' if latest.get('IBS', np.nan) < 0.3 else '❌'}")

        # metrics
        qdf_clean = qdf.dropna(subset=['Strategy_Daily_Return', 'BuyHold_Daily_Return'])
        strat_metrics = calculate_metrics(qdf_clean['Strategy_Daily_Return'])
        bm_metrics = calculate_metrics(qdf_clean['BuyHold_Daily_Return'])

        st.header("📈 성과 지표 (Strategy vs BM)")
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
            st.metric("총 거래 수", len(trades_df))
            if len(trades_df) > 0:
                st.metric("승률", f"{(trades_df['Return'] > 0).sum() / len(trades_df) * 100:.1f}%")

        # charts
        st.header("📊 차트")
        tab1, tab2, tab3 = st.tabs(["누적 수익률", "낙폭", "거래 분석"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=qdf_clean.index, y=qdf_clean['Strategy_Cumulative'], name='Strategy', line=dict(color='#2E86AB')))
            fig.add_trace(go.Scatter(x=qdf_clean.index, y=qdf_clean['BuyHold_Cumulative'], name='BM', line=dict(color='#A23B72')))
            fig.update_layout(title=f'{asset} 누적 수익률 (Strategy vs BM)', xaxis_title='날짜', yaxis_title='누적 수익률', template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            strat_cum = qdf_clean['Strategy_Cumulative']
            running_max = strat_cum.expanding().max()
            drawdown = (strat_cum - running_max) / running_max * 100
            bh_cum = qdf_clean['BuyHold_Cumulative']
            bh_max = bh_cum.expanding().max()
            bh_draw = (bh_cum - bh_max) / bh_max * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=qdf_clean.index, y=drawdown, name='Strategy Drawdown', fill='tozeroy', line=dict(color='#EF553B')))
            fig2.add_trace(go.Scatter(x=qdf_clean.index, y=bh_draw, name='BM Drawdown', fill='tozeroy', line=dict(color='#FFA500')))
            fig2.update_layout(title='Drawdown', xaxis_title='날짜', yaxis_title='낙폭 (%)', template='plotly_white', height=500)
            st.plotly_chart(fig2, use_container_width=True)
        with tab3:
            if len(trades_df) > 0:
                fig3 = go.Figure()
                returns_pct = trades_df['Return'] * 100
                fig3.add_trace(go.Histogram(x=returns_pct, nbinsx=50, marker=dict(color='#2E86AB')))
                fig3.add_vline(x=0, line_dash="dash", line_color="red")
                fig3.update_layout(title='거래별 수익률 분포', xaxis_title='수익률 (%)', yaxis_title='빈도', height=400, template='plotly_white')
                st.plotly_chart(fig3, use_container_width=True)
                st.dataframe(trades_df.tail(20).assign(Return=lambda d: d['Return']*100), use_container_width=True)
            else:
                st.info("거래 데이터가 없습니다.")

    else:
        # Portfolio mode
        with st.spinner('다중 자산 데이터 다운로드 및 처리 중...'):
            result = build_portfolio(ASSETS, start_date, end_date, div_method, rebalance_freq, vol_lookback)

        per_asset_dfs = result['per_asset_dfs']
        per_asset_trades = result['per_asset_trades']
        portfolio_df = result['portfolio_df']
        daily_weights = result['daily_weights']

        st.header("📊 Diversified Portfolio (Strategy aggregated across assets)")
        # show weights snapshot
        st.subheader("Weights snapshot (most recent rebalance)")
        last_weights = daily_weights.loc[portfolio_df.index.intersection(daily_weights.index)].ffill().iloc[-1]
        st.bar_chart(last_weights)

        latest = portfolio_df.iloc[-1]
        st.metric("날짜", portfolio_df.index[-1].strftime('%Y-%m-%d'))
        st.metric("포트폴리오 전략 누적수익률", f"{(portfolio_df['Strategy_Cumulative'].iloc[-1]-1)*100:.2f}%")
        st.metric("포트폴리오 BM 누적수익률", f"{(portfolio_df['BuyHold_Cumulative'].iloc[-1]-1)*100:.2f}%")

        # metrics
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
            total_trades = sum([len(t) for t in per_asset_trades.values()])
            st.metric("총 거래 수 (모든 자산 합)", total_trades)

        # charts
        st.header("📊 차트 (Portfolio)")
        tab1, tab2, tab3 = st.tabs(["누적 수익률", "낙폭", "개별 자산 기여"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df['Strategy_Cumulative'], name='Strategy', line=dict(color='#2E86AB')))
            fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df['BuyHold_Cumulative'], name='BM', line=dict(color='#A23B72')))
            fig.update_layout(title=f'Portfolio 누적 수익률 ({div_method} / rebalance={rebalance_freq})', template='plotly_white', height=500)
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
            # compute contribution: weights * per-asset cumulative
            contributions = {}
            for t, df in per_asset_dfs.items():
                # align with portfolio index
                tmp = df.reindex(portfolio_df.index).copy()
                tmp['Strat_Cum'] = (1 + tmp['Strategy_Daily_Return']).cumprod().fillna(method='ffill').fillna(1)
                contributions[t] = tmp['Strat_Cum']
            contrib_df = pd.DataFrame(contributions)
            st.line_chart(contrib_df)

        # show recent trades across assets
        st.markdown("---")
        st.subheader("최근 거래 (자산별)")
        recent_trades_list = []
        for t, trades in per_asset_trades.items():
            if trades is None or len(trades) == 0:
                continue
            tmp = trades.copy()
            tmp['Asset'] = t
            recent_trades_list.append(tmp)
        if recent_trades_list:
            all_recent = pd.concat(recent_trades_list, ignore_index=True)
            all_recent = all_recent.sort_values('Exit_Date', ascending=False).head(50)
            all_recent['Return (%)'] = all_recent['Return'] * 100
            st.dataframe(all_recent[['Asset', 'Trade_Number', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Return (%)', 'Days', 'Exit_Reason']], use_container_width=True)
        else:
            st.info("최근 거래가 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.exception(e)
