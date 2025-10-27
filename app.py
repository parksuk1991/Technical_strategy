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
    page_title="Mean Reversion Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Mean Reversion Strategy Dashboard")
st.markdown("---")

# 자산 리스트 (선택 가능한 티커들)
ASSETS = ["QQQ", "SPY", "EWY", "IEMG", "IDEV", "ACWI"]

# 사이드바: 자산 선택 및 날짜 등 설정
with st.sidebar:
    st.header("⚙️ 설정")
    asset = st.selectbox("자산 선택", options=ASSETS, index=0)
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
    st.markdown("### 전략 규칙")
    st.markdown("""
    1. Compute the rolling mean of High minus Low over the last 25 days.
    2. Compute the IBS indicator: (Close - Low) / (High - Low).
    3. Compute a lower band as the rolling High over the last 10 days minus 2.5 × the rolling mean from (1).
    4. Go long whenever the asset closes under the lower band (3) and IBS < 0.3.
    5. Close the trade whenever close price is higher than yesterday's high.
    6. Also, close the trade whenever the price is lower than the 300-SMA.
    """)
    st.markdown("---")
    st.caption("BM (benchmark) is the asset's Buy & Hold and will be shown as 'BM' in comparisons.")

# 데이터 다운로드 (캐시)
@st.cache_data(ttl=3600, show_spinner=False)
def download_data(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# 지표 계산: 규칙 1~3 및 6 (300-SMA)
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Avg_25'] = df['HL_Range'].rolling(window=25, min_periods=1).mean()
    # IBS: 분모 0 방지
    denom = (df['High'] - df['Low']).replace(0, np.nan)
    df['IBS'] = (df['Close'] - df['Low']) / denom
    df['Rolling_High_10'] = df['High'].rolling(window=10, min_periods=1).max()
    df['Lower_Band'] = df['Rolling_High_10'] - (2.5 * df['HL_Avg_25'])
    df['Prev_High'] = df['High'].shift(1)
    # 300-SMA (rule 6)
    df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    # Entry condition (rule 4)
    df['Entry_Condition'] = (df['Close'] < df['Lower_Band']) & (df['IBS'] < 0.3)
    # Exit conditions (rule 5 and 6)
    df['Exit_By_PrevHigh'] = df['Close'] > df['Prev_High']
    df['Exit_By_SMA300'] = df['Close'] < df['SMA_200']
    df['Exit_Condition'] = df['Exit_By_PrevHigh'] | df['Exit_By_SMA300']
    return df

# 시그널 생성: entry/exit + 트레이드 기록 (exit reason 포함)
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
        # skip if entry condition cannot be evaluated yet
        if pd.isna(df['Entry_Condition'].iloc[i]):
            continue

        if position == 0 and df['Entry_Condition'].iloc[i]:
            # Enter long
            position = 1
            entry_price = df['Close'].iloc[i]
            entry_date = df.index[i]
            trade_number += 1
            df.at[df.index[i], 'Signal'] = 1
            df.at[df.index[i], 'Trade_Number'] = trade_number

        elif position == 1:
            df.at[df.index[i], 'Trade_Number'] = trade_number
            # Exit if any exit condition true
            if df['Exit_Condition'].iloc[i]:
                position = 0
                exit_price = df['Close'].iloc[i]
                exit_date = df.index[i]
                df.at[df.index[i], 'Signal'] = -1

                # determine reason(s)
                reasons = []
                if df['Exit_By_PrevHigh'].iloc[i]:
                    reasons.append('Prev_High')
                if df['Exit_By_SMA300'].iloc[i]:
                    reasons.append('SMA_200')
                reason_text = "|".join(reasons) if reasons else "Exit"

                df.at[df.index[i], 'Exit_Reason'] = reason_text

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
        df.at[df.index[i], 'Position'] = position

    trades_df = pd.DataFrame(trades)
    return df, trades_df, position

# 수익률 계산: 전략 vs BM (Buy & Hold)
def calculate_returns(df: pd.DataFrame):
    df = df.copy()
    df['Strategy_Daily_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['BuyHold_Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Cumulative'] = (1 + df['Strategy_Daily_Return']).cumprod().fillna(method='ffill').fillna(1)
    df['BuyHold_Cumulative'] = (1 + df['BuyHold_Daily_Return']).cumprod().fillna(method='ffill').fillna(1)
    return df

# 성과지표 계산 (간단하고 안정적으로)
def calculate_metrics(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return {'Total Return': 0, 'CAGR': 0, 'Volatility': 0, 'Sharpe Ratio': 0,
                'Sortino Ratio': 0, 'Max Drawdown': 0, 'Calmar Ratio': 0, 'Win Rate': 0}
    total_return = (1 + returns).prod() - 1
    n_days = len(returns)
    n_years = n_days / 252 if n_days > 0 else 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility_annual = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility_annual if volatility_annual != 0 else 0
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino = (returns.mean() * 252) / downside_std if (not np.isnan(downside_std) and downside_std != 0) else 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    return {'Total Return': total_return, 'CAGR': cagr, 'Volatility': volatility_annual,
            'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino, 'Max Drawdown': max_dd,
            'Calmar Ratio': calmar, 'Win Rate': win_rate}

# 메인
try:
    with st.spinner(f'{asset} 데이터 다운로드 중...'):
        qdf = download_data(asset, start_date, end_date)

    if qdf is None or len(qdf) == 0:
        st.error("데이터를 불러올 수 없습니다. 날짜 범위를 확인하거나 네트워크 문제를 점검하세요.")
        st.stop()

    with st.spinner('지표 및 시그널 계산 중...'):
        qdf = calculate_indicators(qdf)
        qdf, trades_df, current_position = generate_signals(qdf)
        qdf = calculate_returns(qdf)

    # 현재 시그널 표시
    st.header("🔔 현재 시그널")
    latest = qdf.iloc[-1]
    latest_date = qdf.index[-1].strftime('%Y-%m-%d')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("날짜", latest_date)
    with c2:
        st.metric(f"{asset} 종가", f"${latest['Close']:.2f}")
    with c3:
        pos_text = "✅ 보유 중" if current_position == 1 else "⏸️ 대기 중"
        st.metric("현재 포지션", pos_text)
    with c4:
        # Show which exit condition is active if any
        if latest['Entry_Condition'] and current_position == 0:
            st.markdown("### 🟢 진입 신호")
        elif latest['Exit_By_SMA300'] and current_position == 1:
            st.markdown("### 🔴 청산 신호 (SMA_200)")
        elif latest['Exit_By_PrevHigh'] and current_position == 1:
            st.markdown("### 🔴 청산 신호 (Prev High)")
        else:
            st.markdown("### ⚪ 신호 없음")

    # 지표 상세
    with st.expander("📊 현재 지표 상세"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("IBS", f"{latest.get('IBS', np.nan):.3f}")
            st.metric("하단밴드", f"${latest.get('Lower_Band', np.nan):.2f}")
        with col2:
            st.metric("전일 고가", f"${latest.get('Prev_High', np.nan):.2f}")
            st.metric("10일 최고가", f"${latest.get('Rolling_High_10', np.nan):.2f}")
        with col3:
            st.metric("200-SMA", f"${latest.get('SMA_200', np.nan):.2f}")
            entry_check = "✅" if latest.get('Close', np.nan) < latest.get('Lower_Band', np.nan) else "❌"
            st.markdown(f"**종가 < 하단밴드:** {entry_check}")
            ibs_check = "✅" if latest.get('IBS', np.nan) < 0.3 else "❌"
            st.markdown(f"**IBS < 0.3:** {ibs_check}")

    st.markdown("---")

    # 성과 지표 계산 (Strategy vs BM)
    qdf_clean = qdf.dropna(subset=['Strategy_Daily_Return', 'BuyHold_Daily_Return'])
    strat_metrics = calculate_metrics(qdf_clean['Strategy_Daily_Return'])
    bm_metrics = calculate_metrics(qdf_clean['BuyHold_Daily_Return'])  # BM is buy & hold of same asset

    st.header("📈 성과 지표 (Strategy vs BM)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("전략 누적수익률", f"{strat_metrics['Total Return']*100:.2f}%")
        st.metric("전략 CAGR", f"{strat_metrics['CAGR']*100:.2f}%")
    with col2:
        st.metric("BM 누적수익률", f"{bm_metrics['Total Return']*100:.2f}%")
        st.metric("BM CAGR", f"{bm_metrics['CAGR']*100:.2f}%")
    with col3:
        st.metric("전략 샤프 비율", f"{strat_metrics['Sharpe Ratio']:.2f}")
        st.metric("전략 최대낙폭", f"{strat_metrics['Max Drawdown']*100:.2f}%")
    with col4:
        st.metric("총 거래 수", len(trades_df))
        if len(trades_df) > 0:
            win_rate = (trades_df['Return'] > 0).sum() / len(trades_df) * 100
            st.metric("승률", f"{win_rate:.1f}%")

    # 차트: 누적수익률, 낙폭, 거래분포, 연간수익률
    st.header("📊 차트")
    tab1, tab2, tab3, tab4 = st.tabs(["누적 수익률", "낙폭", "거래 분석", "연간 수익률"])

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=qdf_clean.index, y=qdf_clean['Strategy_Cumulative'],
                                  name='Strategy', line=dict(color='#2E86AB', width=2)))
        fig1.add_trace(go.Scatter(x=qdf_clean.index, y=qdf_clean['BuyHold_Cumulative'],
                                  name='BM', line=dict(color='#A23B72', width=2)))
        fig1.update_layout(title=f'{asset} 누적 수익률 (Strategy vs BM)',
                           xaxis_title='날짜', yaxis_title='누적 수익률',
                           hovermode='x unified', template='plotly_white', height=500)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        strategy_cum = qdf_clean['Strategy_Cumulative']
        running_max = strategy_cum.expanding().max()
        drawdown = (strategy_cum - running_max) / running_max * 100
        bh_cum = qdf_clean['BuyHold_Cumulative']
        bh_max = bh_cum.expanding().max()
        bh_drawdown = (bh_cum - bh_max) / bh_max * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=qdf_clean.index, y=drawdown, name='Strategy Drawdown',
                                  fill='tozeroy', line=dict(color='#EF553B', width=1)))
        fig2.add_trace(go.Scatter(x=qdf_clean.index, y=bh_drawdown, name='BM Drawdown',
                                  fill='tozeroy', line=dict(color='#FFA500', width=1)))
        fig2.update_layout(title='Drawdown', xaxis_title='날짜', yaxis_title='낙폭 (%)',
                           hovermode='x unified', template='plotly_white', height=500)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        if len(trades_df) > 0:
            colA, colB = st.columns(2)
            with colA:
                fig3 = go.Figure()
                returns_pct = trades_df['Return'] * 100
                fig3.add_trace(go.Histogram(x=returns_pct, nbinsx=50, name='Trade Returns',
                                            marker=dict(color='#2E86AB')))
                fig3.add_vline(x=0, line_dash="dash", line_color="red")
                fig3.update_layout(title='거래별 수익률 분포', xaxis_title='수익률 (%)', yaxis_title='빈도',
                                   template='plotly_white', height=400)
                st.plotly_chart(fig3, use_container_width=True)
            with colB:
                st.markdown("### 거래 통계")
                st.metric("총 거래", len(trades_df))
                st.metric("평균 수익률", f"{trades_df['Return'].mean()*100:.2f}%")
                st.metric("평균 보유 기간", f"{trades_df['Days'].mean():.1f}일")
                st.metric("최대 수익", f"{trades_df['Return'].max()*100:.2f}%")
                st.metric("최대 손실", f"{trades_df['Return'].min()*100:.2f}%")
            with st.expander("📋 최근 거래 내역"):
                recent_trades = trades_df.tail(10).copy()
                recent_trades['Entry_Date'] = pd.to_datetime(recent_trades['Entry_Date']).dt.strftime('%Y-%m-%d')
                recent_trades['Exit_Date'] = pd.to_datetime(recent_trades['Exit_Date']).dt.strftime('%Y-%m-%d')
                recent_trades['Return (%)'] = recent_trades['Return'] * 100
                st.dataframe(recent_trades[['Trade_Number', 'Entry_Date', 'Exit_Date', 'Entry_Price',
                                            'Exit_Price', 'Return (%)', 'Days', 'Exit_Reason']], use_container_width=True)
        else:
            st.info("거래 데이터가 없습니다.")

    with tab4:
        yearly_strategy = qdf_clean['Strategy_Daily_Return'].resample('YE').apply(lambda x: (1+x).prod()-1)
        yearly_bh = qdf_clean['BuyHold_Daily_Return'].resample('YE').apply(lambda x: (1+x).prod()-1)
        years = [d.year for d in yearly_strategy.index]
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=years, y=yearly_strategy * 100, name='Strategy', marker_color='#2E86AB'))
        fig4.add_trace(go.Bar(x=years, y=yearly_bh * 100, name='BM', marker_color='#A23B72'))
        fig4.add_hline(y=0, line_dash="solid", line_color="black")
        fig4.update_layout(title='연간 수익률', xaxis_title='연도', yaxis_title='수익률 (%)',
                           template='plotly_white', height=500, barmode='group')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.header("💾 데이터 다운로드")
    c1, c2 = st.columns(2)
    with c1:
        csv_data = qdf.to_csv()
        st.download_button(
            label=f"📥 전체 데이터 다운로드 (CSV) - {asset}",
            data=csv_data,
            file_name=f"{asset.lower()}_strategy_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with c2:
        if len(trades_df) > 0:
            trades_csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 거래 내역 다운로드 (CSV)",
                data=trades_csv,
                file_name=f"{asset.lower()}_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.exception(e)
