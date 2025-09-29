from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.tools.api import get_prices, prices_to_df
import json
import numpy as np
import pandas as pd
from src.utils.api_key import get_api_key_from_state

##### 风险管理代理 #####
def risk_management_agent(state: AgentState, agent_id: str = "risk_management_agent"):
    """根据多个股票的波动率调整风险因素来控制头寸规模。"""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    # 初始化每个股票的风险分析
    risk_analysis = {}
    current_prices = {}  # 在此处存储价格以避免冗余的API调用
    volatility_data = {}  # 存储波动率指标
    returns_by_ticker: dict[str, pd.Series] = {}  # 用于相关性分析

    # 首先，获取所有相关股票的价格并计算波动率
    all_tickers = set(tickers) | set(portfolio.get("positions", {}).keys())
    
    for ticker in all_tickers:
        progress.update_status(agent_id, ticker, "Fetching price data and calculating volatility")
        
        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
            api_key=api_key,
        )

        if not prices:
            progress.update_status(agent_id, ticker, "Warning: No price data found")
            volatility_data[ticker] = {
                "daily_volatility": 0.05,  # 默认回退波动率（每日5%）
                "annualized_volatility": 0.05 * np.sqrt(252),
                "volatility_percentile": 100,  # 如果没有数据，则假定高风险
                "data_points": 0
            }
            continue

        prices_df = prices_to_df(prices)
        
        if not prices_df.empty and len(prices_df) > 1:
            current_price = prices_df["close"].iloc[-1]
            current_prices[ticker] = current_price
            
            # 计算波动率指标
            volatility_metrics = calculate_volatility_metrics(prices_df)
            volatility_data[ticker] = volatility_metrics

            # 存储收益率用于相关性分析（使用收盘价对收盘价的收益率）
            daily_returns = prices_df["close"].pct_change().dropna()
            if len(daily_returns) > 0:
                returns_by_ticker[ticker] = daily_returns
            
            progress.update_status(
                agent_id, 
                ticker, 
                f"Price: {current_price:.2f}, Ann. Vol: {volatility_metrics['annualized_volatility']:.1%}"
            )
        else:
            progress.update_status(agent_id, ticker, "Warning: Insufficient price data")
            current_prices[ticker] = 0
            volatility_data[ticker] = {
                "daily_volatility": 0.05,
                "annualized_volatility": 0.05 * np.sqrt(252),
                "volatility_percentile": 100,
                "data_points": len(prices_df) if not prices_df.empty else 0
            }

    # 构建跨股票对齐的收益率DataFrame以进行相关性分析
    correlation_matrix = None
    if len(returns_by_ticker) >= 2:
        try:
            returns_df = pd.DataFrame(returns_by_ticker).dropna(how="any")
            if returns_df.shape[1] >= 2 and returns_df.shape[0] >= 5:
                correlation_matrix = returns_df.corr()
        except Exception:
            correlation_matrix = None

    # 确定当前哪些股票有风险敞口（非零绝对头寸）
    active_positions = {
        t for t, pos in portfolio.get("positions", {}).items()
        if abs(pos.get("long", 0) - pos.get("short", 0)) > 0
    }

    # 根据当前市价计算总投资组合价值（净清算价值）
    total_portfolio_value = portfolio.get("cash", 0.0)
    
    for ticker, position in portfolio.get("positions", {}).items():
        if ticker in current_prices:
            # 加上多头头寸的市值
            total_portfolio_value += position.get("long", 0) * current_prices[ticker]
            # 减去空头头寸的市值
            total_portfolio_value -= position.get("short", 0) * current_prices[ticker]
    
    progress.update_status(agent_id, None, f"Total portfolio value: {total_portfolio_value:.2f}")

    # 计算每个股票经波动率和相关性调整后的风险限额
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Calculating volatility- and correlation-adjusted limits")
        
        if ticker not in current_prices or current_prices[ticker] <= 0:
            progress.update_status(agent_id, ticker, "Failed: No valid price data")
            risk_analysis[ticker] = {
                "remaining_position_limit": 0.0,
                "current_price": 0.0,
                "reasoning": {
                    "error": "Missing price data for risk calculation"
                }
            }
            continue
            
        current_price = current_prices[ticker]
        vol_data = volatility_data.get(ticker, {})
        
        # 计算此头寸的当前市值
        position = portfolio.get("positions", {}).get(ticker, {})
        long_value = position.get("long", 0) * current_price
        short_value = position.get("short", 0) * current_price
        current_position_value = abs(long_value - short_value)  # 使用绝对风险敞口
        
        # 波动率调整后的限额百分比
        vol_adjusted_limit_pct = calculate_volatility_adjusted_limit(
            vol_data.get("annualized_volatility", 0.25)
        )

        # 相关性调整
        corr_metrics = {
            "avg_correlation_with_active": None,
            "max_correlation_with_active": None,
            "top_correlated_tickers": [],
        }
        corr_multiplier = 1.0
        if correlation_matrix is not None and ticker in correlation_matrix.columns:
            # 计算与活跃头寸的相关性（排除自身）
            comparable = [t for t in active_positions if t in correlation_matrix.columns and t != ticker]
            if not comparable:
                # 如果没有活跃头寸，则与所有其他可用股票进行比较
                comparable = [t for t in correlation_matrix.columns if t != ticker]
            if comparable:
                series = correlation_matrix.loc[ticker, comparable]
                # 以防万一，删除NaN
                series = series.dropna()
                if len(series) > 0:
                    avg_corr = float(series.mean())
                    max_corr = float(series.max())
                    corr_metrics["avg_correlation_with_active"] = avg_corr
                    corr_metrics["max_correlation_with_active"] = max_corr
                    # 相关性最高的3个股票
                    top_corr = series.sort_values(ascending=False).head(3)
                    corr_metrics["top_correlated_tickers"] = [
                        {"ticker": idx, "correlation": float(val)} for idx, val in top_corr.items()
                    ]
                    corr_multiplier = calculate_correlation_multiplier(avg_corr)
        
        # 合并波动率和相关性调整
        combined_limit_pct = vol_adjusted_limit_pct * corr_multiplier
        # 转换为美元头寸限额
        position_limit = total_portfolio_value * combined_limit_pct
        
        # 计算该头寸的剩余限额
        remaining_position_limit = position_limit - current_position_value
        
        # 确保我们不超过可用现金
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))
        
        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "volatility_metrics": {
                "daily_volatility": float(vol_data.get("daily_volatility", 0.05)),
                "annualized_volatility": float(vol_data.get("annualized_volatility", 0.25)),
                "volatility_percentile": float(vol_data.get("volatility_percentile", 100)),
                "data_points": int(vol_data.get("data_points", 0))
            },
            "correlation_metrics": corr_metrics,
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position_value": float(current_position_value),
                "base_position_limit_pct": float(vol_adjusted_limit_pct),
                "correlation_multiplier": float(corr_multiplier),
                "combined_position_limit_pct": float(combined_limit_pct),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "risk_adjustment": f"Volatility x Correlation adjusted: {combined_limit_pct:.1%} (base {vol_adjusted_limit_pct:.1%})"
            },
        }
        
        progress.update_status(
            agent_id, 
            ticker, 
            f"Adj. limit: {combined_limit_pct:.1%}, Available: ${max_position_size:.0f}"
        )

    progress.update_status(agent_id, None, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Volatility-Adjusted Risk Management Agent")

    # 将信号添加到analyst_signals列表中
    state["data"]["analyst_signals"][agent_id] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def calculate_volatility_metrics(prices_df: pd.DataFrame, lookback_days: int = 60) -> dict:
    """从价格数据计算综合波动率指标。"""
    if len(prices_df) < 2:
        return {
            "daily_volatility": 0.05,
            "annualized_volatility": 0.05 * np.sqrt(252),
            "volatility_percentile": 100,
            "data_points": len(prices_df)
        }
    
    # 计算每日收益率
    daily_returns = prices_df["close"].pct_change().dropna()
    
    if len(daily_returns) < 2:
        return {
            "daily_volatility": 0.05,
            "annualized_volatility": 0.05 * np.sqrt(252),
            "volatility_percentile": 100,
            "data_points": len(daily_returns)
        }
    
    # 使用最近的lookback_days进行波动率计算
    recent_returns = daily_returns.tail(min(lookback_days, len(daily_returns)))
    
    # 计算波动率指标
    daily_vol = recent_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)  # 假设252个交易日进行年化
    
    # 计算近期波动率与历史波动率的百分位排名
    if len(daily_returns) >= 30:  # 需要足够的历史数据来进行百分位计算
        # 计算整个历史记录的30天滚动波动率
        rolling_vol = daily_returns.rolling(window=30).std().dropna()
        if len(rolling_vol) > 0:
            # 将当前波动率与历史滚动波动率进行比较
            current_vol_percentile = (rolling_vol <= daily_vol).mean() * 100
        else:
            current_vol_percentile = 50  # 默认为中位数
    else:
        current_vol_percentile = 50  # 如果数据不足，则默认为中位数
    
    return {
        "daily_volatility": float(daily_vol) if not np.isnan(daily_vol) else 0.025,
        "annualized_volatility": float(annualized_vol) if not np.isnan(annualized_vol) else 0.25,
        "volatility_percentile": float(current_vol_percentile) if not np.isnan(current_vol_percentile) else 50.0,
        "data_points": len(recent_returns)
    }


def calculate_volatility_adjusted_limit(annualized_volatility: float) -> float:
    """
    根据波动率计算头寸限额占投资组合的百分比。
    
    逻辑：
    - 低波动率 (<15%): 最高25%的配置
    - 中等波动率 (15-30%): 15-20%的配置
    - 高波动率 (>30%): 10-15%的配置
    - 非常高的波动率 (>50%): 最高10%的配置
    """
    base_limit = 0.20  # 20% 基准
    
    if annualized_volatility < 0.15:  # 低波动率
        # 允许为稳定型股票提供更高的配置
        vol_multiplier = 1.25  # 最高25%
    elif annualized_volatility < 0.30:  # 中等波动率
        # 标准配置，根据波动率进行微调
        vol_multiplier = 1.0 - (annualized_volatility - 0.15) * 0.5  # 20% -> 12.5%
    elif annualized_volatility < 0.50:  # 高波动率
        # 大幅减少配置
        vol_multiplier = 0.75 - (annualized_volatility - 0.30) * 0.5  # 15% -> 5%
    else:  # 非常高的波动率 (>50%)
        # 对风险非常高的股票进行最低配置
        vol_multiplier = 0.50  # 最高10%
    
    # 应用边界以确保合理的限制
    vol_multiplier = max(0.25, min(1.25, vol_multiplier))  # 5%到25%的范围
    
    return base_limit * vol_multiplier


def calculate_correlation_multiplier(avg_correlation: float) -> float:
    """将平均相关性映射到调整乘数。
    - 非常高的相关性 (>= 0.8): 大幅降低限额 (0.7倍)
    - 高相关性 (0.6-0.8): 降低 (0.85倍)
    - 中等相关性 (0.4-0.6): 中性 (1.0倍)
    - 低相关性 (0.2-0.4): 略微增加 (1.05倍)
    - 非常低的相关性 (< 0.2): 增加 (1.10倍)
    """
    if avg_correlation >= 0.80:
        return 0.70
    if avg_correlation >= 0.60:
        return 0.85
    if avg_correlation >= 0.40:
        return 1.00
    if avg_correlation >= 0.20:
        return 1.05
    return 1.10
