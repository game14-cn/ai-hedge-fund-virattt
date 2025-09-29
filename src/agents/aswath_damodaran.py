from __future__ import annotations

import json
from typing_extensions import Literal
from pydantic import BaseModel

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class AswathDamodaranSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float          # 0-100
    reasoning: str


def aswath_damodaran_agent(state: AgentState, agent_id: str = "aswath_damodaran_agent"):
    """
    通过Aswath Damodaran的内在价值视角分析美国股票：
      • 通过CAPM计算股权成本（无风险利率 + β·ERP）
      • 5年收入/自由现金流增长趋势和再投资效率
      • 公司自由现金流DCF → 股权价值 → 每股内在价值
      • 与相对估值进行交叉检验（市盈率 vs. 远期市盈率行业中位数代理）
    以Damodaran的分析口吻生成交易信号和解释。
    """
    data      = state["data"]
    end_date  = data["end_date"]
    tickers   = data["tickers"]
    api_key  = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data: dict[str, dict] = {}
    damodaran_signals: dict[str, dict] = {}

    for ticker in tickers:
        # ─── 获取核心数据 ────────────────────────────────────────────────────
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching financial line items")
        line_items = search_line_items(
            ticker,
            [
                "free_cash_flow",
                "ebit",
                "interest_expense",
                "capital_expenditure",
                "depreciation_and_amortization",
                "outstanding_shares",
                "net_income",
                "total_debt",
            ],
            end_date,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # ─── 分析 ───────────────────────────────────────────────────────────
        progress.update_status(agent_id, ticker, "Analyzing growth and reinvestment")
        growth_analysis = analyze_growth_and_reinvestment(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing risk profile")
        risk_analysis = analyze_risk_profile(metrics, line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value (DCF)")
        intrinsic_val_analysis = calculate_intrinsic_value_dcf(metrics, line_items, risk_analysis)

        progress.update_status(agent_id, ticker, "Assessing relative valuation")
        relative_val_analysis = analyze_relative_valuation(metrics)

        # ─── 评分和安全边际 ──────────────────────────────────────────
        total_score = (
            growth_analysis["score"]
            + risk_analysis["score"]
            + relative_val_analysis["score"]
        )
        max_score = growth_analysis["max_score"] + risk_analysis["max_score"] + relative_val_analysis["max_score"]

        intrinsic_value = intrinsic_val_analysis["intrinsic_value"]
        margin_of_safety = (
            (intrinsic_value - market_cap) / market_cap if intrinsic_value and market_cap else None
        )

        # 决策规则 (Damodaran倾向于在约20-25%的安全边际时采取行动)
        if margin_of_safety is not None and margin_of_safety >= 0.25:
            signal = "bullish"
        elif margin_of_safety is not None and margin_of_safety <= -0.25:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "margin_of_safety": margin_of_safety,
            "growth_analysis": growth_analysis,
            "risk_analysis": risk_analysis,
            "relative_val_analysis": relative_val_analysis,
            "intrinsic_val_analysis": intrinsic_val_analysis,
            "market_cap": market_cap,
        }

        # ─── LLM: 构建Damodaran风格的叙述 ──────────────────────────────
        progress.update_status(agent_id, ticker, "Generating Damodaran analysis")
        damodaran_output = generate_damodaran_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        damodaran_signals[ticker] = damodaran_output.model_dump()

        progress.update_status(agent_id, ticker, "Done", analysis=damodaran_output.reasoning)

    # ─── 将消息推送回图状态 ──────────────────────────────────────
    message = HumanMessage(content=json.dumps(damodaran_signals), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(damodaran_signals, "Aswath Damodaran Agent")

    state["data"]["analyst_signals"][agent_id] = damodaran_signals
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ────────────────────────────────────────────────────────────────────────────────
# 辅助分析
# ────────────────────────────────────────────────────────────────────────────────
def analyze_growth_and_reinvestment(metrics: list, line_items: list) -> dict[str, any]:
    """
    增长分数 (0-4):
      +2  5年收入复合年增长率 > 8 %
      +1  5年收入复合年增长率 > 3 %
      +1  5年内自由现金流正增长
    再投资效率 (ROIC > WACC) 加1分
    """
    max_score = 4
    if len(metrics) < 2:
        return {"score": 0, "max_score": max_score, "details": "Insufficient history"}

    # 收入复合年增长率 (从最旧到最新)
    revs = [m.revenue for m in reversed(metrics) if hasattr(m, "revenue") and m.revenue]
    if len(revs) >= 2 and revs[0] > 0:
        cagr = (revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1
    else:
        cagr = None

    score, details = 0, []

    if cagr is not None:
        if cagr > 0.08:
            score += 2
            details.append(f"Revenue CAGR {cagr:.1%} (> 8 %)")
        elif cagr > 0.03:
            score += 1
            details.append(f"Revenue CAGR {cagr:.1%} (> 3 %)")
        else:
            details.append(f"Sluggish revenue CAGR {cagr:.1%}")
    else:
        details.append("Revenue data incomplete")

    # 自由现金流增长 (代理: free_cash_flow趋势)
    fcfs = [li.free_cash_flow for li in reversed(line_items) if li.free_cash_flow]
    if len(fcfs) >= 2 and fcfs[-1] > fcfs[0]:
        score += 1
        details.append("Positive FCFF growth")
    else:
        details.append("Flat or declining FCFF")

    # 再投资效率 (ROIC vs. 10 % 门槛)
    latest = metrics[0]
    if latest.return_on_invested_capital and latest.return_on_invested_capital > 0.10:
        score += 1
        details.append(f"ROIC {latest.return_on_invested_capital:.1%} (> 10 %)")

    return {"score": score, "max_score": max_score, "details": "; ".join(details), "metrics": latest.model_dump()}


def analyze_risk_profile(metrics: list, line_items: list) -> dict[str, any]:
    """
    风险分数 (0-3):
      +1  Beta < 1.3
      +1  负债/权益比 < 1
      +1  利息保障倍数 > 3×
    """
    max_score = 3
    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics"}

    latest = metrics[0]
    score, details = 0, []

    # Beta系数
    beta = getattr(latest, "beta", None)
    if beta is not None:
        if beta < 1.3:
            score += 1
            details.append(f"Beta {beta:.2f}")
        else:
            details.append(f"High beta {beta:.2f}")
    else:
        details.append("Beta NA")

    # 负债/权益比
    dte = getattr(latest, "debt_to_equity", None)
    if dte is not None:
        if dte < 1:
            score += 1
            details.append(f"D/E {dte:.1f}")
        else:
            details.append(f"High D/E {dte:.1f}")
    else:
        details.append("D/E NA")

    # 利息保障倍数
    ebit = getattr(latest, "ebit", None)
    interest = getattr(latest, "interest_expense", None)
    if ebit and interest and interest != 0:
        coverage = ebit / abs(interest)
        if coverage > 3:
            score += 1
            details.append(f"Interest coverage × {coverage:.1f}")
        else:
            details.append(f"Weak coverage × {coverage:.1f}")
    else:
        details.append("Interest coverage NA")

    # 计算股权成本以备后用
    cost_of_equity = estimate_cost_of_equity(beta)

    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details),
        "beta": beta,
        "cost_of_equity": cost_of_equity,
    }


def analyze_relative_valuation(metrics: list) -> dict[str, any]:
    """
    简单的市盈率检查与历史中位数对比 (由于缺少行业可比数据，使用代理):
      +1 如果 TTM P/E < 5年市盈率中位数的 70 %
      +0 如果在 70 %-130 % 之间
      ‑1 如果 >130 %
    """
    max_score = 1
    if not metrics or len(metrics) < 5:
        return {"score": 0, "max_score": max_score, "details": "Insufficient P/E history"}

    pes = [m.price_to_earnings_ratio for m in metrics if m.price_to_earnings_ratio]
    if len(pes) < 5:
        return {"score": 0, "max_score": max_score, "details": "P/E data sparse"}

    ttm_pe = pes[0]
    median_pe = sorted(pes)[len(pes) // 2]

    if ttm_pe < 0.7 * median_pe:
        score, desc = 1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (cheap)"
    elif ttm_pe > 1.3 * median_pe:
        score, desc = -1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (expensive)"
    else:
        score, desc = 0, f"P/E inline with history"

    return {"score": score, "max_score": max_score, "details": desc}


# ────────────────────────────────────────────────────────────────────────────────
# 通过FCFF DCF计算内在价值 (Damodaran风格)
# ────────────────────────────────────────────────────────────────────────────────
def calculate_intrinsic_value_dcf(metrics: list, line_items: list, risk_analysis: dict) -> dict[str, any]:
    """
    FCFF DCF模型参数:
      • 基础FCFF = 最近的自由现金流
      • 增长率 = 5年收入复合年增长率 (上限 12 %)
      • 到第10年线性递减至2.5%的永续增长率
      • 折现率 @ 股权成本 (由于数据限制，不区分债务)
    """
    if not metrics or len(metrics) < 2 or not line_items:
        return {"intrinsic_value": None, "details": ["Insufficient data"]}

    latest_m = metrics[0]
    fcff0 = getattr(latest_m, "free_cash_flow", None)
    shares = getattr(line_items[0], "outstanding_shares", None)
    if not fcff0 or not shares:
        return {"intrinsic_value": None, "details": ["Missing FCFF or share count"]}

    # 增长假设
    revs = [m.revenue for m in reversed(metrics) if m.revenue]
    if len(revs) >= 2 and revs[0] > 0:
        base_growth = min((revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1, 0.12)
    else:
        base_growth = 0.04  # 后备值

    terminal_growth = 0.025
    years = 10

    # 折现率
    discount = risk_analysis.get("cost_of_equity") or 0.09

    # 预测FCFF并折现
    pv_sum = 0.0
    g = base_growth
    g_step = (terminal_growth - base_growth) / (years - 1)
    for yr in range(1, years + 1):
        fcff_t = fcff0 * (1 + g)
        pv = fcff_t / (1 + discount) ** yr
        pv_sum += pv
        g += g_step

    # 终值 (永续增长模型)
    tv = (
        fcff0
        * (1 + terminal_growth)
        / (discount - terminal_growth)
        / (1 + discount) ** years
    )

    equity_value = pv_sum + tv
    intrinsic_per_share = equity_value / shares

    return {
        "intrinsic_value": equity_value,
        "intrinsic_per_share": intrinsic_per_share,
        "assumptions": {
            "base_fcff": fcff0,
            "base_growth": base_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount,
            "projection_years": years,
        },
        "details": ["FCFF DCF completed"],
    }


def estimate_cost_of_equity(beta: float | None) -> float:
    """CAPM: r_e = r_f + β × ERP (使用Damodaran的长期平均值)."""
    risk_free = 0.04          # 10年期美国国债代理
    erp = 0.05                # 长期美国股权风险溢价
    beta = beta if beta is not None else 1.0
    return risk_free + beta * erp


# ────────────────────────────────────────────────────────────────────────────────
# LLM生成
# ────────────────────────────────────────────────────────────────────────────────
def generate_damodaran_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> AswathDamodaranSignal:
    """
    要求LLM模仿Damodaran教授的分析风格:
      • 故事 → 数字 → 价值叙述
      • 强调风险、增长和现金流假设
      • 引用资本成本、隐含的安全边际和估值交叉检验
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你（指LLM）是纽约大学斯特恩商学院的金融学教授Aswath Damodaran。
                使用你的估值框架的美国股票发布交易信号。

                请用你一贯清晰、数据驱动的口吻说话：
                  ◦ 从公司“故事”（定性）开始
                  ◦ 将该故事与关键数字驱动因素联系起来：收入增长、利润率、再投资、风险
                  ◦ 最后总结价值：你的FCFF DCF估算、安全边际和相对估值的合理性检查
                  ◦ 强调主要的不确定性及其如何影响价值
                只返回下面指定的JSON。""",
            ),
            (
                "human",
                """股票代码: {ticker}

                分析数据:
                {analysis_data}

                请严格按照此JSON模式回应:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float (0-100),
                  "reasoning": "string"
                }}""",
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def default_signal():
        return AswathDamodaranSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="解析错误；默认为中性",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=AswathDamodaranSignal,
        agent_name=agent_id,
        state=state,
        default_factory=default_signal,
    )
