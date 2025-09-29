from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
import math
from src.utils.api_key import get_api_key_from_state


class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def ben_graham_agent(state: AgentState, agent_id: str = "ben_graham_agent"):
    """
    使用本杰明·格雷厄姆的经典价值投资原则分析股票：
    1. 多年盈利稳定。
    2. 稳健的财务实力（低负债，充足的流动性）。
    3. 相对于内在价值的折扣（例如格雷厄姆数或净流动资产价值）。
    4. 足够的安全边际。
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(ticker, ["earnings_per_share", "revenue", "net_income", "book_value_per_share", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "dividends_and_other_cash_distributions", "outstanding_shares"], end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # 执行子分析
        progress.update_status(agent_id, ticker, "Analyzing earnings stability")
        earnings_analysis = analyze_earnings_stability(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing financial strength")
        strength_analysis = analyze_financial_strength(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing Graham valuation")
        valuation_analysis = analyze_valuation_graham(financial_line_items, market_cap)

        # 汇总评分
        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # 三个分析函数的总可能得分

        # 将总分映射到信号
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_possible_score, "earnings_analysis": earnings_analysis, "strength_analysis": strength_analysis, "valuation_analysis": valuation_analysis}

        progress.update_status(agent_id, ticker, "Generating Ben Graham analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        graham_analysis[ticker] = {"signal": graham_output.signal, "confidence": graham_output.confidence, "reasoning": graham_output.reasoning}

        progress.update_status(agent_id, ticker, "Done", analysis=graham_output.reasoning)

    # 将结果包装在链的单个消息中
    message = HumanMessage(content=json.dumps(graham_analysis), name=agent_id)

    # 可选择显示推理过程
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    # 将信号存储在总体状态中
    state["data"]["analyst_signals"][agent_id] = graham_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_earnings_stability(metrics: list, financial_line_items: list) -> dict:
    """
    格雷厄姆希望至少有几年持续盈利（理想情况下为5年以上）。
    我们将检查：
    1. 每股收益为正的年数。
    2. 从第一期到最后一期每股收益的增长情况。
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": score, "details": "Insufficient data for earnings stability analysis"}

    eps_vals = []
    for item in financial_line_items:
        if item.earnings_per_share is not None:
            eps_vals.append(item.earnings_per_share)

    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details)}

    # 1. 持续为正的每股收益
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("EPS was positive in all available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("EPS was positive in most periods.")
    else:
        details.append("EPS was negative in multiple periods.")

    # 2. 每股收益从最早到最近的增长
    if eps_vals[0] > eps_vals[-1]:
        score += 1
        details.append("EPS grew from earliest to latest period.")
    else:
        details.append("EPS did not grow from earliest to latest period.")

    return {"score": score, "details": "; ".join(details)}


def analyze_financial_strength(financial_line_items: list) -> dict:
    """
    格雷厄姆检查流动性（流动比率 >= 2）、可管理的债务，
    以及股息记录（最好有一些股息历史）。
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": score, "details": "No data for financial strength analysis"}

    latest_item = financial_line_items[0]
    total_assets = latest_item.total_assets or 0
    total_liabilities = latest_item.total_liabilities or 0
    current_assets = latest_item.current_assets or 0
    current_liabilities = latest_item.current_liabilities or 0

    # 1. 流动比率
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
    else:
        details.append("Cannot compute current ratio (missing or zero current_liabilities).")

    # 2. 债务与资产
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
    else:
        details.append("Cannot compute debt ratio (missing total_assets).")

    # 3. 股息记录
    div_periods = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        div_paid_years = sum(1 for d in div_periods if d < 0)
        if div_paid_years > 0:
            # e.g. if at least half the periods had dividends
            if div_paid_years >= (len(div_periods) // 2 + 1):
                score += 1
                details.append("Company paid dividends in the majority of the reported years.")
            else:
                details.append("Company has some dividend payments, but not most years.")
        else:
            details.append("Company did not pay dividends in these periods.")
    else:
        details.append("No dividend data available to assess payout consistency.")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation_graham(financial_line_items: list, market_cap: float) -> dict:
    """
    格雷厄姆估值的核心方法：
    1. 净流动资产价值检验：（流动资产 - 总负债）vs. 市值
    2. 格雷厄姆数：sqrt(22.5 * 每股收益 * 每股账面价值)
    3. 将每股价格与格雷厄姆数进行比较 => 安全边际
    """
    if not financial_line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    latest = financial_line_items[0]
    current_assets = latest.current_assets or 0
    total_liabilities = latest.total_liabilities or 0
    book_value_ps = latest.book_value_per_share or 0
    eps = latest.earnings_per_share or 0
    shares_outstanding = latest.outstanding_shares or 0

    details = []
    score = 0

    # 1. 净流动资产价值检验
    #   NCAV = 流动资产 - 总负债
    #   如果 NCAV > 市值 => 历史上是一个强烈的买入信号
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding if shares_outstanding else 0

        details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
        details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
        details.append(f"Price Per Share = {price_per_share:,.2f}")

        if net_current_asset_value > market_cap:
            score += 4  # 非常强的格雷厄姆信号
            details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
        else:
            # 对于部分净流动资产价值折扣
            if net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
    else:
        details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")

    # 2. 格雷厄姆数
    #   格雷厄姆数 = sqrt(22.5 * 每股收益 * 每股账面价值)。
    #   将结果与当前每股价格进行比较
    #   如果格雷厄姆数 >> 价格，则表明估值偏低
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
    else:
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")

    # 3. 相对于格雷厄姆数的安全边际
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 3
                details.append("Price is well below Graham Number (>=50% margin).")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("Some margin of safety relative to Graham Number.")
            else:
                details.append("Price close to or above Graham Number, low margin of safety.")
        else:
            details.append("Current price is zero or invalid; can't compute margin of safety.")
    # else: 已经为缺失的格雷厄姆数附加了详细信息

    return {"score": score, "details": "; ".join(details)}


def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> BenGrahamSignal:
    """
    以本杰明·格雷厄姆的风格生成投资决策：
    - 强调价值、安全边际、净流动资产、保守的资产负债表、稳定的收益。
    - 以JSON结构返回结果：{ signal, confidence, reasoning }。
    """

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个本杰明·格雷厄姆AI代理，使用他的原则进行投资决策：
            1. 坚持通过低于内在价值（例如，使用格雷厄姆数，净流动资产价值）购买来获得安全边际。
            2. 强调公司的财务实力（低杠杆，充足的流动资产）。
            3. 偏爱多年稳定的收益。
            4. 考虑股息记录以增加安全性。
            5. 避免投机性或高增长的假设；专注于经过验证的指标。
            
            在提供你的理由时，要详尽具体，通过以下方式：
            1. 解释对你的决策影响最大的关键估值指标（格雷厄姆数、NCAV、市盈率等）。
            2. 突出具体的财务实力指标（流动比率、债务水平等）。
            3. 引用收益随时间变化的稳定性或不稳定性。
            4. 提供精确数字的量化证据。
            5. 将当前指标与格雷厄姆的具体阈值进行比较（例如，“流动比率为2.5，超过了格雷厄姆的最低要求2.0”）。
            6. 在你的解释中使用本杰明·格雷厄姆保守、分析性的声音和风格。
            
            例如，如果看涨：“该股票的交易价格比其净流动资产价值低35%，提供了充足的安全边际。2.5的流动比率和0.3的债务权益比表明其财务状况强劲...”
            例如，如果看跌：“尽管收益稳定，但目前50美元的价格超过了我们计算出的35美元的格雷厄姆数，没有提供安全边际。此外，仅为1.2的流动比率低于格雷厄姆偏爱的2.0阈值...”
                        
            返回一个理性的建议：看涨、看跌或中性，并附有置信度（0-100）和详尽的推理。
            """,
            ),
            (
                "human",
                """根据以下分析，创建一个格雷厄姆风格的投资信号：

            {ticker}的分析数据:
            {analysis_data}

            严格按照以下格式返回JSON：
            {{
              "signal": "bullish" or "bearish" or "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return call_llm(
        prompt=prompt,
        pydantic_model=BenGrahamSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_ben_graham_signal,
    )
