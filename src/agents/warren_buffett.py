from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
from typing_extensions import Literal
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="置信度 0-100")
    reasoning: str = Field(description="决策理由")


def warren_buffett_agent(state: AgentState, agent_id: str = "warren_buffett_agent"):
    """使用巴菲特的原则和LLM推理来分析股票。"""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # 为LLM推理收集所有分析
    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "获取财务指标")
        # 获取所需数据 - 请求更多周期以进行更好的趋势分析
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "收集财务项目")
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
                "shareholders_equity",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
                "gross_profit",
                "revenue",
                "free_cash_flow",
            ],
            end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "获取市值")
        # 获取当前市值
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "分析基本面")
        # 分析基本面
        fundamental_analysis = analyze_fundamentals(metrics)

        progress.update_status(agent_id, ticker, "分析一致性")
        consistency_analysis = analyze_consistency(financial_line_items)

        progress.update_status(agent_id, ticker, "分析竞争护城河")
        moat_analysis = analyze_moat(metrics)

        progress.update_status(agent_id, ticker, "分析定价权")
        pricing_power_analysis = analyze_pricing_power(financial_line_items)

        progress.update_status(agent_id, ticker, "分析账面价值增长")
        book_value_analysis = analyze_book_value_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "分析管理质量")
        mgmt_analysis = analyze_management_quality(financial_line_items)

        progress.update_status(agent_id, ticker, "计算内在价值")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

        # 计算总分，不包括能力圈（LLM将处理）
        total_score = (
                fundamental_analysis["score"] +
                consistency_analysis["score"] +
                moat_analysis["score"] +
                mgmt_analysis["score"] +
                pricing_power_analysis["score"] +
                book_value_analysis["score"]
        )

        # 更新最高可能得分计算
        max_possible_score = (
                10 +  # 基本面分析 (ROE、债务、利润率、流动比率)
                moat_analysis["max_score"] +
                mgmt_analysis["max_score"] +
                5 +  # 定价权 (0-5)
                5  # 账面价值增长 (0-5)
        )

        # 如果我们同时拥有内在价值和当前价格，则增加安全边际分析
        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

        # 合并所有分析结果以供LLM评估
        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "pricing_power_analysis": pricing_power_analysis,
            "book_value_analysis": book_value_analysis,
            "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        progress.update_status(agent_id, ticker, "生成巴菲特分析")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        # 以与其他代理一致的格式存储分析
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "完成", analysis=buffett_output.reasoning)

    # 创建消息
    message = HumanMessage(content=json.dumps(buffett_analysis), name=agent_id)

    # 如果请求，则显示推理
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, agent_id)

    # 将信号添加到analyst_signals列表中
    state["data"]["analyst_signals"][agent_id] = buffett_analysis

    progress.update_status(agent_id, None, "完成")

    return {"messages": [message], "data": state["data"]}


def analyze_fundamentals(metrics: list) -> dict[str, any]:
    """根据巴菲特的标准分析公司基本面。"""
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}

    latest_metrics = metrics[0]

    score = 0
    reasoning = []

    # 检查ROE (净资产收益率)
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:  # 15% ROE 阈值
        score += 2
        reasoning.append(f"强劲的净资产收益率 {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"疲弱的净资产收益率 {latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("净资产收益率数据不可用")

    # 检查债务股本比
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("保守的债务水平")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"高债务股本比 {latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("债务股本比数据不可用")

    # 检查营业利润率
    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("强劲的营业利润率")
    elif latest_metrics.operating_margin:
        reasoning.append(f"疲弱的营业利润率 {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("营业利润率数据不可用")

    # 检查流动比率
    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("良好的流动性状况")
    elif latest_metrics.current_ratio:
        reasoning.append(f"流动性疲弱，流动比率为 {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("流动比率数据不可用")

    return {"score": score, "details": "; ".join(reasoning), "metrics": latest_metrics.model_dump()}


def analyze_consistency(financial_line_items: list) -> dict[str, any]:
    """分析收益的一致性和增长。"""
    if len(financial_line_items) < 4:  # 趋势分析至少需要4个周期
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    # 检查盈利增长趋势
    earnings_values = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings_values) >= 4:
        # 简单检查：每个周期的收益是否都比下一个周期大？
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

        if earnings_growth:
            score += 3
            reasoning.append("过去几个时期的收益持续增长")
        else:
            reasoning.append("收益增长模式不一致")

        # 计算从最旧到最新的总增长率
        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"在过去 {len(earnings_values)} 个时期内，总收益增长率为 {growth_rate:.1%}")
    else:
        reasoning.append("用于趋势分析的收益数据不足")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }


def analyze_moat(metrics: list) -> dict[str, any]:
    """
    评估公司是否可能拥有持久的竞争优势（护城河）。
    增强版，包括巴菲特实际寻找的多个护城河指标：
    1. 持续的高资本回报率
    2. 定价权（稳定/增长的利润率）
    3. 规模优势（随规模改善的指标）
    4. 品牌实力（从利润率和一致性推断）
    5. 转换成本（从客户保留率推断）
    """
    if not metrics or len(metrics) < 5:  # 需要更多数据才能进行适当的护城河分析
        return {"score": 0, "max_score": 5, "details": "Insufficient data for comprehensive moat analysis"}

    reasoning = []
    moat_score = 0
    max_score = 5

    # 1. 资本回报率一致性（巴菲特最喜欢的护城河指标）
    historical_roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    historical_roics = [m.return_on_invested_capital for m in metrics if
                        hasattr(m, 'return_on_invested_capital') and m.return_on_invested_capital is not None]

    if len(historical_roes) >= 5:
        # 检查持续高ROE（大部分时期 >15%）
        high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
        roe_consistency = high_roe_periods / len(historical_roes)

        if roe_consistency >= 0.8:  # 80%以上的时期的ROE > 15%
            moat_score += 2
            avg_roe = sum(historical_roes) / len(historical_roes)
            reasoning.append(
                f"优秀的ROE一致性：{high_roe_periods}/{len(historical_roes)} 个时期 >15% (平均: {avg_roe:.1%}) - 表明持久的竞争优势")
        elif roe_consistency >= 0.6:
            moat_score += 1
            reasoning.append(f"良好的ROE表现：{high_roe_periods}/{len(historical_roes)} 个时期 >15%")
        else:
            reasoning.append(f"不一致的ROE：只有 {high_roe_periods}/{len(historical_roes)} 个时期 >15%")
    else:
        reasoning.append("护城河分析的ROE历史数据不足")

    # 2. 营业利润率稳定性（定价权指标）
    historical_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(historical_margins) >= 5:
        # 检查稳定或改善的利润率（定价权的标志）
        avg_margin = sum(historical_margins) / len(historical_margins)
        recent_margins = historical_margins[:3]  # 最近3个时期
        older_margins = historical_margins[-3:]  # 最早3个时期

        recent_avg = sum(recent_margins) / len(recent_margins)
        older_avg = sum(older_margins) / len(older_margins)

        if avg_margin > 0.2 and recent_avg >= older_avg:  # 20%以上的利润率且稳定/改善
            moat_score += 1
            reasoning.append(f"强大而稳定的营业利润率 (平均: {avg_margin:.1%}) 表明定价权护城河")
        elif avg_margin > 0.15:  # 至少不错的利润率
            reasoning.append(f"不错的营业利润率 (平均: {avg_margin:.1%}) 表明具有一定的竞争优势")
        else:
            reasoning.append(f"低营业利润率 (平均: {avg_margin:.1%}) 表明定价权有限")

    # 3. 资产效率和规模优势
    if len(metrics) >= 5:
        # 检查资产周转率趋势（收入效率）
        asset_turnovers = []
        for m in metrics:
            if hasattr(m, 'asset_turnover') and m.asset_turnover is not None:
                asset_turnovers.append(m.asset_turnover)

        if len(asset_turnovers) >= 3:
            if any(turnover > 1.0 for turnover in asset_turnovers):  # 高效的资产使用
                moat_score += 1
                reasoning.append("高效的资产利用率表明运营护城河")

    # 4. 竞争地位强度（从趋势稳定性推断）
    if len(historical_roes) >= 5 and len(historical_margins) >= 5:
        # 计算变异系数（稳定性度量）
        roe_avg = sum(historical_roes) / len(historical_roes)
        roe_variance = sum((roe - roe_avg) ** 2 for roe in historical_roes) / len(historical_roes)
        roe_stability = 1 - (roe_variance ** 0.5) / roe_avg if roe_avg > 0 else 0

        margin_avg = sum(historical_margins) / len(historical_margins)
        margin_variance = sum((margin - margin_avg) ** 2 for margin in historical_margins) / len(historical_margins)
        margin_stability = 1 - (margin_variance ** 0.5) / margin_avg if margin_avg > 0 else 0

        overall_stability = (roe_stability + margin_stability) / 2

        if overall_stability > 0.7:  # 高稳定性表明强大的竞争地位
            moat_score += 1
            reasoning.append(f"高绩效稳定性 ({overall_stability:.1%}) 表明强大的竞争护城河")

    # 将分数限制在最高分
    moat_score = min(moat_score, max_score)

    return {
        "score": moat_score,
        "max_score": max_score,
        "details": "; ".join(reasoning) if reasoning else "Limited moat analysis available",
    }


def analyze_management_quality(financial_line_items: list) -> dict[str, any]:
    """
    检查股份稀释或持续回购，以及一些股息记录。
    一种简化的方法：
      - 如果有净股份回购或稳定的股份数量，这表明管理层
        可能是对股东友好的。
      - 如果有大量新发行，这可能是一个负面信号（稀释）。
    """
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0

    latest = financial_line_items[0]
    if hasattr(latest,
               "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares < 0:
        # 负数表示公司花钱回购
        mgmt_score += 1
        reasoning.append("公司一直在回购股票（对股东友好）")

    if hasattr(latest,
               "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares > 0:
        # 正数发行意味着新股 => 可能稀释
        reasoning.append("最近发行了普通股（潜在稀释）")
    else:
        reasoning.append("未检测到重大的新股发行")

    # 检查是否有任何股息
    if hasattr(latest,
               "dividends_and_other_cash_distributions") and latest.dividends_and_other_cash_distributions and latest.dividends_and_other_cash_distributions < 0:
        mgmt_score += 1
        reasoning.append("公司有支付股息的记录")
    else:
        reasoning.append("没有或很少支付股息")

    return {
        "score": mgmt_score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }


def calculate_owner_earnings(financial_line_items: list) -> dict[str, any]:
    """
    计算所有者收益（巴菲特偏爱的真实收益能力衡量标准）。
    增强方法：净收入 + 折旧/摊销 - 维护性资本支出 - 营运资本变动
    使用多期分析以更好地估算维护性资本支出。
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]
    details = []

    # 核心组成部分
    net_income = latest.net_income
    depreciation = latest.depreciation_and_amortization
    capex = latest.capital_expenditure

    if not all([net_income is not None, depreciation is not None, capex is not None]):
        missing = []
        if net_income is None: missing.append("net income")
        if depreciation is None: missing.append("depreciation")
        if capex is None: missing.append("capital expenditure")
        return {"owner_earnings": None, "details": [f"Missing components: {', '.join(missing)}"]}

    # 使用历史分析增强维护性资本支出估算
    maintenance_capex = estimate_maintenance_capex(financial_line_items)

    # 营运资本变动分析（如果数据可用）
    working_capital_change = 0
    if len(financial_line_items) >= 2:
        try:
            current_assets_current = getattr(latest, 'current_assets', None)
            current_liab_current = getattr(latest, 'current_liabilities', None)

            previous = financial_line_items[1]
            current_assets_previous = getattr(previous, 'current_assets', None)
            current_liab_previous = getattr(previous, 'current_liabilities', None)

            if all([current_assets_current, current_liab_current, current_assets_previous, current_liab_previous]):
                wc_current = current_assets_current - current_liab_current
                wc_previous = current_assets_previous - current_liab_previous
                working_capital_change = wc_current - wc_previous
                details.append(f"Working capital change: ${working_capital_change:,.0f}")
        except:
            pass  # 如果数据不可用，则跳过营运资本调整

    # 计算所有者收益
    owner_earnings = net_income + depreciation - maintenance_capex - working_capital_change

    # 合理性检查
    if owner_earnings < net_income * 0.3:  # 所有者收益通常不应低于净收入的30%
        details.append("Warning: Owner earnings significantly below net income - high capex intensity")

    if maintenance_capex > depreciation * 2:  # 维护性资本支出通常不应超过折旧的2倍
        details.append("Warning: Estimated maintenance capex seems high relative to depreciation")

    details.extend([
        f"Net income: ${net_income:,.0f}",
        f"Depreciation: ${depreciation:,.0f}",
        f"Estimated maintenance capex: ${maintenance_capex:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}"
    ])

    return {
        "owner_earnings": owner_earnings,
        "components": {
            "net_income": net_income,
            "depreciation": depreciation,
            "maintenance_capex": maintenance_capex,
            "working_capital_change": working_capital_change,
            "total_capex": abs(capex) if capex else 0
        },
        "details": details,
    }


def estimate_maintenance_capex(financial_line_items: list) -> float:
    """
    使用多种方法估算维护性资本支出。
    巴菲特认为这对于理解真实的所有者收益至关重要。
    """
    if not financial_line_items:
        return 0

    # 方法1：占收入的历史平均百分比
    capex_ratios = []
    depreciation_values = []

    for item in financial_line_items[:5]:  # 最近5个时期
        if hasattr(item, 'capital_expenditure') and hasattr(item, 'revenue'):
            if item.capital_expenditure and item.revenue and item.revenue > 0:
                capex_ratio = abs(item.capital_expenditure) / item.revenue
                capex_ratios.append(capex_ratio)

        if hasattr(item, 'depreciation_and_amortization') and item.depreciation_and_amortization:
            depreciation_values.append(item.depreciation_and_amortization)

    # 方法2：折旧的百分比（通常为维护的80-120%）
    latest_depreciation = financial_line_items[0].depreciation_and_amortization if financial_line_items[
        0].depreciation_and_amortization else 0

    # 方法3：行业特定的启发式方法
    latest_capex = abs(financial_line_items[0].capital_expenditure) if financial_line_items[
        0].capital_expenditure else 0

    # 保守估计：使用以下各项中的较高者：
    # 1. 总资本支出的85%（假设15%是增长性资本支出）
    # 2. 折旧的100%（更换磨损资产）
    # 3. 如果稳定，则为历史平均值

    method_1 = latest_capex * 0.85  # 总资本支出的85%
    method_2 = latest_depreciation  # 折旧的100%

    # 如果我们有历史数据，则使用平均资本支出比率
    if len(capex_ratios) >= 3:
        avg_capex_ratio = sum(capex_ratios) / len(capex_ratios)
        latest_revenue = financial_line_items[0].revenue if hasattr(financial_line_items[0], 'revenue') and \
                                                            financial_line_items[0].revenue else 0
        method_3 = avg_capex_ratio * latest_revenue if latest_revenue else 0

        # 为保守起见，使用三种方法的中位数
        estimates = sorted([method_1, method_2, method_3])
        return estimates[1]  # 中位数
    else:
        # 使用方法1和方法2中较高者
        return max(method_1, method_2)


def calculate_intrinsic_value(financial_line_items: list) -> dict[str, any]:
    """
    使用增强的DCF和所有者收益计算内在价值。
    使用更复杂的假设和像巴菲特一样的保守方法。
    """
    if not financial_line_items or len(financial_line_items) < 3:
        return {"intrinsic_value": None, "details": ["Insufficient data for reliable valuation"]}

    # 使用更好的方法计算所有者收益
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]
    latest_financial_line_items = financial_line_items[0]
    shares_outstanding = latest_financial_line_items.outstanding_shares

    if not shares_outstanding or shares_outstanding <= 0:
        return {"intrinsic_value": None, "details": ["Missing or invalid shares outstanding data"]}

    # 具有更现实假设的增强型DCF
    details = []

    # 根据历史表现估算增长率（更保守）
    historical_earnings = []
    for item in financial_line_items[:5]:  # 最近5年
        if hasattr(item, 'net_income') and item.net_income:
            historical_earnings.append(item.net_income)

    # 计算历史增长率
    if len(historical_earnings) >= 3:
        oldest_earnings = historical_earnings[-1]
        latest_earnings = historical_earnings[0]
        years = len(historical_earnings) - 1

        if oldest_earnings > 0:
            historical_growth = ((latest_earnings / oldest_earnings) ** (1 / years)) - 1
            # 保守调整 - 限制增长并进行折减
            historical_growth = max(-0.05, min(historical_growth, 0.15))  # 限制在-5%和15%之间
            conservative_growth = historical_growth * 0.7  # 为保守起见，应用30%的折减
        else:
            conservative_growth = 0.03  # 如果基数为负，则默认为3%
    else:
        conservative_growth = 0.03  # 默认保守增长

    # 巴菲特的保守假设
    stage1_growth = min(conservative_growth, 0.08)  # 阶段1：上限为8%
    stage2_growth = min(conservative_growth * 0.5, 0.04)  # 阶段2：阶段1的一半，上限为4%
    terminal_growth = 0.025  # 长期GDP增长率

    # 基于业务质量的风险调整贴现率
    base_discount_rate = 0.09  # 基准9%

    # 根据分析分数进行调整（如果在调用上下文中可用）
    # 目前，使用保守的10%
    discount_rate = 0.10

    # 三阶段DCF模型
    stage1_years = 5  # 高增长阶段
    stage2_years = 5  # 过渡阶段

    present_value = 0
    details.append(
        f"使用三阶段DCF：阶段1 ({stage1_growth:.1%}, {stage1_years}年), 阶段2 ({stage2_growth:.1%}, {stage2_years}年), 永续期 ({terminal_growth:.1%})")

    # 阶段1：较高增长
    stage1_pv = 0
    for year in range(1, stage1_years + 1):
        future_earnings = owner_earnings * (1 + stage1_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** year
        stage1_pv += pv

    # 阶段2：过渡增长
    stage2_pv = 0
    stage1_final_earnings = owner_earnings * (1 + stage1_growth) ** stage1_years
    for year in range(1, stage2_years + 1):
        future_earnings = stage1_final_earnings * (1 + stage2_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** (stage1_years + year)
        stage2_pv += pv

    # 使用戈登增长模型计算终值
    final_earnings = stage1_final_earnings * (1 + stage2_growth) ** stage2_years
    terminal_earnings = final_earnings * (1 + terminal_growth)
    terminal_value = terminal_earnings / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / (1 + discount_rate) ** (stage1_years + stage2_years)

    # 总内在价值
    intrinsic_value = stage1_pv + stage2_pv + terminal_pv

    # 应用额外的安全边际（巴菲特的保守主义）
    conservative_intrinsic_value = intrinsic_value * 0.85  # 额外的15%折减

    details.extend([
        f"Stage 1 PV: ${stage1_pv:,.0f}",
        f"Stage 2 PV: ${stage2_pv:,.0f}",
        f"Terminal PV: ${terminal_pv:,.0f}",
        f"Total IV: ${intrinsic_value:,.0f}",
        f"Conservative IV (15% haircut): ${conservative_intrinsic_value:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}",
        f"Discount rate: {discount_rate:.1%}"
    ])

    return {
        "intrinsic_value": conservative_intrinsic_value,
        "raw_intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "stage1_growth": stage1_growth,
            "stage2_growth": stage2_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount_rate,
            "stage1_years": stage1_years,
            "stage2_years": stage2_years,
            "historical_growth": conservative_growth if 'conservative_growth' in locals() else None,
        },
        "details": details,
    }


def analyze_book_value_growth(financial_line_items: list) -> dict[str, any]:
    """分析每股账面价值增长 - 巴菲特的一个关键指标。"""
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data for book value analysis"}

    # 提取每股账面价值
    book_values = [
        item.shareholders_equity / item.outstanding_shares
        for item in financial_line_items
        if hasattr(item, 'shareholders_equity') and hasattr(item, 'outstanding_shares')
        and item.shareholders_equity and item.outstanding_shares
    ]

    if len(book_values) < 3:
        return {"score": 0, "details": "Insufficient book value data for growth analysis"}

    score = 0
    reasoning = []

    # 分析增长一致性
    growth_periods = sum(1 for i in range(len(book_values) - 1) if book_values[i] > book_values[i + 1])
    growth_rate = growth_periods / (len(book_values) - 1)

    # 根据一致性评分
    if growth_rate >= 0.8:
        score += 3
        reasoning.append("持续的每股账面价值增长（巴菲特最喜欢的指标）")
    elif growth_rate >= 0.6:
        score += 2
        reasoning.append("良好的每股账面价值增长模式")
    elif growth_rate >= 0.4:
        score += 1
        reasoning.append("中等的每股账面价值增长")
    else:
        reasoning.append("不一致的每股账面价值增长")

    # 计算并评分CAGR
    cagr_score, cagr_reason = _calculate_book_value_cagr(book_values)
    score += cagr_score
    reasoning.append(cagr_reason)

    return {"score": score, "details": "; ".join(reasoning)}


def _calculate_book_value_cagr(book_values: list) -> tuple[int, str]:
    """辅助函数，用于安全地计算账面价值的复合年增长率（CAGR）并返回分数和理由。"""
    if len(book_values) < 2:
        return 0, "计算CAGR数据不足"

    oldest_bv, latest_bv = book_values[-1], book_values[0]
    years = len(book_values) - 1

    # 处理不同情况
    if oldest_bv > 0 and latest_bv > 0:
        cagr = ((latest_bv / oldest_bv) ** (1 / years)) - 1
        if cagr > 0.15:
            return 2, f"优秀的账面价值CAGR: {cagr:.1%}"
        elif cagr > 0.1:
            return 1, f"良好的账面价值CAGR: {cagr:.1%}"
        else:
            return 0, f"账面价值CAGR: {cagr:.1%}"
    elif oldest_bv < 0 < latest_bv:
        return 3, "优秀：公司账面价值从负转正"
    elif oldest_bv > 0 > latest_bv:
        return 0, "警告：公司账面价值从正转负"
    else:
        return 0, "由于存在负值，无法计算有意义的账面价值CAGR"


def analyze_pricing_power(financial_line_items: list, metrics: list) -> dict[str, any]:
    """
    分析定价能力 - 巴菲特衡量护城河的关键指标。
    考察在不流失客户的情况下提高价格的能力（通货膨胀期间的利润率扩张）。
    """
    if not financial_line_items or not metrics:
        return {"score": 0, "details": "Insufficient data for pricing power analysis"}

    score = 0
    reasoning = []

    # 检查毛利率趋势（维持/扩大毛利率的能力）
    gross_margins = []
    for item in financial_line_items:
        if hasattr(item, 'gross_margin') and item.gross_margin is not None:
            gross_margins.append(item.gross_margin)

    if len(gross_margins) >= 3:
        # 检查利润率的稳定性/改善情况
        recent_avg = sum(gross_margins[:2]) / 2 if len(gross_margins) >= 2 else gross_margins[0]
        older_avg = sum(gross_margins[-2:]) / 2 if len(gross_margins) >= 2 else gross_margins[-1]

        if recent_avg > older_avg + 0.02:  # 2%以上的改善
            score += 3
            reasoning.append("Expanding gross margins indicate strong pricing power")
        elif recent_avg > older_avg:
            score += 2
            reasoning.append("Improving gross margins suggest good pricing power")
        elif abs(recent_avg - older_avg) < 0.01:  # 稳定在1%以内
            score += 1
            reasoning.append("Stable gross margins during economic uncertainty")
        else:
            reasoning.append("Declining gross margins may indicate pricing pressure")

    # 检查公司是否能够持续保持高利润率
    if gross_margins:
        avg_margin = sum(gross_margins) / len(gross_margins)
        if avg_margin > 0.5:  # 50%以上的毛利率
            score += 2
            reasoning.append(f"Consistently high gross margins ({avg_margin:.1%}) indicate strong pricing power")
        elif avg_margin > 0.3:  # 30%以上的毛利率
            score += 1
            reasoning.append(f"Good gross margins ({avg_margin:.1%}) suggest decent pricing power")

    return {
        "score": score,
        "details": "; ".join(reasoning) if reasoning else "Limited pricing power analysis available"
    }


def generate_buffett_output(
        ticker: str,
        analysis_data: dict[str, any],
        state: AgentState,
        agent_id: str = "warren_buffett_agent",
) -> WarrenBuffettSignal:
    """使用紧凑的提示从LLM获取投资决策。"""

    # --- 在此构建紧凑的事实 ---
    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "fundamentals": analysis_data.get("fundamental_analysis", {}).get("details"),
        "consistency": analysis_data.get("consistency_analysis", {}).get("details"),
        "moat": analysis_data.get("moat_analysis", {}).get("details"),
        "pricing_power": analysis_data.get("pricing_power_analysis", {}).get("details"),
        "book_value": analysis_data.get("book_value_analysis", {}).get("details"),
        "management": analysis_data.get("management_analysis", {}).get("details"),
        "intrinsic_value": analysis_data.get("intrinsic_value_analysis", {}).get("intrinsic_value"),
        "market_cap": analysis_data.get("market_cap"),
        "margin_of_safety": analysis_data.get("margin_of_safety"),
    }

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是沃伦·巴菲特。仅使用提供的事实来决定看涨、看跌或中性。\n"
                "\n"
                "决策清单：\n"
                "- 能力圈\n"
                "- 竞争护城河\n"
                "- 管理质量\n"
                "- 财务实力\n"
                "- 估值与内在价值\n"
                "- 长期前景\n"
                "\n"
                "信号规则：\n"
                "- 看涨：强大的业务 AND 安全边际 > 0。\n"
                "- 看跌：业务不佳 OR 明显高估。\n"
                "- 中性：业务良好但安全边际 <= 0，或证据混杂。\n"
                "\n"
                "置信度等级：\n"
                "- 90-100%：我圈子里的优秀企业，交易价格诱人\n"
                "- 70-89%：具有良好护城河的优秀企业，估值合理\n"
                "- 50-69%：信号混杂，需要更多信息或更好的价格\n"
                "- 30-49%：超出我的专业范围或基本面令人担忧\n"
                "- 10-29%：业务不佳或被严重高估\n"
                "\n"
                "将理由保持在120个字符以内。不要捏造数据。只返回JSON。"
            ),
            (
                "human",
                "股票代码: {ticker}\n"
                "事实:\n{facts}\n\n"
                "请严格返回:\n"
                "{{\n"
                '  "signal": "bullish" | "bearish" | "neutral",\n'
                '  "confidence": int,\n'
                '  "reasoning": "简短的理由"\n'
                "}}"
            ),
        ]
    )

    prompt = template.invoke({
        "facts": json.dumps(facts, separators=( ",", ":"), ensure_ascii=False),
        "ticker": ticker,
    })

    # 默认回退使用整数置信度以匹配模式并避免解析重试
    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=50, reasoning="数据不足")

    return call_llm(
        prompt=prompt,
        pydantic_model=WarrenBuffettSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_warren_buffett_signal,
    )
