import json
import time
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="要交易的股票数量")
    confidence: int = Field(description="置信度 0-100")
    reasoning: str = Field(description="决策理由")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="股票代码到交易决策的字典")


##### 投资组合管理代理 #####
def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    """为多个股票代码做出最终交易决策并生成订单"""

    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "正在处理分析师信号")

        # 为该投资组合管理器找到相应的风险管理器
        if agent_id.startswith("portfolio_manager_"):
            suffix = agent_id.split('_')[-1]
            risk_manager_id = f"risk_management_agent_{suffix}"
        else:
            risk_manager_id = "risk_management_agent"  # CLI的备用方案

        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0.0)
        current_prices[ticker] = float(risk_data.get("current_price", 0.0))

        # 根据仓位限制和价格计算允许的最大股数
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] // current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # 将分析师信号压缩为 {sig, conf}
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if not agent.startswith("risk_management_agent") and ticker in signals:
                sig = signals[ticker].get("signal")
                conf = signals[ticker].get("confidence")
                if sig is not None and conf is not None:
                    ticker_signals[agent] = {"sig": sig, "conf": conf}
        signals_by_ticker[ticker] = ticker_signals

    state["data"]["current_prices"] = current_prices

    progress.update_status(agent_id, None, "正在生成交易决策")

    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        agent_id=agent_id,
        state=state,
    )
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()},
                             "投资组合经理")

    progress.update_status(agent_id, None, "完成")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def compute_allowed_actions(
        tickers: list[str],
        current_prices: dict[str, float],
        max_shares: dict[str, int],
        portfolio: dict[str, float],
) -> dict[str, dict[str, int]]:
    """确定性地计算每个股票代码允许的行动和最大数量。"""
    allowed = {}
    cash = float(portfolio.get("cash", 0.0))
    positions = portfolio.get("positions", {}) or {}
    margin_requirement = float(portfolio.get("margin_requirement", 0.5))
    margin_used = float(portfolio.get("margin_used", 0.0))
    equity = float(portfolio.get("equity", cash))

    for ticker in tickers:
        price = float(current_prices.get(ticker, 0.0))
        pos = positions.get(
            ticker,
            {"long": 0, "long_cost_basis": 0.0, "short": 0, "short_cost_basis": 0.0},
        )
        long_shares = int(pos.get("long", 0) or 0)
        short_shares = int(pos.get("short", 0) or 0)
        max_qty = int(max_shares.get(ticker, 0) or 0)

        # 从零开始
        actions = {"buy": 0, "sell": 0, "short": 0, "cover": 0, "hold": 0}

        # 多头方面
        if long_shares > 0:
            actions["sell"] = long_shares
        if cash > 0 and price > 0:
            max_buy_cash = int(cash // price)
            max_buy = max(0, min(max_qty, max_buy_cash))
            if max_buy > 0:
                actions["buy"] = max_buy

        # 空头方面
        if short_shares > 0:
            actions["cover"] = short_shares
        if price > 0 and max_qty > 0:
            if margin_requirement <= 0.0:
                # 如果保证金要求为零或未设置，则仅受最大数量限制
                max_short = max_qty
            else:
                available_margin = max(0.0, (equity / margin_requirement) - margin_used)
                max_short_margin = int(available_margin // price)
                max_short = max(0, min(max_qty, max_short_margin))
            if max_short > 0:
                actions["short"] = max_short

        # 持有总是有效的
        actions["hold"] = 0

        # 修剪零容量的动作以减少令牌，保留持有
        pruned = {"hold": 0}
        for k, v in actions.items():
            if k != "hold" and v > 0:
                pruned[k] = v

        allowed[ticker] = pruned

    return allowed


def _compact_signals(signals_by_ticker: dict[str, dict]) -> dict[str, dict]:
    """仅保留 {agent: {sig, conf}} 并删除空代理。"""
    out = {}
    for t, agents in signals_by_ticker.items():
        if not agents:
            out[t] = {}
            continue
        compact = {}
        for agent, payload in agents.items():
            sig = payload.get("sig") or payload.get("signal")
            conf = payload.get("conf") if "conf" in payload else payload.get("confidence")
            if sig is not None and conf is not None:
                compact[agent] = {"sig": sig, "conf": conf}
        out[t] = compact
    return out


def generate_trading_decision(
        tickers: list[str],
        signals_by_ticker: dict[str, dict],
        current_prices: dict[str, float],
        max_shares: dict[str, int],
        portfolio: dict[str, float],
        agent_id: str,
        state: AgentState,
) -> PortfolioManagerOutput:
    """通过确定性约束和最小化提示从LLM获取决策。"""

    # 确定性约束
    allowed_actions_full = compute_allowed_actions(tickers, current_prices, max_shares, portfolio)

    # 预先填充纯持有，以避免将它们完全发送给LLM
    prefilled_decisions: dict[str, PortfolioDecision] = {}
    tickers_for_llm: list[str] = []
    for t in tickers:
        aa = allowed_actions_full.get(t, {"hold": 0})
        # 如果只有 'hold' 键存在，则没有可行的交易
        if set(aa.keys()) == {"hold"}:
            prefilled_decisions[t] = PortfolioDecision(
                action="hold", quantity=0, confidence=100.0, reasoning="没有可用的有效交易"
            )
        else:
            tickers_for_llm.append(t)

    if not tickers_for_llm:
        return PortfolioManagerOutput(decisions=prefilled_decisions)

    # 仅为发送到LLM的股票代码构建紧凑的有效负载
    compact_signals = _compact_signals({t: signals_by_ticker.get(t, {}) for t in tickers_for_llm})
    compact_allowed = {t: allowed_actions_full[t] for t in tickers_for_llm}

    # 最小化提示模板
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一位投资组合经理。\n"
                "每个股票代码的输入：分析师信号和允许的行动及最大数量（已验证）。\n"
                "为每个股票代码选择一个允许的行动和≤最大值的数量。 "
                "保持推理非常简洁（最多100个字符）。不要进行现金或保证金计算。只返回JSON。"
            ),
            (
                "human",
                "信号：\n{signals}\n\n"
                "允许的操作：\n{allowed}\n\n"
                "格式：\n"
                "{{\n"
                '  "decisions": {{\n'
                '    "TICKER": {{"action":"...","quantity":int,"confidence":int,"reasoning":"..."}}\n'
                "  }}\n"
                "}}"
            ),
        ]
    )

    prompt_data = {
        "signals": json.dumps(compact_signals, separators=(",", ":"), ensure_ascii=False),
        "allowed": json.dumps(compact_allowed, separators=(",", ":"), ensure_ascii=False),
    }
    prompt = template.invoke(prompt_data)

    # 如果LLM失败，默认工厂将剩余的股票代码填充为持有
    def create_default_portfolio_output():
        # 从预填充的决策开始
        decisions = dict(prefilled_decisions)
        for t in tickers_for_llm:
            decisions[t] = PortfolioDecision(
                action="hold", quantity=0, confidence=0.0, reasoning="默认决策：持有"
            )
        return PortfolioManagerOutput(decisions=decisions)

    llm_out = call_llm(
        prompt=prompt,
        pydantic_model=PortfolioManagerOutput,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_portfolio_output,
    )

    # 将预填充的持有与LLM结果合并
    merged = dict(prefilled_decisions)
    merged.update(llm_out.decisions)
    return PortfolioManagerOutput(decisions=merged)
