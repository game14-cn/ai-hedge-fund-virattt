# AI 对冲基金

这是一个由 AI 驱动的对冲基金概念验证项目。该项目旨在探索使用 AI 做出交易决策的可能性。此项目仅用于“教育”目的，不用于真实交易或投资。

该系统由多个协作的智能体组成：

1. Aswath Damodaran 智能体——估值之父，专注于故事、数字与有纪律的估值
2. Ben Graham 智能体——价值投资教父，只在有安全边际时买入隐藏的宝石
3. Bill Ackman 智能体——激进投资者，采取大胆仓位并推动改变
4. Cathie Wood 智能体——成长投资女王，坚信创新与颠覆的力量
5. Charlie Munger 智能体——巴菲特的合伙人，只买杰出企业的公平价格
6. Michael Burry 智能体——“大空头”逆向者，专注深度价值
7. Mohnish Pabrai 智能体——Dhandho 投资者，低风险追求翻倍机会
8. Peter Lynch 智能体——务实型投资者，寻找日常企业里的“十倍股”
9. Phil Fisher 智能体——严谨的成长型投资者，依靠深度“闲聊调查”研究
10. Rakesh Jhunjhunwala 智能体——印度股神 Big Bull
11. Stanley Druckenmiller 智能体——宏观传奇，寻找具备增长潜力的不对称机会
12. Warren Buffett 智能体——奥马哈先知，追求以合理价格买入杰出企业
13. 估值智能体——计算股票内在价值并生成交易信号
14. 情绪智能体——分析市场情绪并生成交易信号
15. 基本面智能体——分析基本面数据并生成交易信号
16. 技术面智能体——分析技术指标并生成交易信号
17. 风险管理智能体——计算风险指标并设定仓位上限
18. 资产组合管理智能体——做出最终交易决策并生成订单

<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

注意：该系统不会实际进行任何交易。

[![Twitter 关注](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## 免责声明

本项目仅用于“教育与研究”目的。

- 不用于真实交易或投资
- 不提供任何投资建议或保证
- 创作者不对任何财务损失承担责任
- 投资决策请咨询专业的金融顾问
- 过往表现不代表未来结果

使用本软件即表示你同意仅将其用于学习目的。

## 目录
- [如何安装](#如何安装)
- [如何运行](#如何运行)
  - [命令行界面](#命令行界面)
  - [Web 应用程序](#web-应用程序)
- [如何贡献](#如何贡献)
- [功能需求](#功能需求)
- [许可证](#许可证)

## 如何安装

在运行 AI 对冲基金之前，你需要安装并配置好你的 API 密钥。这些步骤同时适用于全栈 Web 应用和命令行界面。

### 1. 克隆仓库

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. 设置 API 密钥

创建一个用于存放 API 密钥的 `.env` 文件：
```bash
# 在根目录为你的 API 密钥创建 .env 文件
cp .env.example .env
```

打开并编辑 `.env` 文件，添加你的 API 密钥：
```bash
# 运行 OpenAI 托管的 LLM（例如 gpt-4o、gpt-4o-mini 等）
OPENAI_API_KEY=your-openai-api-key

# 获取驱动对冲基金的金融数据
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

重要提示：你至少需要设置一个 LLM 的 API 密钥（例如 `OPENAI_API_KEY`、`GROQ_API_KEY`、`ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY`），系统才能正常工作。

金融数据：AAPL、GOOGL、MSFT、NVDA 和 TSLA 的数据是免费的，不需要 API 密钥。若使用其他股票代码，你需要在 .env 文件中设置 `FINANCIAL_DATASETS_API_KEY`。

## 如何运行

### 命令行界面

你可以直接通过终端运行 AI 对冲基金。这种方式提供更细粒度的控制，适合自动化、脚本化和集成使用。

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

#### 快速开始

1. 安装 Poetry（如果尚未安装）：
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 安装依赖：
```bash
poetry install
```

#### 运行 AI 对冲基金
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

你也可以指定 `--ollama` 参数，使用本地 LLM 运行 AI 对冲基金。

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

你可以选择指定开始和结束日期，在特定时间范围内做出决策。

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

#### 运行回测器
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**示例输出：**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />


注意：`--ollama`、`--start-date` 和 `--end-date` 参数同样适用于回测器！

### Web 应用程序

运行 AI 对冲基金的新方式是通过我们的 Web 应用程序，它提供了一个用户友好的界面。对于不熟悉命令行的用户，我们推荐使用此方式。

请在此处查看如何安装和运行 Web 应用程序的详细说明：[这里](https://github.com/virattt/ai-hedge-fund/tree/main/app)。

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />


## 如何贡献

1. Fork 仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

重要提示：请保持你的 Pull Request 小而集中，这将更容易审核与合并。

## 功能需求

如果你有功能需求，请在此处提交 [issue](https://github.com/virattt/ai-hedge-fund/issues)，并为其添加 `enhancement` 标签。

## 许可证

本项目基于 MIT 许可证开源——详见 LICENSE 文件。