# AI 对冲基金 - 后端 [开发中] 🚧
该项目目前正在开发中。要跟踪进度，请在此处获取更新：[链接](https://x.com/virattt)。

这是 AI 对冲基金项目的后端服务器。它提供了一个简单的 REST API 来与 AI 对冲基金系统进行交互，让您可以通过 Web 界面运行对冲基金。

## 概述

该后端项目是一个 FastAPI 应用程序，作为 AI 对冲基金系统的服务器端组件。它公开了用于运行对冲基金交易系统和回测器的端点。

该后端旨在与未来的前端应用程序配合使用，该应用程序将允许用户通过浏览器与 AI 对冲基金系统进行交互。

## 安装

### 使用 Poetry

1. 克隆仓库：
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

2. 安装 Poetry (如果尚未安装)：
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. 安装依赖项：
```bash
# 从根目录
poetry install
```

4. 设置您的环境变量：
```bash
# 在根目录为您的 API 密钥创建 .env 文件
cp .env.example .env
```

5. 编辑 .env 文件以添加您的 API 密钥：
```bash
# 用于运行由 openai 托管的 LLM (gpt-4o, gpt-4o-mini 等)
OPENAI_API_KEY=your-openai-api-key

# 用于运行由 groq 托管的 LLM (deepseek, llama3 等)
GROQ_API_KEY=your-groq-api-key

# 用于获取金融数据以支持对冲基金
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

## 运行服务器

要运行开发服务器：

```bash
# 导航到后端目录
cd app/backend

# 使用 uvicorn 启动 FastAPI 服务器
poetry run uvicorn main:app --reload
```

这将启动启用热重载的 FastAPI 服务器。

API 将在以下地址可用：
- API 端点: http://localhost:8000
- API 文档: http://localhost:8000/docs

## API 端点

- `POST /hedge-fund/run`: 使用指定参数运行 AI 对冲基金
- `GET /ping`: 用于测试服务器连接性的简单端点

## 项目结构