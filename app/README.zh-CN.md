# Web 应用程序
AI 对冲基金应用是一个包含前端和后端组件的完整系统，让你可以在自己的电脑上通过网页界面运行一个由 AI 驱动的对冲基金交易系统。

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />

## 概览

AI 对冲基金包括：

- **后端**：一个 FastAPI 应用程序，提供 REST API 来运行对冲基金交易系统和回测器。
- **前端**：一个 React/Vite 应用程序，提供用户友好的界面来可视化和控制对冲基金的操作。

## 目录

- [🚀 快速入门 (非技术用户)](#-快速入门-非技术用户)
  - [选项 1: 使用单行 Shell 脚本 (推荐)](#选项-1-使用单行-shell-脚本-推荐)
  - [选项 2: 使用 npm (备选)](#选项-2-使用-npm-备选)
- [🛠️ 手动设置 (开发者)](#️-手动设置-开发者)
  - [先决条件](#先决条件)
  - [安装](#安装)
  - [运行应用程序](#运行应用程序)
- [详细文档](#详细文档)
- [免责声明](#免责声明)
- [问题排查](#问题排查)

## 🚀 快速入门 (非技术用户)

**一键设置并运行命令：**

### 选项 1: 使用单行 Shell 脚本 (推荐)

#### Mac/Linux:
```bash
./run.sh
```

如果遇到 "permission denied" (权限被拒绝) 错误，请先运行：
```bash
chmod +x run.sh && ./run.sh
```

或者，你也可以运行：
```bash
bash run.sh
```

#### Windows:
```cmd
run.bat
```

### 选项 2: 使用 npm (备选)
```bash
cd app && npm install && npm run setup
```

**就这样！** 这些脚本将会：
1. 检查所需的依赖项 (Node.js, Python, Poetry)
2. 自动安装所有依赖项
3. 启动前端和后端服务
4. **自动在你的网络浏览器中打开**应用程序

**要求:**
- [Node.js](https://nodejs.org/) (包含 npm)
- [Python 3](https://python.org/)
- [Poetry](https://python-poetry.org/)

**运行后，你可以访问:**
- 前端 (Web 界面): http://localhost:5173
- 后端 API: http://localhost:8000
- API 文档: http://localhost:8000/docs

---

## 🛠️ 手动设置 (开发者)

如果你倾向于手动设置每个组件或需要更多控制：

### 先决条件

- 用于前端的 Node.js 和 npm
- 用于后端的 Python 3.8+ 和 Poetry

### 安装

1. 克隆仓库:
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

2. 设置你的环境变量:
```bash
# 在根目录为你的 API 密钥创建 .env 文件
cp .env.example .env
```

3. 编辑 .env 文件以添加你的 API 密钥:
```bash
# 用于运行由 openai 托管的 LLM (gpt-4o, gpt-4o-mini, 等)
OPENAI_API_KEY=your-openai-api-key

# 用于运行由 groq 托管的 LLM (deepseek, llama3, 等)
GROQ_API_KEY=your-groq-api-key

# 用于获取金融数据以驱动对冲基金
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

4. 安装 Poetry (如果尚未安装):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

5. 安装根项目依赖:
```bash
# 从根目录
poetry install
```

6. 安装后端应用依赖:
```bash
# 导航到后端目录
cd app/backend
pip install -r requirements.txt  # 如果有 requirements.txt 文件
# 或
poetry install  # 如果后端目录中有 pyproject.toml
```

7. 安装前端应用依赖:
```bash
cd app/frontend
npm install  # 或 pnpm install 或 yarn install
```

### 运行应用程序

1. 启动后端服务器:
```bash
# 在一个终端中，从后端目录
cd app/backend
poetry run uvicorn main:app --reload
```

2. 启动前端应用程序:
```bash
# 在另一个终端中，从前端目录
cd app/frontend
npm run dev
```

现在你可以访问:
- 前端应用程序: http://localhost:5173
- 后端 API: http://localhost:8000
- API 文档: http://localhost:8000/docs

## 详细文档

更多详细信息:
- [后端文档](./backend/README.md)
- [前端文档](./frontend/README.md)

## 免责声明

本项目仅用于**教育和研究目的**。

- 不用于真实交易或投资
- 不提供任何保证
- 创作者对财务损失不承担任何责任
- 投资决策请咨询财务顾问

使用本软件即表示您同意仅将其用于学习目的。

## 问题排查

### 常见问题

#### "Command not found: uvicorn" 错误
如果在运行设置脚本时看到此错误：

```bash
[ERROR] Backend failed to start. Check the logs:
Command not found: uvicorn
```

**解决方案:**
1. **清理 Poetry 环境:**
   ```bash
   cd app/backend
   poetry env remove --all
   poetry install
   ```

2. **或强制重新安装:**
   ```bash
   cd app/backend
   poetry install --sync
   ```

3. **验证安装:**
   ```bash
   cd app/backend
   poetry run python -c "import uvicorn; import fastapi"
   ```

#### Python 版本问题
- **使用 Python 3.11**: Python 3.13+ 可能存在兼容性问题
- **检查你的 Python 版本:** `python --version`
- **如果需要，切换 Python 版本** (使用 pyenv, conda, 等)

#### 环境变量问题
- **确保 .env 文件存在**于项目根目录中
- **从模板复制:** `cp .env.example .env`
- **将你的 API 密钥添加**到 .env 文件中

#### 权限问题 (Mac/Linux)
如果遇到 "permission denied":
```bash
chmod +x run.sh
./run.sh
```

#### 端口已被占用
如果端口 8000 或 5173 已被占用:
- **终止现有进程:** `pkill -f "uvicorn\|vite"`
- **或通过修改脚本使用不同的端口**

### 获取帮助
- 查看 [GitHub Issues](https://github.com/virattt/ai-hedge-fund/issues)
- 在 [Twitter](https://x.com/virattt) 上关注更新