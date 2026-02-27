# 🏥 免疫治疗患者教育助手
### Immuno-Oncology Patient Education Assistant

> 基于大语言模型的医疗对话智能体，帮助肿瘤免疫治疗患者识别和评估免疫相关不良反应（irAEs），提供基于 NCCN 指南的初步建议。

---

## ✨ 核心功能

- **多轮主动引导问诊** — 智能追问，逐步收集关键症状信息，而非单次问答
- **6 类 irAE 症状识别** — 皮肤毒性、胃肠道毒性、肺毒性、肝毒性、肾毒性、内分泌毒性
- **NCCN 风险分级（G1-G4）** — 基于指南的结构化风险评估
- **口语化意图识别** — 症状映射系统（SymptomMapper）支持自然语言描述
- **紧急症状检测** — 自动识别需要立即就医的危急情况
- **会话超时管理** — 30 分钟超时保护，5 分钟预警

---

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.12+ | 主语言 |
| LangGraph | 状态机驱动的多轮对话工作流 |
| Agent Framework | LLM 封装与会话管理 |
| GPT-4o-mini (GitHub Models) | 推理服务 |
| python-dotenv | 环境变量管理 |

---

## 🏗️ 系统架构

```
用户输入
   ↓
[initial_inquiry]        # 初步问诊节点
   ↓
[symptom_collection]     # 症状收集节点（可中断等待用户）
   ↕
[detail_clarification]   # 细节追问节点
   ↓
[risk_assessment]        # NCCN 风险评估节点
   ↓
[recommendation]         # 建议生成节点
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/hosAgent.git
cd hosAgent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 GitHub Token
```

获取 GitHub Token：[GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)

### 4. 运行

```bash
# 运行 v2.0（推荐）
python hosAgent_v2.py

# 运行 v1.0（单轮问答版）
python hosAgent.py
```

---

## 💬 使用示例

```
助手：您好！我是免疫治疗患者教育助手。请描述您目前的症状。

用户：我最近身上起了一些红点，很痒

助手：我了解了，您出现了皮肤方面的症状。我需要了解更多细节：
     皮疹是什么时候开始的？出现几天了？

用户：大概3天前开始的

助手：皮疹大约占身体面积的多少？（如：手掌大小、半个背部等）

...（多轮问诊后）...

助手：根据您描述的情况，初步评估为 G1 级皮肤毒性（轻度）。
     建议：继续观察，使用保湿霜缓解症状，下次复诊时告知医生...

---
⚠️ 重要免责声明：本助手提供的信息仅供参考，不能替代专业医疗诊断。
```

---

## 📁 项目结构

```
hosAgent/
├── README.md                 # 项目说明
├── DEVELOPMENT_HISTORY.md    # 完整开发历程与 Bug 记录
├── LICENSE                   # MIT License
├── .gitignore
├── .env.example              # 环境变量模板
├── requirements.txt          # 依赖列表
├── hosAgent.py               # v1.0 单轮问答版本
├── hosAgent_v2.py            # v2.1 多轮引导版本（推荐）
└── assets/                   # 截图与演示资料
```

---

## 📦 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|----------|
| v1.0.0 | 2026-02-26 | 单轮问答，基础症状评估 |
| v2.0.0 | 2026-02-27 | LangGraph 状态机，多轮主动引导 |
| v2.0.1 | 2026-02-27 | 修复 LangGraph 中断机制 Bug |
| v2.1.0 | 2026-02-27 | 新增 SymptomMapper、会话超时管理 |

详细开发历程见 [DEVELOPMENT_HISTORY.md](./DEVELOPMENT_HISTORY.md)

---

## ⚠️ 免责声明

本工具仅供教育和参考用途，**不能替代专业医疗诊断和治疗建议**。任何情况下，请以主治医生的判断为准。若症状严重或持续加重，请立即就医。

---

## 📄 License

[MIT](./LICENSE)
