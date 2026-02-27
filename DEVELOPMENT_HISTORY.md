# hosAgent 项目开发历程文档

> **项目名称**: 免疫治疗患者教育助手 (Immuno-Oncology Patient Education Assistant)  
> **文档版本**: 1.0.0  
> **最后更新**: 2026-02-27  
> **维护者**: AI Agents for Beginners 项目组

---

## 目录

1. [项目概述](#1-项目概述)
2. [版本迭代历史](#2-版本迭代历史)
3. [Bug修复记录与分析报告](#3-bug修复记录与分析报告)
4. [功能更新日志](#4-功能更新日志)
5. [架构设计与技术决策](#5-架构设计与技术决策)
6. [技术债务与优化措施](#6-技术债务与优化措施)
7. [重要决策记录](#7-重要决策记录)
8. [兼容性处理方案](#8-兼容性处理方案)
9. [附录](#9-附录)

---

## 1. 项目概述

### 1.1 项目背景

免疫治疗（Immuno-Oncology, IO）是肿瘤治疗的重要手段，但其独特的免疫相关不良反应（immune-related Adverse Events, irAEs）常常让患者困惑。本项目旨在开发一个智能患者教育助手，帮助患者：

- 识别和评估免疫治疗相关症状
- 理解症状的严重程度
- 获得基于 NCCN 指南的初步建议
- 判断何时需要就医

### 1.2 核心目标

| 目标 | 描述 | 优先级 |
|------|------|--------|
| 安全性 | 保守建议策略，始终倾向于建议就医 | P0 |
| 准确性 | 基于 NCCN 指南进行风险评估 | P0 |
| 可用性 | 支持自然语言交互，降低使用门槛 | P1 |
| 可扩展性 | 模块化设计，便于添加新功能 | P2 |

### 1.3 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 主要开发语言 |
| Agent Framework | Latest | LLM 封装层 |
| LangGraph | Latest | 状态机管理 |
| GitHub Models | gpt-4o-mini | LLM 推理服务 |
| python-dotenv | Latest | 环境变量管理 |

---

## 2. 版本迭代历史

### 2.1 版本发布时间线

```
v1.0.0 (2026-02-26)
    │
    │  [单轮问答版本]
    │  - 基础症状评估功能
    │  - 6种免疫相关副作用识别
    │  - NCCN 风险分级
    │
    ▼
v2.0.0 (2026-02-27)
    │
    │  [多轮主动引导版本]
    │  - LangGraph 状态机架构
    │  - 主动追问机制
    │  - 结构化问诊流程
    │
    ▼
v2.0.1 (2026-02-27)
    │
    │  [Bug修复版本]
    │  - 修复多轮对话中断机制
    │  - 修复未知症状提示问题
    │  - 修复补充信息未纳入评估
    │
    ▼
v2.1.0 (2026-02-27)
       │
       [功能增强版本]
       - 新增症状映射系统 (SymptomMapper)
       - 新增会话超时管理 (SessionTimeoutManager)
       - 扩展问题库（肝毒性、肾毒性）
```

### 2.2 版本详情

#### v1.0.0 - 单轮问答版本

**发布日期**: 2026-02-26

**主要变更**:
- 实现基础的 Agent Framework 集成
- 配置 GitHub Models 作为 LLM 后端
- 定义 6 种免疫相关副作用类型
- 实现 NCCN 风险分级逻辑
- 单轮问答模式，无对话历史

**技术特点**:
```python
# 核心架构
Agent + OpenAIChatClient + Session

# 每次调用创建新会话
session = agent.create_session()
response = await agent.run(prompt, session=session)
```

**已知限制**:
- 无法进行追问，信息收集不完整
- 单次描述可能导致评估不准确
- 无法处理复杂的多症状场景

---

#### v2.0.0 - 多轮主动引导版本

**发布日期**: 2026-02-27

**主要变更**:
- 引入 LangGraph 状态机架构
- 实现主动追问机制
- 定义结构化问诊流程
- 添加核心问题库 (CORE_QUESTIONS)

**架构升级**:
```
v1.0 架构:
用户输入 → Agent → 单次响应

v2.0 架构:
用户输入 → [状态机] → 多轮交互 → 最终评估
         ↓
    initial_inquiry
         ↓
    symptom_collection ←→ detail_clarification
         ↓
    risk_assessment
         ↓
    recommendation
```

**关键代码变更**:
```python
# 新增状态定义
class ConversationState(TypedDict):
    current_state: str
    symptoms: List[str]
    symptom_type: Optional[str]
    collected_info: Dict[str, Any]
    questions_asked: List[str]
    answers: Dict[str, str]
    risk_grade: Optional[str]
    ...

# 新增图构建
def build_conversation_graph():
    workflow = StateGraph(ConversationState)
    workflow.add_node("initial_inquiry", initial_inquiry)
    workflow.add_node("symptom_collection", symptom_collection)
    ...
```

---

#### v2.0.1 - Bug修复版本

**发布日期**: 2026-02-27

**修复内容**:
1. **LangGraph 中断机制**: 修复 `ainvoke()` 执行整个图的问题
2. **未知症状提示**: 修复无标准问题时显示空提示
3. **补充信息处理**: 修复补充信息未纳入风险评估

**关键修复**:
```python
# 修复前：每次从头执行
result = await self.graph.ainvoke(self.current_state, self.config)

# 修复后：使用中断机制
await self.graph.aupdate_state(self.config, updates)
result = await self.graph.ainvoke(None, self.config)  # 关键：传入 None
```

---

#### v2.1.0 - 功能增强版本

**发布日期**: 2026-02-27

**新增功能**:
1. **症状映射系统 (SymptomMapper)**
   - 支持关键词匹配
   - 支持医学术语识别
   - 支持口语化描述理解
   - 紧急症状检测

2. **会话超时管理 (SessionTimeoutManager)**
   - 30分钟默认超时
   - 5分钟预警机制
   - 自动保存功能

3. **问题库扩展**
   - 新增肝毒性问题集
   - 新增肾毒性问题集

---

## 3. Bug修复记录与分析报告

### 3.1 Bug #001: API密钥配置错误

#### 基本信息

| 属性 | 值 |
|------|-----|
| Bug ID | BUG-001 |
| 发现日期 | 2026-02-26 |
| 修复日期 | 2026-02-26 |
| 影响版本 | v1.0.0 |
| 严重程度 | 阻塞 (Blocker) |
| 发现者 | 用户测试 |

#### 问题描述

```
ValueError: OpenAI API key is required
```

程序启动时抛出异常，无法创建 Agent 实例。

#### 复现步骤

1. 配置 `.env` 文件
2. 运行 `python hosAgent.py`
3. 程序抛出 `ValueError`

#### 根因分析

```python
# 错误代码
client = OpenAIChatClient(
    api_key=os.environ.get("in"),  # 错误：变量名错误
    ...
)
```

**问题层级分析**:

| 层级 | 问题 | 代码位置 |
|------|------|----------|
| 配置层 | 环境变量名称不匹配 | `.env` 文件 |
| 代码层 | `os.getenv("in")` 使用了错误的变量名 | `hosAgent.py:245-249` |
| 验证层 | 缺少环境变量验证 | 启动时无检查 |

#### 修复方案

```python
# 修复后代码
required_vars = ["GITHUB_TOKEN", "GITHUB_ENDPOINT", "GITHUB_MODEL_ID"]
missing = [var for var in required_vars if not os.environ.get(var)]

if missing:
    raise ValueError(f"缺少必需的环境变量: {', '.join(missing)}")

client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),
    api_key=os.environ.get("GITHUB_TOKEN"),
    model_id=os.environ.get("GITHUB_MODEL_ID")
)
```

#### 验证结果

- [x] 程序正常启动
- [x] 环境变量验证通过
- [x] Agent 实例创建成功

#### 预防措施

1. 添加 `validate_environment()` 函数在启动时验证
2. 使用常量定义环境变量名称
3. 添加单元测试覆盖配置验证逻辑

---

### 3.2 Bug #002: Agent框架使用错误

#### 基本信息

| 属性 | 值 |
|------|-----|
| Bug ID | BUG-002 |
| 发现日期 | 2026-02-26 |
| 修复日期 | 2026-02-26 |
| 影响版本 | v1.0.0 |
| 严重程度 | 阻塞 (Blocker) |
| 发现者 | 用户测试 |

#### 问题描述

```
AttributeError: 'OpenAIChatClient' object has no attribute 'chat'
```

尝试调用 `client.chat.completions.create()` 时抛出属性错误。

#### 复现步骤

1. 创建 `OpenAIChatClient` 实例
2. 尝试调用 `client.chat.completions.create()`
3. 程序抛出 `AttributeError`

#### 根因分析

```python
# 错误用法
client = OpenAIChatClient(...)
response = client.chat.completions.create(...)  # OpenAIChatClient 不支持此接口
```

**问题分析**:
- `OpenAIChatClient` 是 Agent Framework 的封装类
- 不直接暴露 OpenAI SDK 的 `chat.completions.create` 接口
- 需要通过 `Agent` 类进行调用

#### 修复方案

```python
# 正确用法
client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),
    api_key=os.environ.get("GITHUB_TOKEN"),
    model_id=os.environ.get("GITHUB_MODEL_ID")
)

agent = Agent(
    name="ImmunoPatientAssistant",
    client=client,
    instructions=SYSTEM_PROMPT,
    tools=[]
)

# 通过 Agent 调用
session = agent.create_session()
response = await agent.run(prompt, session=session)
```

#### 验证结果

- [x] Agent 正常创建
- [x] 对话正常进行
- [x] 响应正确返回

#### 经验总结

1. Agent Framework 的 `OpenAIChatClient` 不是 OpenAI SDK 的直接封装
2. 必须通过 `Agent` 类进行对话交互
3. 使用 `Session` 管理对话上下文

---

### 3.3 Bug #003: LangGraph多轮对话中断失败

#### 基本信息

| 属性 | 值 |
|------|-----|
| Bug ID | BUG-003 |
| 发现日期 | 2026-02-27 |
| 修复日期 | 2026-02-27 |
| 影响版本 | v2.0.0 |
| 严重程度 | 严重 (Critical) |
| 发现者 | 功能测试 |

#### 问题描述

```
问题：LangGraph 执行整个图，没有在 symptom_collection 节点后暂停等待用户输入
现象：用户输入一次，系统自动完成所有问答，无法进行多轮交互
```

#### 复现步骤

1. 启动 `hosAgent_v2.py`
2. 输入初始症状描述
3. 系统自动输出所有问题和评估结果，没有等待用户回答

#### 根因分析

**问题代码**:
```python
# 错误用法：每次都从头开始执行
result = await self.graph.ainvoke(self.current_state, self.config)
```

**技术分析**:

| 问题点 | 说明 |
|--------|------|
| `ainvoke(state)` | 每次调用都从 START 开始执行，忽略中断点 |
| 状态覆盖 | 传入的 state 会覆盖已保存的状态 |
| 中断机制失效 | `interrupt_after` 配置被忽略 |

**LangGraph 工作原理**:
```
正确流程:
START → initial_inquiry → symptom_collection [INTERRUPT]
                                              ↓
用户输入 → update_state → ainvoke(None) → 从中断点恢复
                                              ↓
                           detail_clarification → ...
```

#### 修复方案

```python
# 修复后的 continue_conversation 方法
async def continue_conversation(self, user_input: str) -> str:
    # 1. 准备状态更新
    updates = {"user_input": user_input}
    updates["messages"] = [{"role": "user", "content": user_input}]
    
    # 2. 关键：使用 aupdate_state 更新状态
    await self.graph.aupdate_state(self.config, updates)
    
    # 3. 关键：使用 ainvoke(None) 从中断点恢复
    result = await self.graph.ainvoke(None, self.config)  # 必须传入 None
    self.current_state = result
    
    return self._get_last_assistant_message()
```

**关键点**:
1. `aupdate_state()` - 更新保存的状态
2. `ainvoke(None)` - 从中断点恢复执行（None 表示使用已保存的状态）

#### 验证结果

```
测试对话流程:
用户: 我最近有皮疹
助手: [识别症状类型] 问题 1/5: 皮疹是什么时候开始的？
用户: 3天前
助手: 问题 2/5: 皮疹大约占身体面积的多少？
用户: 手掌大小
助手: 问题 3/5: 瘙痒程度如何？
...
```

- [x] 系统正确暂停等待用户输入
- [x] 多轮对话正常进行
- [x] 状态正确保存和恢复

#### 经验总结

1. LangGraph 的 `interrupt_after` 需要配合 `ainvoke(None)` 使用
2. 状态更新使用 `aupdate_state()` 而非直接修改状态对象
3. 理解状态机的"暂停-恢复"模式是关键

---

### 3.4 Bug #004: 未知症状无提示问题

#### 基本信息

| 属性 | 值 |
|------|-----|
| Bug ID | BUG-004 |
| 发现日期 | 2026-02-27 |
| 修复日期 | 2026-02-27 |
| 影响版本 | v2.0.0 |
| 严重程度 | 中等 (Major) |
| 发现者 | 功能测试 |

#### 问题描述

```
遇到不可识别的症状时：
    根据您的描述，我初步判断可能与【其他】相关。
    为了更准确地评估您的情况，我需要问您几个问题。
    请回答上述问题:
没有给出问题
```

系统承诺"问您几个问题"但实际没有问题可问。

#### 复现步骤

1. 输入不在问题库中的症状描述（如"我头痛"）
2. 系统显示"请回答上述问题"
3. 但没有显示任何问题

#### 根因分析

**问题层级分析**:

| 层级 | 问题 | 代码位置 |
|------|------|----------|
| 消息断裂 | `symptom_collection` 发现症状不在 CORE_QUESTIONS 时直接返回，未添加消息 | `hosAgent_v2.py:468-476` |
| 过度承诺 | `initial_inquiry` 无条件承诺"问您几个问题" | `hosAgent_v2.py:424-434` |
| 覆盖不全 | CORE_QUESTIONS 只定义了6种症状类型 | `hosAgent_v2.py:49-134` |

**问题代码**:
```python
# symptom_collection 节点
if symptom_type not in CORE_QUESTIONS:
    logger.warning(f"未找到 {symptom_type} 的问题库，跳转到评估")
    state["current_state"] = "risk_assessment"
    return state  # 直接返回，没有添加任何消息
```

#### 修复方案

**修复1: 添加补充信息提示**
```python
if symptom_type not in CORE_QUESTIONS:
    state["messages"].append({
        "role": "assistant",
        "content": f"\n由于【{symptom_type}】暂无标准问题库，请您补充以下信息以帮助评估：\n"
                   f"• 症状持续时间\n"
                   f"• 症状严重程度\n"
                   f"• 是否影响日常生活\n"
                   f"• 有无其他伴随症状\n"
    })
    state["waiting_for_supplement"] = True
    state["current_state"] = "risk_assessment"
    return state
```

**修复2: 条件化欢迎消息**
```python
# initial_inquiry 节点
if symptom_type in CORE_QUESTIONS:
    welcome_msg += f"为了更准确地评估您的情况，我需要问您几个问题。"
else:
    welcome_msg += f"我将基于您的描述进行初步评估。"
```

**修复3: 智能提示语**
```python
# 主循环中
elif assistant.current_state.get("waiting_for_supplement"):
    prompt = "请补充上述信息"
elif assistant.current_state.get("current_question"):
    prompt = "请回答上述问题"
```

#### 验证结果

```
测试对话:
用户: 我头痛
助手: 根据您的描述，我初步判断可能与【未知】相关。
      我将基于您的描述进行初步评估。
      请补充以下信息以帮助评估：
      • 症状持续时间
      • 症状严重程度
      • 是否影响日常生活
      • 有无其他伴随症状

请补充上述信息: 头疼欲裂，咳血
助手: [基于补充信息进行评估]
```

- [x] 未知症状显示补充信息提示
- [x] 不再出现空问题提示
- [x] 提示语与实际状态匹配

---

### 3.5 Bug #005: 补充信息未纳入评估

#### 基本信息

| 属性 | 值 |
|------|-----|
| Bug ID | BUG-005 |
| 发现日期 | 2026-02-27 |
| 修复日期 | 2026-02-27 |
| 影响版本 | v2.0.0 |
| 严重程度 | 严重 (Critical) |
| 发现者 | 功能测试 |

#### 问题描述

```
用户输入: 我头痛
系统识别: 未知
用户补充: 头疼欲裂，咳血
风险评估: G1（轻度）← 错误！咳血应该是G3-G4
```

用户补充的关键信息（"咳血"）未被纳入风险评估。

#### 复现步骤

1. 输入不在问题库中的症状
2. 补充严重症状信息
3. 观察风险评估结果

#### 根因分析

**问题分析**:

| 问题点 | 说明 |
|--------|------|
| 信息未追加 | 补充信息没有追加到原始描述 |
| 评估使用错误字段 | `risk_assessment` 只使用 `initial_description` |

**问题代码**:
```python
# continue_conversation - 补充信息未处理
if self.current_state.get("waiting_for_supplement"):
    updates["waiting_for_supplement"] = False
    # 没有将补充信息追加到描述中

# risk_assessment - 只使用初始描述
context = f"""## 患者描述
{state['collected_info'].get('initial_description', '无')}  # 缺少补充信息
"""
```

#### 修复方案

**修复1: 追加补充信息**
```python
# continue_conversation
if self.current_state.get("waiting_for_supplement"):
    original_description = self.current_state["collected_info"].get("initial_description", "")
    supplemented_description = f"{original_description}\n补充信息：{user_input}"
    updates["collected_info"] = {
        **self.current_state["collected_info"],
        "initial_description": original_description,
        "supplement": user_input,
        "full_description": supplemented_description
    }
    updates["waiting_for_supplement"] = False
```

**修复2: 使用完整描述评估**
```python
# risk_assessment
full_description = state['collected_info'].get('full_description', 
                                                state['collected_info'].get('initial_description', '无'))
context = f"""## 患者完整描述
{full_description}
...
"""
```

#### 验证结果

```
测试对话:
用户: 我头痛
助手: 请补充以下信息...
用户: 头疼欲裂，咳血
助手: [评估上下文包含完整描述]
      风险等级: G3-G4（重度）
      建议: 立即就医
```

- [x] 补充信息正确追加
- [x] 风险评估使用完整描述
- [x] 严重症状正确识别

---

### 3.6 Bug修复统计

| Bug ID | 严重程度 | 修复耗时 | 根因类型 |
|--------|----------|----------|----------|
| BUG-001 | Blocker | 15分钟 | 配置错误 |
| BUG-002 | Blocker | 20分钟 | API误用 |
| BUG-003 | Critical | 60分钟 | 框架理解不足 |
| BUG-004 | Major | 30分钟 | 边界条件遗漏 |
| BUG-005 | Critical | 25分钟 | 数据流断裂 |

**根因分类统计**:
- 配置/环境问题: 20%
- API/框架误用: 40%
- 边界条件遗漏: 40%

---

## 4. 功能更新日志

### 4.1 v1.0.0 功能清单

#### F001: 基础症状评估

**功能描述**: 接收用户症状描述，返回基于 NCCN 指南的评估建议。

**实现方案**:
```python
async def assess_symptom(patient_description: str, agent: Agent) -> str:
    session = agent.create_session()
    prompt = f"""患者症状描述:
{patient_description}

请根据上述症状描述，提供评估建议。"""
    response = await agent.run(prompt, session=session)
    return response.messages[-1].contents[0].text + DISCLAIMER
```

**使用说明**:
```python
agent = create_immuno_patient_agent()
assessment = await assess_symptom("我最近有皮疹，有点痒", agent)
print(assessment)
```

#### F002: 免疫相关副作用识别

**功能描述**: 基于关键词匹配快速识别症状类型。

**支持的症状类型**:
| 类型 | 关键词示例 |
|------|------------|
| 皮肤毒性 | 皮疹、瘙痒、皮肤干燥 |
| 胃肠道毒性 | 腹泻、腹痛、便血 |
| 肺毒性 | 咳嗽、呼吸困难、胸闷 |
| 内分泌毒性 | 乏力、体重变化、情绪改变 |
| 肝毒性 | 黄疸、食欲下降、肝区不适 |
| 肾毒性 | 尿量变化、水肿 |

#### F003: NCCN风险分级

**分级标准**:
| 等级 | 定义 | 建议 |
|------|------|------|
| G1 | 轻度，不影响日常生活 | 继续治疗，观察等待 |
| G2 | 中度，部分影响日常活动 | 48小时内联系医疗团队 |
| G3-G4 | 重度，危及生命 | 立即就医 |

---

### 4.2 v2.0.0 功能清单

#### F004: 多轮主动问诊

**功能描述**: 根据症状类型主动提出针对性问题，收集详细信息。

**设计思路**:
1. 症状识别后，从问题库选择相关问题
2. 按优先级顺序提问
3. 收集回答用于风险评估

**实现方案**:
```python
CORE_QUESTIONS = {
    "皮肤毒性": {
        "questions": [
            {"id": "onset_time", "text": "皮疹是什么时候开始的？", "priority": 1},
            {"id": "body_area", "text": "皮疹大约占身体面积的多少？", "priority": 2},
            ...
        ],
        "severity_keywords": {...}
    },
    ...
}
```

**使用说明**:
```python
assistant = ImmunoPatientAssistantV2()
response = await assistant.start_conversation("我最近有皮疹")
print(response)  # 显示第一个问题

response = await assistant.continue_conversation("3天前开始的")
print(response)  # 显示第二个问题或评估结果
```

#### F005: 状态机驱动的问诊流程

**功能描述**: 使用 LangGraph 管理问诊流程状态转换。

**状态转换图**:
```
START
  │
  ▼
initial_inquiry (症状识别)
  │
  ▼
symptom_collection (选择问题) ←──┐
  │                              │
  ▼                              │
detail_clarification (等待回答) ──┘
  │
  ▼
risk_assessment (风险评估)
  │
  ▼
recommendation (生成建议)
  │
  ▼
END
```

**实现方案**:
```python
def build_conversation_graph():
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("initial_inquiry", initial_inquiry)
    workflow.add_node("symptom_collection", symptom_collection)
    workflow.add_node("detail_clarification", detail_clarification)
    workflow.add_node("risk_assessment", risk_assessment)
    workflow.add_node("recommendation", recommendation)
    
    workflow.add_conditional_edges("symptom_collection", should_continue_collecting, {...})
    workflow.add_conditional_edges("detail_clarification", should_ask_more_questions, {...})
    
    return workflow.compile(checkpointer=MemorySaver(), interrupt_after=["symptom_collection"])
```

---

### 4.3 v2.1.0 功能清单

#### F006: 症状映射系统 (SymptomMapper)

**功能描述**: 智能识别用户描述中的症状类型，支持医学术语和口语化描述。

**设计思路**:
1. 关键词匹配：基础症状词汇
2. 医学术语：专业术语识别
3. 口语化描述：日常用语理解
4. 紧急检测：危险症状识别

**实现方案**:
```python
class SymptomMapper:
    SYMPTOM_KEYWORDS = {
        "皮肤毒性": {
            "keywords": ["皮疹", "疹子", "痒", "瘙痒"],
            "medical_terms": ["dermatitis", "rash", "pruritus"],
            "colloquial": ["身上起红点", "皮肤发痒", "起疙瘩"]
        },
        ...
    }
    
    EMERGENCY_KEYWORDS = ["呼吸困难", "咳血", "意识模糊", ...]
    
    @classmethod
    def map_symptom(cls, user_input: str) -> Dict:
        # 返回: symptom_type, confidence, matched_keywords, is_emergency
```

**使用说明**:
```python
result = SymptomMapper.map_symptom("我身上起红点，很痒")
# result = {
#     "symptom_type": "皮肤毒性",
#     "confidence": "高",
#     "matched_keywords": ["起红点", "痒"],
#     "is_emergency": False
# }
```

#### F007: 会话超时管理 (SessionTimeoutManager)

**功能描述**: 管理会话生命周期，防止长时间未操作的会话占用资源。

**设计思路**:
1. 超时检测：基于最后活动时间
2. 预警机制：超时前提醒用户
3. 自动保存：定期保存会话数据

**实现方案**:
```python
class SessionTimeoutManager:
    DEFAULT_TIMEOUT_MINUTES = 30
    WARNING_BEFORE_MINUTES = 5
    
    def check_timeout(self) -> Dict:
        # 返回: is_timeout, is_warning, remaining_seconds
    
    def auto_save(self, data: Dict) -> None:
        # 保存会话数据
```

**使用说明**:
```python
timeout_mgr = SessionTimeoutManager(timeout_minutes=30)

# 检查超时
status = timeout_mgr.check_timeout()
if status["is_warning"]:
    print(f"会话将在 {status['remaining_seconds']} 秒后超时")

# 更新活动时间
timeout_mgr.update_activity()
```

#### F008: 扩展问题库

**功能描述**: 新增肝毒性和肾毒性的问题集。

**新增内容**:
```python
CORE_QUESTIONS["肝毒性"] = {
    "questions": [
        {"id": "jaundice", "text": "皮肤或眼睛有没有发黄？", "priority": 1},
        {"id": "appetite", "text": "食欲有没有下降？恶心吗？", "priority": 2},
        {"id": "liver_pain", "text": "右上腹有没有不适或疼痛？", "priority": 3},
        {"id": "lab_results", "text": "最近肝功能检查结果如何？", "priority": 4},
        {"id": "other_symptoms", "text": "有没有皮肤瘙痒、尿液变深？", "priority": 5}
    ],
    "severity_keywords": {...}
}

CORE_QUESTIONS["肾毒性"] = {
    "questions": [
        {"id": "urine_change", "text": "尿量有没有明显变化？", "priority": 1},
        {"id": "edema", "text": "有没有水肿？在哪些部位？", "priority": 2},
        ...
    ],
    "severity_keywords": {...}
}
```

---

## 5. 架构设计与技术决策

### 5.1 架构演进

#### v1.0 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户界面层                            │
│                    (main_async CLI)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        业务逻辑层                            │
│              (assess_symptom, quick_classification)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Agent 层                              │
│              (Agent + OpenAIChatClient)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        LLM 服务层                            │
│                    (GitHub Models)                          │
└─────────────────────────────────────────────────────────────┘
```

**特点**:
- 单层调用，简单直接
- 无状态管理
- 适合快速原型开发

**局限**:
- 无法进行多轮对话
- 信息收集不完整
- 评估准确性受限

#### v2.0 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户界面层                            │
│              (ImmunoPatientAssistantV2 CLI)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      对话管理层                              │
│              (ImmunoPatientAssistantV2)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              状态管理 (ConversationState)            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     状态机层 (LangGraph)                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐       │
│  │ initial  │→ │   symptom    │→ │      risk       │       │
│  │ inquiry  │  │  collection  │  │   assessment    │       │
│  └──────────┘  └──────────────┘  └──────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Agent 层                              │
│              (Agent + OpenAIChatClient)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        LLM 服务层                            │
│                    (GitHub Models)                          │
└─────────────────────────────────────────────────────────────┘
```

**特点**:
- 状态机驱动，流程可控
- 多轮对话支持
- 中断恢复机制

**优势**:
- 信息收集完整
- 评估准确性提升
- 可扩展性强

### 5.2 技术决策记录

#### 决策 #001: 选择 Agent Framework

**背景**: 需要选择 LLM 封装框架。

**选项**:
| 选项 | 优点 | 缺点 |
|------|------|------|
| OpenAI SDK | 直接控制，文档完善 | 需要手动管理会话 |
| LangChain | 生态丰富，工具多 | 学习曲线陡峭 |
| Agent Framework | 微软官方，与 GitHub Models 兼容 | 相对较新 |

**决策**: 选择 Agent Framework

**理由**:
1. 与 GitHub Models 无缝集成
2. 简洁的 API 设计
3. 微软官方支持，长期维护有保障

#### 决策 #002: 选择 LangGraph 作为状态机

**背景**: v2.0 需要实现多轮对话的状态管理。

**选项**:
| 选项 | 优点 | 缺点 |
|------|------|------|
| 手动状态管理 | 简单直接 | 难以维护，易出错 |
| LangChain | 成熟稳定 | 不支持中断恢复 |
| LangGraph | 原生中断支持，可视化 | 相对较新 |

**决策**: 选择 LangGraph

**理由**:
1. 原生支持 `interrupt_after` 中断机制
2. 状态持久化开箱即用
3. 支持可视化调试

#### 决策 #003: 选择 GitHub Models 作为 LLM 后端

**背景**: 需要选择 LLM 服务提供商。

**选项**:
| 选项 | 优点 | 缺点 |
|------|------|------|
| OpenAI API | 模型选择多 | 需要付费 |
| Azure OpenAI | 企业级支持 | 需要 Azure 订阅 |
| GitHub Models | 免费层可用 | 速率限制 |

**决策**: 选择 GitHub Models

**理由**:
1. 免费层足够开发测试使用
2. 与 Agent Framework 兼容性好
3. 降低用户使用门槛

#### 决策 #004: NCCN 风险分级标准

**背景**: 需要选择症状评估标准。

**选项**:
| 选项 | 优点 | 缺点 |
|------|------|------|
| CTCAE | 肿瘤学标准 | 过于复杂 |
| NCCN | 临床指南，实用性强 | 需要适配 |
| 自定义 | 灵活 | 缺乏权威性 |

**决策**: 选择 NCCN 指南

**理由**:
1. 国际公认的肿瘤治疗指南
2. 分级标准清晰（G1-G4）
3. 患者易于理解

### 5.3 设计模式应用

#### 状态模式 (State Pattern)

**应用场景**: 问诊流程状态转换

**实现**:
```python
# 状态定义
class ConversationState(TypedDict):
    current_state: str  # "initial", "symptom_collection", "risk_assessment", ...
    ...

# 状态转换
def should_continue_collecting(state: ConversationState) -> str:
    if state["question_count"] >= state["max_questions"]:
        return "risk_assessment"
    return "detail_clarification"
```

#### 工厂模式 (Factory Pattern)

**应用场景**: Agent 实例创建

**实现**:
```python
def create_immuno_agent() -> Agent:
    """工厂函数，封装 Agent 创建逻辑"""
    validate_environment()
    client = OpenAIChatClient(...)
    return Agent(name="ImmunoPatientAssistant", client=client, ...)
```

#### 策略模式 (Strategy Pattern)

**应用场景**: 风险评估策略

**实现**:
```python
# 不同风险等级的建议策略
recommendations = {
    "G1": generate_g1_recommendation,
    "G2": generate_g2_recommendation,
    "G3": generate_g3_recommendation,
    "G4": generate_g4_recommendation,
}

def generate_recommendation(state: ConversationState) -> str:
    strategy = recommendations.get(state["risk_grade"], default_strategy)
    return strategy(state)
```

---

## 6. 技术债务与优化措施

### 6.1 当前技术债务

#### TD-001: LLM 调用无重试机制

**描述**: 当前 LLM 调用失败时直接抛出异常，没有重试机制。

**影响**: 网络抖动可能导致服务不可用。

**优先级**: 中

**建议方案**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_llm_for_analysis(agent: Agent, prompt: str) -> str:
    ...
```

#### TD-002: 问题库硬编码

**描述**: CORE_QUESTIONS 硬编码在代码中，难以维护和扩展。

**影响**: 添加新问题需要修改代码并重新部署。

**优先级**: 中

**建议方案**:
```python
# 使用 YAML 或 JSON 配置文件
# questions.yaml
皮肤毒性:
  questions:
    - id: onset_time
      text: "皮疹是什么时候开始的？"
      priority: 1
  severity_keywords:
    G1: ["轻微", "小范围"]
    ...

# 加载配置
import yaml
with open("questions.yaml", "r", encoding="utf-8") as f:
    CORE_QUESTIONS = yaml.safe_load(f)
```

#### TD-003: 无对话历史持久化

**描述**: 对话状态仅保存在内存中，程序重启后丢失。

**影响**: 无法恢复中断的对话，无法进行事后分析。

**优先级**: 低

**建议方案**:
```python
# 使用 SQLite 或 Redis 持久化
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("checkpoints.db")
app = workflow.compile(checkpointer=memory, ...)
```

#### TD-004: 缺少单元测试

**描述**: 当前没有单元测试覆盖。

**影响**: 重构风险高，回归问题难以发现。

**优先级**: 高

**建议方案**:
```python
# tests/test_symptom_mapper.py
import pytest
from hosAgent_v2 import SymptomMapper

def test_map_symptom_skin():
    result = SymptomMapper.map_symptom("我身上起红点，很痒")
    assert result["symptom_type"] == "皮肤毒性"
    assert result["confidence"] in ["高", "中"]

def test_emergency_detection():
    result = SymptomMapper.map_symptom("我咳血了")
    assert result["is_emergency"] == True
```

### 6.2 性能优化措施

#### 优化 #001: LLM 响应缓存

**问题**: 相同症状描述重复调用 LLM。

**方案**: 添加语义相似度缓存

```python
from functools import lru_cache
import hashlib

# 简单缓存
@lru_cache(maxsize=100)
def get_cached_response(description_hash: str) -> Optional[str]:
    return cache.get(description_hash)

async def assess_with_cache(description: str, agent: Agent) -> str:
    desc_hash = hashlib.md5(description.encode()).hexdigest()
    cached = get_cached_response(desc_hash)
    if cached:
        return cached
    
    response = await assess_symptom(description, agent)
    cache.set(desc_hash, response)
    return response
```

#### 优化 #002: 并行问题生成

**问题**: 问题按顺序生成，延迟较高。

**方案**: 预生成所有问题

```python
async def symptom_collection(state: ConversationState) -> ConversationState:
    # 预生成所有问题，避免多次 LLM 调用
    if state["symptom_type"] in CORE_QUESTIONS:
        all_questions = CORE_QUESTIONS[state["symptom_type"]]["questions"]
        state["pending_questions"] = all_questions
    ...
```

#### 优化 #003: 流式响应

**问题**: 用户等待完整响应时间较长。

**方案**: 实现流式输出

```python
async def stream_response(agent: Agent, prompt: str):
    async for chunk in agent.stream(prompt):
        yield chunk

# 使用
async for chunk in stream_response(agent, prompt):
    print(chunk, end="", flush=True)
```

### 6.3 代码质量改进

#### 改进 #001: 类型注解完善

**当前状态**: 部分函数缺少类型注解

**目标**: 100% 类型注解覆盖

```python
# 改进前
def quick_symptom_classification(description):
    ...

# 改进后
def quick_symptom_classification(description: str) -> Optional[str]:
    ...
```

#### 改进 #002: 日志规范化

**当前状态**: 日志格式不统一

**目标**: 结构化日志

```python
import structlog

logger = structlog.get_logger()

# 结构化日志
logger.info("symptom_identified", 
            symptom_type="皮肤毒性", 
            confidence="高",
            user_input_length=len(user_input))
```

#### 改进 #003: 错误处理统一

**当前状态**: 错误处理分散

**目标**: 统一错误处理中间件

```python
from functools import wraps

def handle_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.error("validation_error", error=str(e))
            return {"error": "输入验证失败", "detail": str(e)}
        except Exception as e:
            logger.error("unexpected_error", error=str(e))
            return {"error": "服务暂时不可用，请稍后重试"}
    return wrapper
```

---

## 7. 重要决策记录

### 7.1 ADR-001: 单轮问答 vs 多轮对话

**状态**: 已采纳

**背景**: v1.0 采用单轮问答模式，评估准确性受限。

**决策**: 升级为多轮主动引导模式

**后果**:
- 正面：评估准确性提升，用户体验改善
- 负面：开发复杂度增加，需要状态管理

### 7.2 ADR-002: 本地部署 vs 云服务

**状态**: 已采纳

**背景**: 需要决定部署方式。

**决策**: 优先支持本地部署，云服务作为可选项

**理由**:
1. 医疗数据隐私要求
2. 降低用户使用成本
3. GitHub Models 免费层满足需求

**后果**:
- 正面：用户数据不离开本地，隐私保护
- 负面：需要用户自行配置环境

### 7.3 ADR-003: 免责声明策略

**状态**: 已采纳

**背景**: 需要明确助手的责任边界。

**决策**: 每次响应都包含免责声明

**实现**:
```python
DISCLAIMER = """
---
**重要免责声明**：
本助手提供的所有信息仅供参考，不能替代专业医疗诊断和治疗建议。
- 本建议基于NCCN指南的一般性原则，具体情况因人而异
- 如症状持续加重或出现新的严重症状，请立即就医
- 在任何情况下，医疗团队的专业判断优先于本助手的建议
"""
```

---

## 8. 兼容性处理方案

### 8.1 Python 版本兼容性

**最低要求**: Python 3.12+

**原因**:
- `asyncio.run()` 需要 3.7+
- TypedDict 完整支持需要 3.9+
- 类型注解语法需要 3.10+

**兼容性检查**:
```python
import sys
if sys.version_info < (3, 12):
    raise RuntimeError("需要 Python 3.12 或更高版本")
```

### 8.2 依赖版本管理

**当前依赖**:
```
agent-framework>=0.1.0
langgraph>=0.0.20
langchain-core>=0.1.0
python-dotenv>=1.0.0
```

**版本锁定策略**:
- 主版本号：允许升级（可能有破坏性变更）
- 次版本号：允许升级（向后兼容）
- 修订号：允许升级（bug修复）

### 8.3 API 兼容性

**GitHub Models API**:
- 端点: `https://models.inference.ai.azure.com`
- 认证: Bearer Token
- 模型: `gpt-4o-mini`（推荐）

**兼容性处理**:
```python
# 支持多种模型
SUPPORTED_MODELS = {
    "gpt-4o-mini": {"max_tokens": 4096, "context_window": 128000},
    "gpt-4o": {"max_tokens": 4096, "context_window": 128000},
    "gpt-35-turbo": {"max_tokens": 4096, "context_window": 16385},
}

def create_client(model_id: str = None) -> OpenAIChatClient:
    model_id = model_id or os.environ.get("GITHUB_MODEL_ID", "gpt-4o-mini")
    if model_id not in SUPPORTED_MODELS:
        logger.warning(f"模型 {model_id} 未在支持列表中，可能存在兼容性问题")
    ...
```

### 8.4 向后兼容性

**v1.0 到 v2.0 迁移指南**:

```python
# v1.0 用法（仍然支持）
from hosAgent import create_immuno_patient_agent, assess_symptom

agent = create_immuno_patient_agent()
response = await assess_symptom("我最近有皮疹", agent)

# v2.0 用法（推荐）
from hosAgent_v2 import ImmunoPatientAssistantV2

assistant = ImmunoPatientAssistantV2()
response = await assistant.start_conversation("我最近有皮疹")
while not assistant.is_conversation_ended():
    user_input = input("回答: ")
    response = await assistant.continue_conversation(user_input)
    print(response)
```

---

## 9. 附录

### 9.1 文件结构

```
hospital_assistant_v2/
├── DEVELOPMENT_HISTORY.md    # 本文档
├── bug.md                    # Bug 记录
└── ...

helperAgent/
├── hosAgent.py               # v1.0 单轮问答版本
├── hosAgent_v2.py            # v2.0+ 多轮引导版本
└── ...
```

### 9.2 环境变量配置

```bash
# .env 文件示例
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_ENDPOINT=https://models.inference.ai.azure.com
GITHUB_MODEL_ID=gpt-4o-mini

# Azure 配置（可选）
AZURE_OPENAI_API_KEY=xxxxxxxx
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
```

### 9.3 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 GITHUB_TOKEN

# 3. 运行 v1.0
python hosAgent.py

# 4. 运行 v2.0
python hosAgent_v2.py
```

### 9.4 参考资料

- [NCCN 免疫治疗相关不良反应管理指南](https://www.nccn.org/)
- [Agent Framework 文档](https://github.com/microsoft/agent-framework)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [GitHub Models 文档](https://docs.github.com/en/models)

---

## 变更记录

| 日期 | 版本 | 变更内容 | 作者 |
|------|------|----------|------|
| 2026-02-27 | 1.0.0 | 初始版本，完整开发历程文档 | AI Assistant |

---

*本文档由 AI Assistant 自动生成，如有疑问请联系项目维护者。*
