"""
免疫治疗患者教育助手 v2.0 - 主动引导版
================================================

[WHY] 从单轮问答升级为多轮主动引导式问诊，提高症状评估准确性
[HOW] 使用 LangGraph 状态机 + Agent Framework 实现结构化问诊流程
[WARN] 需要 pip install langgraph langchain-core

核心设计原则：
- 状态机驱动：LangGraph 管理问诊流程状态转换
- 主动追问：根据症状类型主动收集关键信息
- 结构化评估：基于收集的信息进行风险评估
- 安全保守：始终倾向于建议就医

依赖安装：
    pip install langgraph langchain-core agent-framework python-dotenv
"""

import os
import asyncio
import logging
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from datetime import datetime
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ImmunoPatientAssistantV2")


# ============================================================================
# 核心问题库 - 按症状类型分类
# ============================================================================

CORE_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "皮肤毒性": {
        "questions": [
            {"id": "onset_time", "text": "皮疹是什么时候开始的？出现几天了？", "priority": 1},
            {"id": "body_area", "text": "皮疹大约占身体面积的多少？（如：手掌大小、半个背部等）", "priority": 2},
            {"id": "itching_level", "text": "瘙痒程度如何？是否影响睡眠？", "priority": 3},
            {"id": "blister", "text": "有没有水泡、皮肤剥脱或溃烂？", "priority": 4},
            {"id": "treatment", "text": "有没有使用过外用药物？效果如何？", "priority": 5}
        ],
        "severity_keywords": {
            "G1": ["轻微", "小范围", "不影响睡眠", "少于10%"],
            "G2": ["中度", "影响睡眠", "10-30%", "明显瘙痒"],
            "G3_G4": ["严重", "大面积", "水泡", "剥脱", "溃烂", "超过30%"]
        }
    },
    "胃肠道毒性": {
        "questions": [
            {"id": "frequency", "text": "每天腹泻大约几次？", "priority": 1},
            {"id": "duration", "text": "腹泻持续了几天？", "priority": 2},
            {"id": "blood", "text": "大便中有没有血或黏液？", "priority": 3},
            {"id": "pain_level", "text": "有没有腹痛？程度如何？（轻度/中度/剧烈）", "priority": 4},
            {"id": "other_symptoms", "text": "有没有发烧、恶心或呕吐？", "priority": 5}
        ],
        "severity_keywords": {
            "G1": ["少于4次", "轻微", "无血", "轻度腹痛"],
            "G2": ["4-6次", "中度", "影响活动", "中度腹痛"],
            "G3_G4": ["7次以上", "严重", "便血", "剧烈腹痛", "发烧"]
        }
    },
    "肺毒性": {
        "questions": [
            {"id": "cough_type", "text": "咳嗽是干咳还是有痰？持续多久了？", "priority": 1},
            {"id": "breathing", "text": "呼吸困难在什么情况下出现？（静息时/活动后）", "priority": 2},
            {"id": "oxygen", "text": "有没有需要吸氧？血氧饱和度是多少？", "priority": 3},
            {"id": "fever", "text": "有没有发烧？体温多少？", "priority": 4},
            {"id": "chest", "text": "有没有胸痛或胸闷？", "priority": 5}
        ],
        "severity_keywords": {
            "G1": ["轻微咳嗽", "活动正常", "无症状"],
            "G2": ["活动后气短", "轻度呼吸困难", "低热"],
            "G3_G4": ["静息时呼吸困难", "严重", "缺氧", "需要吸氧", "高热"]
        }
    },
    "内分泌毒性": {
        "questions": [
            {"id": "fatigue", "text": "乏力程度如何？是否影响日常活动？", "priority": 1},
            {"id": "weight", "text": "近期体重有没有明显变化？（增加/减少多少公斤）", "priority": 2},
            {"id": "mood", "text": "情绪有没有明显变化？（如焦虑、抑郁、易怒）", "priority": 3},
            {"id": "thyroid", "text": "有没有做过甲状腺功能检查？结果如何？", "priority": 4},
            {"id": "other_endocrine", "text": "有没有多饮、多尿、头痛或视力变化？", "priority": 5}
        ],
        "severity_keywords": {
            "G1": ["无症状", "指标轻度异常", "轻微乏力"],
            "G2": ["轻度症状", "需要药物", "中度乏力"],
            "G3_G4": ["严重症状", "危象", "意识改变", "严重乏力"]
        }
    }
}

# ============================================================================
# 症状映射系统 - 智能症状识别与分类
# ============================================================================

class SymptomMapper:
    """
    [WHY] 症状映射系统：将用户描述映射到已知症状类型
    [HOW] 使用关键词匹配 + 语义相似度 + LLM 辅助
    [WARN] 准确率目标 90%，低置信度结果需标记
    """
    
    # 症状关键词映射表（支持医学术语和口语化描述）
    SYMPTOM_KEYWORDS = {
        "皮肤毒性": {
            "keywords": ["皮疹", "疹子", "痒", "瘙痒", "皮肤红", "红斑", "脱皮", "水泡", "白癜风", "皮肤干燥"],
            "medical_terms": ["dermatitis", "rash", "pruritus", "urticaria", "vitiligo"],
            "colloquial": ["身上起红点", "皮肤发痒", "起疙瘩", "皮肤过敏", "皮肤问题"]
        },
        "胃肠道毒性": {
            "keywords": ["腹泻", "拉肚子", "腹痛", "肚子痛", "便血", "结肠炎", "恶心", "呕吐", "拉稀"],
            "medical_terms": ["diarrhea", "colitis", "enteritis", "gastroenteritis"],
            "colloquial": ["拉肚子", "肚子不舒服", "肠胃不好", "大便不正常", "跑厕所"]
        },
        "肺毒性": {
            "keywords": ["咳嗽", "呼吸困难", "气短", "胸闷", "肺炎", "喘", "缺氧", "气喘"],
            "medical_terms": ["pneumonitis", "pneumonia", "dyspnea", "hypoxia"],
            "colloquial": ["喘不上气", "呼吸不顺畅", "气不够用", "胸口闷", "咳嗽不停"]
        },
        "内分泌毒性": {
            "keywords": ["甲状腺", "乏力", "疲劳", "体重变化", "情绪", "甲亢", "甲减", "内分泌"],
            "medical_terms": ["thyroiditis", "hypophysitis", "adrenal", "hypothyroidism"],
            "colloquial": ["没力气", "容易累", "体重掉", "情绪不好", "脾气大"]
        },
        "肝毒性": {
            "keywords": ["肝", "黄疸", "食欲", "皮肤发黄", "眼睛发黄", "转氨酶", "肝功能"],
            "medical_terms": ["hepatitis", "hepatic", "jaundice", "bilirubin"],
            "colloquial": ["眼睛黄", "皮肤黄", "不想吃饭", "肝不好"]
        },
        "肾毒性": {
            "keywords": ["肾", "尿", "水肿", "肌酐", "浮肿", "尿量", "少尿"],
            "medical_terms": ["nephritis", "renal", "creatinine", "proteinuria"],
            "colloquial": ["小便少", "脸肿", "腿肿", "尿有问题"]
        }
    }
    
    # 紧急症状关键词（立即就医）
    EMERGENCY_KEYWORDS = [
        "呼吸困难", "胸痛", "咯血", "咳血", "意识模糊", "昏迷", "高热", "抽搐",
        "剧烈头痛", "视力模糊", "严重出血", "过敏性休克"
    ]
    
    @classmethod
    def map_symptom(cls, user_input: str) -> Dict[str, Any]:
        """
        [WHY] 将用户输入映射到症状类型
        [HOW] 关键词匹配 + 置信度计算
        [WARN] 低置信度结果标记为"未知"
        
        Args:
            user_input: 用户症状描述
            
        Returns:
            Dict: {
                "symptom_type": "症状类型",
                "confidence": "高/中/低",
                "matched_keywords": ["匹配的关键词"],
                "is_emergency": False
            }
        """
        input_lower = user_input.lower()
        matched_type = None
        matched_keywords = []
        max_score = 0
        
        # 检查紧急症状
        is_emergency = any(kw in input_lower for kw in cls.EMERGENCY_KEYWORDS)
        
        # 遍历所有症状类型
        for symptom_type, mapping in cls.SYMPTOM_KEYWORDS.items():
            score = 0
            type_keywords = []
            
            # 匹配关键词
            for kw in mapping["keywords"]:
                if kw in input_lower:
                    score += 2
                    type_keywords.append(kw)
            
            # 匹配口语化描述
            for colloquial in mapping.get("colloquial", []):
                if colloquial in input_lower:
                    score += 1.5
                    type_keywords.append(colloquial)
            
            # 匹配医学术语
            for term in mapping.get("medical_terms", []):
                if term in input_lower:
                    score += 2.5
                    type_keywords.append(term)
            
            if score > max_score:
                max_score = score
                matched_type = symptom_type
                matched_keywords = type_keywords
        
        # 计算置信度
        if max_score >= 4:
            confidence = "高"
        elif max_score >= 2:
            confidence = "中"
        else:
            confidence = "低"
            matched_type = "未知"
        
        return {
            "symptom_type": matched_type,
            "confidence": confidence,
            "matched_keywords": matched_keywords,
            "is_emergency": is_emergency,
            "score": max_score
        }


# ============================================================================
# 会话超时管理机制
# ============================================================================

import time
from datetime import datetime, timedelta

class SessionTimeoutManager:
    """
    [WHY] 会话超时管理：防止长时间未操作的会话占用资源
    [HOW] 基于用户活动时间戳，自动超时重置
    [WARN] 默认超时30分钟，可配置
    """
    
    DEFAULT_TIMEOUT_MINUTES = 30
    WARNING_BEFORE_MINUTES = 5
    AUTO_SAVE_INTERVAL_SECONDS = 30
    
    def __init__(self, timeout_minutes: int = None):
        """
        Args:
            timeout_minutes: 超时时间（分钟），默认30
        """
        self.timeout_minutes = timeout_minutes or self.DEFAULT_TIMEOUT_MINUTES
        self.last_activity_time = datetime.now()
        self.session_start_time = datetime.now()
        self.warning_sent = False
        self.saved_data = {}
        
    def update_activity(self) -> None:
        """更新用户活动时间"""
        self.last_activity_time = datetime.now()
        self.warning_sent = False
        
    def check_timeout(self) -> Dict[str, Any]:
        """
        检查是否超时
        
        Returns:
            Dict: {
                "is_timeout": bool,
                "is_warning": bool,
                "remaining_seconds": int
            }
        """
        now = datetime.now()
        elapsed = now - self.last_activity_time
        remaining = timedelta(minutes=self.timeout_minutes) - elapsed
        
        is_timeout = elapsed >= timedelta(minutes=self.timeout_minutes)
        is_warning = (
            remaining <= timedelta(minutes=self.WARNING_BEFORE_MINUTES) 
            and remaining > timedelta(0)
            and not self.warning_sent
        )
        
        if is_warning:
            self.warning_sent = True
            
        return {
            "is_timeout": is_timeout,
            "is_warning": is_warning,
            "remaining_seconds": max(0, int(remaining.total_seconds())),
            "elapsed_minutes": int(elapsed.total_seconds() / 60)
        }
    
    def auto_save(self, data: Dict[str, Any]) -> None:
        """自动保存会话数据"""
        self.saved_data = {
            "data": data,
            "saved_at": datetime.now().isoformat()
        }
        
    def get_saved_data(self) -> Optional[Dict[str, Any]]:
        """获取保存的数据"""
        return self.saved_data.get("data")
    
    def reset(self) -> None:
        """重置会话计时器"""
        self.last_activity_time = datetime.now()
        self.session_start_time = datetime.now()
        self.warning_sent = False
        self.saved_data = {}


# ============================================================================
# 核心问题库补充 - 肝毒性、肾毒性
# ============================================================================

CORE_QUESTIONS["肝毒性"] = {
    "questions": [
        {"id": "jaundice", "text": "皮肤或眼睛有没有发黄？", "priority": 1},
        {"id": "appetite", "text": "食欲有没有下降？恶心吗？", "priority": 2},
        {"id": "liver_pain", "text": "右上腹有没有不适或疼痛？", "priority": 3},
        {"id": "lab_results", "text": "最近肝功能检查结果如何？（ALT/AST数值）", "priority": 4},
        {"id": "other_symptoms", "text": "有没有皮肤瘙痒、尿液变深或大便变白？", "priority": 5}
    ],
    "severity_keywords": {
        "G1": ["指标轻度升高", "无症状", "1-3倍"],
        "G2": ["指标中度升高", "轻度症状", "3-5倍"],
        "G3_G4": ["黄疸", "严重升高", "超过5倍", "意识模糊"]
    }
}

CORE_QUESTIONS["肾毒性"] = {
    "questions": [
        {"id": "urine_change", "text": "尿量有没有明显变化？（增多/减少）", "priority": 1},
        {"id": "edema", "text": "有没有水肿？在哪些部位？", "priority": 2},
        {"id": "fatigue", "text": "有没有明显乏力或食欲下降？", "priority": 3},
        {"id": "lab_results", "text": "最近肾功能检查结果如何？（肌酐数值）", "priority": 4},
        {"id": "other_symptoms", "text": "有没有腰痛、血尿或高血压？", "priority": 5}
    ],
    "severity_keywords": {
        "G1": ["肌酐轻度升高", "无症状", "1.5倍基线"],
        "G2": ["肌酐中度升高", "轻度水肿", "1.5-3倍"],
        "G3_G4": ["少尿", "严重水肿", "超过3倍", "需要透析"]
    }
}

# 免责声明模板
DISCLAIMER = """
---
**重要免责声明**：
本助手提供的所有信息仅供参考，不能替代专业医疗诊断和治疗建议。
- 本建议基于NCCN指南的一般性原则，具体情况因人而异
- 如症状持续加重或出现新的严重症状，请立即就医
- 在任何情况下，医疗团队的专业判断优先于本助手的建议
- 如有疑问，请直接联系您的主治医生或医疗团队
"""


# ============================================================================
# 状态定义 - 使用 TypedDict
# ============================================================================

class ConversationState(TypedDict):
    """
    [WHY] 定义对话状态结构，LangGraph 依赖此结构进行状态管理
    [HOW] 使用 TypedDict 提供类型提示和字段文档
    [WARN] 所有字段必须有默认值或在初始化时提供

    状态字段说明：
    - current_state: 当前所处的问诊阶段
    - symptoms: 已识别的症状列表
    - symptom_type: 主要症状类型（如"皮肤毒性"）
    - collected_info: 已收集的患者信息字典
    - questions_asked: 已提问的问题ID列表
    - answers: 患者回答字典 {question_id: answer}
    - risk_grade: 风险分级结果（G1-G4）
    - max_questions: 最大追问次数限制
    - question_count: 当前已追问次数
    - messages: 对话历史列表
    - current_question: 当前正在询问的问题
    - user_input: 用户最新输入
    - final_recommendation: 最终建议
    - waiting_for_supplement: 是否等待用户补充信息
    """
    current_state: str
    symptoms: List[str]
    symptom_type: Optional[str]
    collected_info: Dict[str, Any]
    questions_asked: List[str]
    answers: Dict[str, str]
    risk_grade: Optional[str]
    max_questions: int
    question_count: int
    messages: Annotated[List[Dict[str, str]], add]  # 使用 add 操作符累积消息
    current_question: Optional[str]
    user_input: str
    final_recommendation: Optional[str]
    waiting_for_supplement: bool


# ============================================================================
# 初始状态工厂函数
# ============================================================================

def create_initial_state(user_input: str = "", max_questions: int = 5) -> ConversationState:
    """
    [WHY] 创建初始状态，确保所有必需字段都有默认值
    [HOW] 返回一个完整的 ConversationState 字典
    [WARN] max_questions 控制追问深度，防止无限循环

    Args:
        user_input: 用户的初始症状描述
        max_questions: 最大追问次数，默认5次

    Returns:
        ConversationState: 初始化的对话状态
    """
    return ConversationState(
        current_state="initial",
        symptoms=[],
        symptom_type=None,
        collected_info={},
        questions_asked=[],
        answers={},
        risk_grade=None,
        max_questions=max_questions,
        question_count=0,
        messages=[],
        current_question=None,
        user_input=user_input,
        final_recommendation=None,
        waiting_for_supplement=False
    )


# ============================================================================
# 系统提示词 - 用于 LLM 调用
# ============================================================================

SYSTEM_PROMPT_V2 = """# 角色定位
你是免疫治疗患者教育助手 v2.0，专注于通过主动问诊帮助患者评估症状严重程度。

# 核心任务
1. 分析患者描述，识别症状类型
2. 根据症状类型提出针对性问题
3. 综合评估风险等级
4. 提供结构化的行动建议

# 严格限制
- 不能进行医学诊断
- 不能开具处方或推荐具体药物
- 所有建议必须保守，倾向于建议就医
- 必须基于收集的信息进行评估

# NCCN风险分级标准
## G1（轻度）
- 症状轻微，不影响日常生活
- 建议：继续治疗，观察等待
- 行动：保持与医疗团队的定期沟通

## G2（中度）
- 症状明显，部分影响日常活动
- 建议：可能需要暂停治疗，建议专科会诊
- 行动：48小时内联系医疗团队

## G3-G4（重度）
- 症状严重，明显影响生活质量或危及生命
- 建议：立即停止治疗，需要紧急医疗干预
- 行动：立即就医或拨打急救电话

# 回复格式要求
根据当前任务返回相应格式的JSON：

## 症状识别任务
```json
{
    "symptom_type": "症状类型",
    "symptoms": ["症状1", "症状2"],
    "confidence": "高/中/低"
}
```

## 风险评估任务
```json
{
    "risk_grade": "G1/G2/G3/G4",
    "reasoning": "评估依据",
    "urgent_signs": ["危险信号1", "危险信号2"]
}
```

## 建议生成任务
返回结构化的建议文本，包含：
1. 症状识别结果
2. 风险分级
3. 建议行动
4. 注意事项
5. 免责声明"""


# ============================================================================
# Agent 创建函数
# ============================================================================

def create_immuno_agent() -> Agent:
    """
    [WHY] 创建 Agent 实例，用于 LLM 推理
    [HOW] 配置 OpenAIChatClient + Agent
    [WARN] GitHub Models 有速率限制

    Returns:
        Agent: 配置好的免疫治疗助手实例
    """
    required_vars = ["GITHUB_TOKEN", "GITHUB_ENDPOINT", "GITHUB_MODEL_ID"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        raise ValueError(f"缺少必需的环境变量: {', '.join(missing)}")

    client = OpenAIChatClient(
        base_url=os.environ.get("GITHUB_ENDPOINT"),
        api_key=os.environ.get("GITHUB_TOKEN"),
        model_id=os.environ.get("GITHUB_MODEL_ID")
    )

    agent = Agent(
        name="ImmunoPatientAssistantV2",
        client=client,
        instructions=SYSTEM_PROMPT_V2,
        tools=[]
    )

    logger.info(f"Agent 已创建，使用模型: {os.environ.get('GITHUB_MODEL_ID')}")
    return agent


# ============================================================================
# LLM 辅助函数
# ============================================================================

async def call_llm_for_analysis(agent: Agent, prompt: str) -> str:
    """
    [WHY] 封装 LLM 调用，统一错误处理
    [HOW] 创建临时会话，发送提示，返回响应
    [WARN] 每次调用创建新会话，无上下文记忆

    Args:
        agent: Agent 实例
        prompt: 提示文本

    Returns:
        str: LLM 响应文本
    """
    try:
        session = agent.create_session()
        response = await agent.run(prompt, session=session)

        if response.messages and len(response.messages) > 0:
            last_message = response.messages[-1]
            if last_message.contents and len(last_message.contents) > 0:
                return last_message.contents[0].text

        raise RuntimeError("LLM 返回空响应")

    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        raise


# ============================================================================
# 状态节点函数实现
# ============================================================================

async def initial_inquiry(state: ConversationState) -> ConversationState:
    """
    [WHY] 初始问诊节点，分析用户输入识别症状类型
    [HOW] 调用 LLM 分析症状描述，提取症状类型和关键词
    [WARN] 如果无法识别症状类型，进入通用问诊流程

    状态转换：initial -> symptom_collection
    """
    logger.info(f"[节点] initial_inquiry - 用户输入: {state['user_input'][:50]}...")

    # [WHY] 创建 Agent 用于症状分析
    agent = create_immuno_agent()

    # [HOW] 构建分析提示
    analysis_prompt = f"""请分析以下患者症状描述，识别症状类型。

患者描述：
{state['user_input']}

请从以下症状类型中选择最匹配的一个：
- 皮肤毒性
- 胃肠道毒性
- 肺毒性
- 内分泌毒性
- 肝毒性
- 肾毒性

返回JSON格式：
{{
    "symptom_type": "症状类型",
    "symptoms": ["识别出的症状关键词"],
    "confidence": "高/中/低"
}}"""

    try:
        response = await call_llm_for_analysis(agent, analysis_prompt)

        # [HOW] 解析 LLM 响应，提取症状类型
        import json
        import re

        # 尝试提取 JSON
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                symptom_type = result.get("symptom_type", "未知")
                symptoms = result.get("symptoms", [])
            except json.JSONDecodeError:
                symptom_type = "未知"
                symptoms = []
        else:
            # [WARN] 无法解析 JSON，使用关键词匹配
            symptom_type = quick_symptom_classification(state['user_input']) or "未知"
            symptoms = []

        # [HOW] 更新状态
        state["symptom_type"] = symptom_type
        state["symptoms"] = symptoms
        state["current_state"] = "symptom_collection"

        # [HOW] 记录初始信息
        state["collected_info"]["initial_description"] = state["user_input"]

        # [HOW] 添加欢迎消息 - 根据症状类型决定是否提问
        welcome_msg = f"您好，我是免疫治疗患者教育助手。\n\n"
        welcome_msg += f"根据您的描述，我初步判断可能与【{symptom_type}】相关。\n\n"
        
        # [WHY] 只有当症状类型在问题库中时才承诺提问
        if symptom_type in CORE_QUESTIONS:
            welcome_msg += f"为了更准确地评估您的情况，我需要问您几个问题。"
        else:
            welcome_msg += f"我将基于您的描述进行初步评估。"

        state["messages"].append({
            "role": "assistant",
            "content": welcome_msg
        })

        logger.info(f"[节点] initial_inquiry 完成 - 症状类型: {symptom_type}")

    except Exception as e:
        logger.error(f"初始问诊失败: {e}")
        state["symptom_type"] = "未知"
        state["current_state"] = "symptom_collection"
        state["messages"].append({
            "role": "assistant",
            "content": "您好，我是免疫治疗患者教育助手。请描述您的症状，我将为您提供帮助。"
        })

    return state


async def symptom_collection(state: ConversationState) -> ConversationState:
    """
    [WHY] 症状收集节点，根据症状类型选择问题
    [HOW] 从核心问题库中选择下一个未问的问题
    [WARN] 如果问题已问完或无法识别症状类型，跳转到评估

    状态转换：
    - 有问题 -> detail_clarification
    - 无问题 -> risk_assessment
    """
    logger.info(f"[节点] symptom_collection - 症状类型: {state['symptom_type']}")

    symptom_type = state["symptom_type"]

    # [WHY] 检查是否有对应的问题库
    if symptom_type not in CORE_QUESTIONS:
        logger.warning(f"未找到 {symptom_type} 的问题库，跳转到评估")
        # [HOW] 添加过渡消息，请求用户补充信息
        state["messages"].append({
            "role": "assistant",
            "content": f"\n由于【{symptom_type}】暂无标准问题库，请您补充以下信息以帮助评估：\n"
                       f"• 症状持续时间\n"
                       f"• 症状严重程度\n"
                       f"• 是否影响日常生活\n"
                       f"• 有无其他伴随症状\n"
        })
        # [HOW] 标记为等待补充信息状态
        state["waiting_for_supplement"] = True
        state["current_state"] = "risk_assessment"
        return state

    # [HOW] 获取问题列表
    questions = CORE_QUESTIONS[symptom_type]["questions"]

    # [HOW] 找到下一个未问的问题
    next_question = None
    for q in sorted(questions, key=lambda x: x["priority"]):
        if q["id"] not in state["questions_asked"]:
            next_question = q
            break

    if next_question is None:
        # [WARN] 所有问题已问完，进入评估
        logger.info("所有问题已问完，进入风险评估")
        state["current_state"] = "risk_assessment"
        return state

    # [HOW] 设置当前问题
    state["current_question"] = next_question["text"]
    state["questions_asked"].append(next_question["id"])
    state["current_state"] = "detail_clarification"

    # [HOW] 添加问题消息
    question_msg = f"\n问题 {len(state['questions_asked'])}/{len(questions)}: {next_question['text']}"
    state["messages"].append({
        "role": "assistant",
        "content": question_msg
    })

    logger.info(f"[节点] symptom_collection - 选择问题: {next_question['id']}")

    return state


async def detail_clarification(state: ConversationState) -> ConversationState:
    """
    [WHY] 细节澄清节点，作为流程中转站
    [HOW] 此节点不执行具体逻辑，由条件边决定下一步
    [WARN] 此节点不再记录回答，回答由 continue_conversation 方法处理

    状态转换：由 should_ask_more_questions 条件边决定
    """
    logger.info(f"[节点] detail_clarification - 当前问题数: {state['question_count']}")
    
    # [WHY] 此节点只作为流程中转，实际逻辑由条件边处理
    return state


async def risk_assessment(state: ConversationState) -> ConversationState:
    """
    [WHY] 风险评估节点，综合所有信息进行风险分级
    [HOW] 调用 LLM 分析收集的信息，给出风险等级
    [WARN] 如果信息不足，给出保守的 G2 评估

    状态转换：risk_assessment -> recommendation
    """
    logger.info("[节点] risk_assessment - 开始风险评估")

    # [HOW] 构建评估上下文 - 使用完整描述（包括补充信息）
    # [WHY] Bug修复：确保补充信息被纳入评估
    full_description = state['collected_info'].get('full_description', 
                                                    state['collected_info'].get('initial_description', '无'))
    
    context = f"""## 患者完整描述
{full_description}

## 症状类型
{state['symptom_type']}

## 已收集信息
"""
    for q_id, answer in state["answers"].items():
        context += f"- {q_id}: {answer}\n"
    
    # [HOW] 添加补充信息（如果有）
    if state['collected_info'].get('supplement'):
        context += f"\n## 用户补充信息\n{state['collected_info'].get('supplement')}\n"

    # [HOW] 添加严重程度关键词参考
    if state["symptom_type"] in CORE_QUESTIONS:
        severity_keywords = CORE_QUESTIONS[state["symptom_type"]]["severity_keywords"]
        context += f"\n## 严重程度参考关键词\n"
        context += f"- G1 (轻度): {', '.join(severity_keywords['G1'])}\n"
        context += f"- G2 (中度): {', '.join(severity_keywords['G2'])}\n"
        context += f"- G3-G4 (重度): {', '.join(severity_keywords['G3_G4'])}\n"

    # [HOW] 调用 LLM 进行评估
    agent = create_immuno_agent()

    assessment_prompt = f"""请根据以下患者信息进行风险评估。

{context}

请返回JSON格式：
{{
    "risk_grade": "G1/G2/G3/G4",
    "reasoning": "评估依据（简要说明为什么是这个等级）",
    "urgent_signs": ["需要警惕的危险信号"]
}}"""

    try:
        response = await call_llm_for_analysis(agent, assessment_prompt)

        # [HOW] 解析评估结果
        import json
        import re

        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                state["risk_grade"] = result.get("risk_grade", "G2")
                reasoning = result.get("reasoning", "信息不足，保守评估")
                urgent_signs = result.get("urgent_signs", [])
            except json.JSONDecodeError:
                state["risk_grade"] = "G2"
                reasoning = "无法解析评估结果，保守评估"
                urgent_signs = []
        else:
            state["risk_grade"] = "G2"
            reasoning = "评估结果格式异常，保守评估"
            urgent_signs = []

        # [HOW] 保存评估详情
        state["collected_info"]["risk_assessment"] = {
            "grade": state["risk_grade"],
            "reasoning": reasoning,
            "urgent_signs": urgent_signs
        }

        state["current_state"] = "recommendation"

        logger.info(f"[节点] risk_assessment 完成 - 风险等级: {state['risk_grade']}")

    except Exception as e:
        logger.error(f"风险评估失败: {e}")
        state["risk_grade"] = "G2"
        state["current_state"] = "recommendation"

    return state


async def recommendation(state: ConversationState) -> ConversationState:
    """
    [WHY] 建议生成节点，根据风险等级生成行动建议
    [HOW] 调用 LLM 生成结构化的建议文本
    [WARN] 必须包含免责声明

    状态转换：recommendation -> END
    """
    logger.info(f"[节点] recommendation - 风险等级: {state['risk_grade']}")

    # [HOW] 构建建议生成提示
    context = f"""## 患者信息
症状类型: {state['symptom_type']}
风险等级: {state['risk_grade']}

## 收集的症状信息
"""
    for q_id, answer in state["answers"].items():
        context += f"- {q_id}: {answer}\n"

    # [HOW] 获取评估详情
    assessment = state["collected_info"].get("risk_assessment", {})
    if assessment:
        context += f"\n## 评估依据\n{assessment.get('reasoning', '无')}\n"
        if assessment.get("urgent_signs"):
            context += f"\n## 需警惕的危险信号\n"
            for sign in assessment["urgent_signs"]:
                context += f"- {sign}\n"

    # [HOW] 调用 LLM 生成建议
    agent = create_immuno_agent()

    recommendation_prompt = f"""请根据以下患者信息生成行动建议。

{context}

请生成包含以下部分的建议：
1. **症状识别**：简要说明症状类型和主要表现
2. **风险分级**：说明风险等级及其含义
3. **建议行动**：
   - 紧急程度说明
   - 具体行动建议
   - 就医时间建议
4. **注意事项**：需要观察的危险信号
5. **免责声明**：提醒本建议仅供参考"""

    try:
        response = await call_llm_for_analysis(agent, recommendation_prompt)

        # [HOW] 组装最终建议
        final_recommendation = response + DISCLAIMER
        state["final_recommendation"] = final_recommendation

        # [HOW] 添加最终建议消息
        state["messages"].append({
            "role": "assistant",
            "content": final_recommendation
        })

        logger.info("[节点] recommendation 完成 - 建议已生成")

    except Exception as e:
        logger.error(f"建议生成失败: {e}")
        # [WARN] 生成默认建议
        default_recommendation = generate_default_recommendation(state)
        state["final_recommendation"] = default_recommendation
        state["messages"].append({
            "role": "assistant",
            "content": default_recommendation
        })

    state["current_state"] = "end"
    return state


# ============================================================================
# 辅助函数
# ============================================================================

def quick_symptom_classification(description: str) -> Optional[str]:
    """
    [WHY] 基于关键词的快速症状分类，用于 LLM 解析失败时的备用
    [HOW] 遍历症状关键词进行匹配
    [WARN] 仅用于辅助参考

    Args:
        description: 症状描述

    Returns:
        Optional[str]: 症状类型或 None
    """
    description_lower = description.lower()

    symptom_keywords = {
        "皮肤毒性": ["皮疹", "痒", "皮肤", "疹", "脱皮", "斑", "红"],
        "胃肠道毒性": ["腹泻", "拉肚子", "腹痛", "肚子痛", "便血", "恶心", "呕吐"],
        "肺毒性": ["咳嗽", "呼吸困难", "气短", "胸闷", "喘", "肺炎"],
        "内分泌毒性": ["甲状腺", "乏力", "疲劳", "体重", "情绪", "内分泌"],
        "肝毒性": ["肝", "黄疸", "食欲", "皮肤发黄", "眼睛发黄"],
        "肾毒性": ["肾", "尿", "水肿", "浮肿", "肌酐"]
    }

    for symptom_type, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in description_lower:
                return symptom_type

    return None


def generate_default_recommendation(state: ConversationState) -> str:
    """
    [WHY] 生成默认建议，用于 LLM 调用失败时的备用
    [HOW] 基于风险等级生成模板化建议
    [WARN] 保守策略，倾向于建议就医

    Args:
        state: 当前状态

    Returns:
        str: 默认建议文本
    """
    risk_grade = state.get("risk_grade", "G2")
    symptom_type = state.get("symptom_type", "未知")

    recommendations = {
        "G1": f"""
**症状识别**：您报告的症状可能与【{symptom_type}】相关，目前表现为轻度。

**风险分级**：G1（轻度）
- 症状轻微，不影响日常生活

**建议行动**：
- 紧急程度：低
- 继续当前治疗，密切观察症状变化
- 记录症状日记，包括出现时间、持续时间、严重程度
- 下次复诊时向医疗团队报告

**注意事项**：
- 如果症状加重或出现新症状，请及时联系医疗团队
""",
        "G2": f"""
**症状识别**：您报告的症状可能与【{symptom_type}】相关，目前表现为中度。

**风险分级**：G2（中度）
- 症状明显，可能影响日常活动

**建议行动**：
- 紧急程度：中等
- 建议在48小时内联系医疗团队
- 可能需要暂停治疗或调整方案
- 准备好症状记录以便医生评估

**注意事项**：
- 密切观察症状变化
- 如症状加重，请立即就医
""",
        "G3": f"""
**症状识别**：您报告的症状可能与【{symptom_type}】相关，目前表现为重度。

**风险分级**：G3（重度）
- 症状严重，明显影响生活质量

**建议行动**：
- 紧急程度：高
- 请立即联系医疗团队或前往医院
- 可能需要暂停治疗并进行紧急处理
- 不要自行用药或等待症状自行缓解

**注意事项**：
- 如出现危及生命的症状，请立即拨打急救电话
- 准备好您的治疗记录和用药清单
""",
        "G4": f"""
**症状识别**：您报告的症状可能与【{symptom_type}】相关，目前表现为危及生命。

**风险分级**：G4（危及生命）
- 需要紧急医疗干预

**建议行动**：
- 紧急程度：极高
- 请立即就医或拨打急救电话（120）
- 停止当前治疗
- 在就医途中保持冷静，记录症状

**注意事项**：
- 这是医疗紧急情况，请立即行动
- 不要延误就医时间
"""
    }

    return recommendations.get(risk_grade, recommendations["G2"]) + DISCLAIMER


# ============================================================================
# 状态转换条件函数
# ============================================================================

def should_continue_collecting(state: ConversationState) -> str:
    """
    [WHY] 判断是否继续收集信息
    [HOW] 检查问题是否问完、是否达到最大问题数
    [WARN] 返回下一个节点名称

    Returns:
        str: 下一个节点名称
    """
    # [HOW] 检查是否已识别症状类型
    if state["symptom_type"] == "未知" or state["symptom_type"] not in CORE_QUESTIONS:
        return "risk_assessment"

    # [HOW] 检查是否达到最大问题数
    if state["question_count"] >= state["max_questions"]:
        return "risk_assessment"

    # [HOW] 检查是否还有未问的问题
    questions = CORE_QUESTIONS[state["symptom_type"]]["questions"]
    asked_count = len(state["questions_asked"])

    if asked_count >= len(questions):
        return "risk_assessment"

    return "detail_clarification"


def should_ask_more_questions(state: ConversationState) -> str:
    """
    [WHY] 判断是否需要更多追问
    [HOW] 基于当前收集的信息量和问题数量
    [WARN] 保守策略：信息不足时继续追问

    Returns:
        str: 下一个节点名称
    """
    # [HOW] 检查是否已达到最大问题数
    if state["question_count"] >= state["max_questions"]:
        logger.info(f"[条件边] 已达最大问题数 ({state['question_count']}/{state['max_questions']})，进入评估")
        return "risk_assessment"

    # [HOW] 检查是否还有未问的问题
    symptom_type = state["symptom_type"]
    if symptom_type not in CORE_QUESTIONS:
        logger.info(f"[条件边] 症状类型 {symptom_type} 无问题库，进入评估")
        return "risk_assessment"

    questions = CORE_QUESTIONS[symptom_type]["questions"]
    asked_count = len(state["questions_asked"])

    if asked_count >= len(questions):
        logger.info(f"[条件边] 所有问题已问完 ({asked_count}/{len(questions)})，进入评估")
        return "risk_assessment"

    logger.info(f"[条件边] 继续追问 ({asked_count}/{len(questions)})")
    return "symptom_collection"


# ============================================================================
# LangGraph 图构建
# ============================================================================

def build_conversation_graph():
    """
    [WHY] 构建问诊流程的状态机图
    [HOW] 使用 LangGraph 的 StateGraph 定义节点和边
    [WARN] 图结构决定了问诊流程的逻辑

    图结构：
    ```
    START -> initial_inquiry -> symptom_collection
                                   |
                                   v
                            detail_clarification
                                   |
                                   v
                            risk_assessment -> recommendation -> END
    ```

    Returns:
        CompiledGraph: 编译后的可执行图
    """
    # [WHY] 创建状态图
    workflow = StateGraph(ConversationState)

    # [HOW] 添加节点
    workflow.add_node("initial_inquiry", initial_inquiry)
    workflow.add_node("symptom_collection", symptom_collection)
    workflow.add_node("detail_clarification", detail_clarification)
    workflow.add_node("risk_assessment", risk_assessment)
    workflow.add_node("recommendation", recommendation)

    # [HOW] 设置入口点
    workflow.set_entry_point("initial_inquiry")

    # [HOW] 添加边 - 初始问诊后进入症状收集
    workflow.add_edge("initial_inquiry", "symptom_collection")

    # [HOW] 添加条件边 - 症状收集后判断是否继续
    workflow.add_conditional_edges(
        "symptom_collection",
        should_continue_collecting,
        {
            "detail_clarification": "detail_clarification",
            "risk_assessment": "risk_assessment"
        }
    )

    # [HOW] 添加条件边 - 细节澄清后判断是否继续追问
    workflow.add_conditional_edges(
        "detail_clarification",
        should_ask_more_questions,
        {
            "symptom_collection": "symptom_collection",
            "risk_assessment": "risk_assessment"
        }
    )

    # [HOW] 添加边 - 风险评估后进入建议生成
    workflow.add_edge("risk_assessment", "recommendation")

    # [HOW] 添加边 - 建议生成后结束
    workflow.add_edge("recommendation", END)

    # [WHY] 编译图，添加内存检查点和中断机制
    # [HOW] MemorySaver 用于保存中间状态，支持断点续传
    # [HOW] interrupt_after 在 symptom_collection 节点后暂停，等待用户输入
    # [WARN] interrupt_after 是实现多轮追问的关键，确保每次提问后暂停
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["symptom_collection"]  # 关键：在提问后暂停
    )

    logger.info("LangGraph 对话图已构建完成（启用 interrupt_after）")
    return app


# ============================================================================
# 对话管理器类
# ============================================================================

class ImmunoPatientAssistantV2:
    """
    [WHY] 封装对话管理逻辑，提供简洁的 API
    [HOW] 使用 LangGraph 图管理对话流程
    [WARN] 每个实例维护独立的对话状态

    使用示例：
    ```python
    assistant = ImmunoPatientAssistantV2()
    response = await assistant.start_conversation("我最近有皮疹")
    print(response)

    # 继续对话
    response = await assistant.continue_conversation("皮疹出现3天了")
    print(response)
    ```
    """

    def __init__(self, max_questions: int = 5):
        """
        [WHY] 初始化对话管理器
        [HOW] 创建 LangGraph 图和初始状态
        [WARN] max_questions 控制最大追问深度

        Args:
            max_questions: 最大追问次数
        """
        self.graph = build_conversation_graph()
        self.max_questions = max_questions
        self.current_state: Optional[ConversationState] = None
        self.thread_id = "default"

        logger.info(f"ImmunoPatientAssistantV2 初始化完成，最大问题数: {max_questions}")

    async def start_conversation(self, user_input: str) -> str:
        """
        [WHY] 开始新对话，处理用户初始输入
        [HOW] 创建初始状态，运行图直到 interrupt 点（symptom_collection 后）
        [WARN] 返回助手的第一个响应（包含第一个问题）

        Args:
            user_input: 用户的初始症状描述

        Returns:
            str: 助手的响应
        """
        # [HOW] 创建初始状态
        self.current_state = create_initial_state(user_input, self.max_questions)

        # [HOW] 配置线程ID，用于状态持久化
        self.config = {"configurable": {"thread_id": self.thread_id}}

        # [WHY] 执行到第一个 interrupt 点（symptom_collection 节点后）
        # [HOW] ainvoke 会在 symptom_collection 后暂停，返回当前状态
        result = await self.graph.ainvoke(self.current_state, self.config)
        self.current_state = result

        # [HOW] 提取助手响应
        return self._get_last_assistant_message()

    async def continue_conversation(self, user_input: str) -> str:
        """
        [WHY] 继续对话，处理用户回答
        [HOW] 使用 update_state 更新状态，然后用 ainvoke(None) 从暂停点恢复
        [WARN] 关键：必须用 ainvoke(None) 而非 ainvoke(state) 才能从暂停点恢复

        Args:
            user_input: 用户的回答

        Returns:
            str: 助手的响应
        """
        if self.current_state is None:
            return "请先开始对话"

        # [HOW] 检查是否已结束
        if self.current_state.get("current_state") == "end":
            return self.current_state.get("final_recommendation", "对话已结束")

        # [WHY] 记录用户回答
        updates = {"user_input": user_input}
        
        # [WHY] Bug修复：处理补充信息 - 需要将补充信息追加到原始描述
        if self.current_state.get("waiting_for_supplement"):
            # [HOW] 将补充信息追加到原始描述，形成完整的症状描述
            original_description = self.current_state["collected_info"].get("initial_description", "")
            supplemented_description = f"{original_description}\n补充信息：{user_input}"
            updates["collected_info"] = {
                **self.current_state["collected_info"],
                "initial_description": original_description,
                "supplement": user_input,
                "full_description": supplemented_description
            }
            updates["waiting_for_supplement"] = False
            logger.info(f"[continue_conversation] 补充信息已追加: {user_input[:50]}...")
        
        # [WHY] 处理标准问题的回答
        elif self.current_state["questions_asked"]:
            last_question_id = self.current_state["questions_asked"][-1]
            updates["answers"] = {**self.current_state["answers"], last_question_id: user_input}
            updates["collected_info"] = {**self.current_state["collected_info"], last_question_id: user_input}
            updates["question_count"] = self.current_state["question_count"] + 1
            logger.info(f"[continue_conversation] 记录回答: {last_question_id} -> {user_input[:30]}...")

        # [HOW] 添加用户回答消息
        updates["messages"] = [{"role": "user", "content": user_input}]

        # [WHY] 关键：使用 update_state 更新状态，然后用 ainvoke(None) 恢复
        await self.graph.aupdate_state(self.config, updates)

        # [WHY] 从暂停点恢复执行，直到下一个 interrupt 或 END
        result = await self.graph.ainvoke(None, self.config)
        self.current_state = result

        # [HOW] 提取助手响应
        return self._get_last_assistant_message()

    def _get_last_assistant_message(self) -> str:
        """
        [WHY] 提取最后一条助手消息
        [HOW] 从消息列表中查找最后一条 role=assistant 的消息
        [WARN] 如果没有找到，返回默认消息

        Returns:
            str: 最后一条助手消息
        """
        if not self.current_state or not self.current_state.get("messages"):
            return "抱歉，我无法处理您的请求。"

        for message in reversed(self.current_state["messages"]):
            if message.get("role") == "assistant":
                return message.get("content", "")

        return "抱歉，我无法处理您的请求。"

    def is_conversation_ended(self) -> bool:
        """
        [WHY] 检查对话是否已结束
        [HOW] 检查当前状态是否为 end

        Returns:
            bool: 对话是否已结束
        """
        return self.current_state is not None and self.current_state.get("current_state") == "end"

    def reset(self) -> None:
        """
        [WHY] 重置对话状态，开始新的问诊
        [HOW] 清空当前状态，更新线程ID
        [WARN] 比重新创建实例更轻量，保留已编译的图
        
        使用场景：
        - 对话结束后自动重置
        - 用户输入 'new' 命令
        """
        self.current_state = None
        self.thread_id = f"thread_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info("对话状态已重置，可以开始新的问诊")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        [WHY] 获取对话摘要，用于日志或调试
        [HOW] 提取关键信息

        Returns:
            Dict[str, Any]: 对话摘要
        """
        if self.current_state is None:
            return {"status": "未开始"}

        return {
            "status": self.current_state.get("current_state", "未知"),
            "symptom_type": self.current_state.get("symptom_type"),
            "risk_grade": self.current_state.get("risk_grade"),
            "questions_asked": len(self.current_state.get("questions_asked", [])),
            "messages_count": len(self.current_state.get("messages", []))
        }


# ============================================================================
# 主程序入口
# ============================================================================

async def main_async() -> None:
    """
    [WHY] 异步主函数，提供交互式命令行界面
    [HOW] 使用 ImmunoPatientAssistantV2 进行多轮对话
    [WARN] 每次对话独立，支持多轮追问
    """
    print("\n" + "=" * 70)
    print("       免疫治疗患者教育助手 v2.0 - 主动引导版")
    print("       Immuno-Oncology Patient Education Assistant")
    print("=" * 70)
    print("\n本助手通过主动问诊帮助您评估免疫治疗相关症状。")
    print("我将根据您的描述提出针对性问题，然后给出建议。")
    print("\n输入 'quit' 或 'exit' 退出程序")
    print("输入 'help' 查看使用说明")
    print("输入 'new' 开始新的问诊")
    print("-" * 70 + "\n")

    try:
        # [WHY] 创建对话管理器
        assistant = ImmunoPatientAssistantV2(max_questions=5)

        while True:
            try:
                # [WHY] 检查对话状态，决定提示语
                if assistant.current_state is None:
                    prompt = "请描述您的症状"
                elif assistant.is_conversation_ended():
                    # [HOW] 对话结束后自动重置，无需用户输入 'new'
                    print("\n" + "-" * 50)
                    print("问诊已完成。")
                    print("-" * 50 + "\n")
                    assistant.reset()
                    prompt = "请描述您的症状（或输入 'quit' 退出）"
                elif assistant.current_state.get("waiting_for_supplement"):
                    # [HOW] 等待用户补充信息
                    prompt = "请补充上述信息"
                elif assistant.current_state.get("current_question"):
                    # [HOW] 等待回答具体问题
                    prompt = "请回答上述问题"
                else:
                    prompt = "请继续描述"

                # [HOW] 获取用户输入
                user_input = input(f"{prompt}: ").strip()

                # [WHY] 处理空输入
                if not user_input:
                    print("请输入内容，或输入 'help' 查看帮助。\n")
                    continue

                # [WHY] 处理退出命令
                if user_input.lower() in ("quit", "exit", "q"):
                    print("\n感谢使用免疫治疗患者教育助手。祝您健康！\n")
                    break

                # [WHY] 处理帮助命令
                if user_input.lower() == "help":
                    print_help()
                    continue

                # [WHY] 处理新对话命令
                if user_input.lower() == "new":
                    assistant.reset()
                    print("\n已开始新的问诊会话。请描述您的症状。\n")
                    continue

                # [WHY] 处理对话
                if assistant.current_state is None:
                    # [HOW] 开始新对话
                    print("\n正在分析您的症状...\n")
                    response = await assistant.start_conversation(user_input)
                else:
                    # [HOW] 继续对话
                    response = await assistant.continue_conversation(user_input)

                print(response)

            except KeyboardInterrupt:
                print("\n\n程序已中断。感谢使用！\n")
                break

            except Exception as e:
                logger.error(f"对话处理错误: {e}")
                print(f"\n处理您的请求时发生错误: {e}")
                print("请稍后重试或输入 'new' 开始新的问诊。\n")

    except ValueError as e:
        logger.error(f"配置错误: {e}")
        print(f"\n错误: {e}")
        print("请检查环境变量配置后重试。\n")
        exit(1)

    except Exception as e:
        logger.error(f"未预期的错误: {e}")
        print(f"\n发生错误: {e}")
        print("请稍后重试或联系技术支持。\n")
        exit(1)


def print_help() -> None:
    """
    [WHY] 打印帮助信息
    [HOW] 显示使用说明和示例
    """
    print("\n" + "=" * 60)
    print("使用说明 - 免疫治疗患者教育助手 v2.0")
    print("=" * 60)
    print("""
本助手采用主动问诊模式，将根据您的症状描述提出针对性问题。

【问诊流程】
1. 首先描述您的症状（如：我最近有皮疹）
2. 助手会识别症状类型并提出相关问题
3. 请如实回答每个问题
4. 收集足够信息后，助手将给出风险评估和建议

【可识别的症状类型】
- 皮肤毒性：皮疹、瘙痒、皮肤干燥等
- 胃肠道毒性：腹泻、腹痛、便血等
- 肺毒性：咳嗽、呼吸困难、胸闷等
- 内分泌毒性：乏力、体重变化、情绪改变等
- 肝毒性：黄疸、食欲下降、肝区不适等
- 肾毒性：尿量变化、水肿等

【示例对话】
用户: 我最近有皮疹，有点痒
助手: 您好，我初步判断可能与【皮肤毒性】相关。
      问题 1/5: 皮疹是什么时候开始的？

用户: 3天前开始的
助手: 问题 2/5: 皮疹大约占身体面积的多少？

用户: 大约手掌大小
助手: ...（继续提问或给出评估）

【注意事项】
- 本助手仅提供教育信息，不能替代医疗诊断
- 严重症状请立即就医
- 保守建议策略：倾向于建议就医而非自我处理
""")
    print("=" * 60 + "\n")


def main() -> None:
    """
    [WHY] 同步入口点，包装异步主函数
    [HOW] 使用 asyncio.run() 运行异步主函数
    [WARN] Python 3.7+ 才支持 asyncio.run()
    """
    asyncio.run(main_async())


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    # [WHY] 加载 .env 文件中的环境变量
    load_dotenv()
    main()


# ============================================================================
# Mental Model Recap
# ============================================================================
"""
本文件实现了免疫治疗患者教育助手的 2.0 版本，核心升级是从单轮问答变为多轮主动引导式问诊。

关键设计决策：
1. 状态机架构：使用 LangGraph 管理问诊流程，状态包括 initial_inquiry -> symptom_collection
   -> detail_clarification -> risk_assessment -> recommendation
2. 主动追问策略：根据症状类型从 CORE_QUESTIONS 库中选择针对性问题，最多追问 5 次
3. 风险评估：综合收集的信息调用 LLM 进行 NCCN 标准分级（G1-G4）
4. 安全保守原则：信息不足时默认 G2 评估，始终倾向于建议就医

技术栈组合：
- LangGraph：状态机框架，管理对话流程和状态转换
- Agent Framework：Agent + OpenAIChatClient，封装 LLM 调用
- TypedDict：类型安全的状态定义

潜在改进方向：
- 添加多症状并行评估能力
- 支持对话历史持久化
- 集成医疗知识图谱增强评估准确性
"""
