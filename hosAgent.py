"""
免疫治疗患者教育助手 (Immuno-Oncology Patient Education Assistant)
================================================================

[WHY] 帮助免疫治疗患者理解和管理治疗相关副作用，提供基于NCCN指南的初步建议
[HOW] 使用 Agent Framework + OpenAIChatClient 实现单轮问答式症状评估
[WARN] 本助手仅提供教育信息，不能替代医疗诊断，严重症状必须立即就医

核心设计原则：
- 单轮问答模式：不保存对话历史，每次交互独立
- 保守建议策略：倾向于建议就医而非自我处理
- NCCN分级参考：基于国际指南的风险评估框架
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

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
logger = logging.getLogger("ImmunoPatientAssistant")

# ============================================================================
# 系统提示词常量
# ============================================================================

SYSTEM_PROMPT = """# 角色定位
你是免疫治疗患者教育助手，专注于帮助患者理解和管理免疫治疗相关症状。

# 核心任务
1. 评估患者报告症状的严重程度
2. 提供基于NCCN指南的初步建议
3. 指导患者采取适当的行动

# 严格限制
- 不能进行医学诊断
- 不能开具处方或推荐具体药物
- 所有建议必须保守，倾向于建议就医
- 不主动追问，不进行复杂推理
- 单轮问答模式，无记忆功能

# NCCN风险分级说明
## G1（轻度）
- 定义：症状轻微，不影响日常生活
- 建议：继续治疗，观察等待，记录症状变化
- 行动：保持与医疗团队的定期沟通

## G2（中度）
- 定义：症状明显，部分影响日常活动
- 建议：可能需要暂停治疗，建议专科会诊
- 行动：48小时内联系医疗团队

## G3-G4（重度）
- 定义：症状严重，明显影响生活质量或危及生命
- 建议：立即停止治疗，需要紧急医疗干预
- 行动：立即就医或拨打急救电话

# 免疫相关副作用参考信息

## 皮肤毒性
- 常见症状：皮疹、瘙痒、皮肤干燥、白癜风样改变
- G1表现：皮疹面积<10%体表面积，轻微瘙痒
- G2表现：皮疹面积10-30%，中度瘙痒，影响睡眠
- G3-G4表现：皮疹面积>30%，严重瘙痒，皮肤剥脱

## 胃肠道毒性
- 常见症状：腹泻、腹痛、便血、结肠炎
- G1表现：每日腹泻<4次，无其他症状
- G2表现：每日腹泻4-6次，轻度腹痛
- G3-G4表现：每日腹泻≥7次，严重腹痛，便血

## 肺毒性
- 常见症状：咳嗽、呼吸困难、胸闷、发热
- G1表现：影像学异常但无症状
- G2表现：轻度呼吸困难，活动后加重
- G3-G4表现：严重呼吸困难，静息时也明显

## 内分泌毒性
- 常见症状：乏力、体重变化、情绪改变、甲状腺功能异常
- G1表现：无症状的实验室指标异常
- G2表现：轻度症状，需要激素替代
- G3-G4表现：严重症状，需要紧急处理

## 肝毒性
- 常见症状：乏力、食欲下降、黄疸、肝区不适
- G1表现：AST/ALT 1-3倍正常上限
- G2表现：AST/ALT 3-5倍正常上限
- G3-G4表现：AST/ALT >5倍正常上限，黄疸

## 肾毒性
- 常见症状：尿量变化、水肿、乏力
- G1表现：肌酐升高1.5倍基线
- G2表现：肌酐升高1.5-3倍基线
- G3-G4表现：肌酐升高>3倍基线，需要透析

# 回复格式要求
每次回复必须包含以下结构：

1. **症状识别**：简要说明患者描述的症状属于哪类免疫相关副作用
2. **风险分级**：基于NCCN标准给出初步分级（G1-G4）
3. **建议行动**：
   - 紧急程度说明
   - 具体行动建议
   - 就医时间建议
4. **注意事项**：需要观察的危险信号
5. **免责声明**：提醒本建议仅供参考，不能替代医疗诊断

# 重要提醒
- 对于任何G3-G4级别症状，必须强烈建议立即就医
- 对于不确定的症状描述，建议联系医疗团队确认
- 始终强调患者安全第一的原则"""

# ============================================================================
# 免疫相关副作用数据（内置知识库）
# ============================================================================

IMMUNE_RELATED_ADVERSE_EVENTS: Dict[str, Dict[str, Any]] = {
    "皮肤毒性": {
        "symptoms": ["皮疹", "瘙痒", "皮肤干燥", "白癜风", "皮肤色素改变"],
        "keywords": ["皮肤", "疹", "痒", "皮", "斑", "红", "脱皮"],
        "severity_indicators": {
            "G1": ["轻微", "轻度", "不影响", "小范围"],
            "G2": ["明显", "中度", "影响睡眠", "面积增大"],
            "G3_G4": ["严重", "大面积", "水泡", "剥脱", "溃烂"]
        }
    },
    "胃肠道毒性": {
        "symptoms": ["腹泻", "腹痛", "结肠炎", "便血", "恶心呕吐"],
        "keywords": ["腹泻", "拉肚子", "腹痛", "肚子痛", "便血", "结肠", "恶心", "呕吐"],
        "severity_indicators": {
            "G1": ["每日少于4次", "轻微", "轻度"],
            "G2": ["每日4-6次", "中度", "影响活动"],
            "G3_G4": ["每日7次以上", "严重", "便血", "剧烈"]
        }
    },
    "肺毒性": {
        "symptoms": ["肺炎", "呼吸困难", "咳嗽", "胸闷", "气短"],
        "keywords": ["咳嗽", "呼吸困难", "气短", "胸闷", "肺炎", "喘", "呼吸"],
        "severity_indicators": {
            "G1": ["轻微咳嗽", "活动正常", "无症状"],
            "G2": ["活动后气短", "轻度呼吸困难"],
            "G3_G4": ["静息时呼吸困难", "严重", "缺氧", "需要吸氧"]
        }
    },
    "内分泌毒性": {
        "symptoms": ["甲状腺功能异常", "乏力", "体重变化", "情绪改变", "内分泌紊乱"],
        "keywords": ["甲状腺", "乏力", "疲劳", "体重", "情绪", "内分泌", "甲亢", "甲减"],
        "severity_indicators": {
            "G1": ["无症状", "指标轻度异常"],
            "G2": ["轻度症状", "需要药物"],
            "G3_G4": ["严重症状", "危象", "意识改变"]
        }
    },
    "肝毒性": {
        "symptoms": ["肝炎", "黄疸", "食欲下降", "肝功能异常"],
        "keywords": ["肝", "黄疸", "食欲", "皮肤发黄", "眼睛发黄", "转氨酶"],
        "severity_indicators": {
            "G1": ["指标轻度升高", "无症状"],
            "G2": ["指标中度升高", "轻度症状"],
            "G3_G4": ["黄疸", "严重升高", "意识模糊"]
        }
    },
    "肾毒性": {
        "symptoms": ["肾炎", "尿量变化", "水肿", "肾功能异常"],
        "keywords": ["肾", "尿", "水肿", "肌酐", "浮肿", "尿量"],
        "severity_indicators": {
            "G1": ["指标轻度升高", "无症状"],
            "G2": ["指标中度升高", "轻度水肿"],
            "G3_G4": ["少尿", "严重水肿", "需要透析"]
        }
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
# 环境验证函数
# ============================================================================

def validate_environment() -> None:
    """
    [WHY] 启动时验证配置，避免运行时才发现问题
    [HOW] 检查必需的环境变量是否存在
    [WARN] 不验证 token 有效性，只检查存在性

    Raises:
        ValueError: 缺少必需的环境变量时抛出
    """
    required_vars = ["GITHUB_TOKEN", "GITHUB_ENDPOINT", "GITHUB_MODEL_ID"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        raise ValueError(
            f"缺少必需的环境变量: {', '.join(missing)}。"
            f"请检查您的 .env 文件配置。"
        )
    logger.info("环境变量验证通过")


# ============================================================================
# Agent 创建函数
# ============================================================================

def create_immuno_patient_agent() -> Agent:
    """
    [WHY] 使用工厂函数创建 Agent，便于测试和依赖注入
    [HOW] 配置 OpenAIChatClient + Agent 组合，注入专业系统提示词
    [WARN] GitHub Models 使用 gpt-4o-mini 等模型，注意模型能力差异

    [学习重点] Agent Framework 架构：
    - OpenAIChatClient: 模型客户端，负责与 LLM API 通信
    - Agent: 智能体，封装对话逻辑和专业领域知识
    - 无需 Session: 单轮问答模式，不保持对话上下文

    Returns:
        Agent: 配置好的免疫治疗患者教育助手实例
    """
    validate_environment()

    # [WHY] 创建与 GitHub Models 兼容的 OpenAI 客户端
    # [HOW] 使用环境变量配置端点、密钥和模型
    # [WARN] GitHub Models 免费层有速率限制
    client = OpenAIChatClient(
        base_url=os.environ.get("GITHUB_ENDPOINT"),
        api_key=os.environ.get("GITHUB_TOKEN"),
        model_id=os.environ.get("GITHUB_MODEL_ID")
    )

    # [WHY] 创建专业化的医疗教育助手
    # [HOW] 注入详细的系统提示词和内置知识库
    # [WARN] 系统提示词较长，确保模型支持足够的上下文窗口
    agent = Agent(
        name="ImmunoPatientAssistant",
        client=client,
        instructions=SYSTEM_PROMPT,
        tools=[]  # [WHY] 无需工具，纯对话式问答
    )

    logger.info(f"免疫治疗患者教育助手已创建，使用模型: {os.environ.get('GITHUB_MODEL_ID')}")
    return agent


# ============================================================================
# 症状评估函数
# ============================================================================

async def assess_symptom(
    patient_description: str,
    agent: Agent
) -> str:
    """
    [WHY] 核心症状评估函数，为患者提供初步建议
    [HOW] 将患者描述发送给 Agent，获取结构化评估响应
    [WARN] 每次调用创建新会话，确保单轮问答无记忆

    Args:
        patient_description: 患者对症状的文字描述
        agent: 免疫治疗患者教育助手实例

    Returns:
        str: 结构化的评估建议，包含分级和行动指南

    Raises:
        ValueError: 输入为空时抛出
        RuntimeError: API 调用失败时抛出
    """
    # [WHY] 输入验证，防止空消息导致 API 错误
    if not patient_description or not patient_description.strip():
        raise ValueError("症状描述不能为空")

    try:
        # [WHY] 记录请求日志，便于问题追踪
        logger.info(f"收到症状评估请求: {patient_description[:100]}...")

        # [WHY] 每次创建新会话，实现单轮问答无记忆
        # [HOW] Agent.create_session() 创建独立会话
        # [WARN] 不保存会话引用，用完即弃
        session = agent.create_session()

        # [WHY] 构建带上下文的提示，帮助模型理解用户意图
        # [HOW] 添加角色引导和时间戳
        prompt = f"""患者报告时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}

患者症状描述:
{patient_description}

请根据上述症状描述，提供评估建议。"""

        # [WHY] 调用 Agent 进行症状评估
        # [HOW] 使用异步 API 提高并发性能
        response = await agent.run(prompt, session=session)

        # [WHY] 验证响应有效性
        if not response.messages or len(response.messages) == 0:
            raise RuntimeError("Agent 返回空消息")

        last_message = response.messages[-1]
        if not last_message.contents or len(last_message.contents) == 0:
            raise RuntimeError("Agent 返回空内容")

        # [WHY] 提取文本响应
        text_content = last_message.contents[0].text

        # [WHY] 追加免责声明
        full_response = text_content + DISCLAIMER

        logger.info(f"症状评估完成，响应长度: {len(full_response)} 字符")
        return full_response

    except Exception as e:
        logger.error(f"症状评估失败: {e}")
        raise RuntimeError(f"无法完成症状评估: {e}") from e


# ============================================================================
# 辅助函数：快速症状分类
# ============================================================================

def quick_symptom_classification(description: str) -> Optional[str]:
    """
    [WHY] 提供快速的症状分类，用于前端预处理或日志记录
    [HOW] 基于关键词匹配识别可能的副作用类型
    [WARN] 仅用于辅助参考，不替代 AI 评估

    Args:
        description: 患者症状描述

    Returns:
        Optional[str]: 识别出的副作用类型，未识别返回 None
    """
    description_lower = description.lower()

    for irae_type, data in IMMUNE_RELATED_ADVERSE_EVENTS.items():
        for keyword in data["keywords"]:
            if keyword in description_lower:
                return irae_type

    return None


# ============================================================================
# 主程序入口
# ============================================================================

async def main_async() -> None:
    """
    [WHY] 异步主函数，提供交互式命令行界面
    [HOW] 循环读取用户输入，调用症状评估函数
    [WARN] 每次评估独立，无对话历史
    """
    print("\n" + "=" * 60)
    print("       免疫治疗患者教育助手")
    print("       Immuno-Oncology Patient Education Assistant")
    print("=" * 60)
    print("\n本助手帮助您理解和管理免疫治疗相关症状。")
    print("请描述您的症状，我将提供基于NCCN指南的初步建议。")
    print("\n输入 'quit' 或 'exit' 退出程序")
    print("输入 'help' 查看使用说明")
    print("-" * 60 + "\n")

    try:
        # [WHY] 创建 Agent 实例（只需创建一次）
        agent = create_immuno_patient_agent()

        while True:
            try:
                # [WHY] 获取用户输入
                user_input = input("请描述您的症状: ").strip()

                # [WHY] 处理空输入
                if not user_input:
                    print("请输入症状描述，或输入 'help' 查看帮助。\n")
                    continue

                # [WHY] 处理退出命令
                if user_input.lower() in ("quit", "exit", "q"):
                    print("\n感谢使用免疫治疗患者教育助手。祝您健康！\n")
                    break

                # [WHY] 处理帮助命令
                if user_input.lower() == "help":
                    print("\n" + "=" * 50)
                    print("使用说明:")
                    print("-" * 50)
                    print("1. 请详细描述您的症状，包括:")
                    print("   - 症状类型（如皮疹、腹泻、呼吸困难等）")
                    print("   - 症状严重程度")
                    print("   - 持续时间")
                    print("   - 是否影响日常生活")
                    print("\n2. 示例描述:")
                    print("   - '我最近出现皮疹，有点痒，范围不大'")
                    print("   - '每天腹泻5次，伴有轻微腹痛'")
                    print("   - '呼吸困难，走路都喘'")
                    print("\n3. 注意事项:")
                    print("   - 本助手仅提供教育信息")
                    print("   - 不能替代医疗诊断")
                    print("   - 严重症状请立即就医")
                    print("=" * 50 + "\n")
                    continue

                # [WHY] 快速分类提示
                quick_class = quick_symptom_classification(user_input)
                if quick_class:
                    print(f"[识别] 可能与 {quick_class} 相关\n")

                # [WHY] 调用症状评估
                print("正在评估您的症状...\n")
                assessment = await assess_symptom(user_input, agent)
                print(assessment)

            except KeyboardInterrupt:
                print("\n\n程序已中断。感谢使用！\n")
                break

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
