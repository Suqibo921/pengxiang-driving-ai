"""
RAG 核心引擎
=============
负责将检索到的知识库内容与 LLM 回答结合。
支持通义千问、DeepSeek、豆包 三模型切换。
"""

import json
from typing import List, Dict, Optional
from http import HTTPStatus

# 通义千问 SDK
from dashscope import Generation as DashScopeGeneration

# DeepSeek / 豆包 共用 OpenAI SDK
from openai import OpenAI


# ============================================
# Prompt 模板
# ============================================

SYSTEM_PROMPT = """你是西安鹏翔驾校的资深招生顾问"小影"，你的核心任务是：通过热情专业的介绍，让咨询的学员产生报名冲动，并主动拨打你的电话 15609130011 来进一步咨询或报名。

## 你的核心目标
- 每一次回答都要让学员觉得"这家驾校靠谱，想报名"
- 在对话中自然地引导学员，让学员主动想打电话或来校报名
- 你的电话是 15609130011，在合适的时机主动提供给学员

## 回答规则
1. 必须基于知识库内容回答，不要编造信息
2. 语气亲切、热情、有感染力，像朋友推荐好东西一样
3. 主动介绍驾校的优势和亮点，不要等学员一个一个问
4. 价格、流程等关键信息要准确，让学员觉得透明放心
5. 当学员表现出兴趣或犹豫时，主动提供电话让学员打电话咨询

## 促单技巧
- 介绍优势时用"我们"拉近距离："我们驾校有..."
- 适当制造紧迫感："最近报名的人很多哦"
- 主动邀约参观："要不您打我电话 15609130011，我开车去接您，带您实地看看咱们的训练场和考试场地，您亲眼看看放心"
- 打消顾虑："您放心，我们XX方面都是正规的"

## 邀约参观（核心转化手段）
当学员表现出对训练场、考场、环境、练车等的兴趣或犹豫时，要主动邀请学员实地参观：
1. 强调"来看一看比听我说强多了"——实地参观是促成报名的关键一步
2. 主动提出专车接送——"您打我电话 15609130011，我安排车去接您"
3. 描述参观亮点——"我们的训练场是按考场1:1还原的，您来了就知道有多规范"
4. 参观是免费的，没有压力——"就是来看看，合适再报名，不合适的您扭头就走，没关系的"

## 知识库内容
{context}"""


class RAGEngine:
    """RAG 核心引擎，连接检索和生成。"""

    def __init__(self, vector_store, config: dict):
        """
        初始化 RAG 引擎。

        参数:
            vector_store: VectorStore 实例
            config: 包含 LLM 配置的字典
        """
        self.vector_store = vector_store
        self.config = config
        self._init_llm()

    def _init_llm(self):
        """根据配置初始化 LLM 客户端。"""
        provider = self.config.get("llm_provider", "qwen")

        if provider == "qwen":
            api_key = self.config.get("dashscope_api_key", "")
            if api_key and api_key != "YOUR_DASHSCOPE_API_KEY_HERE":
                import dashscope
                dashscope.api_key = api_key
                self.llm_ready = True
            else:
                print("⚠️ 通义千问 API Key 未配置，将使用模拟模式")
                self.llm_ready = False

        elif provider == "deepseek":
            api_key = self.config.get("deepseek_api_key", "")
            if api_key and api_key != "YOUR_DEEPSEEK_API_KEY_HERE":
                self.deepseek_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                self.llm_ready = True
            else:
                print("⚠️ DeepSeek API Key 未配置，将使用模拟模式")
                self.llm_ready = False

        elif provider == "doubao":
            api_key = self.config.get("doubao_api_key", "")
            base_url = self.config.get("doubao_base_url", "https://ark.cn-beijing.volces.com/api/v3")
            if api_key and api_key != "YOUR_DOUBAO_API_KEY_HERE":
                self.doubao_client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                self.llm_ready = True
                print("✅ 豆包 API 客户端已初始化")
            else:
                print("⚠️ 豆包 API Key 未配置，将使用模拟模式")
                self.llm_ready = False

    def _build_messages(self, question: str, context: str, history: str = "") -> List[Dict]:
        """
        构建消息列表（system + user 格式）。

        参数:
            question: 用户问题
            context: 检索到的知识库上下文
            history: 历史对话（预留）

        返回:
            消息列表，格式为 [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        # 填充系统提示中的上下文
        if context.strip():
            system_content = SYSTEM_PROMPT.format(context=context)
        else:
            system_content = SYSTEM_PROMPT.format(context="（暂无相关知识库内容）")

        # 如果有历史对话，加入用户消息中
        if history and history.strip():
            user_content = f"【历史对话】\n{history}\n\n【新问题】\n{question}"
        else:
            user_content = question

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def _call_qwen(self, messages: List[Dict]) -> str:
        """调用通义千问。"""
        try:
            response = DashScopeGeneration.call(
                model=self.config.get("qwen_model_name", "qwen3-max"),
                messages=messages,
                result_format="message"
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                return f"❌ API 调用失败: {response.message}"
        except Exception as e:
            return f"❌ API 调用异常: {str(e)}"

    def _call_deepseek(self, messages: List[Dict]) -> str:
        """调用 DeepSeek。"""
        try:
            response = self.deepseek_client.chat.completions.create(
                model=self.config.get("deepseek_model_name", "deepseek-chat"),
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ API 调用异常: {str(e)}"

    def _call_doubao(self, messages: List[Dict]) -> str:
        """调用豆包（火山引擎）。"""
        try:
            response = self.doubao_client.chat.completions.create(
                model=self.config.get("doubao_model_name", "doubao-1.5-pro-32k"),
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ API 调用异常: {str(e)}"

    def _simulate_answer(self, context: str, question: str) -> str:
        """模拟回答（API Key 未配置时使用）。"""
        if not context.strip():
            return "这个我需要确认一下，稍后给您回复哦～"

        question_lower = question.lower()
        context_lower = context.lower()

        if "价格" in question or "多少钱" in question or "费用" in question:
            lines = [l for l in context.split("\n") if "元" in l or "费" in l]
            if lines:
                return f"根据我们驾校的信息：\n\n" + "\n".join(lines[:5]) + "\n\n具体详情欢迎来校咨询或拨打 13772409494 了解哦 😊"
        elif "地址" in question or "在哪" in question or "位置" in question:
            return f"我们鹏翔驾校总部在西安市未央区石化大道与楼尤路东北角，另外还有南校区在长安区韦斗路西段。你可以坐班车过来，也可以预约上门接送～"
        elif "电话" in question or "联系" in question:
            return f"我们的服务热线是 13772409494，QQ 是 846530883，也可以直接来校咨询哦～"
        elif "科目" in question or "考试" in question:
            exams = [l for l in context.split("\n") if "科目" in l or "考试" in l or "及格" in l]
            if exams:
                return f"关于考试这边给您说一下：\n\n" + "\n".join(exams[:6])
        else:
            return f"根据我们驾校的信息：\n\n{context[:500]}\n\n如果还有其他问题，随时问我哦 😊"

    def answer(self, question: str, history: str = "", top_k: int = 5) -> Dict:
        """
        根据问题生成回答。

        参数:
            question: 用户的问题
            history: 历史对话记录（用于第二阶段记忆功能）
            top_k: 检索相关文档数量

        返回:
            {
                "answer": "AI 的回答",
                "sources": ["来源文件1", "来源文件2", ...],
                "context": "检索到的知识库原文片段"
            }
        """
        # 1. 检索相关知识
        retrieved = self.vector_store.search(question, top_k=top_k)

        # 2. 拼接上下文
        context_parts = []
        sources = []
        seen_sources = set()
        for r in retrieved:
            context_parts.append(f"【{r['source']}】\n{r['text']}")
            if r["source"] not in seen_sources:
                sources.append(r["source"])
                seen_sources.add(r["source"])

        context = "\n\n".join(context_parts)

        # 3. 构建消息列表
        messages = self._build_messages(question, context, history)

        # 4. 调用 LLM
        provider = self.config.get("llm_provider", "qwen")

        if not self.llm_ready:
            answer_text = self._simulate_answer(context, question)
        elif provider == "qwen":
            answer_text = self._call_qwen(messages)
        elif provider == "deepseek":
            answer_text = self._call_deepseek(messages)
        elif provider == "doubao":
            answer_text = self._call_doubao(messages)
        else:
            answer_text = self._simulate_answer(context, question)

        return {
            "answer": answer_text,
            "sources": sources,
            "context": context
        }