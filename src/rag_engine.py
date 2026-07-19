"""
RAG 核心引擎
=============
负责将检索到的知识库内容与 LLM 回答结合。
使用豆包（火山引擎）API。
"""

import os
import re
from typing import List, Dict
from openai import OpenAI


# ============================================
# Prompt 模板
# ============================================

SYSTEM_PROMPT = """你是西安鹏翔驾校的资深招生顾问"小影"，你的核心任务是：通过热情专业的介绍，让咨询的学员产生报名冲动，并主动拨打你的电话 15609130011 来进一步咨询或报名。

## 你的核心目标
- 让学员觉得"这家驾校靠谱，想报名"
- 在对话中自然地引导学员，让学员主动想打电话或来校报名
- 你的电话是 15609130011，在合适的时机提供给学员

## 回答规则
1. 必须基于知识库内容回答，不要编造信息
2. 语气亲切、热情、有感染力，像朋友推荐好东西一样
3. 主动介绍驾校的优势和亮点，不要等学员一个一个问
4. 价格、流程等关键信息要准确，让学员觉得透明放心

## 对话节奏控制（重要）
- 前2-3轮对话：以介绍和解答为主，不要主动推电话
- 仅在以下时机提供电话和邀约参观：
  * 顾客询问价格/费用时
  * 顾客询问训练场/场地环境时
  * 顾客询问班车路线时
  * 顾客询问考试流程或报名时
  * 顾客表达犹豫或不确定时
  * 顾客表示感兴趣想进一步了解时
- 电话引出的方式要自然："要不您打我电话 15609130011，我详细跟您说"
- 不要在一段话开头就放电话，放在回答的后半段或结尾

## 照片展示规则
- 当顾客第一次表现出对鹏翔驾校的兴趣时（比如"给我介绍一下驾校"、"你们驾校怎么样"、"想了解一下你们"），在回答中嵌入 [图片:beiyuan_01] 等标记
- 注意：图片标记只嵌入一次，后续对话不要再重复嵌入
- 图片标记嵌入在回答中介绍驾校环境的部分，不要放在开头或结尾

## 班车路线推荐流程
当顾客询问班车时：
1. 先问顾客在哪个区
2. 顾客说出区后，列出该区覆盖的站点让顾客选择
3. 顾客选站点 → 告知对应路线
4. 如果顾客不确定 → 提供几个选项让顾客选
5. 顾客仍不确定 → 引导打 15609130011 小影人工安排

## 促单技巧
- 介绍优势时用"我们"拉近距离："我们驾校有..."
- 适当制造紧迫感："最近报名的人很多哦"
- 打消顾虑："您放心，我们XX方面都是正规的"

## 邀约参观
当学员表现出对训练场、考场、环境、练车等的兴趣或犹豫时，可以邀请学员实地参观：
1. 强调"来看一看比听我说强多了"
2. 提出专车接送——"您打我电话 15609130011，我安排车去接您"
3. 描述参观亮点——"我们的训练场是按考场1:1还原的"
4. 参观是免费的，没有压力——"就是来看看，合适再报名"

## 知识库内容
{context}"""


class RAGEngine:
    """RAG 核心引擎，连接检索和生成。"""

    def __init__(self, vector_store, config: dict):
        self.vector_store = vector_store
        self.config = config
        self._init_llm()

    def _init_llm(self):
        """初始化豆包 LLM 客户端。"""
        api_key = self.config.get("doubao_api_key", "")
        base_url = self.config.get("doubao_base_url", "https://ark.cn-beijing.volces.com/api/v3")
        if api_key and api_key != "YOUR_DOUBAO_API_KEY_HERE":
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.llm_ready = True
            print("✅ 豆包 API 客户端已初始化")
        else:
            print("⚠️ 豆包 API Key 未配置，将使用模拟模式")
            self.llm_ready = False

    def _build_messages(self, question: str, context: str, history: str = "") -> List[Dict]:
        if context.strip():
            system_content = SYSTEM_PROMPT.format(context=context)
        else:
            system_content = SYSTEM_PROMPT.format(context="（暂无相关知识库内容）")

        if history and history.strip():
            user_content = f"【历史对话】\n{history}\n\n【新问题】\n{question}"
        else:
            user_content = question

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def _call_doubao(self, messages: List[Dict]) -> str:
        """调用豆包（火山引擎）。"""
        try:
            response = self.client.chat.completions.create(
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

        # 判断是否询问价格
        if "价格" in question or "多少钱" in question or "费用" in question:
            # 优先查找暑期优惠价信息
            promo_lines = [l for l in context.split("\n") if "优惠" in l or "暑期" in l]
            price_lines = [l for l in context.split("\n") if "元" in l or "费" in l]
            total_lines = [l for l in context.split("\n") if "总费用" in l or "4054" in l or "3954" in l]

            parts = []
            if promo_lines:
                parts.append("🔥 **暑期优惠价**")
                parts.extend(promo_lines[:3])
            if price_lines:
                parts.extend(price_lines[:5])
            if total_lines:
                parts.extend(total_lines[:2])

            content = "\n".join(parts) if parts else "\n".join(price_lines[:5])
            return (
                f"咱们鹏翔驾校现在暑期优惠，价格很划算哦！\n\n"
                f"{content}\n\n"
                f"我们是全城一费制，所有费用上墙公示，签订正规合同，后期没有任何隐形消费。"
                f"科二科三补考费全部包含，还有考前模拟，挂科免费复训，您完全不用担心额外费用！\n\n"
                f"要不您打我电话 15609130011，我详细给您介绍一下，合适的话还可以安排车免费接您来实地看看～"
            )

        # 判断是否询问照片/图片/环境
        if any(kw in question for kw in ["照片", "图片", "看看", "环境", "场地", "训练场", "校区"]):
            return (
                "当然可以！给您看看我们鹏翔驾校的实景照片～\n\n"
                "【图片:beiyuan_01】\n\n"
                "这是我们**北校区（总部）**的训练场地，千亩级全封闭独立训练园区，"
                "场地都是按考场1:1还原的，练车就是摸考场！\n\n"
                "【图片:nanyuan_01】\n\n"
                "这是**南校区**，位于长安区新韦斗路，环境优美，训练设施齐全，"
                "423亩的大场地，单人单车随到随学，不用排队等车。\n\n"
                "【图片:qinhan_01】\n\n"
                "这是**秦汉考务中心**，我们的自有考场，考试好预约，通过率高！\n\n"
                "照片看着不错吧？不过实地来看更震撼！您打我电话 15609130011，"
                "我安排车免费接您来参观，合适再报名，完全没有压力～"
            )

        # 判断是否询问地址/位置
        if "地址" in question or "在哪" in question or "位置" in question:
            return (
                "我们鹏翔驾校目前有 **四个校区**、**两个考场**：\n\n"
                "1️⃣ **南校区**：长安区新韦斗路\n"
                "2️⃣ **汉城湖公园校区**：汉城湖公园北门\n"
                "3️⃣ **秦汉新城考务中心**\n"
                "4️⃣ **沣西训练场地**\n\n"
                "全城还有20+直营门店，覆盖西安主要区域，您到哪个店都行！\n\n"
                "平时练车我们有免费大巴车接送，一天三趟，20条线路覆盖全西安市、"
                "咸阳、秦汉、沣西，您在哪都方便～\n\n"
                "要不您打我电话 15609130011，我给您安排最近的路线！"
            )

        if "电话" in question or "联系" in question:
            return f"我的电话是 15609130011，欢迎随时来电咨询哦～"

        if "科目" in question or "考试" in question:
            exams = [l for l in context.split("\n") if "科目" in l or "考试" in l or "及格" in l]
            if exams:
                return f"关于考试这边给您说一下：\n\n" + "\n".join(exams[:8])

        if "班车" in question or "接送" in question or "怎么去" in question:
            return (
                "咱们驾校有 **免费大巴车接送** 服务哦！\n\n"
                "一天三趟班车，20条线路覆盖全西安市、咸阳、秦汉、沣西，"
                "点对点上门接送，您不用自己奔波通勤。\n\n"
                "您方便告诉我您在哪个区吗？我帮您看看附近有没有合适的站点～"
            )

        # 默认回答
        # 截取最相关的上下文段落
        lines = context.split("\n")
        # 去重并保留有意义的行
        meaningful = [l for l in lines if len(l.strip()) > 5 and not l.startswith("#")]
        shown = meaningful[:8] if meaningful else lines[:10]
        context_text = "\n".join(shown)

        return (
            f"根据我们驾校的信息：\n\n{context_text}\n\n"
            f"如果您想了解更多详情，欢迎随时问我，或者直接打我电话 15609130011，"
            f"我安排车免费接您来实地参观，合适再报名，完全没有压力哦～"
        )

    def answer(self, question: str, history: str = "", top_k: int = 5) -> Dict:
        """
        根据问题生成回答。

        返回:
            {
                "answer": "AI 的回答",
                "sources": ["来源文件1", ...],
                "context": "检索到的知识库原文片段"
            }
        """
        # 1. 检索
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

        # 3. 构建消息并调用 LLM
        messages = self._build_messages(question, context, history)

        if self.llm_ready:
            answer_text = self._call_doubao(messages)
        else:
            answer_text = self._simulate_answer(context, question)

        return {
            "answer": answer_text,
            "sources": sources,
            "context": context
        }