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

SYSTEM_PROMPT = """你是西安鹏翔驾校的资深招生顾问"小影"，你的核心任务是：热情专业地解答学员的问题，让学员信任鹏翔驾校。

## 核心原则
1. 必须基于知识库内容回答，不要编造信息
2. 语气亲切、热情、自然，像朋友推荐一样
3. 价格、流程等关键信息要准确清晰
4. 主动介绍驾校优势，但不要过度推销

## 电话推广策略（严格执行）
- **前2轮对话：绝对不要主动提供电话 15609130011**
- 第3轮起，仅在以下情况可以在回答末尾自然提供电话：
  * 顾客主动问联系方式
  * 顾客说"想报名"、"想参观"、"想看看"
  * 顾客表示犹豫不决，你说"要不打我电话..."
- 电话必须放在回答的**最后一句**，不能放在开头或中间
- 如果顾客只是问价格、问班车、问流程，**不要推电话**，直接回答即可

## 照片展示规则（重要）
- 照片标记 [图片:beiyuan_01] 等**仅限**在以下场景使用：
  * 顾客问"你们驾校环境怎么样？"时
  * 顾客问"有照片吗？""有图片吗？"时
  * 顾客问"训练场/校区什么样子？"时
- 回答价格、考试流程、报名、班车等问题时**不要**嵌入照片标记
- 照片标记只在一个会话中嵌入一次，一次最多嵌入3张照片

## 班车路线推荐流程
当顾客询问班车时：
1. 先问顾客在哪个区
2. 顾客说出区后，直接列出该区覆盖的具体站点
3. 如果顾客进一步问具体路线，告知对应线路编号
4. 如果顾客不确定推荐哪个站点，提供几个选项让顾客选
5. 回答要具体，比如"雁塔区有小寨、大雁塔、电视塔等站点"

## 回答风格
- 用"我们"拉近距离
- 信息要具体，不要模糊
- 价格要准确说出数字
- 回答完主要问题后，自然过渡到"还有什么想了解的吗？"
- 不要每段话都推销，真诚解答最重要

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

    def _simulate_answer(self, context: str, question: str, turn_count: int = 0) -> str:
        """模拟回答（API Key 未配置时使用）。
        
        使用真实检索到的知识库上下文来回答，避免硬编码。
        turn_count: 当前对话轮数，用于控制电话推广时机。
        """
        if not context.strip():
            return "这个我需要确认一下，稍后给您回复哦～"

        # 是否应该推电话：顾客明确表达兴趣时才推
        should_offer_phone = (
            turn_count >= 2 and (
                "价格" in question or "多少钱" in question or "费用" in question or
                "报名" in question or "怎么去" in question or
                ("看" in question and ("场地" in question or "环境" in question or "校区" in question)) or
                "犹豫" in question or "不确定" in question or "考虑" in question or
                "想了解" in question or "想报" in question or "想学" in question
            )
        )

        # 提取上下文中的关键信息用于构建回答
        lines = context.split("\n")
        # 去掉标题行和空行
        text_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("##") and not l.strip().startswith("###")]

        # ===== 价格查询 =====
        if "价格" in question or "多少钱" in question or "费用" in question:
            # 从上下文中提取价格信息
            price_info = []
            for line in text_lines:
                if any(kw in line for kw in ["元", "费", "优惠", "暑期", "总费用", "C1", "C2", "包含", "不含"]):
                    price_info.append(line)
            # 去重取前8条
            seen = set()
            unique_price = []
            for p in price_info:
                if p not in seen:
                    seen.add(p)
                    unique_price.append(p)
            price_content = "\n".join(unique_price[:8]) if unique_price else "\n".join(text_lines[:6])

            # 检查是否包含摩托车信息
            is_motorcycle = "摩托" in question or "D" in question or "E" in question or "F" in question

            if is_motorcycle:
                return (
                    f"有的哦亲～我们驾校有摩托车（D/E/F）驾照培训，"
                    f"现在暑假有专属特价活动，报名费只要499元，总费用大概813元，"
                    f"包含了报名费、考试费、体检费等全部相关费用。\n\n"
                    f"训练场地在石化总校和秦汉考务中心，拿证周期大概2周左右。\n\n"
                    f"如果您想了解更多细节或者想报名，要不打我电话15609130011，"
                    f"我详细给您介绍呀～"
                )

            # 构建价格回答
            # 寻找C1/C2的具体价格
            c1_price = ""
            c2_price = ""
            c1_total = ""
            c2_total = ""
            for line in text_lines:
                if "C1" in line and "手动" in line:
                    c1_price = line
                elif "C2" in line and "自动" in line:
                    c2_price = line
                elif "C1：" in line or "C1：" in line or "C1:" in line or "C1:" in line:
                    c1_total = line
                elif "C2：" in line or "C2：" in line or "C2:" in line or "C2:" in line:
                    c2_total = line

            answer = f"咱们鹏翔驾校现在暑期优惠，价格很划算哦！\n\n"
            if c1_price:
                answer += f"🔥 C1手动挡暑期优惠价：{c1_price}\n"
            else:
                answer += "🔥 C1手动挡暑期优惠价：3490元\n"
            if c2_price:
                answer += f"🔥 C2自动挡暑期优惠价：{c2_price}\n"
            else:
                answer += "🔥 C2自动挡暑期优惠价：3390元\n"
            answer += "\n费用包含：报名费、受理费、保险费、计时培训费、科二科三所有补考费和考前模拟\n"
            if c1_total:
                answer += f"\n📊 C1总费用：{c1_total}"
            else:
                answer += "\n📊 C1总费用：3490+520+44=4054元"
            if c2_total:
                answer += f"\n📊 C2总费用：{c2_total}"
            else:
                answer += "\n📊 C2总费用：3390+520+44=3954元"
            answer += "\n\n我们是全城一费制，所有费用上墙公示，签订正规合同，后期没有任何隐形消费。"

            if should_offer_phone:
                answer += "\n\n要不您打我电话15609130011，我详细给您介绍一下，合适的话还可以安排车免费接您来实地看看～"
            else:
                answer += "\n\n如果您想了解更多，随时问我哦～"

            return answer

        # ===== 照片/环境查询 =====
        if any(kw in question for kw in ["照片", "图片", "看看", "环境", "场地", "训练场", "校区"]):
            # 从context中提取校区描述
            campus_lines = []
            for line in text_lines:
                if any(kw in line for kw in ["校区", "场地", "训练", "环境", "北校区", "南校区", "汉城湖", "秦汉", "沣西"]):
                    campus_lines.append(line)

            # 构建回答，使用照片标记
            answer = "当然可以！给您看看我们鹏翔驾校的实景照片～\n\n"
            answer += "【图片:beiyuan_01】\n\n"
            answer += "这是我们**北校区（总部）**，位于未央区石化大道，千亩级全封闭独立训练园区，按考场1:1还原，生活设施配套齐全。\n\n"
            answer += "【图片:nanyuan_01】\n\n"
            answer += "这是我们**南校区**，位于长安区新韦斗路，环境优美，训练设施齐全，423亩大场地，单人单车随到随学。\n\n"
            answer += "【图片:qinhan_01】\n\n"
            answer += "这是**秦汉考务中心**，我们的自有考场，考试好预约，通过率高！\n\n"

            if should_offer_phone:
                answer += "实地来看更震撼！您打我电话15609130011，我安排车免费接您来参观，合适再报名，完全没有压力～"
            else:
                answer += "照片看着不错吧？实地来看更震撼哦～"

            return answer

        # ===== 地址/位置查询 =====
        if "地址" in question or "在哪" in question or "位置" in question:
            answer = "我们鹏翔驾校目前有 **四个校区**、**两个考场**：\n\n"
            answer += "1️⃣ **南校区**：长安区新韦斗路\n"
            answer += "2️⃣ **汉城湖公园校区**：汉城湖公园北门\n"
            answer += "3️⃣ **秦汉新城考务中心**\n"
            answer += "4️⃣ **沣西训练场地**\n\n"
            answer += "全城还有20+直营门店，覆盖西安主要区域，您到哪个店都行！\n\n"
            answer += "平时练车我们有免费大巴车接送，一天三趟，20条线路覆盖全西安市、咸阳、秦汉、沣西。"

            if should_offer_phone:
                answer += "\n\n要不您打我电话15609130011，我给您安排最近的路线！"
            return answer

        # ===== 电话/联系 =====
        if "电话" in question or "联系" in question:
            return "您随时可以拨打我的电话 15609130011，也可以直接来我们驾校实地参观，我安排车去接您～"

        # ===== 考试流程 =====
        if "科目" in question or "考试" in question:
            exam_lines = [l for l in text_lines if "科目" in l or "考试" in l or "及格" in l or "学时" in l or "约" in l]
            if exam_lines:
                result = "关于考试这边给您说一下：\n\n"
                for line in exam_lines[:10]:
                    result += f"• {line}\n"
                if should_offer_phone:
                    result += "\n如果您想了解更详细的考试安排，打我电话15609130011，我给您详细说～"
                return result

        # ===== 班车/接送 =====
        if "班车" in question or "接送" in question or "怎么去" in question:
            # 提取区域覆盖信息
            area_lines = []
            for line in text_lines:
                if any(kw in line for kw in ["覆盖", "站点", "线路", "区", "线", "班车"]):
                    area_lines.append(line)

            area_info = "\n".join(area_lines[:6]) if area_lines else ""

            # 检查是否提到了具体区域
            has_area = any(qu in question for qu in ["雁塔", "长安", "未央", "新城", "碑林", "莲湖", "灞桥", "曲江", "经开", "高新", "咸阳", "秦汉", "沣西", "泾渭", "高陵"])

            if has_area:
                # 构建针对具体区域的回答
                area_name = ""
                for qu in ["雁塔区", "雁塔", "长安区", "长安", "未央区", "未央", "碑林区", "碑林", "莲湖区", "莲湖", "灞桥区", "灞桥", "新城区", "新城", "曲江新区", "曲江", "经开区", "经开", "高新区", "高新", "咸阳", "秦汉", "沣西", "泾渭", "高陵"]:
                    if qu in question:
                        area_name = qu
                        break

                # 从上下文找该区域的具体站点
                specific_stations = []
                for line in text_lines:
                    if area_name in line or any(q in line for q in ["站点", "线路", "线"]):
                        specific_stations.append(line)

                answer = f"有的呀！咱们的班车线路覆盖{area_name}地区哦～\n\n"
                if specific_stations:
                    for s in specific_stations[:5]:
                        answer += f"• {s}\n"
                else:
                    answer += "您方便告诉我具体在哪个位置吗？我帮您看看附近有没有合适的站点～\n"
                    answer += "我们班车覆盖雁塔区丈八北路、南二环、科技路等站点，长安区有韦曲南、长安街等，未央区有龙首村、方新村等。\n"

                if should_offer_phone:
                    answer += "\n要是您想确认离您最近的具体站点和发车时间，打我电话15609130011，我给您详细说～"
                else:
                    answer += "\n如果方便的话，告诉我您具体在哪个位置，我帮您推荐最近的站点～"
                return answer
            else:
                return (
                    "咱们驾校有 **免费大巴车接送** 服务哦！\n\n"
                    "一天三趟班车，20条线路覆盖全西安市、咸阳、秦汉、沣西，"
                    "点对点上门接送，您不用自己奔波通勤。\n\n"
                    "您方便告诉我您在哪个区吗？我帮您看看附近有没有合适的站点～"
                )

        # ===== 报名流程 =====
        if "报名" in question and ("流程" in question or "需要" in question or "材料" in question or "怎么" in question):
            step_lines = [l for l in text_lines if "步" in l or "身份证" in l or "体检" in l or "材料" in l or "面签" in l or "系统" in l]
            result = "咱们报名很简单，一站式搞定！\n\n"
            if step_lines:
                for s in step_lines[:6]:
                    result += f"• {s}\n"
            else:
                result += "• 带身份证来校\n• 自助体检机体检（44元）\n• 录入系统、面签\n• 报完名回家刷题备考科目一\n"
            if should_offer_phone:
                result += "\n要不您打我电话15609130011，我详细给您说说报名需要准备什么～"
            return result

        # ===== 驾校介绍（通用） =====
        if "介绍" in question or "了解" in question or "驾校" in question or "鹏翔" in question:
            # 提取驾校亮点
            highlight_lines = [l for l in text_lines if len(l) > 4 and not l.startswith("(")]
            shown = highlight_lines[:10] if len(highlight_lines) >= 10 else text_lines[:12]
            result = "当然可以！给您介绍一下我们鹏翔驾校的亮点～\n\n"
            for s in shown[:10]:
                result += f"• {s}\n"

            # 首次介绍附照片
            if turn_count <= 1:
                result += "\n【图片:beiyuan_01】\n"
                result += "这是我们北校区（总部）的训练场地～\n"

            result += "\n如果您想了解具体的价格、班型或者考试流程，随时问我哦～"
            return result

        # ===== 默认回答 - 使用真实上下文 =====
        meaningful = [l for l in text_lines if len(l) > 4]
        shown = meaningful[:8] if len(meaningful) >= 8 else text_lines[:10]
        context_text = "\n".join(shown)

        answer = f"根据我们驾校的信息：\n\n{context_text}"

        if should_offer_phone:
            answer += f"\n\n如果您想了解更多详情，欢迎随时问我，或者直接打我电话15609130011，我安排车免费接您来实地参观～"
        else:
            answer += "\n\n还有什么想了解的，随时问我哦～"

        return answer

    def answer(self, question: str, history: str = "", top_k: int = 5, turn_count: int = 0) -> Dict:
        """
        根据问题生成回答。

        参数:
            question: 用户问题
            history: 历史对话上下文
            top_k: 检索返回的文档片段数
            turn_count: 当前对话轮数（用于控制电话推广时机）

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
            answer_text = self._simulate_answer(context, question, turn_count=turn_count)

        return {
            "answer": answer_text,
            "sources": sources,
            "context": context
        }