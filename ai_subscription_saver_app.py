"""
AI Agent 安全购物与订阅省钱管家 - 单文件可运行 MVP

运行方式：
1. 安装依赖：pip install streamlit pandas
2. 启动应用：streamlit run ai_subscription_saver_app.py
3. 在浏览器中打开页面，粘贴账单/邮件文本，或直接使用示例数据测试。

说明：
- 这是一个无需外部 API 的本地原型，适合用于申请 Agent / AI 驱动项目额度。
- 核心包含四类 Agent：消费解析、订阅识别、比价省钱、风险控制。
- 可后续接入邮箱、支付宝/微信账单、电商订单、价格 API、大模型 API 等。
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st


# =========================
# 数据结构
# =========================

@dataclass
class Transaction:
    date: str
    merchant: str
    item: str
    amount: float
    currency: str = "CNY"
    source: str = "manual"
    raw_text: str = ""


@dataclass
class Subscription:
    merchant: str
    item: str
    amount: float
    cycle: str
    next_charge_date: Optional[str]
    confidence: float
    reason: str


@dataclass
class PriceSuggestion:
    item: str
    current_price: float
    reference_low_price: float
    saving: float
    suggestion: str


@dataclass
class RiskAlert:
    level: str
    title: str
    description: str
    action: str


# =========================
# Agent 1：消费解析 Agent
# =========================

class ExpenseParserAgent:
    """
    从账单、邮件、订单通知文本中解析消费记录。
    该版本使用规则解析，适合作为 MVP。
    后续可替换为 LLM、OCR、邮箱 API 或银行账单解析器。
    """

    merchant_keywords = [
        "Netflix", "Spotify", "Apple", "iCloud", "腾讯视频", "爱奇艺", "优酷",
        "京东", "淘宝", "天猫", "美团", "饿了么", "亚马逊", "ChatGPT", "Notion",
        "WPS", "百度网盘", "哔哩哔哩", "B站", "Adobe", "Microsoft", "GitHub"
    ]

    amount_patterns = [
        r"(?:￥|¥|CNY\s*)\s*(\d+(?:\.\d{1,2})?)",
        r"(\d+(?:\.\d{1,2})?)\s*(?:元|人民币)",
        r"(?:USD|\$)\s*(\d+(?:\.\d{1,2})?)",
    ]

    date_patterns = [
        r"(20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)",
        r"(\d{1,2}[-/.月]\d{1,2}日?)",
    ]

    def parse(self, text: str) -> List[Transaction]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        transactions: List[Transaction] = []

        for line in lines:
            amount = self._extract_amount(line)
            if amount is None:
                continue

            merchant = self._extract_merchant(line)
            item = self._extract_item(line, merchant)
            date = self._extract_date(line)
            currency = "USD" if "$" in line or "USD" in line.upper() else "CNY"

            transactions.append(
                Transaction(
                    date=date,
                    merchant=merchant,
                    item=item,
                    amount=amount,
                    currency=currency,
                    source="text",
                    raw_text=line,
                )
            )

        return transactions

    def _extract_amount(self, line: str) -> Optional[float]:
        for pattern in self.amount_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_merchant(self, line: str) -> str:
        for keyword in self.merchant_keywords:
            if keyword.lower() in line.lower():
                return keyword

        # 简单兜底：取“商户/平台/来自”后的短文本
        match = re.search(r"(?:商户|平台|来自|订单)[:：]?\s*([\w\u4e00-\u9fa5\-]+)", line)
        if match:
            return match.group(1)
        return "未知商户"

    def _extract_item(self, line: str, merchant: str) -> str:
        cleaned = line.replace(merchant, "")
        cleaned = re.sub(r"(?:￥|¥|CNY|USD|\$)?\s*\d+(?:\.\d{1,2})?\s*(?:元|人民币)?", "", cleaned)
        cleaned = re.sub(r"20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}日?", "", cleaned)
        cleaned = cleaned.strip(" -—｜|，,。:：")
        return cleaned[:40] if cleaned else merchant

    def _extract_date(self, line: str) -> str:
        for pattern in self.date_patterns:
            match = re.search(pattern, line)
            if match:
                raw = match.group(1)
                return self._normalize_date(raw)
        return datetime.today().strftime("%Y-%m-%d")

    def _normalize_date(self, raw: str) -> str:
        raw = raw.replace("年", "-").replace("月", "-").replace("日", "")
        raw = raw.replace("/", "-").replace(".", "-")
        parts = raw.split("-")
        try:
            if len(parts) == 3:
                y, m, d = parts
                return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            if len(parts) == 2:
                y = datetime.today().year
                m, d = parts
                return f"{y:04d}-{int(m):02d}-{int(d):02d}"
        except ValueError:
            pass
        return datetime.today().strftime("%Y-%m-%d")


# =========================
# Agent 2：订阅识别 Agent
# =========================

class SubscriptionDetectorAgent:
    """
    根据交易记录识别疑似订阅。
    逻辑：
    - 同一商户多次扣款
    - 金额相近
    - 时间间隔接近月度/年度周期
    - 文本中出现自动续费、会员、订阅、premium 等关键词
    """

    subscription_keywords = [
        "会员", "自动续费", "连续包月", "连续包年", "订阅", "subscription", "premium",
        "plus", "pro", "renew", "recurring", "月费", "年费"
    ]

    def detect(self, transactions: List[Transaction]) -> List[Subscription]:
        if not transactions:
            return []

        df = pd.DataFrame([asdict(t) for t in transactions])
        subscriptions: List[Subscription] = []

        for merchant, group in df.groupby("merchant"):
            group = group.sort_values("date")
            raw_joined = " ".join(group["raw_text"].fillna("").tolist()).lower()
            keyword_hit = any(k.lower() in raw_joined for k in self.subscription_keywords)

            repeated = len(group) >= 2
            amount_std = group["amount"].std() if len(group) >= 2 else 0
            amount_mean = group["amount"].mean()
            amount_stable = amount_std <= max(2.0, amount_mean * 0.15)

            cycle, next_date, cycle_reason = self._infer_cycle(group["date"].tolist())

            confidence = 0.0
            reasons = []
            if keyword_hit:
                confidence += 0.45
                reasons.append("文本中出现会员/订阅/自动续费等关键词")
            if repeated:
                confidence += 0.25
                reasons.append("同一商户存在多次扣款")
            if amount_stable and repeated:
                confidence += 0.20
                reasons.append("扣款金额较稳定")
            if cycle != "未知":
                confidence += 0.10
                reasons.append(cycle_reason)

            if confidence >= 0.45:
                latest = group.iloc[-1]
                subscriptions.append(
                    Subscription(
                        merchant=merchant,
                        item=str(latest["item"]),
                        amount=float(latest["amount"]),
                        cycle=cycle,
                        next_charge_date=next_date,
                        confidence=round(min(confidence, 0.98), 2),
                        reason="；".join(reasons),
                    )
                )

        return subscriptions

    def _infer_cycle(self, date_list: List[str]) -> Tuple[str, Optional[str], str]:
        parsed = []
        for d in date_list:
            try:
                parsed.append(datetime.strptime(d, "%Y-%m-%d"))
            except ValueError:
                continue

        parsed = sorted(parsed)
        if len(parsed) < 2:
            return "未知", None, "样本不足，无法判断扣费周期"

        intervals = [(parsed[i] - parsed[i - 1]).days for i in range(1, len(parsed))]
        avg_interval = sum(intervals) / len(intervals)
        latest = parsed[-1]

        if 25 <= avg_interval <= 35:
            next_date = latest + timedelta(days=30)
            return "月度", next_date.strftime("%Y-%m-%d"), "扣款间隔接近月度周期"
        if 350 <= avg_interval <= 380:
            next_date = latest + timedelta(days=365)
            return "年度", next_date.strftime("%Y-%m-%d"), "扣款间隔接近年度周期"
        if 6 <= avg_interval <= 8:
            next_date = latest + timedelta(days=7)
            return "周度", next_date.strftime("%Y-%m-%d"), "扣款间隔接近周度周期"
        return "未知", None, "扣款周期不稳定"


# =========================
# Agent 3：比价省钱 Agent
# =========================

class PriceSavingAgent:
    """
    基于历史低价/参考低价给出省钱建议。
    真实产品中可接入电商价格 API、浏览器插件、爬虫或用户历史订单。
    """

    reference_prices = {
        "iCloud": 6.0,
        "Netflix": 35.0,
        "Spotify": 12.0,
        "腾讯视频": 15.0,
        "爱奇艺": 15.0,
        "优酷": 12.0,
        "百度网盘": 10.0,
        "WPS": 8.0,
        "Notion": 35.0,
        "ChatGPT": 145.0,
        "Adobe": 120.0,
        "Microsoft": 30.0,
        "GitHub": 28.0,
    }

    def suggest(self, transactions: List[Transaction], subscriptions: List[Subscription]) -> List[PriceSuggestion]:
        suggestions: List[PriceSuggestion] = []

        for sub in subscriptions:
            ref = self.reference_prices.get(sub.merchant)
            if ref is None:
                continue
            saving = max(0, sub.amount - ref)
            if saving > 0:
                suggestions.append(
                    PriceSuggestion(
                        item=f"{sub.merchant} - {sub.item}",
                        current_price=sub.amount,
                        reference_low_price=ref,
                        saving=round(saving, 2),
                        suggestion="当前扣费高于参考低价，可考虑家庭组、学生优惠、年度折扣或替代服务。",
                    )
                )

        # 检测重复影音会员
        video_merchants = {"Netflix", "腾讯视频", "爱奇艺", "优酷", "哔哩哔哩", "B站"}
        active_video = [s for s in subscriptions if s.merchant in video_merchants]
        if len(active_video) >= 2:
            total = sum(s.amount for s in active_video)
            suggestions.append(
                PriceSuggestion(
                    item="多平台影音会员",
                    current_price=round(total, 2),
                    reference_low_price=round(total * 0.65, 2),
                    saving=round(total * 0.35, 2),
                    suggestion="检测到多个影音会员，可按观看周期轮换订阅，避免长期同时续费。",
                )
            )

        return suggestions


# =========================
# Agent 4：风险控制 Agent
# =========================

class RiskControlAgent:
    """
    识别隐藏续费、钓鱼链接、异常扣费、重复订阅、支付风险。
    """

    suspicious_keywords = [
        "免费试用", "试用结束", "自动扣费", "免密支付", "点击领取", "限时解锁",
        "账户异常", "立即验证", "verify", "urgent", "password", "gift", "中奖"
    ]

    suspicious_domains = [
        "bit.ly", "tinyurl", "t.cn", "goo.gl", "free-vip", "account-verify", "login-security"
    ]

    def scan(self, raw_text: str, transactions: List[Transaction], subscriptions: List[Subscription]) -> List[RiskAlert]:
        alerts: List[RiskAlert] = []
        lower_text = raw_text.lower()

        # 关键词风险
        hits = [k for k in self.suspicious_keywords if k.lower() in lower_text]
        if hits:
            alerts.append(
                RiskAlert(
                    level="中风险",
                    title="发现可能诱导续费或钓鱼的关键词",
                    description=f"文本中出现：{', '.join(hits[:6])}",
                    action="不要直接点击陌生链接；付款前确认商户、金额和续费规则。",
                )
            )

        # 短链/可疑域名
        domain_hits = [d for d in self.suspicious_domains if d in lower_text]
        if domain_hits:
            alerts.append(
                RiskAlert(
                    level="高风险",
                    title="发现短链或可疑域名",
                    description=f"疑似风险链接包含：{', '.join(domain_hits)}",
                    action="建议不要在该链接中输入账号、密码、验证码或支付信息。",
                )
            )

        # 金额异常
        if transactions:
            df = pd.DataFrame([asdict(t) for t in transactions])
            for merchant, group in df.groupby("merchant"):
                if len(group) >= 3:
                    median = group["amount"].median()
                    latest = group.sort_values("date").iloc[-1]
                    if latest["amount"] > median * 2 and latest["amount"] - median > 20:
                        alerts.append(
                            RiskAlert(
                                level="高风险",
                                title=f"{merchant} 出现异常高额扣费",
                                description=f"最新扣费 {latest['amount']}，明显高于历史中位数 {median:.2f}。",
                                action="建议核对是否升级套餐、误开服务或被重复扣费。",
                            )
                        )

        # 即将续费
        for sub in subscriptions:
            if not sub.next_charge_date:
                continue
            try:
                next_dt = datetime.strptime(sub.next_charge_date, "%Y-%m-%d")
            except ValueError:
                continue
            days_left = (next_dt - datetime.today()).days
            if 0 <= days_left <= 7:
                alerts.append(
                    RiskAlert(
                        level="提醒",
                        title=f"{sub.merchant} 即将自动续费",
                        description=f"预计下次扣费日期：{sub.next_charge_date}，金额约 {sub.amount}。",
                        action="若近期不用，建议提前取消自动续费或改为手动续费。",
                    )
                )

        return alerts


# =========================
# Agent 5：报告生成 Agent
# =========================

class ReportAgent:
    def generate(
        self,
        transactions: List[Transaction],
        subscriptions: List[Subscription],
        suggestions: List[PriceSuggestion],
        alerts: List[RiskAlert],
    ) -> str:
        total_spending = sum(t.amount for t in transactions)
        subscription_spending = sum(s.amount for s in subscriptions)
        estimated_saving = sum(s.saving for s in suggestions)

        lines = []
        lines.append("# AI Agent 消费安全与订阅省钱报告")
        lines.append("")
        lines.append("## 1. 总览")
        lines.append(f"- 已解析消费记录：{len(transactions)} 条")
        lines.append(f"- 疑似订阅服务：{len(subscriptions)} 个")
        lines.append(f"- 当前样本消费总额：{total_spending:.2f}")
        lines.append(f"- 疑似订阅扣费总额：{subscription_spending:.2f}")
        lines.append(f"- 预计可节省金额：{estimated_saving:.2f}")
        lines.append("")

        lines.append("## 2. 疑似订阅")
        if subscriptions:
            for s in subscriptions:
                lines.append(f"- {s.merchant}：{s.amount:.2f} / {s.cycle}，下次预计扣费：{s.next_charge_date or '未知'}，置信度：{s.confidence}")
                lines.append(f"  - 判断依据：{s.reason}")
        else:
            lines.append("- 暂未发现明显订阅。")
        lines.append("")

        lines.append("## 3. 省钱建议")
        if suggestions:
            for item in suggestions:
                lines.append(f"- {item.item}：当前 {item.current_price:.2f}，参考低价 {item.reference_low_price:.2f}，预计可省 {item.saving:.2f}")
                lines.append(f"  - 建议：{item.suggestion}")
        else:
            lines.append("- 暂未发现明显可优化项目。")
        lines.append("")

        lines.append("## 4. 风险提醒")
        if alerts:
            for a in alerts:
                lines.append(f"- [{a.level}] {a.title}")
                lines.append(f"  - 说明：{a.description}")
                lines.append(f"  - 操作建议：{a.action}")
        else:
            lines.append("- 暂未发现明显风险。")
        lines.append("")

        lines.append("## 5. Agent 安全机制")
        lines.append("- 高风险操作不自动执行，只生成草稿或提醒。")
        lines.append("- 支付、退订、修改套餐等操作必须二次确认。")
        lines.append("- 账单文本仅在本地解析，默认不上传第三方平台。")
        lines.append("- 每一次建议都保留可解释理由，方便用户回溯。")

        return "\n".join(lines)


# =========================
# 示例数据
# =========================

SAMPLE_TEXT = """
2026-01-02 Netflix Premium 自动续费 ¥45
2026-02-02 Netflix Premium 自动续费 ¥45
2026-03-02 Netflix Premium 自动续费 ¥45
2026-04-02 Netflix Premium 自动续费 ¥45
2026-01-05 Spotify Premium 会员 ¥18
2026-02-05 Spotify Premium 会员 ¥18
2026-03-05 Spotify Premium 会员 ¥18
2026-04-05 Spotify Premium 会员 ¥18
2026-04-07 爱奇艺 连续包月会员 25元
2026-04-08 腾讯视频 VIP 连续包月 30元
2026-04-12 百度网盘 超级会员 自动续费 30元
2026-04-15 Apple iCloud 2TB 存储空间 ¥68
2026-04-19 某平台 免费试用结束后自动扣费 ¥198 点击领取补偿：http://free-vip-login-security.example.com
2026-04-20 京东 机械键盘 ¥399
2026-04-22 美团 外卖订单 ¥42.5
"""


# =========================
# Streamlit 页面
# =========================

st.set_page_config(
    page_title="AI Agent 安全购物与订阅省钱管家",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI Agent 安全购物与订阅省钱管家")
st.caption("本地可运行 MVP：消费解析、多 Agent 协作、订阅识别、比价建议、支付风险控制")

with st.sidebar:
    st.header("功能说明")
    st.markdown(
        """
        **内置 Agent：**
        1. 消费解析 Agent  
        2. 订阅识别 Agent  
        3. 比价省钱 Agent  
        4. 风险控制 Agent  
        5. 报告生成 Agent

        **适合展示的项目亮点：**
        - 多 Agent 协作
        - 长账单解析
        - 支付前风险控制
        - 自动续费治理
        - 可解释消费建议
        """
    )

    use_sample = st.button("填入示例账单")

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if use_sample:
    st.session_state.input_text = SAMPLE_TEXT

input_text = st.text_area(
    "请粘贴账单、订单通知、邮箱扣费提醒或订阅邮件文本：",
    value=st.session_state.input_text,
    height=260,
    placeholder="例如：2026-04-02 Netflix Premium 自动续费 ¥45",
)

col_run, col_clear = st.columns([1, 5])
run = col_run.button("开始分析", type="primary")
if col_clear.button("清空"):
    st.session_state.input_text = ""
    st.rerun()

if run:
    parser = ExpenseParserAgent()
    detector = SubscriptionDetectorAgent()
    saver = PriceSavingAgent()
    risker = RiskControlAgent()
    reporter = ReportAgent()

    transactions = parser.parse(input_text)
    subscriptions = detector.detect(transactions)
    suggestions = saver.suggest(transactions, subscriptions)
    alerts = risker.scan(input_text, transactions, subscriptions)
    report = reporter.generate(transactions, subscriptions, suggestions, alerts)

    total = sum(t.amount for t in transactions)
    sub_total = sum(s.amount for s in subscriptions)
    saving_total = sum(s.saving for s in suggestions)

    st.subheader("分析结果总览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("消费记录", f"{len(transactions)} 条")
    c2.metric("疑似订阅", f"{len(subscriptions)} 个")
    c3.metric("样本总消费", f"¥{total:.2f}")
    c4.metric("预计可省", f"¥{saving_total:.2f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["消费记录", "订阅识别", "省钱建议", "风险提醒", "完整报告"])

    with tab1:
        st.subheader("消费解析 Agent 输出")
        if transactions:
            df = pd.DataFrame([asdict(t) for t in transactions])
            st.dataframe(df[["date", "merchant", "item", "amount", "currency", "raw_text"]], use_container_width=True)
        else:
            st.info("未解析到消费记录。请确认文本中包含金额信息，例如 ¥45、45元、$9.99。")

    with tab2:
        st.subheader("订阅识别 Agent 输出")
        if subscriptions:
            df = pd.DataFrame([asdict(s) for s in subscriptions])
            st.dataframe(df, use_container_width=True)
            st.info(f"疑似订阅扣费总额：¥{sub_total:.2f}")
        else:
            st.info("暂未发现明显订阅。")

    with tab3:
        st.subheader("比价省钱 Agent 输出")
        if suggestions:
            df = pd.DataFrame([asdict(s) for s in suggestions])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂未发现明显省钱机会。")

    with tab4:
        st.subheader("风险控制 Agent 输出")
        if alerts:
            for alert in alerts:
                if alert.level == "高风险":
                    st.error(f"{alert.title}\n\n{alert.description}\n\n建议：{alert.action}")
                elif alert.level == "中风险":
                    st.warning(f"{alert.title}\n\n{alert.description}\n\n建议：{alert.action}")
                else:
                    st.info(f"{alert.title}\n\n{alert.description}\n\n建议：{alert.action}")
        else:
            st.success("暂未发现明显风险。")

    with tab5:
        st.subheader("报告生成 Agent 输出")
        st.markdown(report)

        export = {
            "transactions": [asdict(t) for t in transactions],
            "subscriptions": [asdict(s) for s in subscriptions],
            "suggestions": [asdict(s) for s in suggestions],
            "alerts": [asdict(a) for a in alerts],
            "report": report,
        }
        st.download_button(
            label="下载 JSON 分析结果",
            data=json.dumps(export, ensure_ascii=False, indent=2),
            file_name="agent_subscription_saver_report.json",
            mime="application/json",
        )

else:
    st.info("点击左侧“填入示例账单”，或粘贴自己的账单文本后点击“开始分析”。")


# =========================
# 可扩展方向
# =========================

"""
后续可扩展：

1. 接入邮箱：
   - Gmail API / IMAP
   - 自动搜索 subject:receipt、自动续费、订阅、invoice、payment 等邮件

2. 接入大模型：
   - 用 LLM 替换规则解析器，提高复杂账单识别能力
   - 生成更自然的退订提醒、预算建议、消费总结

3. 接入电商比价：
   - 商品链接解析
   - 历史价格曲线
   - 优惠券组合推荐

4. 安全执行机制：
   - 所有支付、退订、改套餐动作只生成草稿
   - 用户确认后才执行
   - 对每次 Agent 行动保存日志

5. 隐私机制：
   - 本地解析
   - 金额与商户脱敏
   - 敏感链接隔离
   - 不保存银行卡、验证码、密码
"""
