# -*- coding: utf-8 -*-
"""
LinkLab 高校成果转化平台 - 全链路集成版（含技术经纪人对接）
@author: xie
"""

import streamlit as st
import requests
import os
import plotly.graph_objects as go
import base64
import pandas as pd
import qrcode
from io import BytesIO
from PIL import Image
import datetime

# ========== 1. 点击埋点日志记录 ==========
def log_event(page, action, user, tech_id):
    log = f"{datetime.datetime.now()}, {page}, {action}, {user}, {tech_id}\n"
    with open("click_log.csv", "a", encoding="utf-8") as f:
        f.write(log)

# ========== 2. 支持多维评分的大模型Prompt ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def call_deepseek_lrem(tech_text):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
你是科技成果转化领域的专家，请从以下七个维度（每项1-5分）对该成果进行评分，并简要解释每项理由，最后输出一句建议：
1. 技术创新性（成果的新颖性、独创性、突破性）
2. 技术简易性与通用性（易理解、易部署、适应多场景）
3. 商业准备度（从实验室到市场的准备程度，TRL等级）
4. 市场匹配度（对当前行业/市场的契合度）
5. 转化潜力与价值贡献（产业/社会/经济效益潜力）
6. 行业适配性（行业落地性、与主流行业需求的匹配度）
7. 政策契合度（与国家或地方政策、战略需求吻合程度）

请按如下结构返回：
[维度1]：x分，理由：……
[维度2]：x分，理由：……
……
建议：……
    """
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是科技成果智能评价领域的专家。"},
            {"role": "user", "content": prompt + "\n科技成果内容如下：\n" + tech_text}
        ],
        "temperature": 0.4
    }
    response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=20)
    result = response.json()
    if "choices" not in result:
        raise Exception(f"调用失败，状态码 {response.status_code}，返回内容：{response.text}")
    return result["choices"][0]["message"]["content"]

def extract_seven_scores(text):
    import re
    def search(pattern):
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 3  # 默认为中值
    return [
        search(r"技术创新性.*?(\d)分"),
        search(r"技术简易性与通用性.*?(\d)分"),
        search(r"商业准备度.*?(\d)分"),
        search(r"市场匹配度.*?(\d)分"),
        search(r"转化潜力与价值贡献.*?(\d)分"),
        search(r"行业适配性.*?(\d)分"),
        search(r"政策契合度.*?(\d)分"),
    ]

# ========== 3. 成果转化成功预测（需模型） ==========
def predict_conversion_success(feature_vector):
    import os
    if not os.path.exists("conversion_predictor.pkl"):
        return None
    import joblib
    clf = joblib.load("conversion_predictor.pkl")
    return clf.predict_proba([feature_vector])[0][1]

# ========== 4. 生成二维码 ==========
def generate_qr(url):
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image()
    buf = BytesIO()
    img.save(buf)
    buf.seek(0)
    return buf

# ========== 5. 教师成果数据库 ==========
TECH_DATABASE = [
    {"title": "智能光纤传感器在结构健康监测中的应用", "desc": "基于智能光纤的新型结构健康监测。", "university": "湖南大学", "professor": "张教授"},
    {"title": "船舶与海洋装备耐波性系列模拟软件", "desc": "用于主力船型及海洋装备耐波性与运动模拟，技术国际领先。", "university": "哈工程", "professor": "李教授"},
    # ... 可扩展
]

# ========== 6. 技术经纪人数据库（示例） ==========
AGENTS = [
    {"姓名": "李智", "领域": "智能制造", "联系方式": "lizhi@example.com"},
    {"姓名": "王琳", "领域": "新能源", "联系方式": "wanglin@example.com"},
    {"姓名": "张超", "领域": "生物医药", "联系方式": "zhangchao@example.com"}
]

# ========== 7. Streamlit页面结构 ==========
st.set_page_config(page_title="LinkLab 高校成果转化平台", layout="wide")
st.sidebar.title("🔗 LinkLab 功能导航")
page = st.sidebar.radio(
    "请选择页面",
    [
        "📄 成果详情页", 
        "🤖 教师评分助手", 
        "🏭 企业项目匹配", 
        "👥 技术经纪人对接",    # 新增
        "📊 后台数据分析",
        "📝 成果转化跟踪"
    ]
)

if st.sidebar.button("📱 手机访问二维码"):
    qr_buf = generate_qr("https://linklab-app-ekb89vezcdlata4ix9arbn.streamlit.app/")
    st.image(Image.open(qr_buf), caption="扫码用手机打开", use_column_width=False)

# ========== 8. 成果详情页 ==========
if page == "📄 成果详情页":
    st.title("🔬 LinkLab | 科技成果详情页")
    user = st.session_state.get('user', '匿名')
    for i, tech in enumerate(TECH_DATABASE):
        log_event("成果详情", "浏览", user, i)
        st.markdown("---")
        st.markdown(f"### 🔖 {tech['title']}")
        st.markdown(f"**🏫 所属高校：** {tech['university']}  ")
        st.markdown(f"**👨‍🏫 教师：** {tech['professor']}  ")
        st.markdown(f"**📃 项目简介：** {tech['desc']}")
        feature_vec = [3,3,3,3,3,3,3,100,5,60]  # 示例
        proba = predict_conversion_success(feature_vec)
        if proba is not None:
            st.info(f"AI预测转化成功概率为：{proba*100:.1f}%")
        st.info("详细信息可由评分助手模块生成智能分析报告。")

# ========== 评分助手页面（含输入检测、结构化建议、专家校正、评分声明） ==========
import streamlit as st
import os
import requests
import plotly.graph_objects as go
import base64

# ========== DeepSeek语义判别函数 ==========
def is_tech_achievement_deepseek(text):
    """
    用DeepSeek大模型判断输入内容是否为科技成果描述。
    只返回True/False和原因说明。
    """
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        "请判断下面的内容是否属于“科技成果/技术描述/技术创新/应用案例”类型？"
        "仅当其明确描述了一项技术、创新点、应用场景、性能指标等，才算属于科技成果。"
        "请你只回复：“是”或“否”，后面简单解释理由（不要复述原文，不要输出其他内容）。\n\n内容如下：\n"
        + text.strip()
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 80
    }
    try:
        response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=20)
        result = response.json()
        out = result["choices"][0]["message"]["content"].strip()
        if out.startswith("是"):
            return True, "内容属于科技成果描述，可进入评分。"
        else:
            return False, "当前内容不属于科技成果/技术描述，请输入完整真实的成果描述。"
    except Exception as e:
        return False, f"【DeepSeek API错误】{e}"

# ========== 输入有效性检测 ==========
def is_valid_tech_text(text):
    """
    输入检测：必须字数>=80，且包含“技术”“应用”“创新”关键词之一
    """
    text = text.strip()
    if len(text) < 80:
        return False, "请输入详细的科技成果内容，字数不少于80字。"
    keywords = ["技术", "应用", "创新"]
    if not any(k in text for k in keywords):
        return False, "请补充技术创新点、应用场景等核心信息。"
    return True, ""

# ========== 调用DeepSeek大模型评分 ==========
def call_deepseek_lrem(tech_text):
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
你是科技成果转化领域的专家，请从以下七个维度（每项1-5分）对该成果进行评分，并简要解释每项理由，最后输出一句建议：
1. 技术创新性（成果的新颖性、独创性、突破性）
2. 技术简易性与通用性（易理解、易部署、适应多场景）
3. 商业准备度（从实验室到市场的准备程度，TRL等级）
4. 市场匹配度（对当前行业/市场的契合度）
5. 转化潜力与价值贡献（产业/社会/经济效益潜力）
6. 行业适配性（行业落地性、与主流行业需求的匹配度）
7. 政策契合度（与国家或地方政策、战略需求吻合程度）

请按如下结构返回：
[维度1]：x分，理由：……
[维度2]：x分，理由：……
……
建议：……
    """
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是科技成果智能评价领域的专家。"},
            {"role": "user", "content": prompt + "\n科技成果内容如下：\n" + tech_text}
        ],
        "temperature": 0.4
    }
    response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=20)
    result = response.json()
    if "choices" not in result:
        raise Exception(f"调用失败，状态码 {response.status_code}，返回内容：{response.text}")
    return result["choices"][0]["message"]["content"]

# ========== 评分抽取 ==========
def extract_seven_scores(text):
    import re
    def search(pattern):
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 3  # 默认为中值
    return [
        search(r"技术创新性.*?(\d)分"),
        search(r"技术简易性与通用性.*?(\d)分"),
        search(r"商业准备度.*?(\d)分"),
        search(r"市场匹配度.*?(\d)分"),
        search(r"转化潜力与价值贡献.*?(\d)分"),
        search(r"行业适配性.*?(\d)分"),
        search(r"政策契合度.*?(\d)分"),
    ]

# ========== UI展示 ==========
st.title("📤 教师成果上传 + 🤖 七维AI智能评分（LinkLab-LREM）")
with st.expander("📑 推荐填写模板及范例"):
    st.markdown("""
**建议结构化输入以下内容，提高评分科学性：**
- 【成果名称】
- 【主要技术原理】
- 【创新点】
- 【应用场景】
- 【性能指标】
- 【成熟度/转化进度】

**示例：**
> 【成果名称】XX超疏水纳米涂层  
> 【主要技术原理】采用XX合成工艺，实现超疏水与耐磨兼容  
> 【创新点】首次在XX领域实现XX  
> 【应用场景】新能源车辆表面、太阳能板防污  
> 【性能指标】接触角≥160°，使用寿命5年  
> 【成熟度】已完成中试阶段，具备量产条件
""")

uploaded_file = st.file_uploader("📎 上传专利PDF/项目文档（可选）", type=["pdf"])
user_input = st.text_area("✍️ 输入科技成果内容（建议结构化描述技术亮点、创新性、应用场景等）", height=240)
categories = [
    "技术创新性", "通用性", "商业准备度", 
    "市场匹配度", "转化潜力", "行业适配", "政策契合"
]

if "last_scores" not in st.session_state:
    st.session_state.last_scores = [3]*7
if "expert_scores" not in st.session_state:
    st.session_state.expert_scores = [3]*7
if "ai_result_text" not in st.session_state:
    st.session_state.ai_result_text = ""

st.info("如输入内容不是实际科技成果描述，系统将自动拒绝评分。\n\n🟢 **AI评分为参考意见，仅供初筛和决策辅助，最终结论请结合专家校正与实际转化数据综合判断。**")

# ========== 评分按钮逻辑（含三重判别） ==========
if st.button("🚀 开始AI七维评分"):
    valid, msg = is_valid_tech_text(user_input)
    if not valid:
        st.warning(msg)
    else:
        is_tech, explain = is_tech_achievement_deepseek(user_input)
        if not is_tech:
            st.error(explain)
        elif not os.getenv("DEEPSEEK_API_KEY"):
            st.error("请设置环境变量 DEEPSEEK_API_KEY 或在代码中填入密钥。")
        else:
            with st.spinner("正在AI智能评分中，请稍候..."):
                try:
                    result = call_deepseek_lrem(user_input)
                    st.session_state.ai_result_text = result
                    scores = extract_seven_scores(result)
                    st.session_state.last_scores = scores
                    st.session_state.expert_scores = scores.copy()
                except Exception as e:
                    st.error(f"调用失败，请检查API Key或网络状态：{e}")

if st.session_state.ai_result_text:
    st.success("AI评分完成 ✅")
    st.markdown("### 🧠 七维评分与解释：")
    lines = [l.strip() for l in st.session_state.ai_result_text.split('\n') if l.strip()]
    for line in lines:
        if "分，理由" in line:
            st.markdown(f"- {line}")
    fig = go.Figure(data=[go.Scatterpolar(
        r=st.session_state.last_scores + [st.session_state.last_scores[0]],
        theta=categories + [categories[0]], fill='toself')])
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False, title="📊 七维雷达图"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🏅 专家校正打分（可选）：")
    for i, cat in enumerate(categories):
        st.session_state.expert_scores[i] = st.slider(
            f"{cat}（AI建议：{st.session_state.last_scores[i]}）",
            1, 5, st.session_state.expert_scores[i], key=f"expert_{i}_slider"
        )
    if st.button("保存专家校正评分"):
        st.success(f"专家评分已保存！当前评分为：{st.session_state.expert_scores}。不会丢失或重置！")
        # 可在此处加入存数据库、导出等操作

    export_text = (
        f"评分分析报告\n\n{st.session_state.ai_result_text}\n\n"
        f"七维雷达图分值：{dict(zip(categories, st.session_state.last_scores))}\n"
        f"专家校正：{st.session_state.expert_scores}"
    )
    b64 = base64.b64encode(export_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="评分分析报告.txt">📤 下载评分报告</a>'
    st.markdown(href, unsafe_allow_html=True)



# ========== 10. 企业项目匹配 ==========
elif page == "🏭 企业项目匹配":
    st.title("🏭 企业项目上传 + 成果智能匹配")
    name = st.text_input("📛 企业名称")
    project_desc = st.text_area("📄 项目需求描述", height=180)
    uploaded_proj = st.file_uploader("📎 上传项目说明文件（可选）", type=["pdf", "docx", "txt"])
    if st.button("🔍 智能匹配高校成果"):
        if not project_desc.strip():
            st.warning("请填写项目描述后再进行匹配。")
        else:
            with st.spinner("正在智能分析与匹配高校成果..."):
                st.success("✅ 匹配完成！推荐如下成果：")
                st.markdown("- 智能光纤传感器在结构健康监测中的应用（湖南大学）")
                st.markdown("- 船舶与海洋装备耐波性系列模拟软件（哈工程）")
                # ...根据智能评分和行业关键词自动推荐

# ========== 11. 技术经纪人对接 ==========
elif page == "👥 技术经纪人对接":
    st.title("👥 技术经纪人资源对接平台")
    for agent in AGENTS:
        with st.expander(f"{agent['姓名']} | 擅长：{agent['领域']}"):
            st.markdown(f"📧 联系方式：{agent['联系方式']}")
            if st.button(f"📨 申请与 {agent['姓名']} 对接", key=agent["姓名"]):
                st.success(f"已收到你的对接申请！请耐心等待{agent['姓名']}联系你。")

# ========== 12. 后台数据分析 ==========
elif page == "📊 后台数据分析":
    st.title("📊 平台后台数据监控")
    col1, col2, col3 = st.columns(3)
    col1.metric("上传成果数", f"{len(TECH_DATABASE)}")
    col2.metric("企业项目数", "53")
    col3.metric("活跃用户数", "41")
    st.markdown("#### 🧾 最近上传成果")
    df_teachers = pd.DataFrame([{"教师": tech["professor"], "成果名": tech["title"], "上传时间": "2025-06-18"} for tech in TECH_DATABASE])
    st.table(df_teachers)
    st.download_button("📥 导出成果数据", data=df_teachers.to_csv(index=False), file_name="teacher_data.csv")
    # 展示点击日志部分统计（如有）
    if os.path.exists("click_log.csv"):
        logs = pd.read_csv("click_log.csv", header=None, names=["时间", "页面", "操作", "用户", "成果ID"])
        st.markdown("#### 📊 用户行为日志统计")
        st.write(logs.tail(20))

# ========== 13. 成果转化跟踪 ==========
elif page == "📝 成果转化跟踪":
    st.title("📝 成果转化跟踪与佐证")
    selected = st.selectbox("选择成果", [f"{i+1}.{tech['title']}" for i, tech in enumerate(TECH_DATABASE)])
    tech_id = int(selected.split('.')[0]) - 1
    status = st.selectbox("转化进度", ["未转化", "样机开发", "中试/试用", "专利转让", "企业应用", "市场化成功", "失败"])
    economic = st.text_input("经济效益/合同金额")
    policy = st.text_input("政策支持/获奖情况")
    uploaded_file = st.file_uploader("上传佐证材料（合同/新闻等）", key="file_"+str(tech_id))
    if st.button("保存转化记录"):
        fname = uploaded_file.name if uploaded_file else ''
        with open("conversion_records.csv", "a", encoding="utf-8") as f:
            f.write(f"{tech_id},{status},{economic},{policy},{fname}\n")
        st.success("记录已保存！")
    # 展示历史记录
    if os.path.exists("conversion_records.csv"):
        df = pd.read_csv("conversion_records.csv", header=None, names=["成果ID","转化进度","经济效益","政策支持","文件名"])
        st.markdown("#### ⏳ 历史转化跟踪记录")
        st.write(df[df["成果ID"]==tech_id])
