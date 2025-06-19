# -*- coding: utf-8 -*-
"""
LinkLab 高校成果转化平台 - 教师成果上传评分 + 企业项目上传匹配 + GPT雷达图展示 + 技术成果库展示（含AI评分） + 教师助手 + 企业对接 + 技术经纪人 + 后台数据分析
@author: xie
"""

import streamlit as st
import requests
import os
import re
import plotly.graph_objects as go
import base64
import pandas as pd
import qrcode
from io import BytesIO
from PIL import Image

def generate_qr(url):
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image()
    buf = BytesIO()
    img.save(buf)
    buf.seek(0)
    return buf


# ✅ 设置你的 DeepSeek API Key（建议用环境变量）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ✅ DeepSeek 接口封装
def call_deepseek(tech_text):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
你是一个技术转化分析专家，请根据以下科研成果内容，对其转化潜力进行分析。请输出：

1. 技术成熟度（TRL）：1-9级；
2. 应用行业建议（最多2个）；
3. 创新程度（高/中/低）；
4. 商业化可能性（高/中/低）；
5. 一句话推荐语（建议是否推进转化）。

成果内容如下：
{tech_text}
请以清晰的结构返回以上5项内容。
    """
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个严谨的科技成果分析专家。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=20)
    try:
        result = response.json()
        if "choices" not in result:
            raise Exception("返回中未包含 'choices' 字段")
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"调用失败，状态码 {response.status_code}，返回内容：{response.text}")

# ✅ 提取评分项函数
def extract_scores_from_text(text):
    trl_match = re.search(r"TRL.*?(\d+)", text)
    trl = int(trl_match.group(1)) if trl_match else 5
    innov_match = re.search(r"创新程度[:：]?\s*(高|中|低)", text)
    innov = {"高": 3, "中": 2, "低": 1}.get(innov_match.group(1), 2) if innov_match else 2
    commer_match = re.search(r"商业化可能性[:：]?\s*(高|中|低)", text)
    commer = {"高": 3, "中": 2, "低": 1}.get(commer_match.group(1), 2) if commer_match else 2
    return trl, innov, commer

# ✅ 成果库（完整 7 条）
TECH_DATABASE = [
    {"title": "智能光纤传感器在结构健康监测中的应用", "desc": "本成果提出了一种基于智能光纤传感的新型结构健康监测方法。", "status": "已部署试点项目", "field": "结构健康监测、土木工程、光纤传感", "summary": "适用于桥梁、高铁、航空等关键结构的状态实时监测。", "TRL": 6, "industry": "新能源、智能制造", "innovation": "高", "commercial": "中", "university": "湖南大学", "school": "机械与运载工程学院", "professor": "张教授"},
    {"title": "船舶与海洋装备耐波性系列模拟软件", "desc": "快速模拟各类船舶在波浪中运动的性能，技术成熟，自主可控。", "status": "样品阶段", "field": "船舶设计、海洋装备设计", "summary": "用于模拟船舶抗波能力，提升设计效率。", "TRL": 7, "industry": "船舶设计、海洋工程", "innovation": "高", "commercial": "中", "university": "哈尔滨工程大学", "school": "船舶工程学院", "professor": "李教授"},
    {"title": "舰船综合电力系统多层建模与仿真平台", "desc": "构建多层级舰船电力系统数字孪生仿真平台，助力海军装备数字化升级。", "status": "正在产业化推进", "field": "船舶电力系统、数字孪生、系统仿真", "summary": "应用于舰船复杂电力网络建模与运行优化。", "TRL": 6, "industry": "军工电子、海洋装备", "innovation": "中", "commercial": "高", "university": "哈尔滨工程大学", "school": "智能科学与工程学院", "professor": "王教授"},
    {"title": "基于运动预报的船舶纵-横联合减摇控制技术", "desc": "可联合控制横摇与纵摇，已完成样机并在拖曳水池测试。", "status": "样品阶段", "field": "减摇系统、舰船控制", "summary": "基于水动力预报模型实现动态联合减摇，适用于恶劣海况作业。", "TRL": 6, "industry": "船舶动力与减摇", "innovation": "中", "commercial": "中", "university": "哈尔滨工程大学", "school": "船舶工程学院", "professor": "赵教授"},
    {"title": "深水浮式结构物运动耦合及外载荷分析研究", "desc": "填补国内软件空白，拥有自主产权，支持平台运动与载荷分析。", "status": "开发完成，待推广", "field": "深海工程、载荷分析", "summary": "支持浮式平台三维响应分析，满足国产化设计需求。", "TRL": 5, "industry": "深海工程设计", "innovation": "高", "commercial": "中", "university": "哈尔滨工程大学", "school": "海洋工程学院", "professor": "陈教授"},
    {"title": "高速船耐波性计算软件", "desc": "适用于弗氏数大于0.4的多体高速船，计算效率高。", "status": "技术成熟", "field": "高速船、波浪载荷分析", "summary": "可快速评估高速船耐波性，适用于多体高速船快速设计阶段。", "TRL": 7, "industry": "船舶动力分析", "innovation": "中", "commercial": "高", "university": "哈尔滨工程大学", "school": "动力与能源学院", "professor": "张教授"},
    {"title": "内燃机燃料喷射特性测试与分析系统", "desc": "支持氨/氢等清洁燃料喷射特性测试，已有多型产品应用。", "status": "成熟产品阶段", "field": "发动机燃料供给系统", "summary": "提供高精度流量分析与喷雾形态检测，打破国外垄断。", "TRL": 9, "industry": "能源动力系统、发动机检测", "innovation": "高", "commercial": "高", "university": "哈尔滨工程大学", "school": "动力与能源学院", "professor": "刘教授"}
]

st.set_page_config(page_title="LinkLab 高校成果转化平台", layout="wide")
st.sidebar.title("🔗 LinkLab 功能导航")
page = st.sidebar.radio("请选择页面", ["📄 成果详情页", "🤖 教师评分助手", "🏭 企业项目匹配", "👥 技术经纪人对接", "📊 后台数据分析"])

if st.sidebar.button("📱 手机访问二维码"):
    qr_buf = generate_qr("https://linklab-app-ekb89vezcdlata4ix9arbn.streamlit.app/")
    st.image(Image.open(qr_buf), caption="扫码用手机打开", use_column_width=False)

# ✅ 页面逻辑
if page == "📄 成果详情页":
    st.title("🔬 LinkLab | 科技成果详情页")
    for idx, tech in enumerate(TECH_DATABASE):
        st.markdown("---")
        st.markdown(f"### 🔖 {tech['title']}")
        st.markdown(f"**🏫 所属高校：** {tech['university']}  ")
        st.markdown(f"**🏢 学院：** {tech['school']}  ")
        st.markdown(f"**👨‍🏫 教师：** {tech['professor']}  ")
        st.markdown(f"**📌 所属领域：** {tech['field']}  ")
        st.markdown(f"**📃 项目简介：** {tech['desc']}")
        score = (tech['TRL'] + {"高": 3, "中": 2, "低": 1}[tech['innovation']] + {"高": 3, "中": 2, "低": 1}[tech['commercial']]) / 3
        st.markdown("### 🧠 AI评分：转化潜力 {:.1f} / 5".format(score))
        st.markdown(f"- 技术成熟度（TRL）：{tech['TRL']}  ")
        st.markdown(f"- 应用行业：{tech['industry']}  ")
        st.markdown(f"- 创新程度：{tech['innovation']}  ")
        st.markdown(f"- 商业化可能性：{tech['commercial']}  ")
        st.info(tech["summary"])
        st.download_button("📄 专利全文", "专利内容示意", file_name="patent.pdf", key=f"download_{idx}")

elif page == "🤖 教师评分助手":
    st.title("📤 教师成果上传 + 🤖 GPT智能评分（由 DeepSeek 支持）")
    uploaded_file = st.file_uploader("📎 上传专利PDF（仅用于展示）", type=["pdf"])
    if uploaded_file:
        st.success(f"已上传文件：{uploaded_file.name}")
    user_input = st.text_area("✍️ 输入技术成果内容", height=200, placeholder="请输入专利或研究摘要...")
    if st.button("🚀 开始AI评分"):
        if not DEEPSEEK_API_KEY:
            st.error("请设置环境变量 DEEPSEEK_API_KEY 或在代码中填入密钥。")
        elif user_input.strip() == "":
            st.warning("请输入成果内容后再点击评分。")
        else:
            with st.spinner("正在调用 DeepSeek 模型中，请稍候..."):
                try:
                    result = call_deepseek(user_input)
                    st.success("AI评分完成 ✅")
                    st.markdown("### 🧠 分析结果如下：")
                    st.markdown(result)
                    trl, innov, commer = extract_scores_from_text(result)
                    categories = ["技术成熟度", "创新程度", "商业化可能性"]
                    values = [trl, innov, commer]
                    fig = go.Figure(data=[go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself')])
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 9])), showlegend=False, title="📊 GPT评分雷达图")
                    st.plotly_chart(fig, use_container_width=True)
                    export_text = f"评分分析报告\n\n{result}\n\n雷达图评分：TRL={trl}, 创新={innov}, 商业化={commer}"
                    b64 = base64.b64encode(export_text.encode()).decode()
                    href = f'<a href="data:file/txt;base64,{b64}" download="评分分析报告.txt">📤 下载评分报告</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"调用失败，请检查 API Key 或网络状态：{e}")

elif page == "🏭 企业项目匹配":
    st.title("🏭 企业项目上传 + 成果智能匹配")
    name = st.text_input("📛 企业名称")
    project_desc = st.text_area("📄 项目需求描述", height=180)
    uploaded_proj = st.file_uploader("📎 上传项目说明文件（可选）", type=["pdf", "docx", "txt"])
    if uploaded_proj:
        st.success(f"已上传文件：{uploaded_proj.name}")
    if st.button("🔍 智能匹配高校成果"):
        if not project_desc.strip():
            st.warning("请填写项目描述后再进行匹配。")
        else:
            with st.spinner("正在匹配高校技术成果..."):
                st.success("✅ 匹配完成！推荐如下成果：")
                st.markdown("- 🔬 智能结构健康监测系统（湖南大学）")
                st.markdown("- ⚙️ 高效能机器人控制模块（浙江大学）")
                st.markdown("- 🌱 农业AI监测平台（中国农业大学）")

elif page == "👥 技术经纪人对接":
    st.title("👥 技术经纪人资源对接平台")
    agents = [
        {"姓名": "李智", "领域": "智能制造", "联系方式": "lizhi@example.com"},
        {"姓名": "王琳", "领域": "新能源", "联系方式": "wanglin@example.com"},
        {"姓名": "张超", "领域": "生物医药", "联系方式": "zhangchao@example.com"}
    ]
    for agent in agents:
        with st.expander(f"{agent['姓名']} | 擅长：{agent['领域']}"):
            st.markdown(f"📧 联系方式：{agent['联系方式']}")
            if st.button(f"📨 申请与 {agent['姓名']} 对接", key=agent["姓名"]):
                st.success("我们已收到你的对接申请！请耐心等待经纪人联系。")

elif page == "📊 后台数据分析":
    st.title("📊 平台后台数据监控")
    col1, col2, col3 = st.columns(3)
    col1.metric("上传成果数", f"{len(TECH_DATABASE)}")
    col2.metric("企业项目数", "53")
    col3.metric("活跃用户数", "41")
    st.markdown("#### 🧾 最近上传成果")
    df_teachers = pd.DataFrame([{"教师": tech["professor"], "成果名": tech["title"], "上传时间": "2025-06-18"} for tech in TECH_DATABASE[:2]])
    st.table(df_teachers)
    st.markdown("#### 🏭 企业项目上传记录")
    df_enterprise = pd.DataFrame([
        {"企业": "湖南智创科技", "项目": "高温感知器", "需求时间": "2025-06-16"},
        {"企业": "启明新能源", "项目": "碳中和监测平台", "需求时间": "2025-06-15"}
    ])
    st.table(df_enterprise)
    st.download_button("📥 导出成果数据", data=df_teachers.to_csv(index=False), file_name="teacher_data.csv")
    st.download_button("📥 导出企业数据", data=df_enterprise.to_csv(index=False), file_name="enterprise_data.csv")
