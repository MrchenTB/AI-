import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="AI 語言口說優化器", page_icon="🌍", layout="centered")

st.title("🌍 AI 語言口說優化器")
st.markdown('<p class="subtitle">輸入一句中文，讓 AI 老師教你用道地、自然的外語表達！</p>', unsafe_allow_html=True)

LANGUAGE_OPTIONS = {
    "🇺🇸 英文 English":     "英文",
    "🇯🇵 日文 Japanese":    "日文",
    "🇰🇷 韓文 Korean":      "韓文",
    "🇫🇷 法文 French":      "法文",
    "🇩🇪 德文 German":      "德文",
    "🇪🇸 西班牙文 Spanish":  "西班牙文",
    "🇮🇹 義大利文 Italian":  "義大利文",
    "🇵🇹 葡萄牙文 Portuguese": "葡萄牙文",
}

@st.cache_resource
def get_langchain_pipeline():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    system_prompt = """
    你是一位精通{target_language}口說與寫作教學的母語人士級老師。
    請根據學生輸入的中文句子，提供專業且易懂的教學。

    請嚴格遵守以下輸出格式：

    ### 🌟 自然的{target_language}表達
    （提供 2 到 3 種不同情境的說法，例如：正式/商業、日常口語）

    ### 📖 重點單字與片語
    （挑選句子中的核心詞彙進行解釋，並標注發音或羅馬拼音）

    ### 💡 語法與用法解析
    （詳細講解這句{target_language}的句型結構與語境用法）

    ### 💬 實用例句
    （提供 2 個額外的相關例句，幫助學生加深印象）
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_input}"),
    ])

    return prompt_template | model | parser

chain = get_langchain_pipeline()

col1, col2 = st.columns([2, 1])
with col1:
    default_text = "我想學習如何在會議上清楚地表達自己的想法。"
    user_input = st.text_area("✏️ 請輸入你想學習的中文句子：", value=default_text, height=110)
with col2:
    selected_label = st.selectbox("🎯 目標語言", list(LANGUAGE_OPTIONS.keys()))
    target_language = LANGUAGE_OPTIONS[selected_label]

st.markdown("<br>", unsafe_allow_html=True)

if st.button("開始學習 🚀"):
    if user_input.strip() == "":
        st.warning("請先輸入句子喔！")
    else:
        with st.spinner(f"AI 老師正在為你準備{target_language}專屬教材..."):
            try:
                response = chain.invoke({
                    "target_language": target_language,
                    "user_input": user_input,
                })
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<span class="lang-badge">{selected_label}</span>', unsafe_allow_html=True)
                st.markdown(response)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"發生錯誤：{e}\n請檢查您的 API 金鑰是否正確設定。")
