from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                               temperature=0.7)

parser = StrOutputParser()

system_prompt = """
你是一位精通{target_language}口說與寫作教學的母語人士級老師。
請根據學生輸入的中文句子，提供專業且易懂的教學。

請嚴格遵守以下輸出格式：

### 🌟 自然的{target_language}表達
（提供 2 到 3 種不同情境的說法，例如：正式/商業、日常口語）

### 📖 重點單字與片語
（挑選句子中的核心詞彙進行解釋）

### 💡 語法與用法解析
（詳細講解這句{target_language}的句型結構與語境用法）

### 💬 實用例句
（提供 2 個額外的相關例句，幫助學生加深印象）
"""

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('human', '{user_input}')
]
)
chain = prompt_template | model | parser

target_language = '英文'

user_input = '怎麼表達這句話：根據使用者痛點分析的結果，使用者在學習英文口說時，最大的痛點是缺乏實際的練習機會和即時的反饋。'

input_data = {
    'target_language': '英文',
    'user_input': user_input
}

response = chain.invoke(input_data)
print(response)