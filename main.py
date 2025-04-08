import streamlit as st
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain_openai import ChatOpenAI

from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page



def init_page():
    st.set_page_config(
        page_title="Web Browsing Agent",
        page_icon="📄"
    )
    st.header("Web Browsing Agent")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear DB", key="clear")

    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "資料についての質問や確認したい点があればどうぞ！"}
        ]

        st.session_state['memory'] = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            k=10
        )


def select_model(temperature=0):
    models = ("GPT-3.5", "GPT-4")
    model = st.sidebar.radio("Choose a model", models)
    if model == "GPT-3.5":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-3.5-turbo"
        )
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-4o"
        )
    
    
CUSTOM_SYSTEM_PROMPT = """
あなたは、読み込んだ資料と、その資料内のユーザーが指摘した部分に基づいて、その指摘部分についてインターネットで調べ
情報の正誤判断や、追加すべき情報を提案してくれるアシスタントです。
読み込んだ資料とユーザーの指摘部分は以下のものに基づいてください。
===
前提知識
{context}

===
ユーザーからの質問
{question}

利用可能なツールを使用して、調査した情報を説明してください。
既に知っていることだけに基づいて答えないでください。回答する前にできる限り検索を行ってください。
(ユーザーが読むページを指定するなど、特別な場合は、検索する必要はありません。)

検索結果ページを見ただけでは情報があまりないと思われる場合は、次の2つのオプションを検討して試してみてください。

- 検索結果のリンクをクリックして、各ページのコンテンツにアクセスし、読んでみてください。
- 1ページが長すぎる場合は、3回以上ページ送りしないでください（メモリの負荷がかかるため）。
- 検索クエリを変更して、新しい検索を実行してください。
- 検索する内容に応じて検索に利用する言語を適切に変更してください。
  - 例えば、プログラミング関連の質問については英語で検索するのがいいでしょう。

ユーザーは非常に忙しく、あなたほど自由ではありません。
そのため、ユーザーの労力を節約するために、直接的な回答を提供してください。

=== 悪い回答の例 ===
- これらのページを参照してください。
- これらのページを参照してコードを書くことができます。
- 次のページが役立つでしょう。

=== 良い回答の例 ===
- これはサンプルコードです。 -- サンプルコードをここに --
- あなたの質問の答えは -- 回答をここに --

回答の最後には、参照したページのURLを**必ず**記載してください。（これにより、ユーザーは回答を検証することができます）

ユーザーが使用している言語で回答するようにしてください。
ユーザーが日本語で質問した場合は、日本語で回答してください。ユーザーがスペイン語で質問した場合は、スペイン語で回答してください。
"""





def create_agent():
    tools = [search_ddg, fetch_page]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = select_model()
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state['memory']
    )

def main():
    init_page()
    init_messages()
    if "vectorstore" not in st.session_state:
        st.warning("まずはUpload PDFからPDFをアップロードしてね")

    web_browsing_agent = create_agent()

    for msg in st.session_state['memory'].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="日本の人口は？"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
    #         retriever = st.session_state.vectorstore.as_retriever(
    #     search_type="similarity",
    #     search_keywards={"k":10}
    # )

            response = web_browsing_agent.invoke(
                {'input': prompt},
                config=RunnableConfig({'callbacks': [st_cb]})
            )
            st.write(response["output"])

if __name__ == '__main__':
    main()