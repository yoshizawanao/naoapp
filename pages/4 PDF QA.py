import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

def init_page():
    st.set_page_config(
        page_title="Ask My PDF",
        page_icon="🥺"
    )
    st.sidebar.title("Options")

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
    
def init_qa_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_template("""
    # 以下の前提知識を用いて、ユーザーからの質問に答えてください。
    以下の前提知識の正誤判定をしてください。
                                            

    ===
    前提知識
    {context}

    ===
    ユーザーからの質問
    {question}
    """
    )
    # retriever = st.session_state.vectorstore.as_retriever(
    #     search_type="similarity"
    #     search_keywards={"k":10}
    # )
    
    text = st.session_state.textstore

    chain = (
        {"context": text, "question": RunnablePassthrough()}
        | prompt
        | llm
        |StrOutputParser()
    )
    return chain

def page_ask_my_pdf():
    select_model()
    chain = init_qa_chain()
    if query := st.text_input("PDFへの質問を書いてね： ", key="input"):
        st.markdown("## Answer")
        st.write_stream(chain.stream(query))

def main():
    init_page()
    st.title("PDF QA")
    if "textstore" not in st.session_state:
        st.warning("まずはUpload PDFからPDFをアップロードしてね")
    else:
        page_ask_my_pdf()

if __name__ == '__main__':
    main()        