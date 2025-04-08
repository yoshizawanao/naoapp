import fitz
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

def init_page():
    st.set_page_config(
        page_title="Upload PDF",
        page_icon="😢"
    )
    st.sidebar.title("Options")

# def select_model(temperature=0):
#     models = ("GPT-3.5", "GPT-4")
#     model = st.sidebar.radio("Choose a model", models)
#     if model == "GPT-3.5":
#         return ChatOpenAI(
#             temperature=temperature,
#             model_name="gpt-3.5-turbo"
#         )
#     elif model == "GPT-4":
#         return ChatOpenAI(
#             temperature=temperature,
#             model_name="gpt-4o"
#         )

# def init_messages():
#     clear_button = st.sidebar.button("Clear DB", key="clear")
#     if clear_button and "vectorstore" in st.session_state:
#         del st.session_state.vectorstores

def get_pdf_text():
    pdf_file=st.file_uploader(
        label='Upload your PDF here',
        type='pdf'
    )
    
    if pdf_file:
        pdf_text = ""

        with st.spinner("Loading PDF..."):
            pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in pdf_doc:
                pdf_text += page.get_text()

        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #     model_name="text-embedding-3-small",
        #     chunk_size=500,
        #     chunk_overlap=0,
        # )
        # return text_splitter.split_text(pdf_text)
        return pdf_text
    else:
        return None
    
# def build_vector_store(pdf_text):
#     with st.spinner("Saiving to vector store..."):
#         if 'vectorstore' in st.session_state:
#             st.session_state.vectorstore.add_texts(pdf_text)
#         else:
#             st.session_state.vectorstore = FAISS.from_texts(
#                 pdf_text,
#                 OpenAIEmbeddings(model="text-embedding-3-small")
#             )


# def page_pdf_upload_and_build_vector_db():
#     st.title("PDF Upload")
#     pdf_text = get_pdf_text()
#     if pdf_text:
#         build_vector_store(pdf_text)

def init_qa_chain():
    llm = ChatOpenAI(
        temperature=0,
#       model_name="gpt-4o"
    )
    prompt = ChatPromptTemplate.from_template("""
    # 以下の前提知識を用いて、ユーザーからの質問に答えてください。
    以下の前提知識の正誤判定をしてください。
                                            

    ===
    前提知識
    {context}

    # ===
    # ユーザーからの質問
    # {question}
    """
    )
    # retriever = st.session_state.vectorstore.as_retriever(
    #     search_type="similarity"
    #     search_keywards={"k":10}
    # )
    
    text = get_pdf_text()

    chain = (
        {"context": text
        #  , "question": RunnablePassthrough()
         }
        | prompt
        | llm
        |StrOutputParser()
    )
    return chain

def page_ask_my_pdf():
    select_model()
    chain = init_qa_chain()
    # if query := st.text_input("PDFへの質問を書いてね： ", key="input"):
        # st.markdown("## Answer")
        # st.write_stream(chain.stream(query))
    st.markdown("## Answer")
    st.write(chain)


def main():
    init_page()
    # page_pdf_upload_and_build_vector_db()
    answer=get_pdf_text()
    st.write(answer)
    text = page_ask_my_pdf()
    st.write(text)

if __name__ == '__main__':
    main()
