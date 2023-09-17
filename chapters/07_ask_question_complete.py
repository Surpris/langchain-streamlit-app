"""
ChatPDF-like App (complete).

Requirements
============
Python >= 3.8.1
pypdf2
streamlit
langchain
openai
tiktoken (for TextSplitter in LangChain)
qdrant-client

You can use any HuggingFace embedding model if you want.
"""

# pip install pycryptodome
from enum import Enum
from glob import glob
from PyPDF2 import PdfReader
import streamlit as st
from typing import List, Union, Iterable, Tuple, Dict, Any

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# parameters for TextSplitter
CHUNK_SIZE_DEFAULT: int = 500
CHUNK_OVERLAP: int = 0
SEPARATORS: List[str] = ["\n\n", "\n", "。", "、", " ", ""]

# parameters for OpenAIEmbeddings
EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"

# parameters for QDRANT DB
QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"
EMBEDDING_VECTOR_SIZE: int = 1536
RETRIEVAL_SEARCH_TYPE: str = "similarity"
RETRIEVAL_SEARCH_KWARGS: Dict[str, Any] = {"k": 10}

# parameters for LangChain
CHAIN_TYPE: str = "stuff"
OPENAI_CHAT_MODELS: Tuple[str] = ("GPT-3.5", "GPT-3.5-16k", "GPT-4")

# parameters of a prompt
MAX_OUTPUT_CHARACTERS: int = 300

# ref: https://github.com/hwchase17/langchain/blob/master/langchain/chains/question_answering/stuff_prompt.py
PROMPT_TEMPLATE = f"""Use the following pieces of context to answer the question at the end.
The character length of the answer should be within {MAX_OUTPUT_CHARACTERS}.
If you don't know the answer, just say that you don't know,
don't try to make up an answer.
""" + """
=== context starts
{context}
=== context ends

Question: {question}
Helpful Answer:"""


class SearchTypes(Enum):
    Similarity = 0
    MMR = 1
    Similarity_Score_Threshold = 2

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_int(key: int):
        assert key in [0, 1, 2], ValueError(f"Invalid key: {key}.")
        if key == 0:
            return SearchTypes.Similarity
        elif key == 1:
            return SearchTypes.MMR
        elif key == 2:
            return SearchTypes.Similarity_Score_Threshold


def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🤗"
    )
    st.sidebar.title("Nav")
    st.session_state.costs: List[float] = []


def select_model() -> ChatOpenAI:
    model = st.sidebar.radio("Choose a model:", OPENAI_CHAT_MODELS)
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    elif model == "GPT-4":
        st.session_state.model_name = "gpt-4"
    else:
        raise ValueError(
            "Unknown model was chosen. This error is not excepted to occur in normal usage."
        )

    # 300: 本文以外の指示のトークン数 (以下同じ)
    st.session_state.max_token = OpenAI.modelname_to_contextsize(
        st.session_state.model_name
    ) - MAX_OUTPUT_CHARACTERS
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


def get_pdf_text() -> Union[None, List[str]]:
    uploaded_file = st.file_uploader(
        label='Upload your PDF here😇',
        type='pdf'
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=EMBEDDING_MODEL_NAME,
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
            # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
            chunk_size=CHUNK_SIZE_DEFAULT,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )
        return text_splitter.split_text(text)
    else:
        return None


def load_qdrant() -> Qdrant:
    client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=OpenAIEmbeddings()
    )


def build_vector_store(pdf_text: Iterable[str]):
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)

    # 以下のようにもできる。この場合は毎回ベクトルDBが初期化される
    # LangChain の Document Loader を利用した場合は `from_documents` にする
    # Qdrant.from_texts(
    #     pdf_text,
    #     OpenAIEmbeddings(),
    #     path="./local_qdrant",
    #     collection_name="my_documents",
    # )


def build_qa_model(llm: ChatOpenAI) -> RetrievalQA:
    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        # "mmr",  "similarity_score_threshold" などもある
        search_type=RETRIEVAL_SEARCH_TYPE,
        # 文書を何個取得するか (default: 4)
        search_kwargs=RETRIEVAL_SEARCH_KWARGS
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=True
    )


def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)


def ask(qa: RetrievalQA, query: str) -> Tuple[Dict[str, Any], float]:
    answer = None
    with get_openai_callback() as cb:
        # query / result / source_documents
        answer = qa(query)

    return answer, cb.total_cost


def page_ask_my_pdf():
    st.title("Ask My PDF(s)")

    llm = select_model()
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)


def main():
    init_page()

    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf()

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    main()
