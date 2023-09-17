"""
ChatGPT summarizing a given YouTube video.
This App uses `map_reduce` in `load_summarize_chain.`

Requirements
============
Python >= 3.8.1
streamlit
langchain
openai
beautifulsoup4
tiktoken (for TextSplitter in LangChain)
"""

import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from typing import List

MODEL_TEMPERATURE: float = 0

# parameters for TextSplitter
CHUNK_OVERLAP: int = 0
SEPARATORS: List[str] = ["\n\n", "\n", "。", "、", " ", ""]

# parameters of a prompt
MAX_SUMMARY_CHARACTERS: int = 300
PROMPT_TEMPLATE: str = f"""Please write a concise Japanese summary of the following transcript of a YouTube Video. The character length of the summary should be within {MAX_SUMMARY_CHARACTERS}.

""" + """=== transcript starts
{text}
=== transcript ends
"""


def init_page():
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="🤗"
    )
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")
    st.session_state.costs = []


def select_model() -> ChatOpenAI:
    model = st.sidebar.radio(
        "Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-0613"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k-0613"
    elif model == "GPT-4":
        st.session_state.model_name = "gpt-4"
    else:
        raise ValueError(
            "Unknown model was chosen. This error is not excepted to occur in normal usage."
        )

    # 300: 本文以外の指示のtoken数 (以下同じ)
    st.session_state.max_token = OpenAI.modelname_to_contextsize(
        st.session_state.model_name
    ) - MAX_SUMMARY_CHARACTERS
    return ChatOpenAI(
        temperature=MODEL_TEMPERATURE,
        model_name=st.session_state.model_name
    )


def get_url_input() -> str:
    url = st.text_input("Youtube URL: ", key="input")
    return url


def get_document(url: str) -> List[Document]:
    with st.spinner("Fetching Content ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # タイトルや再生数も取得できる
            language=['en', 'ja']  # 英語→日本語の優先順位で字幕を取得
        )
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.model_name,
            chunk_size=st.session_state.max_token,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )
        return loader.load_and_split(text_splitter=text_splitter)


def summarize(llm, docs):
    PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            verbose=True,
            map_prompt=PROMPT,
            combine_prompt=PROMPT
        )
        response = chain(
            {
                "input_documents": docs,
                # token_max を指示しないと、GPT3.5など通常の
                # モデルサイズに合わせた内部処理になってしまうので注意
                "token_max": st.session_state.max_token
            },
            return_only_outputs=True
        )

    return response['output_text'], cb.total_cost


def main():
    init_page()
    llm = select_model()

    container = st.container()
    response_container = st.container()

    url: str = ""
    output_text: List[Document] = None
    with container:
        url = get_url_input()
        if url:
            document = get_document(url)
            if document:
                with st.spinner("ChatGPT is typing ..."):
                    output_text, cost = summarize(llm, document)
                st.session_state.costs.append(cost)
            else:
                output_text = None

    if not url:
        with response_container:
            st.write("No URL give.")

    if output_text:
        with response_container:
            st.markdown("## Summary")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    main()
