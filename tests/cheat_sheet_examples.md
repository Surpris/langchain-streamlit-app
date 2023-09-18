# Overview 

The codes in this file are based on [KDnuggets: LangChain Cheat Sheet](https://www.kdnuggets.com/2023/08/langchain-cheat-sheet.html).

# LLMs

## An interface for OpenAI GPT-3.5-turbo LLM

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
text = "What do you know about KDnuggets?"
llm(text)
```

## An interface for HugginFace LLM

```python
from langchain import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="togethercomputer/LLaMA-2-7B-32K",
    model_kwargs={"temperature":0,"max_length":64}
)
llm("How old is KDnuggets?")
```

# Prompt Templates

## LangChain facilitates prompt management and optimizationthrough the use of prompt templates.

```python
from langchain import PromptTemplate

template = """Question: {question}
Make the answer more engaging by incorporating puns.
Answer: """

prompt = PromptTemplate.from_template(template)

llm(prompt.format(
    question="Could you provide someinformation on the impact of global warming?"
))
```

# Chains

## Combining LLMs and prompt template can enhance multi-step workflows.

```python
from langchain import LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Could you provide some information on theimpact of global warming?"
llm_chain.run(question)
```

# Agents and Tools

## Tool refers to a function that performs a specific task, suchas Google Search, database lookup, or Python REPL. Agentsuse LLMs to choose a sequence of actions to execute.

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent

tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("Can you tell me the distance between Earth andthe moon? And could you please convert it into miles?Thank you.")
```

# Memory

## LangChain simplifies persistent state management in chainor agent calls with a standard interface

```python
from langchain.chains import ConversationChain
from langchain.memory importConversationBufferMemory

conversation = ConversationChain(
    llm=llm, verbose=True,
    memory=ConversationBufferMemory()
)

conversation.predict(input="How can one overcomeanxiety?")

conversation.predict(input="Tell me more..")
```

# Document Loaders

## By combining language models with your own text data, youcan answer personalized queries. You can load CSV,Markdown, PDF, and more.

```python
from langchain.document_loaders import TextLoader

raw_document =TextLoader("/work/data/Gregory.txt").load()
```

# Vector Stores

## One common method for storing and searchingunstructured data is to embed it as vectors, then embedqueries and retrieve the most similar vectors.

```python
from langchain.embeddings.openai importOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
documents = text_splitter.split_documents(raw_document)

# Vector Store
db = FAISS.from_documents(documents,OpenAIEmbeddings())

# Similarity Search
query = "When was Gregory born?"
docs = db.similarity_search(query)

print(docs[0].page_content)
```

## A retriever is an interface that returns documents based onan unstructured query. When combined with LLM, itgenerates a natural response instead of simply displayingthe text from the document.

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
qa_chain({"query": "When was Gregory born?"})
```
