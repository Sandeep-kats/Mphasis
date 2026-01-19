import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

 
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


tickets_df = pd.read_csv("Details.csv")  
tickets_df["clean_text"] = tickets_df["Description"].apply(lambda x: x.strip().replace("\n", " "))


articles = [
    Document(page_content="Resetting your password can be done via the account settings page."),
    Document(page_content="To integrate Slack, go to the integrations tab and enable Slack notifications."),
    Document(page_content="Google Sheets integration requires API credentials and enabling the Sheets API."),
    Document(page_content="Troubleshooting login issues often involves clearing cookies or resetting credentials."),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
article_chunks = splitter.split_documents(articles)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = DocArrayInMemorySearch.from_documents(article_chunks, embeddings)
retriever = vectorstore.as_retriever()


llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=HF_TOKEN
    )
)


tag_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that tags support tickets.
Analyze the ticket content and suggest 2-3 relevant tags.

Ticket: {ticket}
Tags:
""")

qa_prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions strictly using the provided context.

Context: {context}

Question: {question}

Answer:
""")

tag_chain = tag_prompt | llm
qa_chain = qa_prompt | llm


def tag_ticket(ticket_text: str):
    response = tag_chain.invoke({"ticket": ticket_text})
    return response.content

def recommend_article(query: str):
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    response = qa_chain.invoke({"context": context, "question": query})
    return response.content, retrieved_docs

st.title("AI-Powered Ticket Tagging & Article Recommendation")

ticket_id = st.number_input("Enter Ticket ID", min_value=1, max_value=len(tickets_df), step=1)
ticket_text = tickets_df.loc[ticket_id - 1, "clean_text"]

st.subheader("Ticket Content")
st.write(ticket_text)

if st.button("Run AI Analysis"):
    tags = tag_ticket(ticket_text)
    st.subheader("Suggested Tags")
    st.write(tags)

    answer, docs = recommend_article(ticket_text)
    st.subheader("Recommended Answer")
    st.write(answer)

    st.subheader("Matched Articles")
    for d in docs:
        st.write("-", d.page_content)
