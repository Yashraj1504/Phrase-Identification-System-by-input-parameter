import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit app title
st.title("Phrase Identification System")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Chat Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
    Identify which of the following phrases are relevant to the given paragraph.
    Paragraph: {context}
    Phrases: {phrases}
    Provide the relevant phrases as 'Matched' and the non-relevant phrases as 'Unmatched'.
    Respond only in this exact format:
    Matched: [list of matched phrases]
    Unmatched: [list of unmatched phrases]
    """
)

# Initialize Embedding Model and Vector Store
def vector_embedding(paragraph, phrases):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    combined_data = [paragraph] + phrases
    vectors = FAISS.from_texts(combined_data, embeddings)
    return vectors

# Input fields for paragraph and phrases
paragraph_input = st.text_area("Enter the paragraph:")
phrases_input = st.text_area("Enter phrases (comma-separated):")

# Process the phrases input
phrases_list = [phrase.strip() for phrase in phrases_input.split(",") if phrase.strip()]

# Button to start processing
if st.button("Identify Relevant Phrases"):
    if paragraph_input and phrases_list:
        vectors = vector_embedding(paragraph_input, phrases_list)
        retriever = vectors.as_retriever()

        document_chain = prompt_template.format(
            context=paragraph_input, phrases=", ".join(phrases_list)
        )

        start = time.process_time()
        response = llm.invoke(document_chain)
        end = time.process_time()

        # Extract content from the AIMessage object
        if hasattr(response, 'content'):
            response_text = response.content.strip()
        else:
            response_text = str(response).strip()

        # Extract matched and unmatched phrases from response
        matched_phrases = []
        unmatched_phrases = []

        # Logic to parse matched and unmatched phrases from LLM output
        if "Matched:" in response_text and "Unmatched:" in response_text:
            matched_section = response_text.split("Matched:")[1].split("Unmatched:")[0].strip()
            unmatched_section = response_text.split("Unmatched:")[1].strip()

            matched_phrases = [phrase.strip() for phrase in matched_section.split("\n") if phrase.strip()]
            unmatched_phrases = [phrase.strip() for phrase in unmatched_section.split("\n") if phrase.strip()]

        # Display results
        st.write(f"Matched: {matched_phrases}")
        st.write(f"Unmatched: {unmatched_phrases}")
        st.write(f"Response Time: {end - start:.2f} seconds")

    else:
        st.error("Please provide both a paragraph and phrases.")
