import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

#LLM Model initialization:

model = Ollama(model='llama3',temperature=0.1,)


def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text_pdf = ""
    for page in pdf_reader.pages:
        text_pdf += page.extract_text()
    return text_pdf


def main():

    st.set_page_config(page_title='Academic Partner',page_icon='📑',
                   layout="centered",initial_sidebar_state="collapsed")
    st.header('Academic Partner PDF QA Generator!📔',anchor=False,divider='blue',help='Academic partner helps your exam prep by making Q&A out of your PDF')

    pdf = st.file_uploader("Upload your PDF here!",type='pdf')
    submit = st.button("Generate Q&A Here!")


    if pdf is not None:
        text_pdf = extract_text_from_pdf(pdf)
        st.success("Your pdf uploaded successfully!!")
    else:
        st.error("Please upload your pdf...")

    if submit:
        st.spinner("Your QA is getting ready!!!")
        if text_pdf:
            # Initialize Text Splitter for question generation
            text_splitter_question_gen = TokenTextSplitter(chunk_size=10000, chunk_overlap=200)

            # Split text into chunks for question generation
            text_chunks_question_gen = text_splitter_question_gen.split_text(text_pdf)

            # Convert chunks into Documents for question generation
            docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]

            # Initialize Text Splitter for answer generation
            text_splitter_answer_gen = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)

            # Split documents into chunks for answer generation
            docs_answer_gen = text_splitter_answer_gen.split_documents(docs_question_gen)


            prompt_template_questions = """
            You are an expert at creating practice questions based on study material.
            Your goal is to prepare a student for their exam. 
            You do this by asking questions about the text below:

            ------------
            {text}
            ------------

            Create questions that will prepare the student for their exam.
            Make sure not to lose any important information.

            QUESTIONS:
            """
            PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])

            refine_template_questions = ("""
            You are an expert at creating practice questions based on study material.
            Your goal is to help a student prepare for an exam.
            We have received some practice questions to a certain extent: {existing_answer}.
            We have the option to refine the existing questions or add new ones.
            (only if necessary) with some more context below.
            ------------
            {text}
            ------------

            Given the new context, refine the original questions in English.
            If the context is not helpful, please provide the original questions.
            QUESTIONS:
            """
            )
            REFINE_PROMPT_QUESTIONS = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template_questions,
            )

            # Initialize Large Language Model for question generation
            llm_question_gen = Ollama(model='llama3',temperature=0.6,)

            from langchain.chains.summarize import load_summarize_chain

            # Initialize question generation chain
            question_gen_chain = load_summarize_chain(llm = llm_question_gen, chain_type = "refine", verbose = False, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)


            # Run question generation chain
            questions = question_gen_chain.invoke(docs_question_gen)

            # Initialize Large Language Model for answer generation
            llm_answer_gen = Ollama(model='llama3',temperature=0.1,)

            from langchain_community.embeddings import HuggingFaceEmbeddings

            # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            from langchain_community.vectorstores import Chroma

            # Initialize vector store for answer generation
            vector_store = Chroma.from_documents(docs_answer_gen, embeddings)

            from langchain.chains import RetrievalQA

            # Initialize retrieval chain for answer generation
            answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever(k=2))


            # Split generated questions into a list of questions
            question_list = questions.split("\\n")

            # Answer each question and save to a file
            for question in question_list:
                st.write("Question: \n", question)
                answer = answer_gen_chain.invoke(question)
                st.write("Answer: \n", answer)
            # Save answer to file
            with open("answers.txt", "a") as f:
                f.write("Question: " + question + "\\n")
                f.write("Answer: " + answer + "\\n")
                f.write("--------------------------------------------------\\n\\n")

if __name__ == '__main__':
    main()