from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import getpass
from ingest import ingestdata

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

def retrievalgeneration(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    You are an e-commerce chatbot that is expert in product recommendations and customer queries.
    You analyze product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from going off-topic.
    Your responses should be concise and helpful.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """


    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__=='__main__':
    vstore = ingestdata("Done")
    chain  = retrievalgeneration(vstore)
    print(chain.invoke("Can you tell me which are the best shoes?"))