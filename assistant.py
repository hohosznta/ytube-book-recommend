from textwrap import dedent
from phi.assistant import Assistant
from langchain_community.embeddings import OpenAIEmbeddings
import os

from typing import Optional

from phi.assistant import Assistant
from phi.llm.ollama import Ollama
from phi.knowledge.langchain import LangChainKnowledgeBase
from openai import OpenAI
import qdrant_client

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

def get_chunk_summarizer(
    model: str = "bnksys/yanolja-eeve-korean-instruct-10.8b",
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq Research Assistant."""

    return Assistant(
        name="groq_youtube_pre_processor",
        llm=Ollama(model=model),
        description="당신은 유튜브 동영상 요약을 담당하는 중앙일보의 시니어 기자입니다. 한국어로 작성하십시오.",
        instructions=[
            "유튜브 동영상의 대본을 제공받게 됩니다.",
            "대본을 신중히 읽고 주요 사실과 세부 사항을 포함한 철저한 보고서를 작성하십시오.",
            "요약에는 가능한 많은 세부 사항과 사실을 포함하십시오.",
            "당신의 보고서는 최종 뉴욕 타임스 보고서를 작성하는 데 사용될 것입니다.",
            "각 섹션에 관련 제목을 제공하고 각 섹션에서 세부 사항/사실/절차를 명시하십시오."
            "기억하십시오: 뉴욕 타임스에 기사를 작성하는 것이므로 보고서의 품질이 중요합니다.",
            "보고서가 <report_format>에 따라 제대로 형식화되었는지 확인하십시오.",
            "총 분량을 500자 내외로 조절하십시오."
        ],
        add_to_system_prompt=dedent(
            """
            <report_format>
            ### 개요
            {동영상의 개요를 제공하세요}

            ### 섹션 1
            {이 섹션에서 세부 사항/사실/절차를 제공하세요}

            ... 필요에 따라 더 많은 섹션 추가 ...

            ### 주요 요점
            {동영상에서 얻은 주요 요점을 제공하세요}
            </report_format>
            """
        ),
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )


def get_video_summarizer(
    model: str = "bnksys/yanolja-eeve-korean-instruct-10.8b",
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq Research Assistant."""

    return Assistant(
        name="groq_video_summarizer",
        llm=Ollama(model=model),
        description="한국어로 작성하십시오. 당신은 중앙일보의 시니어 기자입니다. 유튜브 비디오에 대한 요약 보고서를 작성하는 것이 목표입니다.",
        instructions=[
            "당신은 다음을 제공받을 것입니다."
            "  1. 유튜브 동영상 링크 및 동영상에 대한 정보"
            "  2. 선후배 연구자들의 요약을 사전 처리"
            "정보를 신중하게 처리하고 내용에 대해 생각해 보십시오.",
            "그런 다음 아래 제공되는 <report_format>에서 중앙일보의 가치 있는 최종 보고서를 생성합니다.",
            "당신의 보고서를 매력적이고, 정보를 제공하고, 잘 구성된 보고서로 만드십시오.",
            "보고서를 섹션으로 나누고 마지막에 중요한 정보를 제공합니다.",
            "제목이 비디오에 대한 마크다운 링크인지 확인하십시오.",
            "섹션에 관련 제목을 부여하고 각 섹션의 세부 정보/사실/프로세스를 제공합니다."
            "당신은 중앙일보에 기고하고 있기 때문에 보고서의 질이 중요하다는 것을 기억하세요. ",
        ],
        add_to_system_prompt=dedent(
          """
            </report_format>
            ### 주요 요점
            {비디오에서 얻은 주요 요점을 세 줄로 요약하여 제공하세요}

            </report_format>
            """
        ),
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )

def query_qdrant(query, collection_name, vector_name='combined', top_k=3):
    qdrant = qdrant_client.QdrantClient(host="localhost", port=6333)
    client = OpenAI( api_key="sk-proj-yHTYlXnG7IY2kjYyz28ST3BlbkFJHuJ9QIpnn5W2CZHSH3Ju")

    embedded_query = client.embeddings.create(
        input=query,
        model="text-embedding-3-small",
    ).data[0].embedding 

    query_results = qdrant.search(
        collection_name=collection_name,
        query_vector=(
            vector_name, embedded_query
        ),
        limit=top_k, 
        query_filter=None
    )
    
    return query_results

def openai_explained():
    client = OpenAI( model="gpt-3.5-turbo", api_key="sk-proj-yHTYlXnG7IY2kjYyz28ST3BlbkFJHuJ9QIpnn5W2CZHSH3Ju")

    embedded_query = client.embeddings.create(
        input=query,
        model="text-embedding-3-small",
    ).data[0].embedding 

    query_results = qdrant.search(
        collection_name=collection_name,
        query_vector=(
            vector_name, embedded_query
        ),
        limit=top_k, 
        query_filter=None
    )
    
    return query_results

