from textwrap import dedent
from typing import Optional

from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

def get_investment_research_assistant(
    llm_model: str = "llama3",
    embeddings_model: str = "llama3",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Local RAG Assistant."""

    # Define the embedder based on the embeddings model
    embedder = OllamaEmbedder(model=embeddings_model, dimensions=4096)
    embeddings_model_clean = embeddings_model.replace("-", "_")
    if embeddings_model == "nomic-embed-text":
        embedder = OllamaEmbedder(model=embeddings_model, dimensions=768)
    elif embeddings_model == "phi3":
        embedder = OllamaEmbedder(model=embeddings_model, dimensions=3072)
    # Define the knowledge base
    knowledge = AssistantKnowledge(
        vector_db=PgVector2(
            db_url=db_url,
            collection=f"local_rag_documents_{embeddings_model_clean}",
            embedder=embedder,
        ),
        # 3 references are added to the prompt
        num_documents=20,
    )

    return Assistant(
        name="investment_research_assistant_ollama",
        run_id=run_id,
        user_id=user_id,
        llm=Ollama(model=llm_model),
        storage=PgAssistantStorage(table_name="local_rag_assistant", db_url=db_url),
        knowledge_base=knowledge,
        description="You are a Senior Investment Analyst for Goldman Sachs tasked with producing a research report for a very important client.",
        instructions=[
            "You will be provided with a stock and information from junior researchers.",
            "Carefully read the research and generate a final - Goldman Sachs worthy investment report.",
            "Make your report engaging, informative, and well-structured.",
            "When you share numbers, make sure to include the units (e.g., millions/billions) and currency.",
            "REMEMBER: This report is for a very important client, so the quality of the report is important.",
            "Make sure your report is properly formatted and follows the <report_format> provided below.",
        ],
        markdown=True,
        add_datetime_to_instructions=True,
        add_to_system_prompt=dedent("""
        <report_format>
        ## [Company Name]: Investment Report

        ### **Overview**
        {give a brief introduction of the company and why the user should read this report}
        {make this section engaging and create a hook for the reader}

        ### Core Metrics
        {provide a summary of core metrics and show the latest data}
        - Current price: {current price}
        - 52-week high: {52-week high}
        - 52-week low: {52-week low}
        - Market Cap: {Market Cap} in billions
        - P/E Ratio: {P/E Ratio}
        - Earnings per Share: {EPS}
        - 50-day average: {50-day average}
        - 200-day average: {200-day average}
        - Analyst Recommendations: {buy, hold, sell} (number of analysts)

        ### Financial Performance
        {provide a detailed analysis of the company's financial performance}

        ### Growth Prospects
        {analyze the company's growth prospects and future potential}

        ### News and Updates
        {summarize relevant news that can impact the stock price}

        ### Insider Purchases                            
        {analyze how insiders bought in the last 6 months}
        {this should be a paragraph not a table}
                                    
        ### Upgrades and Downgrades
        {share 2 upgrades or downgrades including the firm, and what they upgraded/downgraded to}
        {this should be a paragraph not a table}

        ### [Summary]
        {give a summary of the report and what are the key takeaways}

        ### [Recommendation]
        {provide a recommendation on the stock along with a thorough reasoning}

        Report generated on: {Month Date, Year (hh:mm AM/PM)}
        </report_format>
        """),
        debug_mode=debug_mode,
    )
