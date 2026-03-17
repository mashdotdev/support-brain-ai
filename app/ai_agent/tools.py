from agents import function_tool

from app.rag.retrieval import query


@function_tool
async def search_docs(question: str, category: str | None = None, k: int = 5) -> str:
    """Search the documentation to answer a support question.
    Use this whenever the user asks anything about the product or needs support.

    Args:
        question: The user's support question.
        category: Optional doc category to restrict the search to.
        k: Number of documents to retrieve (default 5).
    """
    result = await query(question=question, category=category, k=k)

    if not result["confident"]:
        return "I don't have enough information in the docs to answer this confidently. This may need human support."

    citations = "\n".join(result["citations"])
    return f"{result['answer']}\n\nSources:\n{citations}"
