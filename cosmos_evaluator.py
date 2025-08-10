import asyncio
import time
import os
from uuid import uuid4
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from openai import AzureOpenAI

load_dotenv()

class CosmosRAGEvaluator:
    """
    RAG evaluation using Azure Cosmos DB vector search and Azure OpenAI.
    Notes:
    - Uses sync Cosmos SDK methods inside async functions for simplicity.
      For production, consider azure.cosmos.aio or run sync calls in a thread pool.
    - Assumes NoSQL API with vector indexing on c.vector.
    - Distances: lower is better. We convert to similarity for readable scoring.
    """

    def __init__(self):
        # Initialize clients
        self.cosmos_client = CosmosClient(
            url=os.getenv('COSMOS_ENDPOINT'),
            credential=os.getenv('COSMOS_KEY')
        )

        self.openai_client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
        )

        # Get database and containers
        self.database = self.cosmos_client.get_database_client(
            os.getenv('COSMOS_DB_NAME', 'eval_db')
        )
        self.docs_container = self.database.get_container_client(
            os.getenv('COSMOS_DOCS_CONTAINER', 'documents')
        )
        self.cache_container = self.database.get_container_client(
            os.getenv('COSMOS_CACHE_CONTAINER', 'cache')
        )

        # Models (must be Azure OpenAI deployment names)
        self.embedding_model = os.getenv('EMBEDDING_MODEL')  # e.g., "text-embedding-3-large" deployment
        self.chat_model = os.getenv('CHAT_MODEL')            # e.g., "gpt-4o" deployment

        if not self.embedding_model or not self.chat_model:
            raise ValueError("Please set EMBEDDING_MODEL and CHAT_MODEL to your Azure OpenAI deployment names.")

    async def evaluate_query(self, question: str, ground_truth: Optional[str] = None,
                             k: int = 3, max_distance: Optional[float] = None) -> Dict[str, Any]:
        """
        Main evaluation method for a single query.
        - Generates embedding
        - Checks semantic cache (distance threshold uses <)
        - Performs vector search if cache miss
        - Generates answer using retrieved contexts
        - Computes retrieval score (distance -> similarity)
        - Caches result
        """
        start_time = time.time()

        # 1) Generate query embedding
        query_embedding = await self.get_embedding(question)

        # 2) Check semantic cache (use distance threshold; lower = more similar)
        # Default threshold can be small (e.g., 0.2–0.3) depending on your embedding/model
        cache_threshold = float(os.getenv('CACHE_DISTANCE_THRESHOLD', '0.25'))
        cached = await self.check_cache(query_embedding, threshold=cache_threshold)

        if cached:
            return {
                'question': question,
                'answer': cached['answer'],
                'contexts': cached['contexts'],
                'retrieval_score': cached.get('retrieval_similarity', 0.0),
                'latency_ms': (time.time() - start_time) * 1000,
                'cached': True
            }

        # 3) Vector search
        contexts = await self.vector_search(query_embedding, k=k, max_distance=max_distance)

        # 4) Generate answer
        answer = await self.generate_answer(question, contexts)

        # 5) Calculate retrieval score as similarity (0,1], higher is better
        retrieval_score = self.calculate_retrieval_similarity(contexts)

        # 6) Cache result
        await self.cache_result(query_embedding, answer, contexts, retrieval_score)

        return {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'retrieval_score': retrieval_score,
            'latency_ms': (time.time() - start_time) * 1000,
            'cached': False
        }

    async def get_embedding(self, text: str) -> List[float]:
        """Get text embedding from Azure OpenAI using deployment name."""
        resp = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return resp.data[0].embedding

    async def vector_search(self, query_vector: List[float], k: int = 3,
                            max_distance: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Vector similarity search using Cosmos DB NoSQL.
        - Uses VectorDistance: lower = better.
        - Orders ascending by distance.
        - Optional WHERE clause to filter by max_distance.
        """
        where_clause = "WHERE VectorDistance(c.vector, @embedding) < @maxDistance" if max_distance is not None else ""
        query = f"""
        SELECT TOP @k c.id, c.content, c.metadata,
               VectorDistance(c.vector, @embedding) AS distance
        FROM c
        {where_clause}
        ORDER BY VectorDistance(c.vector, @embedding)
        """

        params = [{"name": "@embedding", "value": query_vector}, {"name": "@k", "value": k}]
        if max_distance is not None:
            params.append({"name": "@maxDistance", "value": max_distance})

        try:
            results = self.docs_container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True
            )
            return list(results)
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    async def generate_answer(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate answer using retrieved contexts via Azure OpenAI chat."""
        context_text = "\n\n".join([ctx.get('content', '') for ctx in contexts])

        messages = [
            {"role": "system", "content": "Answer based only on the provided context. If context is insufficient, say so clearly."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ]

        try:
            resp = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.1,
                max_tokens=400
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {e}"

    def calculate_retrieval_similarity(self, contexts: List[Dict[str, Any]]) -> float:
        """
        Convert distances to similarities for a readable, higher-is-better score.
        similarity = 1 / (1 + distance) in (0, 1].
        """
        if not contexts:
            return 0.0
        sims = [1.0 / (1.0 + float(ctx.get('distance', 0.0))) for ctx in contexts]
        return sum(sims) / len(sims) if sims else 0.0

    async def check_cache(self, query_vector: List[float], threshold: float = 0.25) -> Optional[Dict[str, Any]]:
        """
        Check semantic cache by distance (lower = more similar).
        Uses '< threshold' and orders by distance ascending to get the closest match.
        """
        query = """
        SELECT TOP 1 *
        FROM c
        WHERE VectorDistance(c.query_vector, @embedding) < @threshold
        ORDER BY VectorDistance(c.query_vector, @embedding)
        """
        try:
            results = self.cache_container.query_items(
                query=query,
                parameters=[
                    {"name": "@embedding", "value": query_vector},
                    {"name": "@threshold", "value": threshold}
                ],
                enable_cross_partition_query=True
            )
            items = list(results)
            return items[0] if items else None
        except Exception:
            return None

    async def cache_result(self, query_vector: List[float], answer: str,
                           contexts: List[Dict[str, Any]], retrieval_similarity: float) -> None:
        """Cache evaluation result with a stable UUID id."""
        cache_item = {
            'id': f"cache_{uuid4()}",
            'query_vector': query_vector,
            'answer': answer,
            'contexts': contexts,
            'retrieval_similarity': retrieval_similarity,
            'timestamp': time.time()
        }
        try:
            self.cache_container.create_item(body=cache_item)
        except Exception as e:
            print(f"Cache error: {e}")


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics for a batch of evaluations.
    - cache_hit_rate
    - avg_latency_ms
    - avg_retrieval_score (similarity)
    """
    if not results:
        return {'total_queries': 0, 'cache_hit_rate': 0.0, 'avg_latency_ms': 0.0, 'avg_retrieval_score': 0.0}

    total = len(results)
    cached = sum(1 for r in results if r.get('cached', False))
    avg_latency = sum(float(r.get('latency_ms', 0.0)) for r in results) / total
    avg_retrieval_score = sum(float(r.get('retrieval_score', 0.0)) for r in results) / total

    return {
        'total_queries': float(total),
        'cache_hit_rate': cached / total if total > 0 else 0.0,
        'avg_latency_ms': avg_latency,
        'avg_retrieval_score': avg_retrieval_score
    }


# Example usage for quick test
async def _demo():
    evaluator = CosmosRAGEvaluator()
    queries = [
        {"q": "What is vector search in Cosmos DB?"},
        {"q": "How does semantic caching reduce cost?"},
        {"q": "Benefits of hybrid search in RAG?"}
    ]

    results = []
    for item in queries:
        res = await evaluator.evaluate_query(item["q"], ground_truth=None, k=3, max_distance=None)
        results.append(res)
        print(f"\nQ: {item['q']}\nA: {res['answer']}\nScore: {res['retrieval_score']:.3f} | Cached: {res['cached']} | Latency: {res['latency_ms']:.1f}ms")

    summary = calculate_metrics(results)
    print("\nSummary:", summary)


if __name__ == "__main__":
    asyncio.run(_demo())
