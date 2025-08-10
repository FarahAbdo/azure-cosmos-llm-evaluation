import os
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv

load_dotenv()

def setup_cosmos_db():
    """Setup Cosmos DB database and containers"""
    client = CosmosClient(
        url=os.getenv('COSMOS_ENDPOINT'),
        credential=os.getenv('COSMOS_KEY')
    )
    
    # Create database
    database = client.create_database_if_not_exists(id=os.getenv('COSMOS_DB_NAME', 'eval_db'))
    print(f"✓ Database '{database.id}' ready")
    
    # Vector policy for 1536-dimension embeddings (text-embedding-3-large)
    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/vector",
                "dataType": "float32", 
                "distanceFunction": "cosine",
                "dimensions": 1536
            },
            {
                "path": "/query_vector", 
                "dataType": "float32",
                "distanceFunction": "cosine", 
                "dimensions": 1536
            }
        ]
    }
    
    vector_indexing_policy = {
        "vectorIndexes": [
            {"path": "/vector", "type": "diskANN"},
            {"path": "/query_vector", "type": "diskANN"}
        ]
    }
    
    # Documents container
    docs_container = database.create_container_if_not_exists(
        id=os.getenv('COSMOS_DOCS_CONTAINER', 'documents'),
        partition_key=PartitionKey(path="/id"),
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=vector_indexing_policy
    )
    print(f"✓ Documents container '{docs_container.id}' ready")
    
    # Cache container  
    cache_container = database.create_container_if_not_exists(
        id=os.getenv('COSMOS_CACHE_CONTAINER', 'cache'),
        partition_key=PartitionKey(path="/id"),
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=vector_indexing_policy
    )
    print(f"✓ Cache container '{cache_container.id}' ready")

def load_sample_data():
    """Load sample documents"""
    from cosmos_evaluator import CosmosRAGEvaluator
    import asyncio
    
    async def _load():
        evaluator = CosmosRAGEvaluator()
        
        sample_docs = [
            {
                "id": "doc_1",
                "content": "Azure Cosmos DB provides integrated vector search capabilities with DiskANN indexing for high-performance similarity searches in AI applications.",
                "metadata": {"topic": "cosmos_vector_search"}
            },
            {
                "id": "doc_2", 
                "content": "Semantic caching in RAG systems reduces costs by storing and reusing responses for similar queries based on vector similarity thresholds.",
                "metadata": {"topic": "semantic_caching"}
            },
            {
                "id": "doc_3",
                "content": "Hybrid search combines vector similarity with traditional keyword search using techniques like Reciprocal Rank Fusion (RRF) for better retrieval accuracy.",
                "metadata": {"topic": "hybrid_search"}
            }
        ]
        
        for doc in sample_docs:
            # Generate embedding
            embedding = await evaluator.get_embedding(doc["content"])
            doc["vector"] = embedding
            
            # Insert document
            evaluator.docs_container.create_item(body=doc)
            print(f"✓ Loaded {doc['id']}")
    
    asyncio.run(_load())

if __name__ == "__main__":
    print("🚀 Setting up Cosmos DB for LLM evaluation...")
    setup_cosmos_db()
    print("\n📚 Loading sample data...")
    load_sample_data()
    print("\n✅ Setup complete! Run your evaluator now.")
