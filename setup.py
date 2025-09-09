import os
from setuptools import setup, find_packages
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv

def setup_cosmos_db():
    """Setup Cosmos DB database and containers"""
    load_dotenv()  # Load environment variables

    client = CosmosClient(
        url=os.getenv('COSMOS_ENDPOINT'),
        credential=os.getenv('COSMOS_KEY')
    )
    
    # Create database
    database_name = os.getenv('COSMOS_DB_NAME', 'llm_evaluation')
    database = client.create_database_if_not_exists(id=database_name)
    print(f"✓ Database '{database.id}' ready")
    
    # Vector policy for 1536-dimension embeddings (text-embedding-3-large)
    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/vector",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": int(os.getenv('VECTOR_DIMENSIONS', 1536))  # Should match the embedding dimension used in llm_evaluation dataset
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
    
    # Evaluations container
    evaluations_container = database.create_container_if_not_exists(
        id=os.getenv('COSMOS_CONTAINER_NAME', 'evaluations'),
        partition_key=PartitionKey(path="/query_id"),
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=vector_indexing_policy
    )
    print(f"✓ Evaluations container '{evaluations_container.id}' ready")
    
    # Semantic cache container  
    cache_container = database.create_container_if_not_exists(
        id=os.getenv('COSMOS_CACHE_CONTAINER', 'semantic_cache'),
        partition_key=PartitionKey(path="/cache_id"),
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=vector_indexing_policy
    )
    print(f"✓ Cache container '{cache_container.id}' ready")

setup(
    name='modern-rag-evaluator',
    version='0.1.0',
    description='RAG Evaluation Toolkit with Azure OpenAI and Cosmos DB',
    long_description="""
    A comprehensive RAG (Retrieval-Augmented Generation) evaluation toolkit 
    that leverages Azure OpenAI and Cosmos DB for advanced metrics and semantic caching.
    
    Features:
    - RAGAS Metrics Evaluation
    - Semantic Caching
    - Azure OpenAI Integration
    - Cosmos DB Vector Search Support
    """,
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/modern-rag-evaluator',
    packages=find_packages(),
    install_requires=[
        'azure-cosmos',
        'python-dotenv',
        'openai',
        'nltk',
        'rouge-score',
        'pandas',
        'rich',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'rag-evaluator=pdf_evaluator:main',
            'setup-cosmos-db=setup:setup_cosmos_db',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    keywords='rag evaluation openai azure nlp machine-learning',
    extras_require={
        'dev': [
            'pytest',
            'pytest-asyncio',
            'mypy',
            'black',
            'flake8'
        ]
    },
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/modern-rag-evaluator/issues',
        'Source': 'https://github.com/yourusername/modern-rag-evaluator',
    },
)

