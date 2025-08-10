import asyncio
from cosmos_evaluator import CosmosRAGEvaluator, calculate_metrics

async def main():
    """Enhanced demo with more detailed output"""
    
    print("🚀 Azure Cosmos DB RAG Evaluation Demo")
    print("=" * 50)
    
    evaluator = CosmosRAGEvaluator()
    
    test_queries = [
        "What is vector search in Cosmos DB?",
        "How does semantic caching reduce cost?", 
        "Benefits of hybrid search in RAG?",
        "What is vector search in Cosmos DB?"  # Duplicate to test caching
    ]
    
    results = []
    
    for i, question in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {question}")
        
        result = await evaluator.evaluate_query(
            question, 
            k=3, 
            max_distance=1.0  # Optional: filter very dissimilar results
        )
        
        results.append(result)
        
        print(f"   💬 Answer: {result['answer'][:120]}...")
        print(f"   📊 Retrieval Score: {result['retrieval_score']:.3f}")
        print(f"   ⏱️  Latency: {result['latency_ms']:.1f}ms")
        print(f"   💾 Cached: {'✅ Yes' if result['cached'] else '❌ No'}")
        print(f"   📚 Retrieved Contexts: {len(result['contexts'])}")
        
        # Show context details
        if result['contexts']:
            print(f"   🔍 Top Context Distance: {result['contexts'][0].get('distance', 0):.3f}")
    
    # Summary metrics
    metrics = calculate_metrics(results)
    
    print("\n" + "=" * 50)
    print("📈 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Queries: {int(metrics['total_queries'])}")
    print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")
    print(f"Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
    print(f"Avg Retrieval Score: {metrics['avg_retrieval_score']:.3f}")
    
    # Performance insights
    cached_count = sum(1 for r in results if r['cached'])
    if cached_count > 0:
        print(f"\n💡 Semantic caching saved {cached_count} expensive LLM generation calls!")
        uncached_avg = sum(r['latency_ms'] for r in results if not r['cached']) / (len(results) - cached_count)
        cached_avg = sum(r['latency_ms'] for r in results if r['cached']) / cached_count
        print(f"   Uncached avg latency: {uncached_avg:.1f}ms")
        print(f"   Cached avg latency: {cached_avg:.1f}ms")
        print(f"   Speedup: {uncached_avg/cached_avg:.1f}x faster")

if __name__ == "__main__":
    asyncio.run(main())
