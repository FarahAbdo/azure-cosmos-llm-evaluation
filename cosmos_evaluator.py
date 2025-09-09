import os
import json
import asyncio
import logging
import pandas as pd
import numpy as np
import hashlib
import time
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid
from azure.cosmos import CosmosClient, PartitionKey
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer
from openai import AzureOpenAI

# Rich CLI imports
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.markdown import Markdown
from rich.align import Align
from rich.padding import Padding

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize rich console
console = Console()

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Azure OpenAI CONFIG ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
LLM_DEPLOYMENT_NAME = os.getenv("LLM_DEPLOYMENT_NAME")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.7))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")

@dataclass
class EvaluationConfig:
    cosmos_endpoint: str
    cosmos_key: str
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    embedding_deployment_name: str
    llm_deployment_name: str
    llm_temperature: float
    llm_top_p: float
    llm_max_tokens: int
    database_name: str = "llm_evaluation"
    container_name: str = "evaluations"
    cache_container: str = "semantic_cache"
    similarity_threshold: float = 0.85
    cache_ttl_hours: int = 24
    batch_size: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 50

@dataclass
class EvaluationResult:
    query_id: str
    query: str
    retrieved_context: List[str]
    generated_answer: str
    expected_answer: Optional[str]
    timestamp: datetime
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_relevancy: float
    rouge_scores: Dict[str, float]
    cached: bool = False
    document_source: Optional[str] = None

class ModernCLILogger:
    """Custom logger for rich console output"""
    def __init__(self, console: Console):
        self.console = console
        self.start_time = time.time()
    
    def info(self, message: str, style: str = "dim"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[{style}]{timestamp}[/{style}] {message}")
    
    def success(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [green]✓[/green] {message}")
    
    def warning(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [yellow]⚠[/yellow] {message}")
    
    def error(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [red]✗[/red] {message}")

logger = ModernCLILogger(console)

def hash_text(text: str) -> str:
    """Deterministic hash for use as cache key."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_azure_openai_embedding_with_cache(texts, evaluator, azure_client, progress_task=None, progress=None):
    """Get embeddings from cache or Azure OpenAI API, store in cache if missing."""
    if not isinstance(texts, list):
        texts = [texts]
    results = []
    
    for i, text in enumerate(texts):
        if progress and progress_task:
            progress.update(progress_task, advance=1)
        
        embedding = None
        hash_key = hash_text(text)
        
        # Look up in semantic_cache
        if evaluator.semantic_cache_container:
            try:
                cache_item = evaluator.semantic_cache_container.read_item(item=hash_key, partition_key=hash_key)
                embedding = cache_item.get("embedding", None)
                if embedding:
                    logger.info(f"[green]Cache hit[/green] for: {text[:50]}...", "dim")
            except Exception:
                embedding = None
        
        # Fallback: query Azure OpenAI API and write result to cache
        if embedding is None:
            logger.info(f"[yellow]API call[/yellow] for: {text[:50]}...", "dim")
            try:
                response = azure_client.embeddings.create(
                    input=text,
                    model=evaluator.config.embedding_deployment_name
                )
                embedding = response.data[0].embedding
                
                # Store in cache
                if evaluator.semantic_cache_container:
                    try:
                        evaluator.semantic_cache_container.upsert_item({
                            "id": hash_key,
                            "cache_id": hash_key,
                            "text": text,
                            "embedding": embedding,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Cache write error: {e}")
            except Exception as e:
                logger.error(f"Embedding API error: {e}")
                embedding = [0.0] * 1536  # fallback
        
        results.append(embedding)
    
    return results if len(results) > 1 else results[0]

def generate_azure_openai_response(prompt, config, progress_task=None, progress=None):
    """Generate LLM response using Azure OpenAI API."""
    if progress and progress_task:
        progress.update(progress_task, advance=1)
    
    try:
        azure_client = AzureOpenAI(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version
        )
        
        response = azure_client.chat.completions.create(
            model=config.llm_deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
            max_tokens=config.llm_max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return f"Error generating response: {str(e)}"

class ModernRAGEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Initialize Azure OpenAI client
        with console.status("[bold blue]Initializing Azure OpenAI client..."):
            try:
                self.azure_client = AzureOpenAI(
                    azure_endpoint=config.azure_openai_endpoint,
                    api_key=config.azure_openai_api_key,
                    api_version=config.azure_openai_api_version
                )
                logger.success("Azure OpenAI client initialized")
            except Exception as e:
                logger.error(f"Azure OpenAI client error: {e}")
                self.azure_client = None
        
        # Initialize Cosmos DB
        with console.status("[bold blue]Setting up Cosmos DB..."):
            try:
                self.cosmos_client = CosmosClient(config.cosmos_endpoint, config.cosmos_key)
                self._setup_evaluation_database()
                self.semantic_cache_container = self._setup_semantic_cache()
                logger.success("Cosmos DB initialized")
            except Exception as e:
                logger.error(f"Cosmos DB error: {e}")
                self.cosmos_client = None
                self.evaluation_container = None
                self.semantic_cache_container = None
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def _setup_semantic_cache(self):
        try:
            if self.cosmos_client:
                database = self.cosmos_client.get_database_client(self.config.database_name)
                return database.create_container_if_not_exists(
                    id=self.config.cache_container,
                    partition_key=PartitionKey(path="/cache_id"),
                    offer_throughput=400
                )
        except Exception as e:
            logger.error(f"Semantic cache setup error: {e}")
        return None

    def _setup_evaluation_database(self):
        try:
            database = self.cosmos_client.create_database_if_not_exists(id=self.config.database_name)
            self.evaluation_container = database.create_container_if_not_exists(
                id=self.config.container_name,
                partition_key=PartitionKey(path="/query_id"),
                offer_throughput=400
            )
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            self.evaluation_container = None

    def calculate_retrieval_metrics(self, query, retrieved_docs, relevant_docs=None, progress=None, task=None):
        metrics = {}
        try:
            if relevant_docs:
                retrieved_set = set(retrieved_docs)
                relevant_set = set(relevant_docs)
                intersection = retrieved_set.intersection(relevant_set)
                precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
                recall = len(intersection) / len(relevant_set) if relevant_set else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                metrics.update({
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "f1_score": f1
                })
            
            if retrieved_docs and self.azure_client:
                query_embedding = np.array(get_azure_openai_embedding_with_cache(query, self, self.azure_client, task, progress))
                doc_embeddings = np.array(get_azure_openai_embedding_with_cache(retrieved_docs, self, self.azure_client, task, progress))
                
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
                similarities = np.dot(doc_norms, query_norm)
                
                metrics.update({
                    "avg_semantic_similarity": float(np.mean(similarities)),
                    "max_semantic_similarity": float(np.max(similarities)),
                    "min_semantic_similarity": float(np.min(similarities))
                })
        except Exception as e:
            logger.error(f"Retrieval metrics error: {e}")
            metrics = {"error": str(e)}
        
        return metrics

    def calculate_generation_metrics(self, query, generated_answer, retrieved_context, expected_answer=None, progress=None, task=None):
        metrics = {}
        try:
            if not self.azure_client:
                return {"error": "Azure OpenAI client not initialized"}
            
            query_embedding = np.array(get_azure_openai_embedding_with_cache(query, self, self.azure_client, task, progress))
            answer_embedding = np.array(get_azure_openai_embedding_with_cache(generated_answer, self, self.azure_client, task, progress))
            
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            answer_norm = answer_embedding / np.linalg.norm(answer_embedding)
            relevance_score = np.dot(query_norm, answer_norm)
            metrics["answer_relevance"] = float(relevance_score)
            
            if retrieved_context:
                context_text = " ".join(retrieved_context)
                context_embedding = np.array(get_azure_openai_embedding_with_cache(context_text, self, self.azure_client, task, progress))
                context_norm = context_embedding / np.linalg.norm(context_embedding)
                faithfulness_score = np.dot(answer_norm, context_norm)
                metrics["faithfulness"] = float(faithfulness_score)
            
            metrics.update({
                "answer_length": len(generated_answer.split()),
                "answer_char_length": len(generated_answer),
                "sentence_count": len(sent_tokenize(generated_answer)),
            })
            
            sentences = len(sent_tokenize(generated_answer))
            words = len(generated_answer.split())
            metrics["coherence_ratio"] = sentences / max(words, 1)
            
        except Exception as e:
            logger.error(f"Generation metrics error: {e}")
            metrics = {"error": str(e)}
        
        return metrics

    def calculate_ragas_metrics(self, query, generated_answer, retrieved_context, expected_answer=None, progress=None, task=None):
        metrics = {}
        try:
            if not self.azure_client:
                return {"error": "Azure OpenAI client not initialized"}
            
            query_embedding = np.array(get_azure_openai_embedding_with_cache(query, self, self.azure_client, task, progress))
            answer_embedding = np.array(get_azure_openai_embedding_with_cache(generated_answer, self, self.azure_client, task, progress))
            
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            answer_norm = answer_embedding / np.linalg.norm(answer_embedding)
            metrics["answer_relevancy"] = float(np.dot(query_norm, answer_norm))
            
            if retrieved_context:
                context_embeddings = np.array(get_azure_openai_embedding_with_cache(retrieved_context, self, self.azure_client, task, progress))
                context_norms = context_embeddings / np.linalg.norm(context_embeddings, axis=1, keepdims=True)
                context_similarities = np.dot(context_norms, query_norm)
                
                metrics["context_relevancy"] = float(np.mean(context_similarities))
                relevant_contexts = np.sum(context_similarities > 0.5)
                metrics["context_precision"] = float(relevant_contexts / len(retrieved_context))
                
                context_text = " ".join(retrieved_context)
                context_embedding = np.array(get_azure_openai_embedding_with_cache(context_text, self, self.azure_client, task, progress))
                context_norm = context_embedding / np.linalg.norm(context_embedding)
                metrics["faithfulness"] = float(np.dot(answer_norm, context_norm))
            
            if expected_answer and retrieved_context:
                expected_embedding = np.array(get_azure_openai_embedding_with_cache(expected_answer, self, self.azure_client, task, progress))
                expected_norm = expected_embedding / np.linalg.norm(expected_embedding)
                context_embeddings = np.array(get_azure_openai_embedding_with_cache(retrieved_context, self, self.azure_client, task, progress))
                context_norms = context_embeddings / np.linalg.norm(context_embeddings, axis=1, keepdims=True)
                context_similarities = np.dot(context_norms, expected_norm)
                metrics["context_recall"] = float(np.max(context_similarities))
                
        except Exception as e:
            logger.error(f"RAGAS metrics error: {e}")
            metrics = {"error": str(e)}
        
        return metrics

    def calculate_rouge_scores(self, generated_answer, expected_answer):
        if not expected_answer:
            return {}
        try:
            scores = self.rouge_scorer.score(expected_answer, generated_answer)
            return {
                "rouge_1_f": scores['rouge1'].fmeasure,
                "rouge_1_p": scores['rouge1'].precision,
                "rouge_1_r": scores['rouge1'].recall,
                "rouge_2_f": scores['rouge2'].fmeasure,
                "rouge_l_f": scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"ROUGE scores error: {e}")
            return {"rouge_error": str(e)}

    async def evaluate_rag_response(self, query_id, query, retrieved_context, generated_answer,
                                   expected_answer=None, relevant_documents=None, document_source=None,
                                   progress=None, task=None):
        try:
            retrieval_metrics = self.calculate_retrieval_metrics(query, retrieved_context, relevant_documents, progress, task)
            generation_metrics = self.calculate_generation_metrics(query, generated_answer, retrieved_context, expected_answer, progress, task)
            ragas_metrics = self.calculate_ragas_metrics(query, generated_answer, retrieved_context, expected_answer, progress, task)
            rouge_scores = self.calculate_rouge_scores(generated_answer, expected_answer) if expected_answer else {}
            
            result = EvaluationResult(
                query_id=query_id,
                query=query,
                retrieved_context=retrieved_context,
                generated_answer=generated_answer,
                expected_answer=expected_answer,
                timestamp=datetime.utcnow(),
                retrieval_metrics=retrieval_metrics,
                generation_metrics=generation_metrics,
                faithfulness=ragas_metrics.get("faithfulness", 0.0),
                answer_relevancy=ragas_metrics.get("answer_relevancy", 0.0),
                context_precision=ragas_metrics.get("context_precision", 0.0),
                context_recall=ragas_metrics.get("context_recall", 0.0),
                context_relevancy=ragas_metrics.get("context_relevancy", 0.0),
                rouge_scores=rouge_scores,
                document_source=document_source
            )
            
            await self._store_evaluation_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return None

    async def _store_evaluation_result(self, result: EvaluationResult):
        if not self.evaluation_container:
            logger.warning("Cosmos DB container not available, skipping storage")
            return
        
        try:
            result_dict = asdict(result)
            result_dict['timestamp'] = result.timestamp.isoformat()
            if "id" not in result_dict or not result_dict["id"]:
                result_dict["id"] = str(uuid.uuid4())
            self.evaluation_container.upsert_item(result_dict)
        except Exception as e:
            logger.error(f"Storage error: {e}")

    async def batch_evaluate(self, evaluation_data: List[Dict]) -> List[EvaluationResult]:
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        ) as progress:
            
            main_task = progress.add_task(
                f"[cyan]Evaluating {len(evaluation_data)} queries...", 
                total=len(evaluation_data)
            )
            
            for idx, item in enumerate(evaluation_data):
                try:
                    query_desc = f"Query {idx + 1}: {item['query'][:50]}..."
                    progress.update(main_task, description=f"[cyan]{query_desc}")
                    
                    generated_answer = item.get('generated_answer')
                    if not generated_answer:
                        context_text = "\n".join(item.get('retrieved_context', []))
                        prompt = f"Context:\n{context_text}\n\nQuestion: {item['query']}\n\nAnswer:"
                        generated_answer = generate_azure_openai_response(prompt, self.config)
                    
                    result = await self.evaluate_rag_response(
                        query_id=item.get('query_id', f"query_{idx}"),
                        query=item['query'],
                        retrieved_context=item.get('retrieved_context', []),
                        generated_answer=generated_answer,
                        expected_answer=item.get('expected_answer'),
                        relevant_documents=item.get('relevant_documents'),
                        document_source=item.get('document_source')
                    )
                    
                    if result:
                        results.append(result)
                        
                    progress.advance(main_task)
                    
                except Exception as e:
                    logger.error(f"Item {idx} error: {e}")
                    progress.advance(main_task)
        
        return results

    def display_results_table(self, results: List[EvaluationResult]):
        """Display results in a modern table format"""
        table = Table(title="🎯 RAG Evaluation Results", box=box.ROUNDED)
        
        table.add_column("Query ID", style="cyan", no_wrap=True)
        table.add_column("Query", style="white", max_width=40)
        table.add_column("Faithfulness", justify="center", style="green")
        table.add_column("Answer Relevancy", justify="center", style="blue")
        table.add_column("Context Precision", justify="center", style="magenta")
        table.add_column("Context Recall", justify="center", style="yellow")
        table.add_column("ROUGE-L", justify="center", style="red")
        
        for result in results:
            if result:
                rouge_l = result.rouge_scores.get('rouge_l_f', 0.0)
                table.add_row(
                    result.query_id,
                    result.query[:40] + "..." if len(result.query) > 40 else result.query,
                    f"{result.faithfulness:.3f}",
                    f"{result.answer_relevancy:.3f}",
                    f"{result.context_precision:.3f}",
                    f"{result.context_recall:.3f}",
                    f"{rouge_l:.3f}"
                )
        
        console.print(table)

    def display_metrics_dashboard(self, results: List[EvaluationResult]):
        """Display a comprehensive metrics dashboard"""
        if not results:
            console.print("[red]No results to display[/red]")
            return
        
        # Calculate overall metrics
        faithfulness_scores = [r.faithfulness for r in results if r and r.faithfulness > 0]
        answer_relevancy_scores = [r.answer_relevancy for r in results if r and r.answer_relevancy > 0]
        context_precision_scores = [r.context_precision for r in results if r and r.context_precision > 0]
        context_recall_scores = [r.context_recall for r in results if r and r.context_recall > 0]
        rouge_l_scores = [r.rouge_scores.get('rouge_l_f', 0) for r in results if r and r.rouge_scores.get('rouge_l_f', 0) > 0]
        
        # Create metrics panels
        panels = []
        
        # Faithfulness panel
        faithfulness_avg = np.mean(faithfulness_scores) if faithfulness_scores else 0
        faithfulness_panel = Panel(
            Align.center(f"[bold green]{faithfulness_avg:.3f}[/bold green]\n{len(faithfulness_scores)} evaluations"),
            title="🎯 Faithfulness",
            border_style="green"
        )
        panels.append(faithfulness_panel)
        
        # Answer Relevancy panel
        relevancy_avg = np.mean(answer_relevancy_scores) if answer_relevancy_scores else 0
        relevancy_panel = Panel(
            Align.center(f"[bold blue]{relevancy_avg:.3f}[/bold blue]\n{len(answer_relevancy_scores)} evaluations"),
            title="📊 Answer Relevancy",
            border_style="blue"
        )
        panels.append(relevancy_panel)
        
        # Context Precision panel
        precision_avg = np.mean(context_precision_scores) if context_precision_scores else 0
        precision_panel = Panel(
            Align.center(f"[bold magenta]{precision_avg:.3f}[/bold magenta]\n{len(context_precision_scores)} evaluations"),
            title="🔍 Context Precision",
            border_style="magenta"
        )
        panels.append(precision_panel)
        
        # Context Recall panel
        recall_avg = np.mean(context_recall_scores) if context_recall_scores else 0
        recall_panel = Panel(
            Align.center(f"[bold yellow]{recall_avg:.3f}[/bold yellow]\n{len(context_recall_scores)} evaluations"),
            title="📚 Context Recall",
            border_style="yellow"
        )
        panels.append(recall_panel)
        
        # ROUGE-L panel
        rouge_avg = np.mean(rouge_l_scores) if rouge_l_scores else 0
        rouge_panel = Panel(
            Align.center(f"[bold red]{rouge_avg:.3f}[/bold red]\n{len(rouge_l_scores)} evaluations"),
            title="📝 ROUGE-L",
            border_style="red"
        )
        panels.append(rouge_panel)
        
        # Overall RAGAS Score
        all_scores = faithfulness_scores + answer_relevancy_scores + context_precision_scores + context_recall_scores
        overall_score = np.mean(all_scores) if all_scores else 0
        overall_panel = Panel(
            Align.center(f"[bold white]{overall_score:.3f}[/bold white]\nRAGAS Score"),
            title="⭐ Overall Score",
            border_style="white"
        )
        panels.append(overall_panel)
        
        console.print(Panel(Columns(panels, equal=True), title="📊 Evaluation Dashboard", border_style="bright_blue"))

def create_sample_evaluation_data() -> List[Dict]:
    return [
        {
            "query_id": "q1",
            "query": "What is Azure Cosmos DB and what are its key features?",
            "retrieved_context": [
                "Azure Cosmos DB is a globally distributed, multi-model database service.",
                "It provides automatic scaling, multi-region replication, and supports multiple APIs.",
                "Cosmos DB offers five consistency models: strong, bounded staleness, session, consistent prefix, and eventual.",
                "The service supports SQL, MongoDB, Cassandra, Gremlin, and Table APIs."
            ],
            "expected_answer": "Azure Cosmos DB is a globally distributed, multi-model database service designed for scalable applications. It provides automatic scaling, multi-region replication, and supports multiple APIs and consistency models.",
            "relevant_documents": [
                "Azure Cosmos DB is a globally distributed, multi-model database service.",
                "It provides automatic scaling, multi-region replication, and supports multiple APIs."
            ],
            "document_source": "azure_documentation.pdf"
        },
        {
            "query_id": "q2",
            "query": "How does machine learning work?",
            "retrieved_context": [
                "Machine learning is a method of data analysis that automates analytical model building.",
                "It uses algorithms that iteratively learn from data without being explicitly programmed.",
                "ML can identify patterns and make decisions with minimal human intervention."
            ],
            "expected_answer": "Machine learning is a data analysis method that uses algorithms to automatically learn from data and identify patterns, enabling systems to make predictions or decisions without explicit programming.",
            "document_source": "ml_basics.pdf"
        },
        {
            "query_id": "q3",
            "query": "What are the benefits of cloud computing?",
            "retrieved_context": [
                "Cloud computing provides on-demand access to computing resources.",
                "It offers scalability, cost-effectiveness, and improved collaboration.",
                "Cloud services reduce the need for physical infrastructure maintenance."
            ],
            "expected_answer": "Cloud computing offers on-demand access to scalable computing resources, providing cost-effectiveness, improved collaboration, and reduced infrastructure maintenance needs.",
            "document_source": "cloud_computing_guide.pdf"
        }
    ]

def display_welcome_banner():
    """Display welcome banner"""
    banner = """
╭──────────────────────────────────────────────────────────────╮
│                                                              │
│     🚀 Modern RAG Evaluator CLI                            │
│                                                              │
│     Evaluate your RAG system with RAGAS metrics            │
│     • Faithfulness  • Answer Relevancy  • Context Quality    │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
    """
    console.print(Panel(banner, border_style="bright_blue"))

def display_config_info(config: EvaluationConfig):
    """Display configuration information"""
    config_tree = Tree("⚙️ Configuration")
    
    azure_branch = config_tree.add("🔵 Azure OpenAI")
    azure_branch.add(f"Endpoint: {config.azure_openai_endpoint}")
    azure_branch.add(f"Embedding Model: {config.embedding_deployment_name}")
    azure_branch.add(f"LLM Model: {config.llm_deployment_name}")
    
    cosmos_branch = config_tree.add("🌌 Cosmos DB")
    cosmos_branch.add(f"Database: {config.database_name}")
    cosmos_branch.add(f"Container: {config.container_name}")
    cosmos_branch.add(f"Cache Container: {config.cache_container}")
    
    params_branch = config_tree.add("🎛️ Parameters")
    params_branch.add(f"Temperature: {config.llm_temperature}")
    params_branch.add(f"Max Tokens: {config.llm_max_tokens}")
    params_branch.add(f"Batch Size: {config.batch_size}")
    
    console.print(config_tree)

async def interactive_mode():
    """Interactive CLI mode"""
    display_welcome_banner()
    
    # Configuration setup
    console.print("\n[bold cyan]🔧 Configuration Setup[/bold cyan]")
    
    use_defaults = Confirm.ask("Use default configuration?", default=True)
    
    if use_defaults:
        config = EvaluationConfig(
            cosmos_endpoint=COSMOS_ENDPOINT,
            cosmos_key=COSMOS_KEY,
            azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_openai_api_key=AZURE_OPENAI_API_KEY,
            azure_openai_api_version=AZURE_OPENAI_API_VERSION,
            embedding_deployment_name=EMBEDDING_DEPLOYMENT_NAME,
            llm_deployment_name=LLM_DEPLOYMENT_NAME,
            llm_temperature=LLM_TEMPERATURE,
            llm_top_p=LLM_TOP_P,
            llm_max_tokens=LLM_MAX_TOKENS,
            batch_size=3
        )
    else:
        # Custom configuration
        azure_endpoint = Prompt.ask("Azure OpenAI Endpoint", default=AZURE_OPENAI_ENDPOINT)
        azure_key = Prompt.ask("Azure OpenAI API Key", default=AZURE_OPENAI_API_KEY, password=True)
        embedding_model = Prompt.ask("Embedding Deployment Name", default=EMBEDDING_DEPLOYMENT_NAME)
        llm_model = Prompt.ask("LLM Deployment Name", default=LLM_DEPLOYMENT_NAME)
        
        config = EvaluationConfig(
            cosmos_endpoint=COSMOS_ENDPOINT,
            cosmos_key=COSMOS_KEY,
            azure_openai_endpoint=azure_endpoint,
            azure_openai_api_key=azure_key,
            azure_openai_api_version=AZURE_OPENAI_API_VERSION,
            embedding_deployment_name=embedding_model,
            llm_deployment_name=llm_model,
            llm_temperature=LLM_TEMPERATURE,
            llm_top_p=LLM_TOP_P,
            llm_max_tokens=LLM_MAX_TOKENS,
            batch_size=3
        )
    
    display_config_info(config)
    
    # Initialize evaluator
    console.print("\n[bold cyan]🚀 Initializing RAG Evaluator...[/bold cyan]")
    evaluator = ModernRAGEvaluator(config)
    
    # Data source selection
    console.print("\n[bold cyan]📊 Data Source Selection[/bold cyan]")
    data_options = {
        "1": "Use sample data (3 queries)",
        "2": "Load from JSON file",
        "3": "Load from CSV file",
        "4": "Manual entry"
    }
    
    for key, value in data_options.items():
        console.print(f"  {key}. {value}")
    
    choice = Prompt.ask("Select data source", choices=list(data_options.keys()), default="1")
    
    if choice == "1":
        evaluation_data = create_sample_evaluation_data()
        logger.success(f"Loaded {len(evaluation_data)} sample queries")
    elif choice == "2":
        filename = Prompt.ask("JSON file path")
        try:
            with open(filename, 'r') as f:
                evaluation_data = json.load(f)
            logger.success(f"Loaded {len(evaluation_data)} queries from JSON")
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            evaluation_data = create_sample_evaluation_data()
    elif choice == "3":
        filename = Prompt.ask("CSV file path")
        try:
            df = pd.read_csv(filename)
            evaluation_data = df.to_dict('records')
            logger.success(f"Loaded {len(evaluation_data)} queries from CSV")
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            evaluation_data = create_sample_evaluation_data()
    else:
        # Manual entry (simplified for demo)
        evaluation_data = create_sample_evaluation_data()
        logger.info("Using sample data for manual entry demo")
    
    # Run evaluation
    console.print(f"\n[bold cyan]🎯 Starting Evaluation of {len(evaluation_data)} queries...[/bold cyan]")
    results = await evaluator.batch_evaluate(evaluation_data)
    
    logger.success(f"Evaluation completed! Processed {len(results)} queries")
    
    # Display results
    console.print("\n[bold cyan]📊 Results Dashboard[/bold cyan]")
    evaluator.display_metrics_dashboard(results)
    
    console.print("\n[bold cyan]📋 Detailed Results Table[/bold cyan]")
    evaluator.display_results_table(results)
    
    # Export options
    console.print("\n[bold cyan]💾 Export Options[/bold cyan]")
    export_choice = Prompt.ask(
        "Export results?", 
        choices=["csv", "json", "both", "none"], 
        default="csv"
    )
    
    if export_choice != "none":
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_choice in ["csv", "both"]:
            csv_filename = f"rag_evaluation_{timestamp}.csv"
            try:
                detailed_results = []
                for r in results:
                    if r:
                        flat = {
                            "query_id": r.query_id,
                            "query": r.query,
                            "generated_answer": r.generated_answer,
                            "expected_answer": r.expected_answer,
                            "document_source": r.document_source,
                            "timestamp": r.timestamp.isoformat(),
                            "faithfulness": r.faithfulness,
                            "answer_relevancy": r.answer_relevancy,
                            "context_precision": r.context_precision,
                            "context_recall": r.context_recall,
                            "context_relevancy": r.context_relevancy
                        }
                        flat.update(r.retrieval_metrics)
                        flat.update(r.generation_metrics)
                        flat.update(r.rouge_scores)
                        detailed_results.append(flat)
                
                pd.DataFrame(detailed_results).to_csv(csv_filename, index=False)
                logger.success(f"Exported to {csv_filename}")
            except Exception as e:
                logger.error(f"CSV export failed: {e}")
        
        if export_choice in ["json", "both"]:
            json_filename = f"rag_evaluation_{timestamp}.json"
            try:
                json_results = []
                for r in results:
                    if r:
                        result_dict = asdict(r)
                        result_dict['timestamp'] = r.timestamp.isoformat()
                        json_results.append(result_dict)
                
                with open(json_filename, 'w') as f:
                    json.dump(json_results, f, indent=2)
                logger.success(f"Exported to {json_filename}")
            except Exception as e:
                logger.error(f"JSON export failed: {e}")
    
    console.print("\n[bold green]✅ Evaluation Complete![/bold green]")
    console.print("Thank you for using Modern RAG Evaluator CLI! 🚀")

def create_cli_parser():
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Modern RAG Evaluator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_evaluator.py                          # Interactive mode
  python rag_evaluator.py --data sample            # Use sample data
  python rag_evaluator.py --data data.json         # Load from JSON file
  python rag_evaluator.py --data data.csv          # Load from CSV file
  python rag_evaluator.py --batch-size 5           # Set batch size
  python rag_evaluator.py --export csv             # Export to CSV
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str,
        help='Data source: "sample", JSON file path, or CSV file path'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing (default: 10)'
    )
    
    parser.add_argument(
        '--export',
        choices=['csv', 'json', 'both', 'none'],
        default='csv',
        help='Export format (default: csv)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file path (JSON)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable semantic caching'
    )
    
    return parser

async def batch_mode(args):
    """Non-interactive batch mode"""
    display_welcome_banner()
    
    # Load configuration
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            config = EvaluationConfig(**config_data)
            logger.success(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            config = EvaluationConfig(
                cosmos_endpoint=COSMOS_ENDPOINT,
                cosmos_key=COSMOS_KEY,
                azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_openai_api_key=AZURE_OPENAI_API_KEY,
                azure_openai_api_version=AZURE_OPENAI_API_VERSION,
                embedding_deployment_name=EMBEDDING_DEPLOYMENT_NAME,
                llm_deployment_name=LLM_DEPLOYMENT_NAME,
                llm_temperature=LLM_TEMPERATURE,
                llm_top_p=LLM_TOP_P,
                llm_max_tokens=LLM_MAX_TOKENS,
                batch_size=args.batch_size
            )
    else:
        config = EvaluationConfig(
            cosmos_endpoint=COSMOS_ENDPOINT,
            cosmos_key=COSMOS_KEY,
            azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_openai_api_key=AZURE_OPENAI_API_KEY,
            azure_openai_api_version=AZURE_OPENAI_API_VERSION,
            embedding_deployment_name=EMBEDDING_DEPLOYMENT_NAME,
            llm_deployment_name=LLM_DEPLOYMENT_NAME,
            llm_temperature=LLM_TEMPERATURE,
            llm_top_p=LLM_TOP_P,
            llm_max_tokens=LLM_MAX_TOKENS,
            batch_size=args.batch_size
        )
    
    if not args.quiet:
        display_config_info(config)
    
    # Initialize evaluator
    evaluator = ModernRAGEvaluator(config)
    
    # Load data
    if args.data == "sample" or not args.data:
        evaluation_data = create_sample_evaluation_data()
        logger.success(f"Using sample data ({len(evaluation_data)} queries)")
    elif args.data.endswith('.json'):
        try:
            with open(args.data, 'r') as f:
                evaluation_data = json.load(f)
            logger.success(f"Loaded {len(evaluation_data)} queries from {args.data}")
        except Exception as e:
            logger.error(f"Failed to load {args.data}: {e}")
            return
    elif args.data.endswith('.csv'):
        try:
            df = pd.read_csv(args.data)
            evaluation_data = df.to_dict('records')
            logger.success(f"Loaded {len(evaluation_data)} queries from {args.data}")
        except Exception as e:
            logger.error(f"Failed to load {args.data}: {e}")
            return
    else:
        logger.error(f"Unsupported data format: {args.data}")
        return
    
    # Run evaluation
    console.print(f"\n[bold cyan]🎯 Evaluating {len(evaluation_data)} queries...[/bold cyan]")
    results = await evaluator.batch_evaluate(evaluation_data)
    
    logger.success(f"Evaluation completed! Processed {len(results)} queries")
    
    # Display results
    if not args.quiet:
        evaluator.display_metrics_dashboard(results)
        evaluator.display_results_table(results)
    
    # Export results
    if args.export != "none":
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if args.export in ["csv", "both"]:
            csv_filename = f"rag_evaluation_{timestamp}.csv"
            try:
                detailed_results = []
                for r in results:
                    if r:
                        flat = {
                            "query_id": r.query_id,
                            "query": r.query,
                            "generated_answer": r.generated_answer,
                            "expected_answer": r.expected_answer,
                            "document_source": r.document_source,
                            "timestamp": r.timestamp.isoformat(),
                            "faithfulness": r.faithfulness,
                            "answer_relevancy": r.answer_relevancy,
                            "context_precision": r.context_precision,
                            "context_recall": r.context_recall,
                            "context_relevancy": r.context_relevancy
                        }
                        flat.update(r.retrieval_metrics)
                        flat.update(r.generation_metrics)
                        flat.update(r.rouge_scores)
                        detailed_results.append(flat)
                
                pd.DataFrame(detailed_results).to_csv(csv_filename, index=False)
                logger.success(f"Results exported to {csv_filename}")
            except Exception as e:
                logger.error(f"CSV export failed: {e}")
        
        if args.export in ["json", "both"]:
            json_filename = f"rag_evaluation_{timestamp}.json"
            try:
                json_results = []
                for r in results:
                    if r:
                        result_dict = asdict(r)
                        result_dict['timestamp'] = r.timestamp.isoformat()
                        json_results.append(result_dict)
                
                with open(json_filename, 'w') as f:
                    json.dump(json_results, f, indent=2)
                logger.success(f"Results exported to {json_filename}")
            except Exception as e:
                logger.error(f"JSON export failed: {e}")
    
    # Print summary
    if results:
        faithfulness_scores = [r.faithfulness for r in results if r and r.faithfulness > 0]
        answer_relevancy_scores = [r.answer_relevancy for r in results if r and r.answer_relevancy > 0]
        
        if faithfulness_scores and answer_relevancy_scores:
            avg_faithfulness = np.mean(faithfulness_scores)
            avg_relevancy = np.mean(answer_relevancy_scores)
            
            summary_table = Table(title="📊 Evaluation Summary", box=box.ROUNDED)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Average Score", justify="center", style="green")
            summary_table.add_column("Count", justify="center", style="yellow")
            
            summary_table.add_row("Faithfulness", f"{avg_faithfulness:.3f}", str(len(faithfulness_scores)))
            summary_table.add_row("Answer Relevancy", f"{avg_relevancy:.3f}", str(len(answer_relevancy_scores)))
            
            console.print(summary_table)

async def main():
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        if args.data:
            await batch_mode(args)
        else:
            await interactive_mode()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Evaluation interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        console.print(f"\n[red]💥 Fatal error: {e}[/red]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
