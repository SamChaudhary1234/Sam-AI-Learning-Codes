# Sam-AI-Learning-Codes

Hands-on code from my journey through LLM Engineering — building progressively from a simple webpage summarizer all the way to a 7-agent autonomous deal-finder system.

Spans frontier APIs, open-source models, RAG, fine-tuning, QLoRA, and multi-agent architectures.

Stack: Python, OpenAI, Anthropic, Google, Groq, HuggingFace, LangChain, Chroma, PyTorch, Gradio, Modal.


## Week 1 — Frontier API Fundamentals

First contact with the OpenAI API. Built a webpage summarizer using BeautifulSoup for scraping and GPT-4o-mini for summarization. Includes alternative versions running locally via Ollama (Llama 3.2) — both through the OpenAI-compatible client and the native ollama package. End-of-week capstone: a dual-model technical-question explainer with streaming output.


## Week 2 — Multi-Provider APIs and Gradio UIs

Connected to OpenAI, Anthropic, Google, and DeepSeek through a unified interface. Explored prompt caching with the full text of Hamlet, built chat interfaces with Gradio, and added tool-calling for an airline booking assistant backed by SQLite. Capstone: multi-modal app combining DALL-E image generation with TTS audio output.


## Week 3 — Open-Source Models on HuggingFace

Moved off frontier APIs and onto HuggingFace transformers. Covered Pipelines (high-level inference for sentiment, NER, QA, summarization, image gen, TTS), Tokenizers across model families (Llama, Phi, Qwen, StarCoder), and direct model loading with 4-bit quantization via bitsandbytes so 7-8B models fit on a free Colab T4. Capstone: meeting minutes generator (Whisper transcription, Llama summarization). Includes a token-prediction visualizer using OpenAI's logprobs API rendered as a NetworkX graph.


## Week 4 — Code Generation: Python to C++ / Rust

Used frontier and open-source models to convert Python to high-performance C++ and Rust. Detected system info (OS, CPU, SIMD, compiler) and Rust toolchain automatically so the LLM could pick optimal compile flags for each platform. Compared GPT-5, Claude 4.5 Sonnet, Grok 4, Gemini 2.5 Pro against open-source models (Qwen Coder, DeepSeek Coder, gpt-oss) running locally via Ollama and hosted via Groq and OpenRouter. Wrapped the whole thing in a polished Gradio UI with custom CSS.


## Week 5 — Retrieval-Augmented Generation (RAG)

Built RAG from the ground up. Started with brute-force similarity, then chunking and Chroma vector store with t-SNE visualization, then a full LangChain RAG pipeline. Added evaluation with MRR, nDCG, and LLM-as-judge. Capstone goes deeper with advanced RAG techniques — LLM-driven semantic chunking with Pydantic schemas, reranking with cross-encoders, and query rewriting for better retrieval.


## Week 6 — The Price is Right (Capstone)

Built a price-prediction model from a curated 800K-item Amazon dataset. Day 1: data curation with weighted sampling to balance categories and price distributions. Day 2: LLM-driven product rewriting via Groq's batch API. Day 3: classical ML baselines (Linear Regression, Random Forest, XGBoost). Day 4: 8-layer PyTorch NN and zero-shot evaluation across 6 frontier LLMs. Day 5: fine-tuning GPT-4.1-nano via OpenAI's API. Bonus: a 289M-parameter deep residual network that achieves $46 mean absolute error — competitive with frontier models.


## Week 7 — Fine-Tuning Open-Source LLMs (QLoRA)

Fine-tuned Llama 3.1 8B for the price-prediction task using QLoRA (Quantized Low-Rank Adaptation). Covered the fundamentals: rank, alpha, target modules, dropout, 4-bit quantization with bitsandbytes, supervised fine-tuning with TRL's SFTTrainer. Includes data preparation (tokenization, prompt formatting, length-distribution analysis) and the pricer package with the Item model and Tester evaluation framework.


## Week 8 — Autonomous Multi-Agent System

The capstone: a 7-agent system that scrapes RSS deal feeds, prices each item using an ensemble of 3 models, and pushes notifications when it finds bargains.

The Specialist Agent runs a fine-tuned Llama on Modal (serverless GPU). The Frontier Agent is a RAG-based pricer using GPT-5.1 with Chroma and sentence-transformers. The Neural Network Agent wraps the 289M-param residual network from week 6. The Ensemble Agent combines all three (80, 10, 10 weighted). The Scanner Agent parses RSS feeds and picks the 5 best deals via structured outputs. The Messaging Agent sends Pushover notifications with message text crafted by Claude. The Planning Agent orchestrates the full workflow, and an autonomous variant uses tool-calling.

UI built with Gradio. Final benchmark: the Ensemble achieves $29.90 mean error.


## Repo Structure

Each week lives in its own folder.

week1 — Frontier APIs, Ollama, BeautifulSoup
week2 — Multi-provider, Gradio, tool-calling, multi-modal
week3 — HuggingFace transformers, tokenizers, quantization
week4 — Code generation (Python to C++ / Rust)
week5 — RAG, LangChain, Chroma, advanced retrieval
week6 — Price prediction: data, ML, NN, fine-tuning
week7 — QLoRA fine-tuning of Llama 3.1
week8 — 7-agent autonomous deal-finder system


## Setup

Clone the repo, then run uv sync to install dependencies.

Add your API keys to a .env file (OpenAI, Anthropic, Google, Groq, HuggingFace, Pushover as needed).

For weeks 7 and 8 you'll need a GPU — Colab works. Drop the trained deep_neural_network.pth weights into week6 and week8 before running the ensemble code.


## Contact

Sam Chaudhary — linkedin.com/in/sam-chaudhary-75351a259 — samchaudharywork@gmail.com
