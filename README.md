# 🚀 RAG-TUI v0.0.2 Beta

## *The Terminal App That Makes Chunking Actually Fun*

> "I used to stare at my RAG pipeline wondering why it sucked. Then I found RAG-TUI and realized my chunks were the size of War and Peace." - A Developer, Probably

---

## 🎭 What Even Is This?

**RAG-TUI** is a beautiful terminal-based debugger for your Retrieval-Augmented Generation (RAG) pipelines. It's like having X-ray vision for your text chunking.

You know that feeling when your LLM hallucinates because your retrieval returned garbage? Yeah, this fixes that.

```
┌─────────────────────────────────────────────────────────────────┐
│  RAG-TUI v0.0.2 Beta                                            │
├─────────────────────────────────────────────────────────────────┤
│  📝 Input   🎨 Chunks   🔍 Search   📊 Batch   ⚙️ Settings      │
│                                                                 │
│  Your text, but now with ✨ colors ✨ and 📊 metrics 📊        │
│                                                                 │
│  Chunk Size: [◀] [200] [▶]  ████████░░░░░░  tokens              │
│  Overlap:    [◀] [10]  [▶]  ██░░░░░░░░░░░░  %                   │
│                                                                 │
│  ⚡ 5 chunks                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 📸 Screenshots

| Input Tab | Chunks Tab |
|-----------|------------|
| ![Input](assets/input-tab.png) | ![Chunks](assets/chunks-tab.png) |

| Search Tab | Chat Tab |
|------------|----------|
| ![Search](assets/search-tab.png) | ![Chat](assets/chat-tab.png) |

---

## 🤔 Why Should I Care?

### The Problem

You're building a RAG app. You chunk your documents. You embed them. You search. And then...

```
User: "What's the company's refund policy?"
LLM: "Based on the context, your refrigerator appears to be running."
```

### The Solution

**See exactly how your text is being chunked.** Tweak parameters in real-time. Test queries. Export settings. Actually understand what's going on.

---

## ⚡ Quick Start (30 Seconds, I Promise)

### Install

```bash
pip install rag-tui
```

### Run

```bash
rag-tui
```

### That's It

No really. You're done. Press `L` to load sample text and start playing.

---

## 🎨 Features (The Good Stuff)

### 1. 🧩 Six Chunking Strategies

Because one size definitely does NOT fit all.

| Strategy | Best For | Vibe |
|----------|----------|------|
| **Token** | General text | "I count tokens for breakfast" |
| **Sentence** | Articles, docs | "Periods are sacred" |
| **Paragraph** | Structured text | "Double newline gang" |
| **Recursive** | Code, mixed | "I'll try everything" |
| **Fixed** | Speed demons | "Just cut every 500 chars lol" |
| **Custom** | You, apparently | "I know better" (you might!) |

### 2. 🔌 Four LLM Providers

Switch between providers like you switch between tabs (too often).

```bash
# Ollama (Free! Local! Private!)
ollama serve
rag-tui

# OpenAI (When you need that GPT juice)
export OPENAI_API_KEY="sk-..." 
rag-tui

# Groq (FAST. LIKE, REALLY FAST.)
export GROQ_API_KEY="gsk_..."
rag-tui

# Google Gemini (Free tier FTW)
export GOOGLE_API_KEY="AI..."
rag-tui
```

### 3. 📁 Load Any File

PDFs? ✅ Markdown? ✅ Python? ✅ That random `.txt` file from 2019? ✅

Supported: `.txt`, `.md`, `.py`, `.js`, `.json`, `.yaml`, `.pdf`, and 10 more!

### 4. 📊 Batch Testing

Test 50 queries at once. See which ones fail. Cry. Fix. Repeat.

```
📊 Batch Test Results
━━━━━━━━━━━━━━━━━━━━
Total Queries: 50
Hit Rate (>0.5): 78%
Avg Top Score: 0.72

You're doing better than average!
(The average is made up, but still, congrats!)
```

### 5. ⚡ Built-in Presets

Don't know what settings to use? We got you.

| Preset | Size | Overlap | For |
|--------|------|---------|-----|
| **Q&A Retrieval** | 200 | 15% | Chatbots, search |
| **Document Summary** | 500 | 5% | Long docs |
| **Code Analysis** | 300 | 20% | Source code |
| **Long Context** | 800 | 10% | GPT-4-128k users |
| **High Precision** | 100 | 25% | When you NEED accuracy |

### 6. 📋 Export Settings

Take your carefully tuned settings and use them in production.

```python
# LangChain Export
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
)

# LlamaIndex Export  
from llama_index.core.node_parser import SentenceSplitter

parser = SentenceSplitter(
    chunk_size=800,
    chunk_overlap=80,
)
```

---

## 🎮 How To Use It

### The Interface

```
┌─────────────────────────────────────────────────────────────────┐
│  Strategy: [Token ▼]  │  File: [path...]  │  [📂 Load]         │
├─────────────────────────────────────────────────────────────────┤
│  📝 Input │ 🎨 Chunks │ 🔍 Search │ 📊 Batch │ ⚙️ Settings │ 💬 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    (Your content here)                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Chunk Size: [◀] [200] [▶]  │  Overlap: [◀] [10] [▶] %          │
│  ⚡ 5 chunks                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### The Tabs

| Tab | What It Does | When To Use It |
|-----|--------------|----------------|
| 📝 **Input** | Paste or load your document | First |
| 🎨 **Chunks** | See colorful chunk cards | To see the magic |
| 🔍 **Search** | Query and see what comes back | Testing retrieval |
| 📊 **Batch** | Test many queries at once | Before production |
| ⚙️ **Settings** | Export config, custom code | When you're done |
| 💬 **Chat** | Talk to your chunks | For fun |

### Keyboard Shortcuts

| Key | Action | Pro Tip |
|-----|--------|---------|
| `Q` | Quit | When you're done procrastinating |
| `L` | Load sample | Start here if confused |
| `R` | Rechunk | After changing params |
| `D` | Dark/Light mode | We default to dark (obviously) |
| `E` | Export config | Save your precious settings |
| `F1` | Help | When this README isn't enough |
| `Tab` | Next tab | Navigate like a pro |

---

## 🔧 The Workflow (AKA What To Actually Do)

### Step 1: Load Your Document

Either:
- Type/paste in the **📝 Input** tab
- Enter a file path and click **📂 Load**
- Press `L` for sample text (recommended for newbies)

### Step 2: Pick Your Strategy

Use the dropdown at the top. If unsure:
- **Text/articles?** → Sentence
- **Code?** → Recursive  
- **Don't know?** → Token (it's the safe choice)

### Step 3: Adjust Parameters

In the **🎨 Chunks** tab, use the sliders:
- **Chunk Size**: How big each chunk should be (in tokens)
- **Overlap**: How much chunks should share (prevents context loss)

**The Golden Rule**: Smaller chunks = more precise, less context. Bigger chunks = more context, less precise.

### Step 4: Test Your Queries

Go to **🔍 Search** tab:
1. Type a question
2. Click **Search**
3. See what chunks come back
4. Cry or celebrate accordingly

### Step 5: Batch Test (The Pro Move)

Go to **📊 Batch** tab:
1. Enter multiple queries (one per line)
2. Click **Run Batch Test**
3. See your hit rate
4. Adjust until it's good enough™

### Step 6: Export

Go to **⚙️ Settings** tab:
- Click **JSON**, **LangChain**, or **LlamaIndex**
- Copy the generated code
- Paste in your project
- Deploy
- Profit???

---

## 🧪 Custom Chunking (For The Brave)

Don't like our strategies? Roll your own!

Go to **⚙️ Settings** tab, paste a function like:

```python
def chunk_by_headers(text, chunk_size, overlap):
    """Split on markdown headers."""
    import re
    sections = re.split(r'\n(?=#{1,3} )', text)
    return [(s, 0, len(s)) for s in sections if s.strip()]
```

Click **⚡ Apply Custom Chunker** and watch the magic.

---

## 🤖 Provider Setup

### Ollama (Recommended for Privacy)

```bash
# Install Ollama
brew install ollama  # macOS
# or download from ollama.ai

# Pull required models
ollama pull nomic-embed-text  # For embeddings
ollama pull llama3.2:1b       # For chat (small & fast)

# Start the server
ollama serve
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Uses: `text-embedding-3-small` + `gpt-4o-mini`

### Groq (Free Tier!)

```bash
export GROQ_API_KEY="gsk_your-key-here"
```

Uses: `llama-3.1-8b-instant` (NO embeddings - pair with Ollama)

### Google Gemini (Also Free Tier!)

```bash
export GOOGLE_API_KEY="your-key-here"
```

Uses: `text-embedding-004` + `gemini-1.5-flash`

---

## 📈 Understanding The Metrics

### In Chunks Tab

```
📊 5 chunks | Avg: 180 chars | Total: 900 chars | ~225 tokens
```

- **5 chunks**: Your document was split into 5 pieces
- **Avg: 180 chars**: Each chunk is ~180 characters
- **Total: 900 chars**: Your whole document size
- **~225 tokens**: Estimated token count (chars ÷ 4)

### In Search Tab

```
#1 ████████░░ 0.89  "The refund policy states..."
#2 ██████░░░░ 0.72  "For returns within 30 days..."
#3 ████░░░░░░ 0.45  "Our customer service team..."
```

- **#1, #2, #3**: Ranking by relevance
- **████████░░**: Visual similarity bar
- **0.89**: Cosine similarity score (0-1, higher = better)

### In Batch Tab

```
Hit Rate (>0.5): 78%
Avg Top Score: 0.72
```

- **Hit Rate**: % of queries where top result scored > 0.5
- **Avg Top Score**: Average of all top-1 scores

**As a rule of thumb:**
- Hit Rate > 80% = Great
- Hit Rate 60-80% = Acceptable
- Hit Rate < 60% = Time to tune

---

## 🐛 Troubleshooting

### "Ollama not available"

```bash
# Is Ollama running?
ollama serve

# Did you pull the models?
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

### "No chunks"

- Is your text too short?
- Is chunk size bigger than your text?
- Try lowering chunk size to 50

### "Search returns garbage"

- Check if embeddings were created (needs Ollama/OpenAI)
- Try a different chunking strategy
- Lower chunk size for more precision

### "App looks weird"

```bash
# Reset terminal
reset

# Try a different terminal (iTerm2, Warp, etc.)
```

---

## 🎓 Chunking 101 (The Theory)

### Why Chunk At All?

LLMs have context limits. Your document is bigger than the limit. So we split it up, find the relevant parts, and only send those.

```
Your 50-page PDF → Split into 100 chunks → Search → Top 3 sent to LLM → Answer!
```

### The Size-Precision Tradeoff

| Chunk Size | Precision | Context | Best For |
|------------|-----------|---------|----------|
| Small (50-100) | High ✅ | Low ❌ | Specific facts |
| Medium (200-400) | Medium | Medium | General Q&A |
| Large (500-1000) | Low ❌ | High ✅ | Summaries |

### The Overlap Question

Overlap = how many tokens chunks share at boundaries.

- **0% overlap**: Chunks are completely separate (risk: losing context at boundaries)
- **10-20% overlap**: Goldilocks zone (recommended)
- **50% overlap**: Lots of redundancy (wastes tokens but very safe)

---

## 📦 Programmatic Usage

Don't want the TUI? Use the library directly:

```python
from rag_tui.core import ChunkingEngine, StrategyType

# Create engine
engine = ChunkingEngine()
engine.set_strategy(StrategyType.SENTENCE)

# Chunk some text
chunks = engine.chunk_text(
    "Your document here...",
    chunk_size=200,
    overlap=20
)

for text, start, end in chunks:
    print(f"[{start}:{end}] {text[:50]}...")
```

### Use Providers Directly

```python
from rag_tui.core.providers import get_provider, ProviderType

# Get Ollama provider
provider = get_provider(ProviderType.OLLAMA)

# Check connection
if await provider.check_connection():
    # Embed text
    embedding = await provider.embed("Hello world")
    
    # Generate response
    response = await provider.generate("What is RAG?")
```

---

## 🤝 Contributing

Found a bug? Have an idea? Want to add support for Claude/Anthropic?

1. Fork the repo
2. Create a branch
3. Make your changes
4. Submit a PR
5. Get famous (in our small community)

---

## 📜 License

MIT License - Do whatever you want, just don't blame us if your RAG app becomes sentient.

---

## 🙏 Credits

Built with:
- [Textual](https://textual.textualize.io/) - The TUI framework that makes terminals beautiful
- [Chonkie](https://github.com/chonkie-ai/chonkie) - Token-based chunking
- [Usearch](https://github.com/unum-cloud/usearch) - Blazing fast vector search
- [Ollama](https://ollama.ai/) - Local LLM inference

---

## 💭 Final Words

RAG is hard. Chunking is an art. But with RAG-TUI, at least you can *see* what you're doing wrong.

Now go forth and chunk responsibly! 🎯

---

<p align="center">
  <b>Made with ❤️ and too much ☕ for RAG developers everywhere</b>
</p>

<p align="center">
  <i>"May your chunks be small and your retrieval be accurate."</i>
</p>
