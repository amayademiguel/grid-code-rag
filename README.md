# Grid code RAG assistant

A local, retrieval-augmented assistant that answers questions about **grid connection requirements** and builds a **requirements report for a power plant project**, citing the article of the regulation behind every answer.

It runs entirely on a laptop with [Ollama](https://ollama.com): documents, embeddings and the language model never leave the machine. That is the constraint a company faces with confidential or paid grid codes, which cannot be sent to a public chatbot.

The corpus is the EU network code **Requirements for Generators (RfG)**, [Commission Regulation (EU) 2016/631](https://eur-lex.europa.eu/eli/reg/2016/631/oj), the baseline that every national grid code in the EU builds on.

## Why

To connect a plant to the grid, every requirement of the applicable grid code has to be met: frequency and voltage ranges, reactive power capability, fault-ride-through, fault current injection, protection, models, compliance tests. Grid codes are long, they change, and they are often read by hand. A requirement that is missed can delay a connection.

The aim is a tool that, given a question or a project, **finds the relevant requirements and shows exactly where they come from**, so an engineer can check them quickly.

## How it works

```
PDF ──► parsing ──► article-level chunks ──► embeddings (bge-m3) ──► index
                     + metadata                                       │
question ──► hybrid search (semantic + BM25, metadata filters) ◄──────┘
                 │
                 ▼
        local LLM (qwen2.5) ──► answer with [Art. N] citations / JSON requirements
```

| Step | Choice | Why |
|---|---|---|
| Parsing | PyMuPDF, running headers removed, table cells re-joined (`Uret: 0`) | PDF tables come out one cell per line |
| Chunking | One chunk per **article**, long articles split by numbered paragraph, each chunk prefixed with its article title | In a regulation the article is the unit of meaning and of citation |
| Metadata | document, jurisdiction, version, valid-from date, article, pages | Filter by country and by the version in force; cite precisely |
| Embeddings | `bge-m3` via Ollama (multilingual) | Grid codes are written in national languages |
| Retrieval | Semantic + BM25 combined with reciprocal rank fusion | Paraphrases ("stay connected during a voltage dip") *and* exact terms ("type D", "LFSM-O") |
| Generation | `qwen2.5` via Ollama, temperature 0, answer only from context, mandatory citations | Traceability; no outside knowledge |
| Structured output | JSON schema enforced by Ollama | A requirements list is data, not prose |

### Project report: an agentic workflow

`python app.py project --mw 300 --kv 220` runs a fixed plan:

1. **Classify** the module with a deterministic tool (Article 5, Table 1): ≥ 110 kV ⇒ type D.
2. **Derive the applicable articles** (requirements are cumulative: a type D power park module follows Articles 13–16 and 20–22).
3. For each requirement category (frequency, voltage, reactive power, fault-ride-through, …), **retrieve only from those articles**.
4. **Extract** the requirements as JSON with the LLM.
5. Return a table with the source of every requirement.

The plan is fixed in code rather than left to the model: with a small local model, a code-controlled workflow is more reliable than free planning. Classification is plain code because thresholds are exact rules.

## Results

### Retrieval

21 test questions (in English, plus one each in Spanish, German and French), each labelled with the article that contains the answer. `python -m eval.evaluate`:

| Search | recall@1 | recall@4 | MRR |
|---|---|---|---|
| Semantic | 0.76 | **1.00** | 0.86 |
| Keyword (BM25) | 0.62 | **1.00** | 0.76 |
| Hybrid | 0.76 | **1.00** | 0.86 |

The correct article reaches the model for every question, including the Spanish, German and French ones (multilingual embeddings). For the French question (*"injection rapide de courant de défaut"*), the semantic search ranks Article 20 first, while BM25 only ranks it third, through the single shared word "injection". The hybrid search keeps it first. The French question also exposed a tokenizer bug: accented letters were dropped ("défaut" became "d" + "faut"). It is now fixed. Caveat: the set is small and was written with knowledge of the article titles, so it is optimistic; a real test set should come from engineers' questions on past projects.

### Generation: where it fails

Both models run locally on a laptop CPU (Intel i7, 16 GB RAM, no GPU).

**1. A question that is ambiguous for a reason.** *"What are the fault-ride-through parameters Uret and tclear for type D power park modules?"*

The correct answer depends on the connection voltage, which the question does not give. Article 16 sends type D modules connected at 110 kV or above to Table 7.2 (Uret = 0 pu, tclear = 0.14–0.15 s) and those connected below 110 kV to Table 3.2 in Article 14 (Uret = 0.05–0.15 pu).

| Model | Answer | Time |
|---|---|---|
| qwen2.5:3b | Uret 0.05–0.15, tclear 0.14–0.15 s, with no condition and no citation: it took Article 14's value without saying why | ~45 s |
| qwen2.5:7b | Same values, but states that they apply to type D modules **connected below 110 kV** and cites Table 3.2 | ~3 min |

The 7B answer is correct for one of the two cases; neither model points out that the answer depends on the voltage. This is why the project report asks for the connection voltage first: the inputs remove the ambiguity before retrieval.

**2. Project report** (`project --mw 300 --kv 220`). Retrieval and classification are identical (type D, Articles 13–16 and 20–22); extraction differs:

| | qwen2.5:3b | qwen2.5:7b |
|---|---|---|
| Frequency response | LFSM-O 50.2–50.5 Hz, LFSM-U 49.8–49.5 Hz, droop 2–12 %, but duplicates and some items in the wrong category | Same values, well organised, with conditions |
| Voltage ranges | missing | Continental Europe values correct (0.85–0.90 pu 60 min, 0.90–1.118 pu unlimited, ...), **plus a wrong row from the Baltic area** |
| Reactive power | U-Q/Pmax and P-Q/Pmax profiles described, no values | Q/Pmax envelope 0.75 for Continental Europe |
| Runtime | several minutes | ~32 min |

**3. A question in another language.** *"Quelles sont les exigences d'injection rapide de courant de défaut pour les parcs non synchrones de générateurs ?"*

Retrieval puts Article 20(2), which contains the answer, first. Yet both models answer "Not found in the provided documents" (3B in ~1 min, 7B in ~3 min). They do not link the French terms to the English ones in the excerpt ("fast fault current", "power park module").

Two changes to the same context make it work:
- Asking the same question in English gives a complete answer citing Art. 20(2).
- Asking in French with the English terms in brackets gives a correct answer in French.

So the multilingual gap is in generation, not retrieval. A fix is a query-rewriting step that translates the question into the language of the document, or a glossary of grid code terms, before generation.

**What it shows.** Retrieval works; the errors come from (a) cumulative requirements with exceptions (a type D module also follows Article 14, except where Article 16 replaces it), and (b) tables flattened by the PDF parser, which lose which row belongs to which synchronous area. Both are known hard problems of regulatory documents. Ways to address them: project inputs and article filters (already in the workflow), table-aware parsing that keeps each row with its area, a larger model, and checking every value against the cited source with an engineer in the loop.

## Usage

Requirements: Python 3.10+, [Ollama](https://ollama.com).

```bash
pip install -r requirements.txt
ollama pull bge-m3
ollama pull qwen2.5:3b          # or qwen2.5:7b for better answers

python app.py build             # download the RfG, chunk it and build the index
python app.py search "fault ride through type D"
python app.py ask "Which frequency ranges must a module withstand in Continental Europe?"
python app.py project --mw 300 --kv 220 --out report.json
python -m eval.evaluate         # retrieval evaluation
```

## Project structure

```
app.py                     command-line interface
src/gridrag/config.py      models, paths and parameters
src/gridrag/documents.py   document registry with version metadata
src/gridrag/ingest.py      PDF parsing and article-level chunking
src/gridrag/index.py       embeddings
src/gridrag/retrieve.py    hybrid search with metadata filters
src/gridrag/answer.py      prompting, citations, structured output
src/gridrag/rfg_types.py   module classification (Article 5)
src/gridrag/workflow.py    project requirements report
eval/                      test questions and retrieval evaluation
```

## Limitations and next steps

- **Tables and figures:** fault-ride-through profiles and P–Q diagrams are figures; only their numeric parameters are captured. A layout-aware parser would handle tables better.
- **Cross-references:** "as specified in Article 14(3)" is not followed automatically. An agent with a `get_article` tool, or a graph of references, would.
- **National codes and versions:** only the European baseline is indexed. The metadata already supports several jurisdictions and validity dates, so a national code (and its updates) can be added and filtered by the version in force.
- **Evaluation of answers:** retrieval is measured; answer correctness and faithfulness still need a test set validated by engineers.
- **Human in the loop:** the tool points to the requirements and their source; an engineer still validates them.

## Author

Amaya de Miguel Mediavilla, MSc student in the Erasmus Mundus master DREAM (Dynamics of Renewables-based Power Systems), École Centrale de Nantes and HTW Berlin.

The RfG text is © European Union, reused under the [Commission's reuse policy](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).
