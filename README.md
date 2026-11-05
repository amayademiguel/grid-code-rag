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
question ──► direct table value? ──► answered by code (tables.py), no LLM      │
                 │ otherwise                                          │
                 ▼                                                    │
             hybrid search (semantic + BM25, metadata filters) ◄──────┘
                 │
                 ▼
        local LLM (qwen2.5:7b) ──► answer with [Art. N] citations / JSON requirements
```

| Step | Choice | Why |
|---|---|---|
| Parsing | PyMuPDF, running headers removed, each table row written with its table, scope and area | PDF tables come out one cell per line, and a row alone does not say what it applies to |
| Chunking | One chunk per **article**, long articles split by numbered paragraph, each chunk prefixed with its article title | In a regulation the article is the unit of meaning and of citation |
| Metadata | document, jurisdiction, version, valid-from date, article, pages | Filter by country and by the version in force; cite precisely |
| Embeddings | `bge-m3` via Ollama (multilingual) | Grid codes are written in national languages |
| Retrieval | Semantic, BM25 and named tables or articles, combined with reciprocal rank fusion; test articles and recitals left out unless asked for | Paraphrases ("stay connected during a voltage dip"), exact terms ("type D", "LFSM-O") and references ("Table 7.2") |
| Exact rules | Plain code: module type, applicable articles, table values, the range that contains a value | These are exact tasks, and an LLM can read the wrong row |
| Generation | `qwen2.5:7b` via Ollama, temperature 0, answer only from context, mandatory citations | Traceability; no outside knowledge |
| Structured output | JSON schema enforced by Ollama | A requirements list is data, not prose |

### Project report: an agentic workflow

`python app.py project --mw 300 --kv 220` runs a fixed plan:

1. **Classify** the module with a deterministic tool (Article 5, Table 1): ≥ 110 kV ⇒ type D.
2. **Derive the applicable articles** (requirements are cumulative: a type D power park module follows Articles 13–16 and 20–22).
3. **Read the requirements that are in tables with code**: frequency and voltage ranges, reactive power capability at maximum capacity, fault-ride-through parameters.
4. For the requirements in the text, **retrieve only from those articles** and **extract** them as JSON with the LLM.
5. Return a JSON report with the source of every requirement.

The plan is fixed in code rather than left to the model: with a small local model, a code-controlled workflow is more reliable than free planning. Classification is plain code because thresholds are exact rules.

## Results

Measured on a laptop CPU (Intel i7, 16 GB RAM, no GPU).

### Retrieval

90 test questions, 86 of them with an article that contains the answer. `python -m eval.evaluate`, share of questions with the right article in the top 4:

| Questions | Semantic | Keyword (BM25) | Hybrid |
|---|---|---|---|
| Direct (48) | 1.00 | 0.94 | 0.98 |
| Varied wording (29) | 0.86 | 0.83 | 1.00 |
| Names a table (6) | 0.67 | 1.00 | 1.00 |
| Other languages (3) | 1.00 | 1.00 | 1.00 |

The right article reaches the model in 85 of 86 questions. `python -m eval.check_tables` compares 258 values of 15 tables with an independent extraction of the PDF, with no differences.

### Answers

110 test questions through the full pipeline: 90 used to improve the system (`python -m eval.assessment`, corrections and reasons in `eval/manual_review.json`) and 20 new ones kept apart as a check (`eval/questions_holdout.json`).

| Questions | Correct | Refused | Wrong or incomplete |
|---|---|---|---|
| Direct | 42/48 | 0 | 6 |
| Names a table | 6/6 | 0 | 0 |
| Varied wording | 26/29 | 1 | 2 |
| Other languages | 0/3 | 0 | 3 |
| Outside the regulation | 4/4 | 0 | 0 |
| **Used to improve it** | **78/90** | **1** | **11** |
| **New, first run** | **17/20** | **0** | **3** |
| **Total** | **95/110** | **1** | **14** |

22 of the 90 questions are answered by code in about a second. The LLM takes about 2 minutes per answer, and a project report 19 minutes.

**How far these numbers go.** Each answer was graded by Claude, an online AI model, against the text of the regulation. Claude also helped to build the system and to write the questions, and no engineer has validated the results, so they are indicative. The 90 questions are optimistic because the system was tuned on them. The three errors in the 20 new ones led to two fixes in the table reader, so that set is no longer unseen, and the table gives its first run. A real test set should come from engineers' questions on past projects.

### Where it still fails

- **Other languages:** the search finds the article, but the answer is wrong. The French question gets a refusal, the German one mixes a row, and the Spanish one mixes two tables. Only the search can be relied on outside English.
- **Wrong details in cited answers:** one answer gets the rule right and adds that ENTSO-E is the TSO in Germany, which is not in the excerpts. A citation does not guarantee that the text is right.
- **Values in the text of an article:** when asked how LFSM-O and LFSM-U work, the answer leaves out the frequency thresholds, although they are in the excerpts.
- **Project report:** the values in tables are read by code, for the case of the project. The rows that the LLM reads from the text need checking: it labels the LFSM-U threshold as LFSM-O, includes an option for type A and leaves some values as N/A.

## Usage

Requirements: Python 3.10+, [Ollama](https://ollama.com).

```bash
pip install -r requirements.txt
ollama pull bge-m3
ollama pull qwen2.5:7b

python app.py build             # download the RfG, chunk it and build the index
python app.py search "fault ride through type D"
python app.py ask "Which frequency ranges must a module withstand in Continental Europe?"
python app.py project --mw 300 --kv 220 --out report.json
python -m eval.evaluate         # retrieval evaluation
python -m eval.evaluate_answers # full pipeline on the test questions (hours on a CPU)
python -m eval.assessment       # results after the review of the answers
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
src/gridrag/tables.py      table values read by code, without the LLM
src/gridrag/rfg_types.py   module classification (Article 5), project facts
src/gridrag/workflow.py    project requirements report
eval/                      test questions, retrieval and answer evaluation, review of the answers
```

## Limitations and next steps

- **Tables and figures:** fault-ride-through profiles and P–Q diagrams are figures; only their numeric parameters are captured. The table reader covers the direct questions it recognises; the rest go to the LLM.
- **Cross-references:** "as specified in Article 14(3)" is not followed automatically. An agent with a `get_article` tool, or a graph of references, would.
- **National codes and versions:** only the European baseline is indexed. The metadata already supports several jurisdictions and validity dates, so a national code (and its updates) can be added and filtered by the version in force.
- **Evaluation of answers:** answers are checked automatically and graded by an AI model; they still need a test set validated by engineers.
- **Human in the loop:** the tool points to the requirements and their source; an engineer still validates them.

## Author

Amaya de Miguel Mediavilla, MSc student in the Erasmus Mundus master DREAM (Dynamics of Renewables-based Power Systems), École Centrale de Nantes and HTW Berlin.

Developed with the help of [Claude Code](https://claude.com/claude-code), Anthropic's AI coding assistant.

The RfG text is © European Union, reused under the [Commission's reuse policy](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).
