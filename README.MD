# DecisionCell  
A modular multi-agent framework for transparent, tool-augmented decision making

DecisionCell is a proof-of-concept multi-agent decision framework designed to support
collaborative reasoning through transparency and tool use.  
It enables multiple agents to think, act, call tools, and exchange observations
to collectively analyze a problem or objective.

The system combines:

- **ReAct-style reasoning** – agents think, take structured actions, call tools, and react to observations  
- **Transparent reasoning traces** – each agent exposes its step-by-step thought process  
- **Retrieval-Augmented Generation (RAG)** – agents can query their own knowledge bases to ground reasoning in data  

This repo is initialized with an example that demonstrates how DecisionCell can support military Course of Action (CoA) selection using agents such as a Commander, Cyber Operations Expert, and Legal Advisor.

## 🔗 Example Reasoning Chain

Below is a simplified trace from an actual experiment run (Llama 3.3 70B, multi-agent configuration). It shows how agents collaborate through structured ReAct steps to reach a decision.

```
Step 1 │ Commander
        Thought:  I must evaluate each CoA based on simplicity, security,
                  and proportionality.
        Action:   ask_cyber_operations
        Line:     Compare the technical feasibility and potential risks of
                  CoA-1, CoA-2, and CoA-3.

Step 2 │ CyberOperationsExpert
        Thought:  I must weigh the technical feasibility, potential impact,
                  and risks of each CoA option.
        Action:   finish
        Line:     RECOMMENDATION: CoA-1 for high impact if detection risk
                  is mitigated, otherwise CoA-3 for stealth.

Step 3 │ Commander
        Thought:  I need to assess the legality of these options, considering
                  civilian impact and proportionality.
        Action:   ask_legal
        Line:     Evaluate the legality of BlackEnergy3 malware or GNSS
                  spoofing under international humanitarian law.

Step 4 │ LegalAdvisor
        Thought:  I must assess legality under IHL, considering distinction,
                  proportionality, and unnecessary suffering.
        Action:   search
        Line:     BlackEnergy3 GNSS spoofing international humanitarian law

Step 5 │ LegalAdvisor
        Thought:  I must weigh military necessity against potential harm
                  to civilians.
        Action:   ask_human
        Line:     What is the potential impact on civilian infrastructure?

Step 6 │ LegalAdvisor
        Thought:  Given the high impact of BlackEnergy3 and medium impact
                  of GNSS spoofing, CoA-3 balances necessity with lower risk.
        Action:   finish
        Line:     RECOMMENDATION: CoA-3 (GNSS spoofing) — lower risk of
                  harm to civilians.

Step 7 │ Commander
        Thought:  I must present this draft decision for mandatory human review.
        Action:   ask_human
        Line:     CONCEPT — WAITING FOR HUMAN FEEDBACK
                  RECOMMENDED: CoA-3 (GNSS spoofing)
                  REJECTED: CoA-1 (disproportionate civilian harm)

Step 8 │ Commander
        Thought:  Human has approved the draft decision.
        Action:   finish
        Line:     FINAL DECISION: CoA-3 (GNSS spoofing)
```

This trace illustrates three key properties of the framework:
1. **Separation of concerns** — each agent contributes domain-specific reasoning (technical feasibility, legal compliance)
2. **Transparency** — every intermediate step is visible and traceable
3. **Human-in-the-Loop** — the Commander presents a draft decision (CONCEPT) and waits for human approval before finalising

## 🚀 Getting Started

This repository includes a Jupyter notebook demonstrating a full DecisionCell workflow. Follow the steps below first to prepare your environment.

## 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Choose an LLM provider

### **Using Ollama (local models)**

Make sure the model is installed and the service is running:

```bash
ollama pull llama3
ollama list
```

Agents use:

```python
provider="ollama", model="llama3"
```

### **Using Hugging Face (cloud models)**

Set your key:

```bash
export HF_KEY=your_api_key
```

Or create a `.env` file:

```
HF_KEY=your_api_key
```

Agents use:

```python
provider="hf", model="meta-llama/Meta-Llama-3-8B-Instruct"
```

---

## 4. Knowledge Bases (optional)

Place per‑agent documents under:

```
agents/kb/<AgentName>/
```

Tools will automatically load these for retrieval‑augmented reasoning.

---

## 5. Run the Notebook

```bash
jupyter lab
```

Open:

```
DecisionCell.ipynb
```
The notebook contains guided setup cells that walk you through the full workflow:

- Configuring your provider (Ollama or HF)
- Creating the staff (agents)
- Enabling tools
- Building the DecisionCell
- Defining and analyzing a mission objective


## 6. Extending

- Create new agents by giving them a name and placing their system prompt in the `agents/` folder (e.g., `agents/prompts/<AgentName>.txt`).  
- Add new tools by subclassing `BaseTool`. 

All experiment-specific configuration (agents, tools, objectives, models) is defined directly in the notebook.
