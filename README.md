# 🏕️ Tasmania Backpacker Bot (Chat-AI-Project)

> A LangChain + Gemini-powered intelligent travel assistant for backpackers exploring Tasmania.  
> The system integrates document retrieval (RAG), external tools, and prompt-based routing to answer travel, weather, and budget queries interactively.

---

## 🌏 Scenario and Motivation

**Scenario Selected:**  
The system is designed for *independent backpackers visiting Tasmania* who often need up-to-date travel, weather, and cost information but lack consistent internet connectivity or unified sources.  
The **Tasmania Backpacker Bot** acts as a hybrid knowledge assistant — retrieving verified information from official PDF guidebooks (offline RAG) and combining it with live tools (DuckDuckGo Search, OpenWeatherMap, Open-Meteo Forecast) to deliver fact-checked and contextual travel insights.

**Why this scenario:**  
- Tasmania is a high-interest region for adventure travelers, with diverse weather and remote attractions.  
- Backpackers often require **cost estimates**, **weather forecasts**, and **route suggestions** — making it an ideal use case for RAG + Tool calling.  
- Demonstrates the integration of **static** and **dynamic** knowledge sources via an agentic reasoning workflow.

---

## 📄 Documents and Tools Used

| Component | Description |
|------------|-------------|
| **PDF Documents** | Tasmanian Backpacker Guides (official PDF documents stored in `/data/docs/`) used as the retrieval base for factual information such as accommodation, attractions, and safety. |
| **RAG Vector Store** | Chroma DB for semantic search over guide content. |
| **DuckDuckGo Search** | Fetches live, web-based updates when information is missing from the guides. |
| **OpenWeatherMap API** | Returns *current* weather conditions for Tasmanian cities. |
| **Open-Meteo API** | Provides *multi-day forecasts* (1–7 days) for temperature and precipitation. |
| **Trip Budget Planner Tool** | Estimates trip costs by querying the RAG index for AUD-based accommodation, food, and rental data. |

---

## 🤖 LLMs and Frameworks Adopted

| Layer | Framework / Model | Purpose |
|-------|-------------------|----------|
| **Language Model (LLM)** | `Gemini 2.5 Flash` via `langchain_google_genai.ChatGoogleGenerativeAI` | ReAct-style reasoning, tool calling, and summarisation. |
| **Framework** | [LangChain](https://python.langchain.com/) | Agent creation, memory management, and retrieval orchestration. |
| **Embeddings** | `HuggingFaceEmbeddings` (MiniLM-L6-v2) | Used to embed PDF chunks for semantic retrieval. |
| **Vector Store** | `Chroma` | Persistent local vector store for document chunks. |
| **UI Framework** | `Gradio` | Lightweight chat interface with example prompts. |

---

## 🔀 Routing Logic and Agent Design

The system uses a **ReAct (Reason + Act) agent** guided by a structured prompt (`agent_prompt`) that enforces deterministic routing rules.  
Each user query is classified through the **prompt logic** rather than hard-coded branching.

**Routing Summary:**

| Query Type | Routed Tool | Example |
|-------------|-------------|----------|
| *“Current weather in Hobart”* | `WeatherTool` (OpenWeatherMap) | → returns real-time conditions |
| *“3-day forecast for Launceston”* | `WeatherForecast` (Open-Meteo) | → returns forecast summary |
| *“Trip budget for 5 days in Hobart”* | `TripBudgetPlanner` | → uses RAG-derived prices to compute AUD estimate |
| *“Best camping spots near Cradle Mountain”* | `RAGTool` | → retrieves facts from PDF guides |
| *“Upcoming festivals in Hobart”* | `DuckDuckGoSearch` | → fetches real-time web info |

**Memory:**  
`ConversationBufferMemory` retains previous exchanges, enabling contextual follow-ups such as “What about tomorrow?” or “Add a car rental estimate too.”

---

## 💻 Running the System Locally

### 1️⃣ Clone and set up environment
```bash
git clone https://github.com/Jayani-123/Chat-AI-Project.git
cd Chat-AI-Project/tas_backpacker_bot
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate
pip install -r requirements.txt


### Set environment variables

Create a .env file:
GOOGLE_API_KEY=your_gemini_key_here
OPENWEATHERMAP_API_KEY=your_openweathermap_key_here

Run the Gradio app
python app.py

Then open http://localhost:7860
 in your browser.

 Example Queries and Expected Outputs
Example 1 — Weather Forecast

User:

Forecast the weather for Hobart for 3 days

Expected:

**3-Day Forecast for Hobart**
• Mon, 20 Oct — Low: 8.2 °C | High: 17.4 °C | Rain: 1.2 mm
• Tue, 21 Oct — Low: 9.1 °C | High: 18.0 °C | Rain: 0.4 mm
• Wed, 22 Oct — Low: 10.0 °C | High: 19.5 °C | Rain: 0.0 mm
Sources: WeatherForecast (Open-Meteo API)

Example 2 — Trip Budget

User:

Estimate a 3-day backpacker budget near Hobart

Expected:

🗺️ Trip Budget & Planner for Hobart (3 days)
🏕️ Accommodation / Camping: $40 × 3 = $120
🍽️ Food (per meal): $25 × 3 = $75
🚗 Vehicle Rental: $32 × 3 = $96
----------------------------------
💰 Estimated Total: $291 AUD
Sources: TripBudgetPlanner (backpacker PDF guides)

⚙️ Error Handling and Limitations

Known Limitations & Insights

⚠️ Document Coverage: The backpacker PDFs don’t always contain prices for food or accommodation in every region.

🌦️ Weather Forecast Geocoding: Queries like “3-day forecast for Hobart” initially failed due to string cleaning issues (hyphens, dashes). Fixed via regex sanitization.

🌐 Tool Constraints: DuckDuckGo sometimes returns duplicate or outdated links when no strong keywords are given.

💬 LLM Limitations: The Gemini model can occasionally over-generate “Sources” twice; handled with regex cleanup.

🧩 RAG Gaps: Retrieval quality depends on the embedding granularity; overly small chunks reduce context coherence.

🕒 Latency: Multi-tool chains (e.g., weather + RAG) increase response time slightly.

🧱 Unrelated Git Histories: Early version control issues were resolved via --allow-unrelated-histories merge.


🧩 Repository Structure
tas_backpacker_bot/
├── app.py                     # Gradio UI launcher
├── agent.py                   # Agent creation & process_query logic
├── tools.py                   # Tool definitions (Weather, RAG, Budget, Search)
├── rag.py                     # RAG setup, Chroma, embeddings
├── prompts.py                 # Prompt templates (RAG, Agent)
├── config.py / config.yml     # Configurations and API keys
├── data/docs/                 # Backpacker PDF guides
└── README.md                  # Project documentation

## 👩‍💻 Contributors

This project was developed collaboratively as part of **KIT719 – Assignment 2** at the **University of Tasmania**.

| Name | Student ID |
|------|-------------|
| **Jayani Madusha Edirisinghe** | 707202 |
| **Rakhitha Uthpalawanna Dassanayake Mudiyanselage** | 706778 |
| **Jahanvi Dasari** | 682393 |
