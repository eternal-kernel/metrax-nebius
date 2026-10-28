# ⚖️ METRAX-Cloud: Autonomous E-Commerce & Packaged Commodity Statutory Inspector

> **Autonomous multi-tier statutory compliance auditor for the Legal Metrology (Packaged Commodities) Rules, 2011 — saving up to 99% in LLM compute credits via tiered routing between NVIDIA Nemotron Nano (30B), deterministic rule engines, Tavily live regulatory intelligence, and Nemotron Ultra (550B).**

🏆 **Submission Tracks:**
- **Best Apps and Agents Track** (Powered by Nebius Token Factory & NVIDIA Nemotron)
- **Best Use of Tavily** (Real-time statutory precedent extraction & marketplace URL ingestion)

---

## 💡 The Core Problem: The E-Commerce Compliance Crisis

In India, pre-packaged commodities sold across digital marketplaces (Amazon, Flipkart, Blinkit, Zepto, Quick Commerce) and physical retail must strictly conform to the **Legal Metrology (Packaged Commodities) Rules, 2011** and the **Legal Metrology Act, 2009**.

Every year, thousands of sellers face severe penal action:
- **Section 36(1) Penalties**: Fines up to **₹25,000** for first offences, **₹50,000** for second offences, and **imprisonment up to 1 year** with fines up to **₹1,00,000** for subsequent offences.
- **Section 15 Seizure**: Full seizure of non-compliant inventory at fulfillment centers.
- **Section 49 Corporate Liability**: Direct personal liability of Company Directors and Compliance Officers.
- **Shrinkflation & Unit Price Distortion**: Sellers reducing pack sizes without updating Unit Sale Price (Rule 6(11)) or using deceptive imperial measures (e.g., "16 oz", "approx 500g").

### Why Naive Single-LLM Approaches Fail:
1. **Mathematical Hallucination**: Generative models cannot reliably verify division for Unit Sale Price (`MRP / Quantity`) with high precision.
2. **Extreme Token Waste**: Sending clean, fully-compliant products to a 550B-parameter model wastes tens of thousands of tokens and hundreds of dollars per day.
3. **Stale Legal Knowledge**: Static LLMs lack real-time gazette notifications, high court precedents, and live case laws.

---

## 🏛️ System Architecture: The 3-Tier Multi-Model Cascade

METRAX-Cloud solves this by coupling **specialized tiered model routing** on **Nebius Token Factory** with a **zero-latency deterministic rule engine** and **Tavily Search/Extract**:

```
                              Raw Product Input
               (E-Commerce Listing Copy / Live URL / Label Image)
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  [Input Normalization Layer]                                │
        │  - Image Upload  ➔ Transcribed via Nemotron-3.5-Super-VL    │
        │  - Product URL   ➔ Content Extracted via Tavily Extract     │
        │  - Raw Text Copy ➔ Direct Ingestion                         │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  TIER 1: Fast Structural Extraction                         │
        │  Model: NVIDIA-Nemotron-3-Nano-30B-A3B (Nebius)             │
        │  - Extracts 8 mandatory fields into strict Pydantic schema  │
        │  - Fallback safety: Nemotron-Super-120B on parse failure    │
        │  - Latency: ~10s | Cost: ~$0.00025                          │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  TIER 2: Deterministic Statutory & Mathematical Audit       │
        │  Engine: rules.py (Pydantic + Python SI Math Engine)        │
        │  - Rule 6(1)(a): Complete Manufacturer/Importer Address     │
        │  - Rule 6(1)(aa): Country of Origin Verification            │
        │  - Rule 6(1)(c) & Rule 12: SI Metric Units & Vague Words    │
        │  - Rule 6(1)(d): Month & Year of Manufacture                │
        │  - Rule 6(1)(e): MRP & Tax-Inclusive Mandatory Phrasing     │
        │  - Rule 6(2): Consumer Care Email & Phone Verification      │
        │  - Rule 6(11): Exact Unit Sale Price Math Audit (5% margin) │
        │  - Latency: < 1 ms | Cost: $0.00000 (Zero Hallucination)    │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                         [SMART COMPUTE ROUTER]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
      [100% Deterministic Compliance]             [Statutory Defect Flagged]
                    │                                     │
      ⚡ ULTRA & TAVILY BYPASSED                   🔍 TAVILY REGULATORY SEARCH
      - Instant Clearance Certificate             - Query live precedent gazettes
      - 99.0% Compute Cost Saved                  - Fetch Section 36 penalties
      - Latency cut from 35s to 10s                       │
                                                          ▼
                                                  🏛️ TIER 3: DEEP LEGAL SYNTHESIS
                                                  Model: Nemotron-3-Ultra-550B
                                                  - Official Show-Cause Notice
                                                  - Clause-by-clause cross-exam
                                                  - Section 36/49 risk exposure
                                                  - Ready-to-print corrected label
```

---

## ⚡ Empirical Benchmarks & Cost Savings

Measured on actual live Nebius Token Factory endpoints and Tavily API during test executions:

| Scenario | Routine Path | Endpoints Invoked | Latency | Pipeline Cost | Compute Savings vs All-Ultra |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Compliant Product** *(Fortune 1L Oil)* | **Fast Bypass** | Nemotron-3-Nano + `rules.py` | **10.7 s** | **$0.00025** | **99.0% Saved** ⚡ |
| **Violating Supplement** *(Titan Beast 32oz)* | **Deep Escalation** | Nemotron-3-Nano + `rules.py` + Tavily + Nemotron-3-Ultra-550B | **33.1 s** | **$0.01612** | **32.9% Saved** (vs Ultra extraction) |
| **Packaging Label OCR** *(Product Photo)* | **Vision + Tiered** | Nemotron-3.5-Super-VL + Nano + rules + Ultra | **38.4 s** | **$0.01750** | **High-Fidelity OCR** |

---

## 🛠️ Tech Stack & Live Models

- **Nebius Token Factory (Inference API)**:
  - `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` (Fast extraction with thinking tokens budget)
  - `nvidia/nemotron-3-super-120b-a12b` (Resilient parsing fallback)
  - `nvidia/Nemotron-3-Ultra-550b-a55b` (550B flagship model for statutory show-cause notices)
  - `nvidia/Nemotron-3.5-Super-VL` (Vision model for high-resolution packaging label OCR)
- **Tavily AI**:
  - `TavilyClient.search()`: Live legal precedents, gazette notifications, and Section 36 case laws.
  - `TavilyClient.extract()`: Real-time DOM markdown extraction from e-commerce product URLs.
- **Deterministic Engine**:
  - `Pydantic v2`: Strict schema validation, data parsing, and serialization.
  - Python SI math: Decoupled unit sale price verification, eliminating LLM arithmetic errors.
- **Frontend & Visualization**:
  - `Streamlit 1.66+`: High-contrast, accessibility-first split dashboard with live `st.status` telemetry.

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.14
- Nebius Token Factory API Key (`NEBIUS_API_KEY`)
- Tavily Search API Key (`TAVILY_API_KEY`)

### 2. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-username/metrax-nebius.git
cd metrax-nebius

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory:
```env
NEBIUS_API_KEY=your_actual_nebius_token_factory_key
TAVILY_API_KEY=your_actual_tavily_key
```
*(Note: You can also enter API keys directly in the Streamlit sidebar if running in a cloud container without `.env`)*

### 4. Run Verification Tests
```bash
# Verify API connections
python test_connections.py

# Verify deterministic rules engine & model list
python test_rules.py

# Run end-to-end multi-model routing pipeline
python test_engine.py
```

### 5. Launch the Streamlit Inspector
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛡️ Documented Failure Cases & Engineering Rigor

In accordance with strict production AI safety standards, the following known edge cases and mitigation strategies are documented:

1. **Multi-Pack & Heterogeneous Combo Bundles (Rule 24)**:
   - *Limitation*: When a single SKU contains distinct items (e.g., a grooming kit with 1 razor, 2 blades, and 50ml shaving gel), the statute requires apportioned unit sale prices for each distinct commodity.
   - *Mitigation*: Flagged under Rule 24 for human-in-the-loop legal review.

2. **Specular Reflections on Curved Bottles in OCR**:
   - *Limitation*: Severe flashlight glare on metallic spray cans or cylindrical cosmetic bottles can obscure consumer care digits or barcode details.
   - *Mitigation*: Nemotron-3.5-Super-VL requests re-upload if character confidence is low, and deterministic fallback flags missing characters as a MAJOR defect rather than passing blindly.

3. **Anti-Scraping Marketplace Walls on URLs**:
   - *Limitation*: Heavy client-side JavaScript SPAs or Cloudflare bot shields on marketplace URLs may return incomplete DOM trees.
   - *Mitigation*: Tavily Extract is coupled with user-facing fallback alerts prompting the user to paste raw listing copy.

4. **Reasoning Token Latency Spikes on Nemotron Nano**:
   - *Limitation*: Nemotron-3-Nano generates chain-of-thought tokens before JSON emission. If unconstrained, it can exhaust token budgets.
   - *Mitigation*: Strict system prompts ("Keep internal reasoning minimal and concise; output JSON immediately") with a generous 2048-token window and regex stripping for `<think>...</think>` tags ensure rapid execution without truncation.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE). Copyright (c) 2026 METRAX-Cloud Contributors.
