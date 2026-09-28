# Mutual Fund Facts Assistant

A facts-only mutual fund information assistant built as a Product Management and AI System Design project.

The prototype is designed as a product concept for Groww and currently supports five Quant Mutual Fund schemes using official public sources.

## Supported Mutual Funds

- Quant Small Cap Fund
- Quant Large Cap Fund
- Quant Flexi Cap Fund
- Quant Multi Cap Fund
- Quant PSU Fund

## What the Assistant Can Answer

The assistant provides factual information about supported mutual funds, including:

- Minimum SIP
- Minimum investment
- Expense Ratio / TER
- Exit load
- Fund managers
- Investment objective
- Asset allocation
- Riskometer
- Benchmark
- Redemption timelines
- Account statements
- Capital gains and taxation information available in the indexed official sources

## Facts-Only Guardrails

This assistant does not provide investment advice.

It will not:

- Recommend whether a user should buy, sell, hold, redeem, or invest in a fund
- Recommend the "best" mutual fund
- Predict investment returns
- Calculate or compare fund performance
- Provide personalized portfolio recommendations
- Process personal or sensitive information such as PAN, Aadhaar, account numbers, OTPs, email addresses, or phone numbers

For performance-related questions, users are directed to official Quant Mutual Fund information.

## RAG Architecture

The application uses a Retrieval-Augmented Generation (RAG) pipeline.

Pipeline:

Official Public Documents  
→ Docling PDF-to-Markdown conversion  
→ Markdown cleaning  
→ Heading-aware chunking  
→ Metadata enrichment  
→ Sentence Transformer embeddings  
→ ChromaDB vector database  
→ Semantic + BM25 hybrid retrieval  
→ Cross-encoder reranking  
→ Groq LLM generation  
→ Factual answer with official source citation

### Main Technologies

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- BM25
- Cross-Encoder reranking
- Groq API
- `openai/gpt-oss-20b`

Embedding model:

`all-MiniLM-L6-v2`

Reranking model:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

## Knowledge Base

The knowledge base contains 23 official public sources from:

- Quant Mutual Fund
- SEBI

The indexed ChromaDB contains approximately 2,401 chunks.

Sources include:

- Scheme Information Documents (SID)
- Scheme presentations
- Scheme documents
- Quant Mutual Fund factsheet
- Statement of Additional Information
- Total Expense Ratio information
- SEBI investor information
- Other official Quant Mutual Fund documents

## Example Questions

### 1. Minimum SIP

**Question:** What is the minimum SIP amount for Quant Small Cap Fund?

**Answer:** The minimum SIP amount is ₹1,000.

### 2. Fund Manager

**Question:** Who manages Quant Large Cap Fund?

**Answer:** Quant Large Cap Fund is managed by Sandeep Tandon and Ankit Pande.

### 3. Benchmark

**Question:** What is the benchmark of Quant Flexi Cap Fund?

**Answer:** The benchmark is NIFTY 500 TRI.

### 4. Riskometer

**Question:** What is the Riskometer level of Quant Multi Cap Fund?

**Answer:** The scheme is classified as Very High Risk.

### 5. Exit Load

**Question:** What is the exit load for Quant PSU Fund?

**Answer:** An exit load of 1% applies if units are redeemed within 15 days from the date of allotment.

The live assistant additionally provides the relevant official source link and source update information with factual responses.

## Source List

1. https://quantmutual.com/Admin/SIDPdf/SID%20quant%20Small%20Cap%20Fund%20March%202026.pdf
2. https://quantmutual.com/Admin/Pdf/quant%20Small%20Cap%20Fund_Presentation.pdf
3. https://quantmutual.com/Admin/Pdf/quant_Small_Cap_Fund.pdf
4. https://quantmutual.com/Admin/SIDPdf/SID_quant_Large_Cap_Fund_Sep_2025.pdf
5. https://quantmutual.com/Admin/Pdf/quant%20Large%20Cap%20Fund%20Presentation.pdf
6. https://quantmutual.com/Admin/Pdf/quant_Large_Cap_Fund.pdf
7. https://quantmutual.com/Admin/SIDPdf/SID_quant_Flexi_Cap_Fund_September_2025.pdf
8. https://quantmutual.com/Admin/Pdf/quant_Flexi_Cap_Fund_Presentation-%20Updated.pdf
9. https://quantmutual.com/Admin/Pdf/quant_Flexi_Cap_Fund.pdf
10. https://quantmutual.com/Admin/SIDPdf/SID%20quant%20PSU%20Fund%20March%202026.pdf
11. https://quantmutual.com/Admin/Pdf/quant%20PSU%20Fund%20Presentation.pdf
12. https://quantmutual.com/Admin/Pdf/quant_PSU_Fund.pdf
13. https://quantmutual.com/Admin/SIDPdf/SID%20quant%20Multi%20Cap%20Fund%20March%202026.pdf
14. https://quantmutual.com/Admin/Pdf/quant%20Multi%20Cap%20Fund_Presentation.pdf
15. https://quantmutual.com/Admin/Pdf/quant_Multi_Cap_Fund.pdf
16. https://quantmutual.com/Admin/Factsheet/quant_Factsheet_-_September_2026.pdf
17. https://quantmutual.com/Pdf/Definitions_and_Interpretation.pdf
18. https://quantmutual.com/Pdf/Penalities.pdf
19. https://quantmutual.com/Admin/SIDPdf/Principles_of_Incentive_Structure_for_Market_Makers.pdf
20. https://quantmutual.com/Admin/SIDPdf/Statement_of_Additional_Information.pdf
21. https://www.sebi.gov.in/sebi_data/faqfiles/sep-2024/1727242783639.pdf
22. https://quantmutual.com/Admin/Pdf/QMF_Schemes.pdf
23. https://www.quantmutual.com/Total-Expense-Ratio

## Running the Project Locally

Create and activate a Python virtual environment.

Install dependencies:

    pip install -r requirements.txt

Set the Groq API key as an environment variable:

    export GROQ_API_KEY="YOUR_GROQ_API_KEY"

Run the Streamlit application:

    streamlit run app.py

## Scope

This prototype currently supports only the five Quant Mutual Fund schemes listed above.

Answers are generated only from information available in the indexed official public documents.

## Known Limitations

- Only five Quant Mutual Fund schemes are currently supported.
- The assistant is not connected to Groww's internal systems or user accounts.
- The project is a product concept and is not an official Groww or Quant Mutual Fund product.
- Information is limited to the documents currently indexed in the knowledge base.
- Source documents may be updated after they are indexed.
- The assistant does not provide investment recommendations or personalized financial advice.
- Performance prediction and performance comparison are outside the scope.
- Personal financial/account information is outside the scope.

## Disclaimer

Facts-only. No investment advice.

This prototype provides factual information from official public sources and does not recommend mutual funds, predict returns, compare investment performance, or provide personalized investment advice.

Users should verify current information using the linked official sources before making financial decisions.
