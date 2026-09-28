import chromadb
import re
import time
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from rank_bm25 import BM25Okapi

DATABASE_FOLDER = "chroma_db"
COLLECTION_NAME = "quant_mutual_fund"
MODEL_NAME = "all-MiniLM-L6-v2"
NUMBER_OF_RESULTS = 10

SOURCE_URLS = {
    "SID_quant_Small_Cap_Fund_March_2026.md": "https://quantmutual.com/Admin/SIDPdf/SID%20quant%20Small%20Cap%20Fund%20March%202026.pdf",
    "Presentation_for_Quant_Small_Cap_Fund.md": "https://quantmutual.com/Admin/Pdf/quant%20Small%20Cap%20Fund_Presentation.pdf",
    "quant_Small_Cap_Fund.md": "https://quantmutual.com/Admin/Pdf/quant_Small_Cap_Fund.pdf",

    "SID_quant_Large_Cap_Fund_Sep_2025.md": "https://quantmutual.com/Admin/SIDPdf/SID_quant_Large_Cap_Fund_Sep_2025.pdf",
    "quant_Large_Cap_Fund_Presentation.md": "https://quantmutual.com/Admin/Pdf/quant%20Large%20Cap%20Fund%20Presentation.pdf",
    "quant_Large_Cap_Fund.md": "https://quantmutual.com/Admin/Pdf/quant_Large_Cap_Fund.pdf",

    "SID_quant_Flexi_Cap_Fund_September_2025.md": "https://quantmutual.com/Admin/SIDPdf/SID_quant_Flexi_Cap_Fund_September_2025.pdf",
    "quant_Flexi_Cap_Fund_Presentation-_Updated.md": "https://quantmutual.com/Admin/Pdf/quant_Flexi_Cap_Fund_Presentation-%20Updated.pdf",
    "quant_Flexi_Cap_Fund.md": "https://quantmutual.com/Admin/Pdf/quant_Flexi_Cap_Fund.pdf",

    "SID_quant_PSU_Fund_March_2026.md": "https://quantmutual.com/Admin/SIDPdf/SID%20quant%20PSU%20Fund%20March%202026.pdf",
    "quant_PSU_Fund_Presentation.md": "https://quantmutual.com/Admin/Pdf/quant%20PSU%20Fund%20Presentation.pdf",
    "quant_PSU_Fund.md": "https://quantmutual.com/Admin/Pdf/quant_PSU_Fund.pdf",

    "SID_quant_Multi_Cap_Fund_March_2026.md": "https://quantmutual.com/Admin/SIDPdf/SID%20quant%20Multi%20Cap%20Fund%20March%202026.pdf",
    "quant_Multi_Cap_Fund_Presentation.md": "https://quantmutual.com/Admin/Pdf/quant%20Multi%20Cap%20Fund_Presentation.pdf",
    "quant_Multi_Cap_Fund.md": "https://quantmutual.com/Admin/Pdf/quant_Multi_Cap_Fund.pdf",

    "quant_Factsheet_-_September_2026.md": "https://quantmutual.com/Admin/Factsheet/quant_Factsheet_-_September_2026.pdf",
    "Definitions_and_Interpretation.md": "https://quantmutual.com/Pdf/Definitions_and_Interpretation.pdf",
    "Penalities.md": "https://quantmutual.com/Pdf/Penalities.pdf",
    "Principles_of_Incentive_Structure_for_Market_Makers.md": "https://quantmutual.com/Admin/SIDPdf/Principles_of_Incentive_Structure_for_Market_Makers.pdf",
    "Statement_of_Additional_Information.md": "https://quantmutual.com/Admin/SIDPdf/Statement_of_Additional_Information.pdf",
    "1727242783639.md": "https://www.sebi.gov.in/sebi_data/faqfiles/sep-2024/1727242783639.pdf",
    "QMF_Schemes.md": "https://quantmutual.com/Admin/Pdf/QMF_Schemes.pdf",
    "quant_TER.md": "https://www.quantmutual.com/Total-Expense-Ratio",
}

SOURCE_DATES = {
    "SID_quant_Small_Cap_Fund_March_2026.md": "March 2026",
    "SID_quant_Large_Cap_Fund_Sep_2025.md": "September 2025",
    "SID_quant_Flexi_Cap_Fund_September_2025.md": "September 2025",
    "SID_quant_PSU_Fund_March_2026.md": "March 2026",
    "SID_quant_Multi_Cap_Fund_March_2026.md": "March 2026",
    "quant_Factsheet_-_September_2026.md": "September 2026",
    "1727242783639.md": "September 2024",
}

print("Loading RAG system...")

embedding_model = SentenceTransformer(MODEL_NAME)
INTENT_EXAMPLES = {
    "exit_load": [
        "What is the exit load?",
        "Will I be charged if I withdraw my investment early?",
        "Is there a penalty if I redeem my units?",
        "Does the fund deduct anything when I cash out?",
    ],
    "minimum_investment": [
        "What is the minimum investment amount?",
        "How much money do I need to start investing?",
        "What is the minimum lump sum investment?",
        "How much is required for my first one-time investment?",
    ],
    "sip": [
        "What is the minimum SIP amount?",
        "How little can I invest every month?",
        "What is the smallest recurring investment allowed?",
        "What is the minimum SIP instalment?",
    ],
    "fund_manager": [
        "Who is the fund manager?",
        "Who manages the fund?",
        "Who handles the investment portfolio?",
        "Who is responsible for managing the scheme?",
    ],
    "ter": [
        "What is the expense ratio?",
        "What is the TER?",
        "What are the charges for Direct and Regular plans?",
        "What are the ongoing annual fund charges?",
    ],
    "investment_objective": [
        "What is the investment objective?",
        "What does the fund aim to achieve?",
        "What outcome is the fund designed to pursue?",
        "What is the purpose of the investment strategy?",
    ],
    "asset_allocation": [
        "What is the asset allocation?",
        "Where does the fund invest its money?",
        "What percentage can be invested in each asset class?",
        "What are the permitted investment ranges?",
    ],
    "redemption_timeline": [
        "When will I receive my redemption proceeds?",
        "How long does redemption payment take?",
        "How soon will I receive my money after redeeming units?",
        "When will the payout reach me after I sell my units?",
    ],
    "riskometer": [
        "What is the Riskometer level of the fund?",
        "What is the risk level of the scheme?",
        "How risky is this mutual fund according to the Riskometer?",
        "What does the scheme Riskometer say?",
    ],
    "benchmark": [
        "What is the benchmark of the fund?",
        "Which benchmark index does the scheme use?",
        "What index is the fund benchmarked against?",
        "What is the scheme benchmark?",
    ],

    "account_statement": [
        "How can I get my account statement?",
        "How do I request an account statement?",
        "When will I receive my account statement?",
        "How can I get my Consolidated Account Statement?",
        "How do I get my CAS?",
    ],
    "capital_gains": [
        "What are the capital gains tax rules?",
        "What tax applies to capital gains from this fund?",
        "What is the long-term capital gains tax?",
        "What is the short-term capital gains tax?",
        "How are capital gains from this mutual fund taxed?",
    ],
}

USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

if USE_RERANKER:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
else:
    reranker = None
intent_names = []
intent_texts = []

for intent_name, examples in INTENT_EXAMPLES.items():
    for example in examples:
        intent_names.append(intent_name)
        intent_texts.append(example)

intent_embeddings = embedding_model.encode(
    intent_texts,
    normalize_embeddings=True
)

chroma_client = chromadb.PersistentClient(
    path=DATABASE_FOLDER
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

# Load all chunks for BM25 keyword search
all_chunks = collection.get(
    include=["documents", "metadatas"]
)

bm25_documents = all_chunks["documents"]
bm25_metadatas = all_chunks["metadatas"]

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

tokenized_documents = [
    tokenize(document)
    for document in bm25_documents
]

bm25 = BM25Okapi(tokenized_documents)

print(f"Ready! Database contains {collection.count()} chunks.")

def add_source_citation(answer, metadata):
    source_file = metadata.get("source_file", "")
    source_url = SOURCE_URLS.get(source_file)
    source_date = SOURCE_DATES.get(source_file, "Date not specified in source")

    if source_url and source_date:
        return (
            f"{answer}\n"
            f"Source: {source_url}\n"
            f"Last updated from sources: {source_date}"
        )

    if source_url:
        return f"{answer}\nSource: {source_url}"

    return answer

def ask_rag(question):

    # Detect which fund the user is asking about
    question_lower = question.lower()
    # Guardrail: refuse investment advice / buy-sell recommendations
    advice_phrases = [
        "should i invest",
        "should i buy",
        "should i sell",
        "should i redeem",
        "should i hold",
        "is it a good investment",
        "is this a good investment",
        "which fund should i buy",
        "which fund should i invest",
        "which fund is best",
        "recommend",
    ]

    if (
        any(phrase in question_lower for phrase in advice_phrases)
        or "a good investment" in question_lower
        or ("which" in question_lower and "best" in question_lower)
    ):
        return (
            "I can provide factual information about mutual funds, "
            "but I can't recommend whether you should buy, sell, hold, "
            "redeem, or invest in a particular fund.\n"
            "Investor education: "
            "https://www.sebi.gov.in/sebi_data/faqfiles/sep-2024/1727242783639.pdf\n"
            "Last updated from sources: September 2024"
        )
   # Guardrail: do not provide, predict, calculate, or compare fund returns
    performance_phrases = [
        "compare the returns",
        "expected return",
        "expected returns",
        "expect from",
        "how much return",
        "how much returns",
        "past return",
        "past returns",
        "historical return",
        "historical returns",
        "fund performance",
        "performance of",
        "performed",
        "cagr",
        "xirr",
    ]

    if (
        any(phrase in question_lower for phrase in performance_phrases)
        or ("returns" in question_lower and "highest" in question_lower)
    ):
        return (
            "I don't provide, predict, calculate, or compare mutual fund returns or performance. "
            "For official performance information, please refer to the Quant Mutual Fund factsheet:\n"
            + "https://quantmutual.com/Admin/Factsheet/quant_Factsheet_-_September_2026.pdf\n"
            + "Last updated from sources: September 2026"
        )
    # Guardrail: do not accept or process personal/sensitive information
    pii_patterns = [
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",          # PAN
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",         # Aadhaar-like number
        r"\b\d{10}\b",                         # Indian phone number
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",         # Email address
    ]

    pii_phrases = [
        "my pan is",
        "my aadhaar",
        "my aadhar",
        "my account number",
        "my otp",
    ]

    if (
        any(re.search(pattern, question, re.IGNORECASE) for pattern in pii_patterns)
        or any(phrase in question_lower for phrase in pii_phrases)
    ):
        return (
            "Please do not share personal or sensitive information such as PAN, Aadhaar, "
            "account numbers, OTPs, email addresses, or phone numbers. "
            "I can only provide general factual information about mutual funds."
        )

    question_intent_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    intent_scores = {}

    for intent_name in INTENT_EXAMPLES:
        scores = [
            float(question_intent_embedding @ intent_embeddings[i])
            for i, name in enumerate(intent_names)
            if name == intent_name
        ]

        intent_scores[intent_name] = max(scores)

    semantic_intent = max(
        intent_scores,
        key=intent_scores.get
    )

    semantic_intent_score = intent_scores[semantic_intent]
    print(
        f"Semantic intent: {semantic_intent} "
        f"(score: {semantic_intent_score:.3f})"
    )

    # Detect user intent from natural-language variations
    is_exit_load = (
        semantic_intent == "exit_load"
        or "exit load" in question_lower
        or "exit charge" in question_lower
        or (
            ("withdraw" in question_lower or "redeem" in question_lower)
            and ("charge" in question_lower or "early" in question_lower)
        )
    )

    is_minimum_investment = (
        "sip" not in question_lower
        and (
            semantic_intent == "minimum_investment"
            or "minimum investment" in question_lower
            or "minimum amount" in question_lower
            or "minimum lump-sum" in question_lower
            or "how much money" in question_lower
            or "start investing" in question_lower
            or "least amount" in question_lower
            or "invest initially" in question_lower
        )
    )

    is_fund_manager = (
        semantic_intent == "fund_manager"
        or "fund manager" in question_lower
        or "fund managers" in question_lower
        or "who manages" in question_lower
    )

    is_asset_allocation = (
        semantic_intent == "asset_allocation"
        or "asset allocation" in question_lower
        or "allocate its assets" in question_lower
        or (
            "percentage" in question_lower
            and "portfolio" in question_lower
            and "asset" in question_lower
        )
        or (
            "where does" in question_lower
            and "invest" in question_lower
            and (
                "proportion" in question_lower
                or "percentage" in question_lower
            )
        )
    )

    is_redemption_timeline = (
        not is_exit_load
        and (
            semantic_intent == "redemption_timeline"
            or "redemption proceeds" in question_lower
            or "receive redemption" in question_lower
            or "redemption timeline" in question_lower
            or (
                "redemption" in question_lower
                and (
                    "working days" in question_lower
                    or "when" in question_lower
                    or "how long" in question_lower
                    or "paid" in question_lower
                )
            )
            or (
                ("redeem" in question_lower or "redeeming" in question_lower)
                and (
                    "when" in question_lower
                    or "how long" in question_lower
                    or "working days" in question_lower
                )
            )
        )
    )

    is_riskometer = (
        semantic_intent == "riskometer"
        or "riskometer" in question_lower
        or "risk-o-meter" in question_lower
        or "risk level" in question_lower
    )    
   
    is_investment_objective = (
        semantic_intent == "investment_objective"
        or "investment objective" in question_lower
        or "objective of" in question_lower
        or "aim to achieve" in question_lower
        or "aims to achieve" in question_lower
        or "trying to accomplish" in question_lower
    )

    if "small cap" in question_lower:
        selected_fund = "Quant Small Cap Fund"
    elif "large cap" in question_lower:
        selected_fund = "Quant Large Cap Fund"
    elif "flexi cap" in question_lower:
        selected_fund = "Quant Flexi Cap Fund"
    elif "multi cap" in question_lower:
        selected_fund = "Quant Multi Cap Fund"
    elif "psu" in question_lower:
        selected_fund = "Quant PSU Fund"
    else:
        selected_fund = None

    # BM25 keyword search
    tokenized_query = tokenize(question_lower)

    eligible_indices = [
        i for i, metadata in enumerate(bm25_metadatas)
        if selected_fund is None
        or metadata.get("fund_name") == selected_fund
    ]

    # Build BM25 using only the relevant fund's chunks.
    # This prevents unrelated funds from affecting BM25 term statistics.
    eligible_tokenized_documents = [
        tokenized_documents[i]
        for i in eligible_indices
    ]

    fund_bm25 = BM25Okapi(eligible_tokenized_documents)

    fund_bm25_scores = fund_bm25.get_scores(tokenized_query)

    # Map the fund-specific scores back to the original Chroma indices.
    bm25_scores = {
        original_index: fund_bm25_scores[position]
        for position, original_index in enumerate(eligible_indices)
    }

    bm25_top_indices = sorted(
        eligible_indices,
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:40]

    for i in eligible_indices:
        if bm25_metadatas[i].get("chunk_id") == 157 and "SID_quant_Small_Cap_Fund" in bm25_metadatas[i].get("source_file", ""):
            print("DEBUG CHUNK 157 BM25 SCORE:", bm25_scores[i])

    print("DEBUG BM25 TOP 10:")
    for rank, i in enumerate(bm25_top_indices, start=1):
        print(rank, bm25_metadatas[i].get("chunk_id"), round(bm25_scores[i], 2))

    bm25_results = []

    for i in bm25_top_indices:
        bm25_results.append({
            "document": bm25_documents[i],
            "metadata": bm25_metadatas[i],
            "score": bm25_scores[i]
        })

    # Expand the search query based on the type of question
    if is_minimum_investment:
        search_question = (
            question
            + " minimum investment minimum application purchase "
            + "lumpsum lump sum initial investment additional purchase "
            + "subsequent investment SIP minimum amount"
        )

    elif is_fund_manager:
        search_question = (
            question
            + " WHO MANAGES THE SCHEME "
            + "Name Age Qualification Tenure for scheme management "
            + "Managing the scheme since"
        )
    elif is_exit_load:
        search_question = (
            question
            + " exit load redemption load withdrawal redeem"
        )

    elif (
        semantic_intent == "ter"
        or "expense ratio" in question_lower
        or re.search(r"\bter\b", question_lower)
    ):
        search_question = (
            question
            + " expense ratio total expense ratio TER"
        )

    elif "performance" in question_lower or "return" in question_lower:
        search_question = (
            question
            + " performance returns scheme returns benchmark"
        )

    elif semantic_intent == "sip" or "sip" in question_lower:
        search_question = (
            question
            + " SIP systematic investment plan minimum SIP amount instalment"
        )

    elif semantic_intent == "capital_gains":
        search_question = (
            question
            + " Capital Gains Long Term period of holding more than 12 months "
            + "Short Term period of holding up to 12 months "
            + "12.50% 20% tax"
        )

    elif is_riskometer:
        search_question = (
            question
            + " scheme riskometer risk of the scheme "
            + "very high risk product suitability"
        )

    elif is_asset_allocation:
        search_question = (
            question
            + " HOW WILL THE SCHEME ALLOCATE ITS ASSETS "
            + " indicative allocations minimum maximum "
            + " equity debt money market InvITs residual portion"
        )

    elif "lock-in" in question_lower or "lock in" in question_lower:
        search_question = (
            question
            + " lock-in lock in period ELSS equity linked savings scheme "
            + "three years units cannot be redeemed"
        )

    elif "tax" in question_lower or "taxation" in question_lower:
        search_question = (
            question
            + " TAXATION capital gains long term short term "
            + "resident investors non-resident investors "
            + "equity oriented fund TDS STT tax rate"
        )

    elif is_redemption_timeline:
        search_question = (
            question
            + " applicable timelines dispatch redemption proceeds "
            + "within three working days valid redemption request"
        )

    elif "redemption" in question_lower:
        search_question = (
            question
            + " redemption rules minimum redemption switch out "
            + "redemption proceeds within three working days "
            + "applicable NAV cut off time exit load "
            + "restrict redemption suspend redemption"
        )

    else:
        search_question = question

    embedding_start = time.time()

    question_embedding = embedding_model.encode(
        search_question,
        normalize_embeddings=True
    )

    print(f"⏱ EMBEDDING TIME: {time.time() - embedding_start:.2f} seconds")

    print("DEBUG selected_fund:", selected_fund)
    print("DEBUG search_question:", search_question)

    if selected_fund is None:
        query_filter = None
    elif is_fund_manager:
        query_filter = {
            "$and": [
                {"fund_name": selected_fund},
                {"document_type": "SID"}
            ]
        }
    else:
        query_filter = {"fund_name": selected_fund}

    # Search only the fund detected in the user's question
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=NUMBER_OF_RESULTS,
        where=query_filter,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    # Preserve the original top semantic candidates before hybrid reranking
    semantic_top_documents = results["documents"][0][:4]
    semantic_top_metadatas = results["metadatas"][0][:4]

    # Combine semantic and BM25 candidates without removing useful results
    combined_documents = []
    combined_metadatas = []
    seen_keys = set()

    # Add semantic-search candidates
    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        key = meta["source_file"] + "_" + str(meta["chunk_id"])

        if key not in seen_keys:
            combined_documents.append(doc)
            combined_metadatas.append(meta)
            seen_keys.add(key)

    # Add BM25 candidates
    for bm25_result in bm25_results:
        doc = bm25_result["document"]
        meta = bm25_result["metadata"]

        key = meta["source_file"] + "_" + str(meta["chunk_id"])

        if key not in seen_keys:
            combined_documents.append(doc)
            combined_metadatas.append(meta)
            seen_keys.add(key)

    results["documents"][0] = combined_documents
    results["metadatas"][0] = combined_metadatas

    # Rerank the hybrid candidates only when CrossEncoder is enabled
    if USE_RERANKER:
        rerank_pairs = [
            [question, doc]
            for doc in results["documents"][0]
        ]

        reranker_start = time.time()

        rerank_scores = reranker.predict(rerank_pairs)

        print(f"⏱ RERANKER TIME: {time.time() - reranker_start:.2f} seconds")

        print("DEBUG RERANKER SCORES:")
        for score, doc, meta in zip(
            rerank_scores,
            results["documents"][0],
            results["metadatas"][0]
        ):
            print(
                "chunk",
                meta.get("chunk_id"),
                "file",
                meta.get("source_file"),
                "score",
                round(float(score), 3)
            )
    else:
        rerank_scores = [0] * len(results["documents"][0])
        print("CrossEncoder reranker disabled.")

    # Rank all candidates using the CrossEncoder
    reranked_all = sorted(
        zip(
            rerank_scores,
            results["documents"][0],
            results["metadatas"][0]
        ),
        key=lambda x: x[0],
        reverse=True
    )

    # Build the final context from multiple retrieval signals.
    # Start with the top 3 semantic results so strong semantic evidence
    # cannot be completely discarded by the CrossEncoder.
    final_candidates = []
    final_keys = set()

    for doc, meta in zip(
        semantic_top_documents,
        semantic_top_metadatas
    ):
        key = meta["source_file"] + "_" + str(meta["chunk_id"])

        if key not in final_keys:
            final_candidates.append((0, doc, meta))
            final_keys.add(key)

    # Add the strongest CrossEncoder results that are not already included.
    for score, doc, meta in reranked_all:
        key = meta["source_file"] + "_" + str(meta["chunk_id"])

        if key not in final_keys:
            final_candidates.append((score, doc, meta))
            final_keys.add(key)

        if len(final_candidates) >= 5:
            break

    # Add the strongest BM25 result that is not already included.
    for bm25_result in bm25_results:
        meta = bm25_result["metadata"]
        key = meta["source_file"] + "_" + str(meta["chunk_id"])

        if key not in final_keys:
            final_candidates.append(
                (
                    bm25_result["score"],
                    bm25_result["document"],
                    meta
                )
            )
            final_keys.add(key)

        if len(final_candidates) >= 6:
            break

    results["documents"][0] = [
        item[1] for item in final_candidates
    ]

    results["metadatas"][0] = [
        item[2] for item in final_candidates
    ]

    # For fund-manager questions, find the actual manager section in the SID
    if is_fund_manager:
        sid_chunks = collection.get(
            where={
                "$and": [
                    {"fund_name": selected_fund},
                    {"document_type": "SID"}
                ]
            },
            include=["documents", "metadatas"]
        )

        for doc, meta in zip(sid_chunks["documents"], sid_chunks["metadatas"]):
            if doc.strip().startswith("## E.") and "WHO MANAGES THE SCHEME" in doc:
                results["documents"][0] = [doc]
                results["metadatas"][0] = [meta]

    # For minimum-investment questions, find the fund document
    # containing the actual lump-sum and subsequent investment amounts
    if is_minimum_investment:
        fund_chunks = collection.get(
            where={
                "fund_name": selected_fund
            },
            include=["documents", "metadatas"]
        )

        for doc, meta in zip(
            fund_chunks["documents"],
            fund_chunks["metadatas"]
        ):
            doc_upper = doc.upper()

            if (
                "LUMPSUM" in doc_upper
                and (
                    "MINIMUM INVESTMENT" in doc_upper
                    or "SUBSEQUENT INVESTMENT" in doc_upper
                )
            ):

                results["documents"][0] = [doc]
                results["metadatas"][0] = [meta]

                # The subsequent-investment heading may be split
                # into the next Chroma chunk.
                next_chunk_id = meta.get("chunk_id", 0) + 1

                for next_doc, next_meta in zip(
                    fund_chunks["documents"],
                    fund_chunks["metadatas"]
                ):
                    if (
                        next_meta.get("source_file") == meta.get("source_file")
                        and next_meta.get("chunk_id") == next_chunk_id
                        and "SUBSEQUENT INVESTMENT" in next_doc.upper()
                    ):
                        results["documents"][0].append(next_doc)
                        results["metadatas"][0].append(next_meta)
                        break

                break

    # For asset-allocation questions, find the actual asset-allocation section in the SID
    if is_asset_allocation:
        sid_chunks = collection.get(
            where={
                "$and": [
                    {"fund_name": selected_fund},
                    {"document_type": "SID"}
                ]
            },
            include=["documents", "metadatas"]
        )

        for doc, meta in zip(
            sid_chunks["documents"],
            sid_chunks["metadatas"]
        ):
            doc_upper = doc.upper()

            if (
                "HOW WILL THE SCHEME ALLOCATE ITS ASSETS" in doc_upper
                and "INDICATIVE ALLOCATIONS" in doc_upper
            ):
                results["documents"][0] = [doc]
                results["metadatas"][0] = [meta]
                break

    # For Riskometer questions, retrieve explicit scheme risk evidence
    if is_riskometer:
        if selected_fund == "Quant Small Cap Fund":
            factsheet_chunks = collection.get(
                where={"source_file": "quant_Factsheet_-_September_2026.md"},
                include=["documents", "metadatas"]
            )

            for doc, meta in zip(
                factsheet_chunks["documents"],
                factsheet_chunks["metadatas"]
            ):
                if (
                    "quant small cap fund" in doc.lower()
                    and "very high risk" in doc.lower()
                ):
                    results["documents"][0] = [doc]
                    results["metadatas"][0] = [meta]
                    break

        else:
            fund_chunks = collection.get(
                where={"fund_name": selected_fund},
                include=["documents", "metadatas"]
            )

            for doc, meta in zip(
                fund_chunks["documents"],
                fund_chunks["metadatas"]
            ):
                if "risk of the scheme is" in doc.lower():
                    results["documents"][0] = [doc]
                    results["metadatas"][0] = [meta]
                    break

    # For expense-ratio questions, use the dedicated TER document
    if "expense ratio" in question_lower or re.search(r"\bter\b", question_lower):
        ter_chunks = collection.get(
            where={
                "$and": [
                    {"fund_name": selected_fund},
                    {"document_type": "TER"}
                ]
            },
            include=["documents", "metadatas"]
        )

        if ter_chunks["documents"]:
            results["documents"][0] = ter_chunks["documents"]
            results["metadatas"][0] = ter_chunks["metadatas"]

    # For exit-load questions, find the actual Load Structure section in the SID
    if is_exit_load:
        sid_chunks = collection.get(
            where={
                "$and": [
                    {"fund_name": selected_fund},
                    {"document_type": "SID"}
                ]
            },
            include=["documents", "metadatas"]
        )

        for doc, meta in zip(sid_chunks["documents"], sid_chunks["metadatas"]):
            if re.match(r"^##\s*D\.\s*LOAD STRUCTURE", doc.strip(), re.IGNORECASE):
                results["documents"][0] = [doc]
                results["metadatas"][0] = [meta]
                break

    # For taxation questions, find the actual Taxation section in the SID
    if "tax" in question_lower or "taxation" in question_lower:
        sid_chunks = collection.get(
            where={
                "$and": [
                    {"fund_name": selected_fund},
                    {"document_type": "SID"}
                ]
            },
            include=["documents", "metadatas"]
        )

        for doc, meta in zip(sid_chunks["documents"], sid_chunks["metadatas"]):
            if "## F.Taxation" in doc:
                results["documents"][0] = [doc]
                results["metadatas"][0] = [meta]
                break

    # For redemption-timeline questions, find the applicable timeline in the SID
    if is_redemption_timeline:
        sid_chunks = collection.get(
            where={
                "$and": [
                    {"fund_name": selected_fund},
                    {"document_type": "SID"}
                ]
            },
            include=["documents", "metadatas"]
        )

        for doc, meta in zip(sid_chunks["documents"], sid_chunks["metadatas"]):
            if (
                "redemption proceeds" in doc.lower()
                and "within three working days" in doc.lower()
                and "valid redemption request" in doc.lower()
            ):
                results["documents"][0] = [doc]
                results["metadatas"][0] = [meta]
                break

    # For investment-objective questions, find the formal objective in the SID
    if is_investment_objective:
        sid_chunks = collection.get(
            where={
                "$and": [
                    {"fund_name": selected_fund},
                    {"document_type": "SID"}
                ]
            },
            include=["documents", "metadatas"]
        )

        for doc, meta in zip(sid_chunks["documents"], sid_chunks["metadatas"]):
            if re.search(
                r"(?:##\s*)?\(?ii\.?\)?\.?\s*Investment Objective",
                doc,
                re.IGNORECASE
            ):
                results["documents"][0] = [doc]
                results["metadatas"][0] = [meta]
                break

    # Add neighboring chunks for better context
    documents = list(results["documents"][0])
    metadatas = list(results["metadatas"][0])
    for metadata in list(metadatas):
        if (
            not is_fund_manager
            or "WHO MANAGES THE SCHEME" not in documents[metadatas.index(metadata)]
        ):
            continue

        next_chunk_id = metadata["chunk_id"] + 1

        neighbor = collection.get(
            where={
                "$and": [
                    {"source_file": metadata["source_file"]},
                    {"chunk_id": next_chunk_id}
                ]
            },
            include=["documents", "metadatas"]
        )

        if neighbor["documents"]:
            documents.append(neighbor["documents"][0])
            metadatas.append(neighbor["metadatas"][0])

    # Normalize malformed minimum-investment text before sending it to the LLM
    if is_minimum_investment:
        normalized_documents = []

        for document in documents:
            document = re.sub(
                r"Rs\.\s*5,000/-\s*LUMPSUM\s*Rs\.\s*1,000/?\s*(?:##\s*)?SUBSEQUENT INVESTMENT",
                "INITIAL LUMPSUM INVESTMENT: Rs. 5,000/-\nSUBSEQUENT INVESTMENT: Rs. 1,000/-",
                document,
                flags=re.IGNORECASE
            )
            normalized_documents.append(document)

        documents = normalized_documents

        # Handle minimum-investment details split across adjacent chunks
        combined_minimum_text = "\n\n".join(documents)

        combined_minimum_text = re.sub(
            r"Rs\.\s*5,000/-\s*LUMPSUM\s*Rs\.\s*1,000/?\s*(?:##\s*)?SUBSEQUENT INVESTMENT",
            "INITIAL LUMPSUM INVESTMENT: Rs. 5,000/-\nSUBSEQUENT INVESTMENT: Rs. 1,000/-",
            combined_minimum_text,
            flags=re.IGNORECASE
        )

        if combined_minimum_text != "\n\n".join(documents):
            documents = [combined_minimum_text]
            metadatas = [metadatas[0]]

    # Combine retrieved chunks
    context_parts = []

    for i in range(len(documents)):

        document = documents[i]
        metadata = metadatas[i]

        context_parts.append(
            f"""
    SOURCE {i + 1}
    Fund: {metadata.get('fund_name')}
    Document Type: {metadata.get('document_type')}
    Source File: {metadata.get('source_file')}

    {document}
    """
        )

    context = "\n\n".join(context_parts)

    print("\n--- RETRIEVED CHUNKS ---")

    for i in range(len(documents)):
        print(f"\nRESULT {i + 1}")
        print("Metadata:", metadatas[i])
        print(documents[i])

    print("\n--- END RETRIEVED CHUNKS ---")

    print("\nGenerating answer locally...\n")

    # Deterministic answer for redemption-timeline questions
    if is_redemption_timeline:
        redemption_text = "\n".join(documents)

        redemption_match = re.search(
            r"within\s+(\w+)\s+working\s+days",
            redemption_text,
            re.IGNORECASE
        )

        if redemption_match:
            answer = (
                f"Redemption proceeds will be dispatched within "
                f"{redemption_match.group(1)} working days from the date of "
                f"receipt of a valid redemption request."
            )

            print("\nRAG ANSWER:")
            print(answer)

            return add_source_citation(answer, metadatas[0])

    # Deterministic answer for Riskometer questions
    if is_riskometer:
        riskometer_text = "\n".join(documents)

        riskometer_match = re.search(
            r"risk of the scheme is\s+((?:low|moderate|moderately high|high|very high)\s+risk)",
            riskometer_text,
            re.IGNORECASE
        )

        # Small Cap factsheet states the risk level directly
        if not riskometer_match and selected_fund == "Quant Small Cap Fund":
            riskometer_match = re.search(
                r"\b(very high risk)\b",
                riskometer_text,
                re.IGNORECASE
            )

        if riskometer_match:
            risk_level = riskometer_match.group(1).title()

            answer = f"Scheme Riskometer: {risk_level}."

            print("\nRAG ANSWER:")
            print(answer)

            return add_source_citation(answer, metadatas[0])    

    # Deterministic answer for exit-load questions
    if is_exit_load:
        exit_load_text = "\n".join(documents)

        exit_load_match = re.search(
            r"(?:within|if\s+exit\s*<=?)\s+(\d+)\s+(day|days|month|months|year|years).*?(\d+(?:\.\d+)?)\s*%",
            exit_load_text,
            re.IGNORECASE | re.DOTALL
        )

        if exit_load_match:
            period_number = exit_load_match.group(1)
            period_unit = exit_load_match.group(2)

            if period_number != "1" and not period_unit.endswith("s"):
                period_unit += "s"

            exit_load_percentage = exit_load_match.group(3)

            answer = (
                f"Exit Load: {exit_load_percentage}% for redemptions/switch outs "
                f"within {period_number} {period_unit} from the date of allotment of units."
            )

            print("\nRAG ANSWER:")
            print(answer)

            return add_source_citation(answer, metadatas[0])

    # Deterministic answer for minimum-investment questions
    if is_minimum_investment:
        minimum_text = "\n".join(documents)

        initial_match = re.search(
            r"INITIAL LUMPSUM INVESTMENT:\s*Rs\.\s*([\d,]+)",
            minimum_text,
            re.IGNORECASE
        )

        subsequent_match = re.search(
            r"SUBSEQUENT INVESTMENT:\s*Rs\.\s*([\d,]+)",
            minimum_text,
            re.IGNORECASE
        )

        if initial_match and subsequent_match:
            answer = (
                f"Initial lump-sum investment: Rs. {initial_match.group(1)}. "
                f"Subsequent investment: Rs. {subsequent_match.group(1)}."
            )

            print("\nRAG ANSWER:")
            print(answer)

            return add_source_citation(answer, metadatas[0])
    prompt = f"""
        You are a mutual fund information assistant.

    Answer the USER QUESTION using ONLY the RETRIEVED CONTEXT.

    Rules:
    - Answer directly and concisely.
    - Keep the answer to a maximum of 3 sentences. Use semicolons to combine multiple factual values when necessary so no required information is omitted.
    - Do not mention the retrieved context, sources, documents, or your reasoning.
    - Do not use outside knowledge.
    - Prefer explicit scheme rules and tables over examples or illustrations.
    - If the answer is not present, say: "I could not find this information in the provided documents."
    - If multiple values directly answer the question, include ALL applicable values.
    - Do not choose one plan, option, category, period, or variant when the question does not specify one.
    - For expense ratio/TER questions, if both Direct Plan and Regular Plan values are present, ALWAYS report both.
    - For exit-load questions, ALWAYS include BOTH the exit-load percentage AND the applicable holding period/time condition. Never answer with only the percentage.
    - For asset-allocation questions, ALWAYS report BOTH the Minimum AND Maximum percentage for EVERY instrument listed in the asset-allocation table. Never report only the Minimum percentage.
    - For SIP questions, report ONLY the SIP amount or SIP amounts stated in the context. Do not report lump-sum or subsequent investment amounts.
    - For minimum-investment questions, if the context contains "INITIAL LUMPSUM INVESTMENT" and "SUBSEQUENT INVESTMENT", those values directly answer the question. ALWAYS report both amounts.

    USER QUESTION:
    {question}

    RETRIEVED CONTEXT:
    {context}
    """

    print(f"📏 PROMPT CHARACTERS: {len(prompt):,}")
    print(f"📏 CONTEXT CHARACTERS: {len(context):,}")

    generation_start = time.time()
    
    client = Groq()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=500
    )

    print(f"⏱ GENERATION TIME: {time.time() - generation_start:.2f} seconds")

    answer = response.choices[0].message.content
    return add_source_citation(answer, metadatas[0])

if __name__ == "__main__":
    question = input("\nAsk a question: ")
    answer = ask_rag(question)

    print("=" * 70)
    print("RAG ANSWER")
    print("=" * 70)
    print(answer)
    print("=" * 70)
