import streamlit as st
from search_rag_groq import ask_rag

st.set_page_config(
    page_title="Mutual Fund Facts Assistant",
    page_icon="📊",
    layout="wide"
)

# Keep the whole interface compact
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }

    div.stButton > button {
        min-height: 55px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title("📊 Mutual Fund Facts Assistant")

st.write(
    "Get factual information about selected Quant Mutual Fund schemes "
    "using official public sources."
)

st.info("Facts-only. No investment advice.")

# Supported funds
st.markdown("#### Supported Quant Mutual Funds")

fund1, fund2, fund3, fund4, fund5 = st.columns(5)

fund1.markdown("**Small Cap Fund**")
fund2.markdown("**Large Cap Fund**")
fund3.markdown("**Flexi Cap Fund**")
fund4.markdown("**Multi Cap Fund**")
fund5.markdown("**PSU Fund**")

st.caption(
    "This prototype currently supports only the five Quant Mutual Fund schemes listed above."
)

# Quick questions
st.markdown("#### Quick Questions")

q1, q2, q3, q4, q5 = st.columns(5)

with q1:
    small_cap = st.button(
        "💰 Small Cap\nMinimum SIP",
        use_container_width=True
    )

with q2:
    large_cap = st.button(
        "👤 Large Cap\nFund Managers",
        use_container_width=True
    )

with q3:
    flexi_cap = st.button(
        "📈 Flexi Cap\nBenchmark",
        use_container_width=True
    )

with q4:
    multi_cap = st.button(
        "⚠️ Multi Cap\nRiskometer",
        use_container_width=True
    )

with q5:
    psu_fund = st.button(
        "📤 PSU Fund\nExit Load",
        use_container_width=True
    )

# Fill question box when quick question is selected
if small_cap:
    st.session_state["question"] = (
        "What is the minimum SIP amount for Quant Small Cap Fund?"
    )

if large_cap:
    st.session_state["question"] = (
        "Who manages Quant Large Cap Fund?"
    )

if flexi_cap:
    st.session_state["question"] = (
        "What is the benchmark of Quant Flexi Cap Fund?"
    )

if multi_cap:
    st.session_state["question"] = (
        "What is the Riskometer level of Quant Multi Cap Fund?"
    )

if psu_fund:
    st.session_state["question"] = (
        "What is the exit load for Quant PSU Fund?"
    )

# User question
question = st.text_input(
    "Ask a mutual fund question",
    placeholder="Type your question here...",
    key="question"
)

ask_button = st.button(
    "Ask",
    type="primary",
    use_container_width=True
)

if ask_button:
    if question.strip():
        with st.spinner("Searching official sources..."):
            answer = ask_rag(question)

        st.markdown("### Answer")

        if "Source:" in answer:
            main_answer, source_section = answer.split("Source:", 1)

            if "Last updated from sources:" in source_section:
                source, updated = source_section.split(
                    "Last updated from sources:", 1
                )

                st.write(main_answer.strip())

                st.markdown("**Source:**")
                st.markdown(source.strip())

                st.markdown("**Last updated from sources:**")
                st.write(updated.strip())

            else:
                st.write(main_answer.strip())
                st.markdown("**Source:**")
                st.markdown(source_section.strip())

        else:
            st.write(answer)

    else:
        st.warning("Please enter a question.")
