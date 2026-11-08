import json
from pathlib import Path

from wellness.data_gen import ASSET_TYPES, LIABILITY_TYPES
from wellness.profile import build_profile
from wellness.metrics import compute_metrics
from wellness.tools import make_tools
from wellness.levels import level2
from wellness.llm import get_llm

import pandas as pd
import streamlit as st

# Set up the Streamlit app
st.set_page_config(page_title="PNC SmartMoney AI", layout="wide"
                   )

# Title and description
st.title("PNC SmartMoney AI")
st.write("Your personal financial wellness dashboard")

# Load synthetic customer data
data_path = Path(__file__).parent / "customers.json"

with open(data_path, "r", encoding="utf-8") as file:
    customers = json.load(file)

# Presentation-only credentials for customers.
DEMO_CUSTOMER_ID = "CUST-0001"
DEMO_USERNAME = "demo"
DEMO_PASSWORD = "abc"

DEMO_CUSTOMER_ID_Two = "CUST-0002"
DEMO_USERNAME_Two = "demo2"
DEMO_PASSWORD_Two = "abc"

DEMO_CUSTOMER_ID_Three = "CUST-0003"
DEMO_USERNAME_Three = "demo3"
DEMO_PASSWORD_Three = "abc"

DEMO_CUSTOMER_ID_FOUR = "CUST-0004"
DEMO_USERNAME_FOUR = "demo4"
DEMO_PASSWORD_FOUR = "abc"

DEMO_CUSTOMER_ID_FIVE = "CUST-0005"
DEMO_USERNAME_FIVE = "demo5"
DEMO_PASSWORD_FIVE = "abc"

# Check if the user is signed in
if st.session_state.get("signed_in_customer") not in [
    DEMO_CUSTOMER_ID,
    DEMO_CUSTOMER_ID_Two,
    DEMO_CUSTOMER_ID_Three,
    DEMO_CUSTOMER_ID_FOUR,
    DEMO_CUSTOMER_ID_FIVE
]:
    _, login_column, _ = st.columns([1, 2, 1])
    with login_column:
        with st.container(border=True):
            st.subheader("Welcome to SmartMoney")
            st.write("Sign in to see your financial picture and explore your next move.")
            with st.form("customer_login", clear_on_submit=True):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                sign_in = st.form_submit_button("Sign in", use_container_width=True)
            if sign_in:
                if username.strip() == DEMO_USERNAME and password == DEMO_PASSWORD:
                    st.session_state["signed_in_customer"] = DEMO_CUSTOMER_ID
                    st.rerun()
                elif username.strip() == DEMO_USERNAME_Two and password == DEMO_PASSWORD_Two:
                    st.session_state["signed_in_customer"] = DEMO_CUSTOMER_ID_Two
                    st.rerun()
                elif username.strip() == DEMO_USERNAME_Three and password == DEMO_PASSWORD_Three:
                    st.session_state["signed_in_customer"] = DEMO_CUSTOMER_ID_Three
                    st.rerun()
                elif username.strip() == DEMO_USERNAME_FOUR and password == DEMO_PASSWORD_FOUR:
                    st.session_state["signed_in_customer"] = DEMO_CUSTOMER_ID_FOUR
                    st.rerun()
                elif username.strip() == DEMO_USERNAME_FIVE and password == DEMO_PASSWORD_FIVE:
                    st.session_state["signed_in_customer"] = DEMO_CUSTOMER_ID_FIVE
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Please try again.")
            st.caption("HackUTD demo • Synthetic financial data only")
    st.stop()

selected_id = st.session_state.get("signed_in_customer")
customer = next((c for c in customers if c["customer_id"] == selected_id), None)
if customer is None:
    st.error("The demo customer is missing from customers.json.")
    st.stop()

with st.sidebar:
    st.subheader("Your SmartMoney")
    st.write("Signed in to your demo account")
    st.caption("Synthetic customer • " + selected_id)
    if st.button("Log out", use_container_width=True):
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()

st.subheader("Welcome back!")

accounts = customer["accounts"]

# Consolidate current balances across every account of each type.
def total_balance(account_types):
    return sum(account["balance"] for account in accounts
               if account["type"] in account_types)


checking = total_balance({"checking"})
savings = total_balance({"savings"})
credit_cards = total_balance({"credit_card"})
investments = total_balance({"cd", "ira", "brokerage"})
loans = total_balance({"auto_loan", "personal_loan", "mortgage"})
total_assets = total_balance(ASSET_TYPES)
total_debt = total_balance(LIABILITY_TYPES)

st.subheader("Financial Overview")
st.caption("Your monthly income and combined current account balances, all in one place.")

assets_column, debt_summary_column = st.columns(2)
assets_column.metric("Total Assets · What you have", f"${total_assets:,.2f}",
                     help="Checking + savings + CDs + IRAs + brokerage balances. Matches the Assets chart; unrecorded property is excluded.")
debt_summary_column.metric("Total Debt · What you owe", f"${total_debt:,.2f}",
                           help="Credit card balances + outstanding loans, including mortgages. Matches the Debt chart.")
st.caption("Total assets = checking + savings + investments. Total debt = credit cards + loans.")

col1, col2, col3 = st.columns(3)
col1.metric("Monthly Income", f"${customer['monthly_income']:,.2f}")
col2.metric("Checking Balance", f"${checking:,.2f}",
            help="Combined balance of all checking accounts.")
col3.metric("Savings Balance", f"${savings:,.2f}",
            help="Combined balance of all savings accounts. CDs appear under Investments.")

col4, col5, col6 = st.columns(3)
col4.metric("Credit Card Balance", f"${credit_cards:,.2f}",
            help="Total currently owed across all credit cards, not historical charges.")
col5.metric("Investments", f"${investments:,.2f}",
            help="Current combined balance of CDs, IRAs, and brokerage accounts, not lifetime contributions.")
col6.metric("Loans · Excludes credit cards", f"${loans:,.2f}",
            help="Total balance owed on auto loans, personal loans, and mortgages. Credit cards are shown separately.")

# Display account details
st.subheader("Financial Accounts")

st.caption("Current balances grouped by account type. Each chart has its own total.")

account_type_labels = {
    "checking": "Checking", "savings": "Savings", "cd": "CDs",
    "ira": "IRAs", "brokerage": "Brokerage", "credit_card": "Credit cards",
    "auto_loan": "Auto loans", "personal_loan": "Personal loans", "mortgage": "Mortgages",
}

asset_column, debt_column = st.columns(2)
for column, title, account_types in (
    (asset_column, "Assets · What you have", ASSET_TYPES),
    (debt_column, "Debt · What you owe", LIABILITY_TYPES),
):
    with column:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            balances = {}
            for account in accounts:
                if account["type"] in account_types:
                    label = account_type_labels.get(account["type"], account["type"])
                    balances[label] = balances.get(label, 0) + account["balance"]
            total = sum(balances.values())
            st.caption(f"Total: ${total:,.2f}")
            if any(balance < 0 for balance in balances.values()):
                st.info("This chart cannot show negative balances. See the account table below.")
            elif total > 0:
                chart_data = pd.DataFrame([
                    {"Account Type": label, "Balance": balance, "Share": balance / total}
                    for label, balance in balances.items() if balance > 0
                ])
                st.vega_lite_chart(chart_data, {
                    # Preserve plot height when fitting the card width so the
                    # legend cannot shrink the plot beneath a fixed radius.
                    "autosize": {"type": "fit-x", "contains": "padding"},
                    "padding": {"top": 16, "right": 16, "bottom": 16, "left": 16},
                    "mark": {
                        "type": "arc",
                        "innerRadius": {"expr": "max(0, min(width, height) * 0.28)"},
                        "outerRadius": {"expr": "max(0, min(width, height) / 2 - 12)"},
                    },
                    "encoding": {
                        "theta": {"field": "Balance", "type": "quantitative", "stack": True},
                        "color": {"field": "Account Type", "type": "nominal",
                                  "scale": {"scheme": "tableau10"},
                                  "legend": {"orient": "bottom", "title": None,
                                             "columns": 2, "labelLimit": 180,
                                             "rowPadding": 8, "offset": 16}},
                        "tooltip": [
                            {"field": "Account Type", "type": "nominal"},
                            {"field": "Balance", "type": "quantitative", "format": "$,.2f"},
                            {"field": "Share", "type": "quantitative", "format": ".1%"},
                        ],
                    },
                    "height": 260,
                    "view": {"stroke": None},
                }, use_container_width=True)
            else:
                st.info("No balances to display in this category.")


df = pd.DataFrame(accounts)
df = df.drop(columns=["id"])

# Account details from customer.json
df = df.rename(columns={
    "name": "Account Name",
    "type": "Account Type",
    "balance": "Balance",
    "apy": "APY",
    "credit_limit": "Credit Limit",
    "apr": "APR",
    "min_payment": "Minimum Payment",
    "monthly_payment": "Monthly Payment"
})

# Display the account details in a table
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# Convert the customer data into a unified financial profile
profile = build_profile(customer)

# Calculate financial health indicators
metrics = compute_metrics(profile)

metric_values = metrics["values"]
metric_status = metrics["status"]

# Financial Health Analysis
st.subheader("Financial Health Analysis")

st.write(
    "Understand your financial situation using "
    "PNC's financial health indicators."
)

# Friendly names for the eight metrics
metric_labels = {
    "emergency_fund_months": "Emergency Fund Coverage",
    "debt_to_income": "Debt-to-Income Ratio",
    "savings_rate": "Savings Rate",
    "credit_utilization": "Credit Utilization",
    "net_worth": "Net Worth",
    "debt_to_asset": "Debt-to-Asset Ratio",
    "housing_cost_ratio": "Housing Cost Ratio",
    "liquidity_ratio": "Liquidity Ratio",
}


# Format the values appropriately
def format_metric(key, value):
    if value is None:
        return "N/A"
    if key == "net_worth":
        return f"${value:,.2f}"
    if key == "emergency_fund_months":
        return f"{value:.2f} months"
    return f"{value * 100:.2f}%"


# Display the eight metrics in two columns
for i, (key, label) in enumerate(metric_labels.items()):
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])

        value = metric_values[key]
        status = metric_status[key]

        with col1:
            st.metric(
                label=label,
                value=format_metric(key, value)
            )

        with col2:
            if status == "good":
                st.success("Good")
            elif status == "watch":
                st.warning("Watch")
            elif status == "risk":
                st.error("Risk")
            else:
                st.info("Unknown")

# Feature 3: What if car purchase simulation
st.subheader("Car Purchase Simulation")

st.write(
    "Explore how purchasing a car could affect "
    "your monthly payments and financial health."
)

# Get the approved PNC tools for the selected customer
tools = make_tools(customer)

# Retrieve the car purchase simulation tool
simulate_car_purchase = tools["simulate_car_purchase"][0]

col1, col2 = st.columns(2)

with col1:
    car_price = st.number_input(
        "Car Price ($)",
        min_value=0.0,
        value=30000.0,
        step=1000.0
    )

    down_payment = st.number_input(
        "Down Payment ($)",
        min_value=0.0,
        max_value=car_price,
        value=min(5000.0, car_price),
        step=500.0
    )

    with col2:
        interest_rate = st.number_input(
            "Annual Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=7.0,
            step=0.5
        )

        loan_term = st.selectbox(
            "Loan Term (Months)",
            [36, 48, 60, 72],
            index=2
        )

    # Run the car purchase simulation when the button is clicked
    if st.button("Calculate Car Purchase"):
        result = simulate_car_purchase(
            price=car_price,
            down=down_payment,
            apr=interest_rate / 100,
            term_months=loan_term
        )

        # Save results so Streamlit can remember them
        st.session_state["car_result"] = result
        st.session_state["car_customer_id"] = selected_id

    # Display results only when they exist
    if ("car_result" in st.session_state
            and st.session_state.get("car_customer_id") == selected_id):

        result = st.session_state["car_result"]

        st.subheader("Simulation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Loan Amount",
            f"${result['loan_amount']:,.2f}"
        )

        col2.metric(
            "Monthly Payment",
            f"${result['monthly_payment']:,.2f}"
        )

        col3.metric(
            "Total Interest",
            f"${result['total_interest']:,.2f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Cash After Down Payment",
            f"${result['cash_after_down_payment']:,.2f}"
        )

        col5.metric(
            "Emergency Fund After Purchase",
            f"{result['emergency_fund_months_after']:.2f} months"
        )

        col6.metric(
            "New Monthly Net Cash Flow",
            f"${result['new_monthly_net_cash_flow']:,.2f}"
        )

        if result["cash_goes_negative_within_12m"]:
            st.error(
                "Your projected cash balance may become "
                "negative within the next 12 months."
            )
        else:
            st.success(
                "Your projected cash balance does not become "
                "negative within the next 12 months."
            )

        if result["emergency_fund_months_after"] < 3:
            st.warning(
                "Your emergency fund would cover fewer than "
                "3 months of expenses after this purchase."
            )

        st.caption(
            "Estimates based on synthetic customer data. "
            "Results are not financial advice."
        )


# Feature 4: AI-driven financial coach

st.subheader("SmartMoney AI Financial Coach")

st.write(
    "Ask questions about your financial health, "
    "accounts, or potential financial decisions."
)

# Initialize the LLM for the financial coach
@st.cache_resource
def load_ai_model():
    return get_llm("ollama:llama3.2")

# Load the AI model
llm = load_ai_model()


# User question form
with st.form("financial_coach_form"):

    question = st.text_input(
        "Ask SmartMoney AI",
        placeholder="Can I afford a $30,000 car with $5,000 down?"
    )

    submitted = st.form_submit_button("Ask AI")

# Process the question
if submitted:
    if question.strip():

        with st.spinner("SmartMoney AI is analyzing your finances..."):
            try:
                answer = level2(llm, customer, question)

                # Save the AI response
                st.session_state["coach_response"] = answer
                st.session_state["coach_customer_id"] = selected_id

            except Exception as error:
                st.session_state.pop("coach_response", None)
                st.error(f"Unable to generate a response: {error}")
    else:
        st.warning("Please enter a question.")

# Display the AI response
if (
    "coach_response" in st.session_state
    and st.session_state.get("coach_customer_id") == selected_id
):
    st.subheader("AI Financial Coach Response")
    st.write(st.session_state["coach_response"])

st.caption(
    "AI-generated explanations based on synthetic financial data. "
    "Not financial advice."
)
