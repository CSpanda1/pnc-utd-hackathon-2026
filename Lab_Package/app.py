import streamlit as st
import pandas as pd
from datetime import date
from financial_tools import (calculate_indicators,simulate_purchase,get_accounts,get_transactions,
                             get_account_history,summarize_transaction_history,get_history_evidence)
from llm_demo import narrate
st.set_page_config(page_title='Synthetic Financial Wellness Lab',layout='wide')
st.title('Financial Wellness AI — Hackathon Starter')
st.caption('Synthetic accounts only. Educational prototype; not banking, credit, investment, or lending advice.')
cid=st.selectbox('Synthetic customer', [f'C{i:03d}' for i in range(1,31)])
ind=calculate_indicators(cid)
a,b,c=st.columns(3)
a.metric('Cash surplus / month',f"${ind['monthly_cash_surplus']:,.0f}")
b.metric('Emergency fund (months)',f"{ind['emergency_fund_months']:.1f}")
c.metric('Debt payment / income',f"{ind['debt_payment_to_income_pct']:.1f}%")
st.subheader('Health indicators and grounded metrics');st.json(ind)
st.caption('These indicators use the current snapshot and profile assumptions. The generated history below is a separate activity view.')
history=summarize_transaction_history(cid)
st.subheader('24-month account and transaction history')
st.caption(f"{history['history_start']} through {history['as_of_date']}. " + history['definitions'])
activity,balances,ledger=st.tabs(['Monthly activity','Account balances','Transactions'])
with activity:
    monthly=pd.DataFrame(history['months']).set_index('month')
    st.line_chart(monthly[['paycheck_income','spending','debt_payments']])
    st.dataframe(monthly,width='stretch')
with balances:
    statements=pd.DataFrame(get_account_history(cid))
    st.line_chart(statements.pivot(index='month',columns='account_type',values='closing_balance'))
    st.caption('Asset and liability balances are both positive. Interest accrual increases an asset or liability; a payment decreases a liability.')
with ledger:
    owned=get_accounts(cid)
    choices={'All accounts':None, **{f"{a['account_type']} — {a['account_id']}":a['account_id'] for a in owned}}
    label=st.selectbox('History account',list(choices))
    start,end=st.columns(2)
    start_day=start.date_input('From date',date.fromisoformat(history['history_start']),key='history_from')
    end_day=end.date_input('Through date',date.fromisoformat(history['as_of_date']),key='history_to')
    page=st.number_input('Transaction page (100 entries per page)',min_value=1,value=1,step=1)
    if start_day>end_day:
        st.warning('From date must not follow through date.')
    else:
        result=get_transactions(cid,choices[label],start_day.isoformat(),end_day.isoformat(),limit=100,offset=(page-1)*100)
        st.caption(f"{result['total']} matching ledger entries. Internal transfers have two linked entries.")
        st.dataframe(pd.DataFrame(result['items']),width='stretch')
st.subheader('What-if: vehicle purchase')
a,b,c=st.columns(3)
price=a.number_input('Price ($)',0,200000,30000,step=1000)
down=b.number_input('Down payment ($)',0,int(price),min(5000,int(price)),step=500)
apr=c.number_input('APR (%)',0.0,40.0,8.0,step=.5)
term=st.slider('Loan term (months)',12,84,60,12)
scenario=simulate_purchase(cid,price,down,apr,term)
st.json(scenario)
provider=st.selectbox('Narration provider',['none','huggingface','openai','ollama'])
if provider=='huggingface':
    import os
    from local_hf import DEFAULT_MODEL
    st.caption(f"Local cached model: {os.getenv('HF_MODEL', DEFAULT_MODEL)}. First request loads the weights.")
if st.button('Explain my synthetic financial picture'):
    try:st.write(narrate({'indicators':ind,'scenario':scenario,'transaction_history':get_history_evidence(cid)},provider))
    except Exception as e:st.error(f'Provider unavailable: {type(e).__name__}: {e}')
