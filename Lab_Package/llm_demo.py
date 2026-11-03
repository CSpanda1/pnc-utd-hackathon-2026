"""Optional commercial or local LLM narration; calculations remain in financial_tools."""
import json, os
from financial_tools import calculate_indicators,simulate_purchase,get_history_evidence

def build_evidence(customer_id='C001'):
    return {'indicators':calculate_indicators(customer_id),'scenario':simulate_purchase(customer_id,30000,5000),
            'transaction_history':get_history_evidence(customer_id)}

def narrate(evidence,provider='none'):
    instructions=('You are an educational financial wellness assistant. Use ONLY the provided synthetic tool results; '
                  'do not invent facts, guarantees, credit scores, or eligibility. Distinguish observations from assumptions. '
                  'Explain three useful considerations in concise plain English. Do not give personalized investment advice. '
                  'Clearly state that data is synthetic and balances are partial. '
                  'Distinguish snapshot assumptions from historical transaction activity. '
                  'Never count internal transfers or debt payments as additional spending; '
                  'do not present calibrated external transfers as earned income.')
    prompt=instructions+'\nEVIDENCE:\n'+json.dumps(evidence)
    if provider=='huggingface':
        from local_hf import generate_local
        return generate_local(prompt)
    if provider=='openai':
        from openai import OpenAI
        if not os.getenv('OPENAI_API_KEY'): raise RuntimeError('Set OPENAI_API_KEY first')
        response=OpenAI().responses.create(model=os.getenv('OPENAI_MODEL','gpt-5.4'),input=prompt)
        return response.output_text
    if provider=='ollama':
        import requests
        url=os.getenv('OLLAMA_URL','http://localhost:11434/api/chat')
        r=requests.post(url,json={'model':os.getenv('OLLAMA_MODEL','qwen3:4b'), 'messages':[{'role':'user','content':prompt}],'stream':False},timeout=180)
        r.raise_for_status()
        return r.json()['message']['content']
    if provider=='foundry':
        from openai import OpenAI
        base_url=os.getenv('FOUNDRY_BASE_URL','http://127.0.0.1:51333/v1')
        model=os.getenv('FOUNDRY_MODEL','qwen2.5-0.5b')
        client=OpenAI(base_url=base_url, api_key=os.getenv('FOUNDRY_API_KEY','foundry-local'))
        try:
            response=client.responses.create(model=model,input=prompt)
            return response.output_text
        except Exception as exc:
            # Some Foundry Local builds expose /v1/chat/completions but not /v1/responses.
            if getattr(exc,'status_code',None)!=404:
                raise
            completion=client.chat.completions.create(
                model=model,
                messages=[{'role':'user','content':prompt}],
                temperature=.2,
                max_tokens=600,
            )
            return completion.choices[0].message.content
    if provider != 'none':
        raise ValueError(f'Unknown narration provider: {provider}')
    return 'Offline mode: inspect the validated evidence below. Configure an LLM provider for a narrative.\n'+json.dumps(evidence,indent=2)

if __name__=='__main__':
    print(narrate(build_evidence(),os.getenv('LLM_PROVIDER','none')))
