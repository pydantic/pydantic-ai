"""N=10 per model. V1 fabricated search_tools | V2 refs on real load_capability | V3 fabricated load_capability."""
import yaml, copy, anthropic
c = anthropic.Anthropic()
d = yaml.safe_load(open('/tmp/fable.yaml'))
body = d['interactions'][2]['request']['parsed_body']
TOOLS = copy.deepcopy(body['tools'])
BASE = copy.deepcopy(body['messages'][:5])
REF = {'type': 'tool_reference', 'tool_name': 'lookup_refund_policy'}

V1 = copy.deepcopy(BASE)

V2 = copy.deepcopy(BASE[:3])
V2[2]['content'][0]['content'] = [REF]

# V3: keep the real load_capability call+result intact (instructions preserved),
# then a fabricated SECOND load_capability call whose result carries the refs.
V3 = copy.deepcopy(BASE[:3])
V3.append({'role': 'assistant', 'content': [
    {'type': 'tool_use', 'id': 'auto_reveal_1', 'name': 'load_capability', 'input': {'id': 'refunds'}}]})
V3.append({'role': 'user', 'content': [
    {'type': 'tool_result', 'tool_use_id': 'auto_reveal_1', 'content': [REF], 'is_error': False}]})

N = 10
print(f'{"model":<20} {"V1 search_tools":<18} {"V2 real anchor":<18} V3 load_capability')
for model in ('claude-fable-5','claude-opus-5','claude-sonnet-5','claude-haiku-4-5','claude-sonnet-4-6'):
    out = []
    for msgs in (V1, V2, V3):
        hits = 0
        for _ in range(N):
            try:
                r = c.messages.create(model=model, max_tokens=80, tools=TOOLS, messages=msgs)
                if any(b.type=='tool_use' and b.name=='lookup_refund_policy' for b in r.content): hits += 1
            except Exception as e:
                out.append('ERR ' + str(e).splitlines()[0][55:110]); break
        else: out.append(f'{hits}/{N}')
    print(f'{model:<20} {out[0]:<18} {out[1]:<18} {out[2]}')
