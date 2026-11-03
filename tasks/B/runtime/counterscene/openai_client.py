"""Responses adapter with exact-model binding, token counting and a spend guard.

Credentials are accepted in memory/environment only. No hosted tools are enabled.
The ledger uses conservative Standard-tier input/cache-write rates, not invoices.
"""
import copy
from contextlib import contextmanager
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid
from .session import write_json


class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    def __init__(self,path,limit=50,*,authorized_ceiling=50):
        self.path=Path(path);self.limit=Decimal(str(limit))
        ceiling=Decimal(str(authorized_ceiling))
        if not self.limit.is_finite() or not ceiling.is_finite() or self.limit<=0 or self.limit>ceiling:raise ValueError('Ledger limit exceeds explicit authorization')
        self.path.parent.mkdir(parents=True,exist_ok=True)

    @contextmanager
    def locked(self):
        lock=self.path.with_suffix('.lock')
        for _ in range(100):
            try:
                descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(descriptor);break
            except FileExistsError:time.sleep(.1)
        else:raise RuntimeError('Spend ledger locked; do not run overlapping remote jobs')
        try:
            data=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else dict(limit_usd=float(self.limit),requests=[],
                pricing_source='https://developers.openai.com/api/docs/models/gpt-6.1-sol',
                pricing_date='2026-10-30',currency='USD',method='Conservative estimate: all input at $2.50/MTok; output at $10/MTok. Not an invoice.')
            if Decimal(str(data['limit_usd']))!=self.limit:raise ValueError('Ledger limit differs from authorization')
            yield data
            write_json(self.path,data)
        finally:lock.unlink(missing_ok=True)

    def reserve(self,input_tokens,max_output_tokens):
        if input_tokens>272000:raise BudgetExceeded('Long-context pricing is outside this run policy')
        bound=(Decimal(input_tokens)*Decimal('2.50')+Decimal(max_output_tokens)*10)/1000000
        with self.locked() as data:
            spent=sum(Decimal(str(r['charge_bound_usd'])) for r in data['requests'])
            if spent+bound>self.limit:raise BudgetExceeded('Authorized spend limit reached; request was not sent')
            rid=uuid.uuid4().hex
            data['requests'].append(dict(reservation_id=rid,status='reserved',input_token_count=input_tokens,
                max_output_tokens=max_output_tokens,charge_bound_usd=float(bound)))
        return rid

    def settle(self,rid,response=None,rejected=False):
        with self.locked() as data:
            row=next(r for r in data['requests'] if r['reservation_id']==rid)
            if row['status']!='reserved':raise RuntimeError('Request already settled')
            if rejected:row.update(status='rejected_before_generation',charge_bound_usd=0)
            elif response is None:row['status']='uncertain_network_charge_reserved'
            else:
                usage=response.get('usage')
                if not usage:row['status']='missing_usage_charge_reserved'
                else:
                    amount=(Decimal(usage['input_tokens'])*Decimal('2.50')+Decimal(usage['output_tokens'])*10)/1000000
                    row.update(status='completed',response_id=response['id'],model=response['model'],usage=usage,
                               charge_bound_usd=float(amount))
            data['conservative_spend_usd']=float(sum(Decimal(str(r['charge_bound_usd'])) for r in data['requests']))


def strict_schema(schema):
    value=copy.deepcopy(schema)
    if isinstance(value,dict):
        if 'const' in value:value['enum']=[value.pop('const')]
        if value.get('type')=='object':
            required=set(value.get('required',[]));properties=value.get('properties',{})
            for name,prop in list(properties.items()):
                normalized=strict_schema(prop)
                properties[name]=normalized if name in required else {'anyOf':[normalized,{'type':'null'}]}
            value['required']=list(properties);value['additionalProperties']=False
        for key in ('anyOf','oneOf'):
            if key in value:value[key]=[strict_schema(s) for s in value[key]]
        if 'items' in value:value['items']=strict_schema(value['items'])
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


class OpenAIResponses:
    def __init__(self,ledger,api_key=None,model='gpt-6.1-sol',reasoning='medium',timeout=180,*,purpose='participant'):
        if (purpose,model) not in (('participant','gpt-6.1-sol'),('auxiliary_judge','gpt-6-astra')):raise ValueError('Exact authorized model and participant/judge purpose required')
        if model=='gpt-6-astra' and (getattr(ledger,'model',None)!=model or getattr(ledger,'input_rate',0)<25 or getattr(ledger,'output_rate',0)<50):raise ValueError('Astra judge requires its verified campaign rate binding')
        self._key=api_key or os.environ.get('OPENAI_API_KEY')
        if not self._key:raise ValueError('OPENAI_API_KEY is not set')
        self.ledger=ledger;self.model=model;self.reasoning=reasoning;self.timeout=timeout
        self.base='https://api.openai.com/v1';self.num_ctx=16384;self._info=None
        self.backend='openai_responses';self.evidence_type='frontier_vlm';self.is_remote=True

    def api(self,path,data=None):
        request=urllib.request.Request(self.base+path,data=None if data is None else json.dumps(data).encode(),
                    headers={'Content-Type':'application/json','Authorization':'Bearer '+self._key})
        try:
            with urllib.request.build_opener(NoRedirect()).open(request,timeout=self.timeout) as r:return json.load(r)
        except urllib.error.HTTPError as error:
            # Never serialize request headers or an unredacted provider error.
            details=error.read().decode(errors='replace').replace(self._key,'[REDACTED_API_KEY]')
            details=re.sub(r'sk-[A-Za-z0-9_-]{16,}','[REDACTED_API_KEY]',details)
            clean=RuntimeError(f'OpenAI HTTP {error.code}: {details[:800]}');clean.http_status=error.code;raise clean from None

    def info(self):
        if self._info is None:
            metadata=self.api('/models/'+self.model)
            if metadata['id']!=self.model:raise ValueError('Requested model ID mismatch')
            self._info=dict(capabilities=['completion','vision'],provider='OpenAI',metadata=metadata,
                docs='https://developers.openai.com/api/docs/models/'+self.model)
        return self._info

    def request(self,path,data=None):
        if path=='/api/version':return dict(version='OpenAI Responses API',model=self.model,reasoning_effort=self.reasoning)
        raise ValueError('Unsupported local-transport compatibility operation')

    def generate(self,messages,seed=0,temperature=.3,max_tokens=2048,schema=None):
        started=time.monotonic();wrapped=bool(schema and 'anyOf' in schema)
        normalized=strict_schema(schema) if schema else None
        if wrapped:normalized={'type':'object','properties':{'action':normalized},'required':['action'],'additionalProperties':False}
        inputs=[]
        for message in messages:
            if message['role']=='user':
                content=[{'type':'input_text','text':message['content']}]
                content.extend({'type':'input_image','image_url':'data:image/png;base64,'+encoded,'detail':'high'} for encoded in message.get('images',[]))
                inputs.append(dict(role='user',content=content))
            else:
                content=message['content']
                if wrapped and message['role']=='assistant':
                    try:content=json.dumps({'action':json.loads(content)})
                    except ValueError:pass
                inputs.append(dict(role=message['role'],content=content))
        if wrapped:inputs.insert(0,dict(role='system',content='Output formatting only: wrap the selected tool action in {"action": ...}, matching the provided schema. Task rules and tool meanings are unchanged.'))
        text={'format':{'type':'json_schema','name':'benchmark_output','strict':True,'schema':normalized}} if normalized else {'format':{'type':'json_object'}}
        counted=self.api('/responses/input_tokens',dict(model=self.model,input=inputs,text=text))
        input_count=counted['input_tokens'];rid=self.ledger.reserve(input_count,max_tokens)
        body=dict(model=self.model,input=inputs,text=text,max_output_tokens=max_tokens,store=False,
                  reasoning={'effort':self.reasoning},service_tier='default')
        try:response=self.api('/responses',body)
        except Exception as error:
            self.ledger.settle(rid,rejected=getattr(error,'http_status',0) in (400,401,403,404,422,429))
            raise
        # Durable response journal permits usage reconciliation if ledger I/O
        # fails after a paid generation. It never contains HTTP auth headers.
        write_json(self.ledger.path.parent/'response_journal'/f'{rid}.json',response)
        self.ledger.settle(rid,response)
        if response['model']!=self.model and not response['model'].startswith(self.model+'-'):raise ValueError('Provider returned a different model')
        content=''.join(block.get('text','') for item in response.get('output',[]) if item['type']=='message' for block in item.get('content',[]) if block['type']=='output_text')
        if wrapped and content:
            parsed=json.loads(content)['action'];content=json.dumps({k:v for k,v in parsed.items() if v is not None})
        usage=response['usage'];upper=(usage['input_tokens']*float(getattr(self.ledger,'input_rate',2.5))+usage['output_tokens']*float(getattr(self.ledger,'output_rate',10)))/1e6
        result=dict(model=response['model'],created_at=response.get('created_at'),message={'role':'assistant','content':content},
            prompt_eval_count=usage['input_tokens'],eval_count=usage['output_tokens'],api_cost_usd=upper,
            cost_kind='conservative Standard-tier upper estimate, not invoice',reservation_id=rid,
            provider_response=response,provider_request=body,input_token_count=counted,
            requested_seed=seed,seed_supported=False,temperature_supported=False,reasoning_effort=self.reasoning,
            output_envelope='action wrapper normalized' if wrapped else 'direct JSON')
        return result,round((time.monotonic()-started)*1000,2)
