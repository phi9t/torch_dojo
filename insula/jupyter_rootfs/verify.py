#!/usr/bin/env python3
"""Exercise the authenticated Jupyter API and a real CUDA kernel."""
import json
import pathlib
import sys
import urllib.request
import uuid

import websocket

base = 'http://127.0.0.1:8899'
token = pathlib.Path('/workspace/torch_dojo/target/insula/shared/token').read_text().strip()
headers = {'Authorization': 'token ' + token, 'Content-Type': 'application/json'}

def api(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read()
        return json.loads(raw) if raw else None

kernel = api('POST', '/api/kernels', {'name': 'torch-dojo-gpu', 'path': ''})
try:
    kid = kernel['id']
    session = str(uuid.uuid4())
    ws = websocket.create_connection(f'ws://127.0.0.1:8899/api/kernels/{kid}/channels?session_id={session}&token={token}', timeout=90, origin=base)
    try:
        msgid = uuid.uuid4().hex
        code = '''import json, os, pathlib, sys, torch
p=pathlib.Path('/workspace/torch_dojo/target/insula/shared/kernel-write.txt')
a=torch.ones(4, device='cuda'); torch.cuda.synchronize()
p.write_text('visible from host\\n')
print(json.dumps({'executable':sys.executable,'cwd':os.getcwd(),'cuda':a.sum().item(),'home':os.environ['HOME']}))'''
        ws.send(json.dumps({'header': {'msg_id': msgid, 'username':'verify', 'session':session, 'msg_type':'execute_request', 'version':'5.3'}, 'parent_header':{}, 'metadata':{}, 'content':{'code':code,'silent':False,'store_history':False,'user_expressions':{},'allow_stdin':False,'stop_on_error':True},'channel':'shell'}))
        output = []
        while True:
            msg = json.loads(ws.recv())
            if msg.get('parent_header', {}).get('msg_id') != msgid:
                continue
            if msg['msg_type'] == 'stream': output.append(msg['content']['text'])
            if msg['msg_type'] == 'error': raise RuntimeError(msg['content'])
            if msg['msg_type'] == 'execute_reply':
                assert msg['content']['status'] == 'ok', msg
                break
        evidence = json.loads(''.join(output).strip())
        assert evidence['executable'].startswith('/usr/'), evidence
        assert evidence['cwd'] == '/workspace/torch_dojo', evidence
        assert evidence['home'] == '/home/dojo', evidence
        assert evidence['cuda'] == 4.0, evidence
        print(json.dumps(evidence, indent=2))
    finally:
        ws.close()
finally:
    api('DELETE', '/api/kernels/' + kernel['id'])
