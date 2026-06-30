"""测试 ComfyUI 工作流"""
import urllib.request, json, uuid, time, sys

COMFY = 'http://127.0.0.1:8188'

def submit_and_wait(workflow_path, param_overrides=None, timeout=180):
    """提交工作流并等待完成"""
    with open(workflow_path, 'r') as f:
        wf = json.load(f)

    if param_overrides:
        for node_id, overrides in param_overrides.items():
            for k, v in overrides.items():
                if node_id in wf:
                    wf[node_id]['inputs'][k] = v

    cid = str(uuid.uuid4())
    data = json.dumps({'prompt': wf, 'client_id': cid}).encode('utf-8')
    req = urllib.request.Request(f'{COMFY}/prompt', data=data,
                                 headers={'Content-Type': 'application/json'})
    resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
    pid = resp['prompt_id']

    wf_name = workflow_path.split('/')[-1]
    print(f'  [{wf_name}] submitted: {pid}')

    for i in range(timeout // 2):
        time.sleep(2)
        try:
            hresp = urllib.request.urlopen(f'{COMFY}/history/{pid}', timeout=5)
            history = json.loads(hresp.read())
            if pid in history:
                h = history[pid]
                status = h.get('status', {})
                if status.get('status_str') == 'error':
                    for m in status.get('messages', []):
                        if m[0] == 'execution_error':
                            print(f'  [{wf_name}] ERROR: {m[1].get("exception_message", "?")[:200]}')
                            return False
                outputs = h.get('outputs', {})
                if outputs:
                    count = sum(len(out.get('images', [])) for out in outputs.values())
                    print(f'  [{wf_name}] SUCCESS: {count} images')
                    return True
        except Exception as e:
            pass
        if i % 15 == 0:
            print(f'  [{wf_name}] waiting... ({i*2}s)')

    print(f'  [{wf_name}] TIMEOUT after {timeout}s')
    return False


if __name__ == '__main__':
    base = 'd:/aigc-project/ComfyUI/workflows'

    # 检查服务
    try:
        urllib.request.urlopen(f'{COMFY}/object_info', timeout=5)
        print('ComfyUI is running.\n')
    except Exception as e:
        print(f'ComfyUI not reachable: {e}')
        sys.exit(1)

    results = {}

    # Test 1: character_concept (quick, 30 steps)
    print('=== Test 1: Character Concept ===')
    results['character'] = submit_and_wait(
        f'{base}/character_concept.json',
        {'2': {'text': '1girl, simple white bg, full body'}},
        timeout=180
    )

    # Test 2: asset_icon_text (quick, 25 steps)
    print('\n=== Test 2: Asset Icon (Text) ===')
    results['asset_icon'] = submit_and_wait(
        f'{base}/asset_icon_text.json',
        {'2': {'text': 'golden sword, game icon, clean design'}},
        timeout=180
    )

    # Test 3: PBR material (slow, 10 steps for test)
    print('\n=== Test 3: PBR Material ===')
    results['pbr'] = submit_and_wait(
        f'{base}/pbr_material.json',
        {'1': {'prompt': 'rough stone wall, gray', 'steps': 10}},
        timeout=300
    )

    print('\n' + '='*50)
    print('Results:')
    for name, ok in results.items():
        status = 'PASS' if ok else 'FAIL'
        print(f'  {status}: {name}')

    passed = sum(1 for v in results.values() if v)
    print(f'\n{passed}/{len(results)} workflows passed')
