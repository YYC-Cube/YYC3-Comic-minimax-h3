#!/usr/bin/env python3
# ==============================================================
# g4t2_submit.py — TC-G4-002 任务提交器（经 H3 网关 claim 鉴权）
# 对齐 agent/h3_agent/gateway.py POST /api/tasks 契约与 security.issue_claim
# 用法：ComfyUI.venv/bin/python scripts/g4t2_submit.py
# ==============================================================
import json
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from agent.h3_agent import security  # noqa: E402

BASE = "http://localhost:8300"
claim = security.issue_claim("trace-g4t2-001", "generate_single")
body = {"task_type": "generate_single",
        "payload": {"batch": "g4t2", "seeds": "42", "variant": "nf4",
                    "preview": True,
                    "prompt_file": "ref_images/g4t2_prompt.txt"}}
req = urllib.request.Request(
    f"{BASE}/api/tasks",
    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json",
             "X-Claim-Token": json.dumps(claim, ensure_ascii=False)})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=14400) as resp:
        result = json.loads(resp.read())
except Exception as e:  # HTTP 错误体含 detail
    body_txt = getattr(e, "read", lambda: b"")().decode("utf-8", "replace")
    print(json.dumps({"error": str(e), "detail": body_txt[:500]},
                     ensure_ascii=False))
    sys.exit(1)
result["elapsed_s"] = round(time.time() - t0, 1)
print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
