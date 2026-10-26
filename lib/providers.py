"""New fail-closed local-only transport, not the historical experiment provider."""
import ipaddress
import json
import os
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from jsonschema import Draft202012Validator


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise ValueError("Model endpoint redirects are disabled.")


def backend():
    return "local"


def local_model():
    return os.environ.get("SUBMISSION_LOCAL_MODEL", "")


def require_local():
    if os.environ.get("SUBMISSION_ALLOW_LOCAL_LLM") != "1":
        raise RuntimeError("Network disabled. Offline examples need no model server.")
    if not local_model():
        raise ValueError("Set SUBMISSION_LOCAL_MODEL explicitly.")


def local_tool_call(*, system, user, tool_name, schema, max_tokens=4096, retries=1):
    require_local()
    base = os.environ.get("SUBMISSION_LOCAL_BASE_URL", "http://localhost:8000/v1")
    parsed = urlsplit(base)
    host = parsed.hostname or ""
    if host != "localhost":
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = False
        if not allowed:
            raise ValueError("Only an explicit loopback model endpoint is permitted.")
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Invalid local endpoint.")
    payload = {
        "model": local_model(),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "tools": [{"type": "function", "function": {
            "name": tool_name, "parameters": schema,
        }}],
        "tool_choice": {"type": "function", "function": {"name": tool_name}},
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    request = Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Do not forward corpus contents through environment-configured HTTP proxies.
    with build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=180) as response:
        result = json.load(response)
    arguments = result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    data = json.loads(arguments) if isinstance(arguments, str) else arguments
    Draft202012Validator(schema).validate(data)
    return data, result.get("usage", {})
