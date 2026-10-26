"""New local-only interface preserving the recovered extraction call contract."""
from lib import providers


def tool_call(*, system, user, tool_name, schema, budget=None, max_tokens=4096, **kwargs):
    data, _ = providers.local_tool_call(
        system=system, user=user, tool_name=tool_name, schema=schema, max_tokens=max_tokens,
    )
    return data


def extract(*, page_type, content_markdown, schema, prompt_template):
    prompt = prompt_template.replace("{{CONTENT}}", content_markdown)
    if content_markdown not in prompt:
        prompt += "\n\n" + content_markdown
    data, usage = providers.local_tool_call(
        system="Extract only supported statements. Follow the supplied schema and instructions.",
        user=prompt, tool_name="emit_extraction", schema=schema, max_tokens=8192,
    )
    return {"data": data, "usage": usage}
