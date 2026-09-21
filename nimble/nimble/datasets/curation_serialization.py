"""Lossless container recovery before the unchanged context and factual audits."""
import json


def wrap_paragraph_for_object(document, request):
    original = json.loads(request['payload_json'])['original_input']['state']
    if not isinstance(original, dict):
        return document
    try:
        paragraph = json.loads(document['base_state_json'])
    except json.JSONDecodeError:
        return document
    if not isinstance(paragraph, str) or not paragraph.strip():
        return document
    spans = document['focus_evidence']
    if len(spans) != 2 or any(s['path'] != [] or not s['text']
                              or paragraph.count(s['text']) != 1 for s in spans):
        return document
    # The model supplied a complete paragraph and root-relative evidence paths.
    # Add only a container; never infer missing facts, fields, or speaker roles.
    return {**document, 'base_state_json': json.dumps({'context': paragraph}, ensure_ascii=False),
            'focus_evidence': [{**s, 'path': ['context']} for s in spans]}
