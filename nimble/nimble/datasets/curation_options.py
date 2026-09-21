"""Shared structured-response options for curation stages."""

from nimble.datasets.create_diverse_dataset import generation_params, generator_backend


def llm_options(model, schema):
    return {"model_name": model, "backend": "openai", "response_format": schema,
            "backend_params": generator_backend(), "generation_params": {
                **generation_params(model, 3500),
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "schema_contrast", "strict": True, "schema": schema.model_json_schema()}}}}
