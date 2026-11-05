"""Shared LangSmith setup helpers for workshop demos."""
import os


def configure_langsmith(project: str | None = None) -> dict[str, str]:
    """Enable LangSmith tracing and return trace metadata for downstream calls."""
    project_name = project or os.getenv('LANGSMITH_PROJECT', 'pnc-utd-workshop-observability')
    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_PROJECT'] = project_name
    if not os.getenv('LANGSMITH_API_KEY'):
        raise RuntimeError('Set LANGSMITH_API_KEY before running LangSmith observability demos')
    return {'project': project_name, 'endpoint': os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')}
