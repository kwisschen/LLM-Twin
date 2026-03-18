from zenml import pipeline

from steps.etl import get_or_create_user, load_chatgpt_gemini


@pipeline
def chatgpt_gemini_etl(user_full_name: str, chatgpt_export_path: str, gemini_export_path: str) -> str:
    user = get_or_create_user(user_full_name)
    last_step = load_chatgpt_gemini(user=user, chatgpt_path=chatgpt_export_path, gemini_path=gemini_export_path)

    return last_step.invocation_id
