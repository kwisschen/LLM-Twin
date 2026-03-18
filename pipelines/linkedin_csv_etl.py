from zenml import pipeline

from steps.etl import get_or_create_user, load_linkedin_csv


@pipeline
def linkedin_csv_etl(user_full_name: str, export_dir: str) -> str:
    user = get_or_create_user(user_full_name)
    last_step = load_linkedin_csv(user=user, export_dir=export_dir)

    return last_step.invocation_id
