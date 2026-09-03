# Orchestration only. This image never trains and never scores: the retrain DAG hands the
# work to the SAME keras-mlops-fraud-trainer image a human runs with `docker compose run`,
# through the host's docker socket. Airflow that carried keras would be a second place for
# the training environment to live, and a second place for it to drift.
#
# Constraints are pinned to the image's own Airflow version so the two extra packages cannot
# drag a different sqlalchemy/pydantic under Airflow's feet.
FROM apache/airflow:3.3.1-python3.12
COPY airflow/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"
