from mlflow.tracking import MlflowClient


def promote_model(model_name, version, stage):
    # 등록된 모델의 스테이지를 변경(추후 사용될 수 있음)
    client = MlflowClient()
    client.transition_model_version_stage(name=model_name, version=version, stage=stage)


def get_latest_registered_model_uri(model_name):
    # MLflow 클라이언트 생성
    client = MlflowClient()

    # 가장 최근 버전의 모델 메타데이터 가져오기
    model_metadata = client.get_latest_versions(model_name, stages=["None"])
    latest_model_version = model_metadata[0].version

    model_uri = f"models:/{model_name}/{latest_model_version}"

    return model_uri


def get_registered_model_info(model_name):
    client = MlflowClient()
    info = {}

    # 등록된 모델의 정보 가져오기
    model = client.get_registered_model(model_name)

    # 모델 정보
    info["Model Name"] = model.name
    info["Model Description"] = model.description
    info["Model Creation Timestamp"] = model.creation_timestamp
    info["Model Last Updated Timestamp"] = model.last_updated_timestamp

    return info


def parse_auto_logged_info(r):
    info = {}
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    info["run_id"] = r.info.run_id
    info["artifacts"] = artifacts
    info["params"] = r.data.params
    info["metrics"] = r.data.metrics
    info["tags"] = tags

    return info
