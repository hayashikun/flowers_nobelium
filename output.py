import json
import os
from datetime import datetime

import boto3

OutPath = os.path.join(os.path.dirname(__file__), "out")
GitPath = os.path.join(os.path.dirname(__file__), ".git")
S3Path = "flowers_nobelium"


def init_train(model_class):
    now = datetime.now()
    name = "{}_{}".format(model_class.__name__, now.strftime("%y%m%d%H%M"))
    try:
        bucket = boto3.resource("s3").Bucket("hayashikun")
        with open(os.path.join(GitPath, "refs/heads/master")) as f:
            commit_hash = f.read()
        body = {
            "name": name,
            "datetime": now.isoformat(),
            "commit": commit_hash.strip()
        }
        bucket.put_object(Key=os.path.join(S3Path, name, "train.log"), Body=json.dumps(body))
    except Exception as e:
        print(e)
        return None
    return name


def upload_result(name):
    bucket = boto3.resource("s3").Bucket("hayashikun")

    for root, dirs, files in os.walk(os.path.join(OutPath, name)):
        for file in files:
            if file.startswith('.'):
                continue
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, OutPath)
            s3_path = os.path.join(S3Path, relative_path)

            with open(local_path, 'rb') as f:
                bucket.put_object(Key=s3_path, Body=f)


def fetch_log(name):
    log_path = os.path.join(OutPath, "{}.log".format(name))
    if not os.path.exists(log_path):
        bucket = boto3.resource("s3").Bucket("hayashikun")
        bucket.download_file(os.path.join(S3Path, name, "log"), log_path)
    with open(log_path) as f:
        return json.load(f)


def log_list():
    bucket = boto3.resource("s3").Bucket("hayashikun")
    result = bucket.meta.client.list_objects(Bucket=bucket.name, Prefix=S3Path + "/", Delimiter='/')
    return [p["Prefix"].replace(S3Path, "").strip("/") for p in result.get("CommonPrefixes") if "Prefix" in p]
