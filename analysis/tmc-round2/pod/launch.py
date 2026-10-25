#!/usr/bin/env python3
"""Launch the autonomous TMC round-2 detection-dump pod and verify it really exists.

Adapted from 09-30-atr-iter/pod/launch.py. The create mutation has been seen to
report failure while provisioning the pod anyway (a $45 orphan in 2026-07), so the
response is never trusted on its own -- this polls `pods` afterwards and prints what
is actually running.
"""
import base64
import json
import os
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
API = "https://api.runpod.io/graphql"
UA = "runpodctl/1.14"
BUCKET = "trafficgroup-ne-data-sb"
BASE = "experiments/round2-dets"
KEY = os.environ["RUNPOD_API_KEY"]


def gql(query):
    req = urllib.request.Request(
        f"{API}?api_key={KEY}", method="POST",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else time.strftime("r%Y%m%d-%H%M%S")
    sites = os.environ.get("SITES", "gunselford rolling sanford md108 1022_meadowbranch")
    # Community supply for any single type comes and goes, so walk a price-ordered
    # list rather than failing the launch. A6000 .33, 4090 .34, L40 .44.
    gpus = [g.strip() for g in os.environ.get(
        "GPU", "NVIDIA RTX A6000,NVIDIA GeForce RTX 4090,NVIDIA L40").split(",")]
    # NOT the prod image: registry.runpod.net is private and a plain pod cannot pull
    # it -- the container never starts and the pod bills doing nothing.
    image = os.environ.get("IMAGE", "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")

    # base64 so nothing has to survive two layers of shell + GraphQL quoting. boto3
    # is how the bootstrap is fetched, so it must exist before the bootstrap runs;
    # the public base image does not ship it.
    inner = (
        "set -u\n"
        "mkdir -p /workspace && cd /workspace\n"
        "python -m pip install --quiet --no-input boto3 requests 2>&1 | tail -3\n"
        "python -c \"import boto3,os\n"
        "c=boto3.client('s3')\n"
        "[c.download_file(os.environ['RESULT_BUCKET'],'%s/code/'+f,'/workspace/'+f) "
        "for f in ['bootstrap.sh','s3util.py']]\"\n"
        "exec bash /workspace/bootstrap.sh\n" % BASE)
    b64 = base64.b64encode(inner.encode()).decode()
    docker_args = f"bash -c 'echo {b64} | base64 -d > /tmp/go.sh && bash /tmp/go.sh'"

    env = {
        "RUN_ID": run_id, "SITES": sites,
        "RESULT_BUCKET": BUCKET, "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "RUNPOD_API_KEY": KEY,
        "DUMP_CONF": os.environ.get("DUMP_CONF", "0.05"),
        "WATCHDOG_SECS": os.environ.get("WATCHDOG_SECS", "21600"),
    }
    env_s = ",".join("{key:%s,value:%s}" % (json.dumps(k), json.dumps(v))
                     for k, v in env.items())
    name = f"tmc-r2-{run_id}"

    def mutation_for(gpu, cloud):
        return """
        mutation { podFindAndDeployOnDemand(input:{
          cloudType: %s, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 60,
          minMemoryInGb: 24, minVcpuCount: 8,
          gpuTypeId: %s, name: %s, imageName: %s,
          dockerArgs: %s, ports: "", volumeMountPath: "/workspace",
          env: [%s]
        }) { id machineId imageName costPerHr } }""" % (
            cloud, json.dumps(gpu), json.dumps(name), json.dumps(image),
            json.dumps(docker_args), env_s)

    print(f"launching {run_id}  sites=[{sites}]  conf={env['DUMP_CONF']}")
    created = False
    for cloud in ("COMMUNITY", "SECURE"):
        for gpu in gpus:
            print(f"  try {gpu} / {cloud} ...", end=" ")
            try:
                resp = gql(mutation_for(gpu, cloud))
            except Exception as e:
                print(f"RAISED {e}")
                created = True        # may have provisioned anyway -- go and look
                break
            errs = resp.get("errors")
            if errs:
                msg = errs[0].get("message", "")[:70]
                print(f"no ({msg})")
                if "no longer any instances" not in msg and "SUPPLY" not in str(errs):
                    created = True
                    break
                continue
            print("accepted:", json.dumps(resp.get("data"))[:200])
            created = True
            break
        if created:
            break
    if not created:
        print("  every GPU/cloud combination reported supply constraint")

    for attempt in range(12):
        time.sleep(10)
        pods = (gql("query { myself { pods { id name desiredStatus costPerHr "
                    "runtime { uptimeInSeconds } } } }")
                .get("data", {}).get("myself", {}).get("pods") or [])
        mine = [p for p in pods if p["name"] == name]
        if mine:
            p = mine[0]
            print(f"\nPOD LIVE  id={p['id']}  status={p['desiredStatus']}  ${p.get('costPerHr')}/hr")
            print(f"results   s3://{BUCKET}/{BASE}/runs/{run_id}/out/")
            (HERE / "last_pod.json").write_text(
                json.dumps({"run_id": run_id, "pod": p}, indent=1))
            return 0
        print(f"  not visible yet ({attempt + 1}/12)")
    print("\nNO POD FOUND after 2 min. Check the console before relaunching -- "
          "a pod may still be provisioning and would be an orphan.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
