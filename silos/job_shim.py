"""run_job.py: the script a SageMaker training job runs (packaged as the source directory's entry point).

It does three things and then hands over to `python -m <module>`:
  1. puts the source directory on sys.path (silos/, tools/evlib.py);
  2. if the organisers' RAN RL SDK is needed (JOB_NEEDS_SDK=1) and not importable, installs it from SDK_S3_URI
     (a wheel, or a zip containing a wheel) with the `maveric` extra. The SDK is licensed: it is read from our
     own bucket at run time and is never copied into this repository or the source archive;
  3. runs the module named by JOB_MODULE with the arguments in JOB_ARGS (shell-quoted).

Environment values are visible in DescribeTrainingJob: nothing secret goes here.
"""

import os
import runpy
import shlex
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def ensure_sdk() -> None:
    try:
        import ran_rl.twin.maveric.twin  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass
    uri = os.environ.get("SDK_S3_URI", "").strip()
    if not uri.startswith("s3://"):
        raise SystemExit("the RAN RL SDK is not installed and SDK_S3_URI is not set")
    import boto3

    bucket, _, key = uri[5:].partition("/")
    tmp = tempfile.mkdtemp(prefix="sdk-")
    local = os.path.join(tmp, os.path.basename(key))
    boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-2")).download_file(bucket, key, local)
    target = local
    if local.endswith(".zip"):
        with zipfile.ZipFile(local) as z:
            wheels = [n for n in z.namelist() if n.endswith(".whl")]
            if not wheels:
                raise SystemExit("no wheel inside the SDK zip")
            z.extract(wheels[0], tmp)
            target = os.path.join(tmp, wheels[0])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", f"{target}[maveric]"])


def main() -> int:
    module = os.environ.get("JOB_MODULE", "").strip()
    if not module.startswith("silos."):
        raise SystemExit("JOB_MODULE must be a silos.* module")
    if os.environ.get("JOB_NEEDS_SDK") == "1":
        ensure_sdk()
    sys.argv = [module] + shlex.split(os.environ.get("JOB_ARGS", ""))
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
