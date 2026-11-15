import json
import subprocess
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
req = json.loads(path.read_text())
what = req["case"]["params"]["what"]
if what=="timeout":
    # A grandchild proves the process-group timeout does not leave training
    # subprocesses behind after their immediate parent is killed.
    child = subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"])
    (path.parent/"child.pid").write_text(str(child.pid))
    time.sleep(120)
elif what=="crash":
    raise SystemExit(7)
elif what=="malformed":
    (path.parent/"outcome.json").write_text("{")
elif what=="exit_after_pass":
    (path.parent/"outcome.json").write_text(json.dumps({"status":"pass"}))
    raise SystemExit(3)
else:
    (path.parent/"outcome.json").write_text(json.dumps({"status":what,"details":{"synthetic":True}}))
