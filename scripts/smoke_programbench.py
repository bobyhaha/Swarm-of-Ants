"""Free Docker adapter check, with a synthetic implementation never used as an agent seed."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from ant_colony.programbench import ProgramBenchWorkbench


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--evaluator-python",type=Path,required=True)
    parser.add_argument("--native",action="store_true")
    args=parser.parse_args()
    bench=ProgramBenchWorkbench(args.task,args.out,args.evaluator_python)
    agent=SimpleNamespace(name="fixture",team="fixture-team")
    bench.observation(agent)
    bench.register_team(agent.team,agent.name)
    isolation=bench.execute(agent,{"kind":"exec","command":"test ! -e /oracle && test -d /reference-docs && test -z \"$OPENROUTER_API_KEY\""})
    assert isolation["exit_code"]==0, isolation
    probe=bench.execute(agent,{"kind":"probe","args":["count"],"stdin":"name\na\nb\n"})
    assert probe["stdout"]["text"] == "2\n",probe
    source="import csv,sys\nif sys.argv[1:] == ['count']:\n print(max(0,len(list(csv.reader(sys.stdin)))-1))\nelse:\n sys.exit(1)\n"
    bench.execute(agent,{"kind":"write","path":"main.py","content":source,"base_sha256":"new"})
    compile_script="#!/bin/sh\nset -eu\nprintf '#!/usr/bin/env python3\\n' > executable\ncat main.py >> executable\nchmod +x executable\n"
    bench.execute(agent,{"kind":"write","path":"compile.sh","content":compile_script,"base_sha256":"new"})
    check=bench.execute(agent,{"kind":"test"})
    assert check["passed"]==check["total"]==1,check
    bench.finish_round(1,limit=2)
    assert bench.hashes(bench.public)==bench.hashes(bench.source(agent.team))
    result={"isolation":True,"probe":True,"build":True,"integration":True,"cost_usd":0}
    if args.native:
        bench.apply(SimpleNamespace(author="__public__"))
        result["native_status"]=bench.final_check["status"]
        assert result["native_status"]=="native_scored"
    (bench.directory / "smoke-result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
