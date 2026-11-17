#!/bin/bash
# Tempestovsky runs for the bulk-density figure (run from the repository root)
R=results/apophis_density
PY=${PY:-/home/dusan/anaconda3/envs/tempest/bin/python}
for n in chk_g100_A0.14 g50_A0.14 g75_A0.14 g50_A0.08 g50_A0.20 g75_A0.08 g75_A0.20 conv_g50_A0.14_s1600; do
  [ -f $R/out/$n.txt ] && continue
  /usr/bin/time -f "$n %e s" $PY tempest.py --config $R/cfg/$n.yaml > $R/out/$n.log 2>&1
  echo "$(tail -1 $R/out/$n.log) | $(cat $R/out/$n.txt 2>/dev/null)"
done
git checkout -- execution_time.txt 2>/dev/null; true
