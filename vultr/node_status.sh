#!/bin/bash
# Stato compatto per il polling da locale via ssh.
for i in 1 2 3; do
  last=$(tr '\r' '\n' < ~/logs/va$i.log 2>/dev/null | grep -aE "it\]|Epoch [0-9]/5:|Invalid|Error at batch" | tail -1)
  done=$(cat ~/logs/va$i.done 2>/dev/null)
  echo "va$i: ${done:-run} | $last"
done
echo "evalq: $(ls ~/evalq/*_val.json 2>/dev/null | wc -l) val, $(ls ~/evalq/*FINAL_test.json 2>/dev/null | wc -l) final"
echo "disk: $(df -h / | tail -1 | awk '{print $4}') free | gpu: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' ')"
