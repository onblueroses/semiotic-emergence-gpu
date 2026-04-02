#!/usr/bin/env bash
# Runs on the Vast.ai instance via a single SSH call.
# Expects VAST_API_KEY and VAST_INSTANCE_ID to be set in environment.
# Installs deps, launches experiment with nohup, returns immediately.
# The nohup job writes done.json and self-destructs on completion.
set -euo pipefail

echo "=== setup: installing JAX (CUDA 12) ==="
pip install --quiet --upgrade "jax[cuda12]"

echo "=== setup: cloning repo ==="
if [ -d /workspace/semgpu/.git ]; then
    git -C /workspace/semgpu pull --ff-only
else
    git clone https://github.com/onblueroses/semiotic-emergence-gpu.git /workspace/semgpu
fi

echo "=== setup: installing package ==="
pip install --quiet -e /workspace/semgpu

echo "=== setup: verifying GPU ==="
python3 -c "
import jax
devs = jax.devices()
print('JAX devices:', devs)
assert devs[0].platform == 'gpu', f'Expected GPU, got {devs[0].platform}'
print('OK')
"

echo "=== launching experiment (nohup) ==="
mkdir -p /workspace/results/7k-seed42

# Write the run-and-teardown wrapper with creds baked in
cat > /workspace/run-teardown.sh << EOF
#!/usr/bin/env bash
set -euo pipefail
echo "Started: \$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /workspace/run.log 2>&1

cd /workspace/results/7k-seed42
python3 -m semgpu.main 42 100000 \\
  --pop 7000 --grid 178 --food 1050 \\
  --zone-radius 71 --zone-speed 4.45 --zone-drain 0.15 \\
  --signal-cost 0.015 --signal-range 71 --ticks 500 \\
  --pred 5 --freeze-zones 2 --poison-ratio 0.0 \\
  --metrics-interval 10 --checkpoint-interval 1000 \\
  >> /workspace/run.log 2>&1

EXIT=\$?
echo "Python exited \$EXIT at \$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /workspace/run.log

echo "{\"status\":\"complete\",\"exit_code\":\$EXIT,\"ts\":\"\$(date -Iseconds)\"}" > /workspace/results/done.json

# Self-destruct
curl -s -X DELETE \\
  "https://console.vast.ai/api/v0/instances/${VAST_INSTANCE_ID}/?api_key=${VAST_API_KEY}" \\
  && echo "Destroy sent." >> /workspace/run.log \\
  || echo "WARNING: destroy failed" >> /workspace/run.log
EOF

chmod +x /workspace/run-teardown.sh
nohup bash /workspace/run-teardown.sh </dev/null >> /workspace/run.log 2>&1 &
echo "PID: $!"
echo "Monitor: tail -f /workspace/run.log"
echo "=== SSH session done - experiment running in background ==="
