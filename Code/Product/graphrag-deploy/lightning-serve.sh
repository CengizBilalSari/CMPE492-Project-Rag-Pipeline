#!/usr/bin/env bash
# =============================================================================
# lightning-serve.sh — Start / Stop / Restart vLLM in Lightning AI Studio
#
# Usage:
#   bash lightning-serve.sh start
#   bash lightning-serve.sh start --gpu-memory-utilization 0.95
#   bash lightning-serve.sh stop
#   bash lightning-serve.sh restart
#   bash lightning-serve.sh status
#   bash lightning-serve.sh logs
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

VLLM_PID_FILE="/tmp/graphrag_vllm.pid"
VLLM_LOG_FILE="/tmp/graphrag_vllm.log"

_is_running() {
  if [ -f "${VLLM_PID_FILE}" ]; then
    local pid
    pid=$(cat "${VLLM_PID_FILE}")
    if kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

_wait_for_health() {
  local url="http://localhost:${VLLM_PORT}/health"
  local max_wait=600
  local waited=0

  echo "    Waiting for vLLM health..."
  while [ ${waited} -lt ${max_wait} ]; do
    if curl -sf "${url}" > /dev/null 2>&1; then
      echo "    ✓ vLLM is healthy on port ${VLLM_PORT}"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
    if [ $((waited % 30)) -eq 0 ]; then
      echo "    ... still waiting (${waited}s elapsed)"
    fi
  done

  echo "    ✗ vLLM did not become healthy within ${max_wait}s"
  echo "    Check logs with: bash lightning-serve.sh logs"
  return 1
}

cmd_start() {
  if _is_running; then
    echo "vLLM is already running (PID: $(cat ${VLLM_PID_FILE}))"
    return 0
  fi

  local extra_args=("$@")
  local extra_str="${extra_args[*]:-}"

  if [[ "${MODEL}" == *"/"* ]] && [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set."
    echo "If the model is gated/private, set it first: export HF_TOKEN='hf_xxxxx'"
  fi

  echo "==> Starting vLLM for GraphRAG..."
  echo "    Model: ${MODEL}"
  echo "    Port:  ${VLLM_PORT}"
  if [ ${#extra_args[@]} -gt 0 ]; then
    echo "    Extra: ${extra_args[*]}"
  fi

  local cli_args=()
  cli_args+=(--port "${VLLM_PORT}")

  [[ "${extra_str}" != *"--max-model-len"* ]]         && cli_args+=(--max-model-len "${MAX_MODEL_LEN}")
  [[ "${extra_str}" != *"--gpu-memory-utilization"* ]] && cli_args+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
  [[ "${extra_str}" != *"--dtype"* ]]                  && cli_args+=(--dtype "${DTYPE}")

  nohup vllm serve "${MODEL}" \
    "${cli_args[@]}" \
    "${extra_args[@]}" \
    > "${VLLM_LOG_FILE}" 2>&1 &

  echo $! > "${VLLM_PID_FILE}"
  echo "    PID: $(cat ${VLLM_PID_FILE})"

  _wait_for_health
}

cmd_stop() {
  if ! _is_running; then
    echo "vLLM is not running."
    rm -f "${VLLM_PID_FILE}"
    return 0
  fi

  local pid
  pid=$(cat "${VLLM_PID_FILE}")
  echo "==> Stopping vLLM (PID: ${pid})..."
  kill "${pid}" 2>/dev/null || true

  local waited=0
  while kill -0 "${pid}" 2>/dev/null && [ ${waited} -lt 15 ]; do
    sleep 1
    waited=$((waited + 1))
  done

  if kill -0 "${pid}" 2>/dev/null; then
    echo "    Force killing..."
    kill -9 "${pid}" 2>/dev/null || true
  fi

  rm -f "${VLLM_PID_FILE}"
  echo "    ✓ vLLM stopped."
}

cmd_restart() {
  cmd_stop
  sleep 2
  cmd_start "$@"
}

cmd_status() {
  if _is_running; then
    local pid
    pid=$(cat "${VLLM_PID_FILE}")
    echo "vLLM is running (PID: ${pid})"
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
      echo "  Health: ✓ OK"
    else
      echo "  Health: ✗ Not responding"
    fi
    echo "  API:    http://localhost:${VLLM_PORT}/v1"
  else
    echo "vLLM is not running."
  fi
}

cmd_logs() {
  if [ -f "${VLLM_LOG_FILE}" ]; then
    tail -f "${VLLM_LOG_FILE}"
  else
    echo "No log file found."
  fi
}

ACTION="${1:-}"
shift || true

case "${ACTION}" in
  start)   cmd_start "$@" ;;
  stop)    cmd_stop ;;
  restart) cmd_restart "$@" ;;
  status)  cmd_status ;;
  logs)    cmd_logs ;;
  *)
    echo "Usage: bash lightning-serve.sh {start|stop|restart|status|logs} [extra vllm args...]"
    exit 1
    ;;
esac
