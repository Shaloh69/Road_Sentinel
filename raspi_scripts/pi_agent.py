#!/usr/bin/env python3
"""
Road Sentinel — Pi Agent
Connects to the Node service's authenticated /admin Socket.IO namespace and
relays terminal commands. The admin page sends commands here; output streams
back in real-time.

Usage:
    python3 pi_agent.py --node http://192.168.8.50:3001 --id pi4 --token <PI_AGENT_TOKEN>
    # or set PI_AGENT_TOKEN in the environment instead of --token

No inbound ports needed on the Pi — the Pi initiates the outbound connection.
PI_AGENT_TOKEN must match the value configured in server/node-service/.env —
the /admin namespace rejects the connection outright without it.
"""

import argparse
import os
import select
import signal
import subprocess
import sys
import threading
import time

import socketio

# ── Globals ────────────────────────────────────────────────────────────────────

NAMESPACE = "/admin"

sio = socketio.Client(
    reconnection=True,
    reconnection_attempts=0,       # retry forever
    reconnection_delay=3,
    reconnection_delay_max=30,
    logger=False,
    engineio_logger=False,
)

current_proc: subprocess.Popen | None = None
current_proc_lock = threading.Lock()
pi_id: str = "pi4"
requester_id: str = ""


# ── Socket events (all scoped to the authenticated /admin namespace) ───────────

@sio.event(namespace=NAMESPACE)
def connect():
    print(f"[pi_agent] Connected to Node service /admin — registering as '{pi_id}'")
    sio.emit("pi_register", pi_id, namespace=NAMESPACE)


@sio.event(namespace=NAMESPACE)
def disconnect():
    print("[pi_agent] Disconnected from Node service — will reconnect...")


@sio.event(namespace=NAMESPACE)
def connect_error(data):
    print(f"[pi_agent] Connection error (check PI_AGENT_TOKEN matches the server's): {data}")


@sio.on("pi_command", namespace=NAMESPACE)
def on_command(data: dict):
    global current_proc, requester_id
    command = data.get("command", "").strip()
    req_id  = data.get("requesterId", "")

    if not command:
        return

    print(f"[pi_agent] Command from admin ({req_id}): {command}")
    requester_id = req_id

    def run():
        global current_proc
        try:
            proc = subprocess.Popen(
                ["sh", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                preexec_fn=os.setsid,  # create process group so we can kill it all
            )
            with current_proc_lock:
                current_proc = proc

            # Read stdout and stderr concurrently using select
            streams = {proc.stdout.fileno(): ("stdout", proc.stdout),
                       proc.stderr.fileno(): ("stderr", proc.stderr)}
            open_fds = set(streams.keys())

            while open_fds:
                try:
                    readable, _, _ = select.select(list(open_fds), [], [], 0.1)
                except ValueError:
                    break
                for fd in readable:
                    kind, stream = streams[fd]
                    chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
                    if chunk:
                        sio.emit("pi_output", {
                            "type": kind,
                            "data": chunk.decode("utf-8", errors="replace"),
                            "requesterId": req_id,
                        }, namespace=NAMESPACE)
                    else:
                        open_fds.discard(fd)

            proc.wait()
            exit_code = proc.returncode

        except Exception as exc:
            sio.emit("pi_output", {
                "type": "stderr",
                "data": f"\n[pi_agent error: {exc}]\n",
                "requesterId": req_id,
            }, namespace=NAMESPACE)
            exit_code = -1

        finally:
            with current_proc_lock:
                current_proc = None

        sio.emit("pi_output", {
            "type": "exit",
            "data": f"\n[Process exited with code {exit_code}]\n",
            "requesterId": req_id,
        }, namespace=NAMESPACE)
        print(f"[pi_agent] Command finished (exit {exit_code}): {command}")

    threading.Thread(target=run, daemon=True).start()


@sio.on("pi_kill", namespace=NAMESPACE)
def on_kill():
    global current_proc
    with current_proc_lock:
        proc = current_proc
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            print("[pi_agent] Sent SIGINT to process group")
        except Exception as exc:
            print(f"[pi_agent] Kill failed: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global pi_id

    parser = argparse.ArgumentParser(description="Road Sentinel Pi Agent")
    parser.add_argument(
        "--node",
        default=os.environ.get("NODE_URL", "http://localhost:3001"),
        help="Node service URL",
    )
    parser.add_argument(
        "--id",
        default=os.environ.get("PI_ID", "pi4"),
        help="Pi identifier shown in admin terminal (e.g. pi4, pi5)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("PI_AGENT_TOKEN", ""),
        help="Shared secret matching PI_AGENT_TOKEN in server/node-service/.env "
             "(required — the /admin namespace rejects connections without it)",
    )
    args = parser.parse_args()

    pi_id = args.id
    node_url = args.node

    if not args.token:
        print("[pi_agent] ERROR: no token provided. Set PI_AGENT_TOKEN in the "
              "environment or pass --token — it must match the Node service's "
              "PI_AGENT_TOKEN.")
        sys.exit(1)

    print(f"[pi_agent] Starting — id={pi_id}  node={node_url}  namespace={NAMESPACE}")

    while True:
        try:
            sio.connect(
                node_url,
                transports=["websocket", "polling"],
                namespaces=[NAMESPACE],
                auth={"token": args.token},
            )
            sio.wait()
        except Exception as exc:
            print(f"[pi_agent] Fatal error: {exc} — retrying in 10s")
            time.sleep(10)


if __name__ == "__main__":
    main()
