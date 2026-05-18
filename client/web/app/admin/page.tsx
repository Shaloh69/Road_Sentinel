"use client";

import type { Socket } from "socket.io-client";

import { useState, useEffect, useRef, useCallback, KeyboardEvent } from "react";

import { getSocket } from "@/lib/socket";

type Target = "server" | "pi4" | "pi5";

interface TerminalLine {
  id: number;
  type: "stdout" | "stderr" | "exit" | "command" | "info";
  data: string;
}

interface PiStatus {
  pi4: boolean;
  pi5: boolean;
}

const TARGETS: { id: Target; label: string; desc: string }[] = [
  {
    id: "server",
    label: "Server",
    desc: "Node.js service host (Render / local)",
  },
  { id: "pi4", label: "Pi 4 — Cam A", desc: "Camera A + LED matrix" },
  { id: "pi5", label: "Pi 5 — Cam B", desc: "Camera B + LED matrix" },
];

const QUICK_COMMANDS: { label: string; cmd: string; targets: Target[] }[] = [
  {
    label: "git pull",
    cmd: "git pull origin main",
    targets: ["server", "pi4", "pi5"],
  },
  { label: "git status", cmd: "git status", targets: ["server", "pi4", "pi5"] },
  {
    label: "git log",
    cmd: "git log --oneline -10",
    targets: ["server", "pi4", "pi5"],
  },
  {
    label: "update scripts",
    cmd: "~/roadsentinel/update.sh",
    targets: ["pi4", "pi5"],
  },
  {
    label: "service status",
    cmd: "sudo systemctl status roadsentinel-camera roadsentinel-display roadsentinel-agent --no-pager 2>&1 | head -50",
    targets: ["pi4", "pi5"],
  },
  {
    label: "restart all",
    cmd: "sudo systemctl restart roadsentinel-camera roadsentinel-display roadsentinel-agent",
    targets: ["pi4", "pi5"],
  },
  {
    label: "camera log",
    cmd: "tail -50 ~/roadsentinel/logs/camera.log",
    targets: ["pi4", "pi5"],
  },
  {
    label: "display log",
    cmd: "tail -50 ~/roadsentinel/logs/display.log",
    targets: ["pi4", "pi5"],
  },
  {
    label: "node -v",
    cmd: "node --version && npm --version",
    targets: ["server"],
  },
  { label: "pwd", cmd: "pwd", targets: ["server", "pi4", "pi5"] },
  { label: "df -h", cmd: "df -h", targets: ["server", "pi4", "pi5"] },
  { label: "free -h", cmd: "free -h", targets: ["pi4", "pi5"] },
  { label: "uptime", cmd: "uptime", targets: ["server", "pi4", "pi5"] },
  {
    label: "ps aux",
    cmd: "ps aux | head -25",
    targets: ["server", "pi4", "pi5"],
  },
];

// Strip ANSI escape sequences before rendering
function stripAnsi(str: string): string {
  return str.replace(/\x1b\[[0-9;]*[mABCDEFGHJKSTfnsu]/g, "");
}

export default function AdminPage() {
  const [target, setTarget] = useState<Target>("server");
  const [piStatus, setPiStatus] = useState<PiStatus>({
    pi4: false,
    pi5: false,
  });
  const [lines, setLines] = useState<TerminalLine[]>([
    {
      id: 0,
      type: "info",
      data: "Road Sentinel — Admin Terminal\nConnecting to server...\n",
    },
  ]);
  const [command, setCommand] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [historyIdx, setHistoryIdx] = useState(-1);

  const socketRef = useRef<Socket | null>(null);
  const termRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const lineId = useRef(1);
  const targetRef = useRef<Target>("server");

  // Keep targetRef in sync for use inside event handlers
  useEffect(() => {
    targetRef.current = target;
  }, [target]);

  const push = useCallback((type: TerminalLine["type"], data: string) => {
    setLines((prev) => [
      ...prev,
      { id: lineId.current++, type, data: stripAnsi(data) },
    ]);
  }, []);

  useEffect(() => {
    const socket = getSocket();

    socketRef.current = socket;

    socket.emit("subscribe_admin");

    const onConnect = () => {
      setConnected(true);
      push("info", "Connected to server terminal.\n");
    };
    const onDisconnect = () => {
      setConnected(false);
      setIsRunning(false);
      push("info", "Disconnected from server.\n");
    };
    const onOutput = ({ type, data }: { type: string; data: string }) => {
      if (type === "exit") setIsRunning(false);
      push(type as TerminalLine["type"], data);
    };
    const onPiStatus = ({
      piId,
      online,
    }: {
      piId: string;
      online: boolean;
    }) => {
      setPiStatus((prev) => ({ ...prev, [piId]: online }));
      push("info", `${piId} ${online ? "connected" : "disconnected"}.\n`);
    };
    const onPiStatusAll = (snapshot: Record<string, boolean>) => {
      setPiStatus((prev) => ({ ...prev, ...snapshot }));
    };

    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("terminal_output", onOutput);
    socket.on("pi_status", onPiStatus);
    socket.on("pi_status_all", onPiStatusAll);

    if (socket.connected) {
      setConnected(true);
      push("info", "Connected to server terminal.\n");
    }

    return () => {
      socket.emit("unsubscribe_admin");
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("terminal_output", onOutput);
      socket.off("pi_status", onPiStatus);
      socket.off("pi_status_all", onPiStatusAll);
    };
  }, []);

  // Auto-scroll on new output
  useEffect(() => {
    if (termRef.current)
      termRef.current.scrollTop = termRef.current.scrollHeight;
  }, [lines]);

  const isTargetOnline = (t: Target) => {
    if (t === "server") return connected;

    return piStatus[t as keyof PiStatus];
  };

  const run = useCallback(
    (cmd: string) => {
      if (!cmd.trim() || isRunning || !connected) return;
      push("command", `[${targetRef.current}] $ ${cmd}\n`);
      setHistory((prev) => [cmd, ...prev.slice(0, 49)]);
      setHistoryIdx(-1);
      setCommand("");
      setIsRunning(true);
      socketRef.current?.emit("terminal_command", {
        command: cmd,
        target: targetRef.current,
      });
    },
    [isRunning, connected, push],
  );

  const kill = useCallback(() => {
    socketRef.current?.emit("terminal_kill", targetRef.current);
    setIsRunning(false);
    push("stderr", "\n^C\n");
  }, [push]);

  const handleKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      run(command);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      const idx = Math.min(historyIdx + 1, history.length - 1);

      setHistoryIdx(idx);
      setCommand(history[idx] ?? "");
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (historyIdx <= 0) {
        setHistoryIdx(-1);
        setCommand("");
      } else {
        const idx = historyIdx - 1;

        setHistoryIdx(idx);
        setCommand(history[idx] ?? "");
      }
    } else if (e.ctrlKey && e.key === "c") {
      kill();
    } else if (e.ctrlKey && e.key === "l") {
      e.preventDefault();
      setLines([]);
    }
  };

  const lineColor = (type: TerminalLine["type"]) => {
    switch (type) {
      case "command":
        return "text-[#ED9E59]";
      case "stderr":
        return "text-red-400";
      case "exit":
        return "text-yellow-400/60";
      case "info":
        return "text-sky-400";
      default:
        return "text-green-300";
    }
  };

  const visibleCommands = QUICK_COMMANDS.filter((qc) =>
    qc.targets.includes(target),
  );

  return (
    <div className="p-6 min-h-screen">
      <div className="max-w-5xl mx-auto space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Admin Terminal</h1>
            <p className="text-white/50 text-sm mt-1">
              Run shell commands on the server or directly on each Raspberry Pi
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${connected ? "bg-green-400 animate-pulse" : "bg-red-400"}`}
            />
            <span className="text-sm text-white/60">
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>

        {/* Target selector */}
        <div className="p-4 bg-white/5 rounded-xl border border-white/10">
          <p className="text-[10px] uppercase tracking-widest text-white/40 mb-3">
            Target Device
          </p>
          <div className="flex gap-3">
            {TARGETS.map((t) => {
              const online = isTargetOnline(t.id);
              const active = target === t.id;

              return (
                <button
                  key={t.id}
                  className={`flex-1 flex flex-col items-start gap-1 px-4 py-3 rounded-xl border transition-all ${
                    active
                      ? "bg-[#ED9E59]/15 border-[#ED9E59]/60 text-white"
                      : "bg-white/5 border-white/10 text-white/60 hover:bg-white/10 hover:text-white"
                  }`}
                  onClick={() => {
                    setTarget(t.id);
                    push("info", `Switched target to ${t.label}.\n`);
                  }}
                >
                  <div className="flex items-center gap-2 w-full">
                    <span
                      className={`w-2 h-2 rounded-full flex-shrink-0 ${
                        online ? "bg-green-400 animate-pulse" : "bg-red-400/60"
                      }`}
                    />
                    <span className="font-semibold text-sm">{t.label}</span>
                    <span
                      className={`ml-auto text-[10px] px-1.5 py-0.5 rounded font-mono ${
                        online
                          ? "bg-green-400/15 text-green-400"
                          : "bg-red-400/10 text-red-400/60"
                      }`}
                    >
                      {online ? "ONLINE" : "OFFLINE"}
                    </span>
                  </div>
                  <span className="text-[11px] text-white/40 pl-4">
                    {t.desc}
                  </span>
                </button>
              );
            })}
          </div>
          {(target === "pi4" || target === "pi5") && !piStatus[target] && (
            <p className="mt-3 text-xs text-yellow-400/70 pl-1">
              {target} is offline — make sure{" "}
              <code className="font-mono bg-white/10 px-1 rounded">
                roadsentinel-agent
              </code>{" "}
              service is running on that Pi. The Pi must be able to reach the
              Node service URL.
            </p>
          )}
        </div>

        {/* Quick commands */}
        <div className="p-4 bg-white/5 rounded-xl border border-white/10">
          <p className="text-[10px] uppercase tracking-widest text-white/40 mb-3">
            Quick Commands — {TARGETS.find((t) => t.id === target)?.label}
          </p>
          <div className="flex flex-wrap gap-2">
            {visibleCommands.map((qc) => (
              <button
                key={qc.cmd}
                className="px-3 py-1.5 text-xs rounded-lg bg-white/10 hover:bg-white/20 text-white/75 hover:text-white border border-white/10 transition-all disabled:opacity-30 disabled:cursor-not-allowed font-mono"
                disabled={isRunning || !connected || !isTargetOnline(target)}
                onClick={() => run(qc.cmd)}
              >
                {qc.label}
              </button>
            ))}
          </div>
        </div>

        {/* Terminal window */}
        <div className="rounded-xl border border-white/10 overflow-hidden shadow-2xl bg-[#0d0d0d]">
          {/* Title bar */}
          <div className="flex items-center gap-2 px-4 py-2.5 bg-[#1c1c1c] border-b border-white/10">
            <span className="w-3 h-3 rounded-full bg-[#ff5f57]" />
            <span className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
            <span className="w-3 h-3 rounded-full bg-[#28c840]" />
            <span className="ml-3 text-xs text-white/30 font-mono">
              road-sentinel @{" "}
              <span className="text-[#ED9E59]">
                {TARGETS.find((t) => t.id === target)?.label ?? target}
              </span>
            </span>
            {isRunning && (
              <button
                className="ml-auto px-2.5 py-0.5 text-xs bg-red-500/20 hover:bg-red-500/40 text-red-400 rounded border border-red-500/30 transition-all font-mono"
                onClick={kill}
              >
                ■ Kill (Ctrl+C)
              </button>
            )}
          </div>

          {/* Output */}
          {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions */}
          <div
            ref={termRef}
            className="h-[420px] overflow-y-auto p-4 font-mono text-xs leading-relaxed cursor-text select-text"
            onClick={() => {
              if (!window.getSelection()?.toString()) {
                inputRef.current?.focus();
              }
            }}
            onKeyDown={() => inputRef.current?.focus()}
          >
            {lines.map((l) => (
              <pre
                key={l.id}
                className={`whitespace-pre-wrap break-all ${lineColor(l.type)}`}
              >
                {l.data}
              </pre>
            ))}
            {isRunning && (
              <span className="inline-flex items-center gap-0.5 ml-1">
                {[0, 100, 200].map((d) => (
                  <span
                    key={d}
                    className="w-1 h-1 bg-white/40 rounded-full animate-bounce"
                    style={{ animationDelay: `${d}ms` }}
                  />
                ))}
              </span>
            )}
          </div>

          {/* Input row */}
          <div className="flex items-center gap-2 px-4 py-3 border-t border-white/10 bg-[#0a0a0a]">
            <span className="text-[#ED9E59] font-mono select-none text-sm">
              [{TARGETS.find((t) => t.id === target)?.label}] $
            </span>
            <input
              ref={inputRef}
              autoComplete="off"
              className="flex-1 bg-transparent text-white/90 font-mono text-sm outline-none placeholder:text-white/20"
              disabled={!connected}
              placeholder={
                !connected
                  ? "Not connected..."
                  : isRunning
                    ? "Running... (Ctrl+C to kill)"
                    : `Type a command for ${TARGETS.find((t) => t.id === target)?.label}`
              }
              spellCheck={false}
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              onKeyDown={handleKey}
            />
            <button
              className="px-3 py-1 text-xs bg-[#ED9E59]/20 hover:bg-[#ED9E59]/30 text-[#ED9E59] rounded border border-[#ED9E59]/30 transition-all disabled:opacity-30 disabled:cursor-not-allowed font-mono"
              disabled={isRunning || !connected || !command.trim()}
              onClick={() => run(command)}
            >
              Run ↵
            </button>
          </div>
        </div>

        {/* Tips */}
        <div className="p-4 bg-white/5 rounded-xl border border-white/10 text-xs text-white/40 space-y-1">
          <p className="text-white/60 font-medium mb-1">
            How Pi terminal works
          </p>
          <p>
            Each Pi runs a{" "}
            <code className="font-mono bg-white/10 px-1 rounded">
              roadsentinel-agent
            </code>{" "}
            service that connects back to this Node server — no SSH or open
            ports needed on the Pi.
          </p>
          <p className="mt-2 text-white/60 font-medium">Keyboard shortcuts</p>
          <p>
            ↑ / ↓ — navigate command history &nbsp;·&nbsp; Ctrl+C — kill process
            &nbsp;·&nbsp; Ctrl+L — clear
          </p>
        </div>
      </div>
    </div>
  );
}
