"use client";

import type { Socket } from "socket.io-client";

import {
  useState,
  useEffect,
  useRef,
  useCallback,
  KeyboardEvent,
  FormEvent,
} from "react";

import {
  getAdminSocket,
  getStoredAdminToken,
  storeAdminToken,
  clearStoredAdminToken,
} from "@/lib/adminSocket";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

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

// ── Login gate ─────────────────────────────────────────────────────────────────

function LoginForm({ onSuccess }: { onSuccess: (token: string) => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!password.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const json = await res.json();

      if (json.success && json.token) {
        storeAdminToken(json.token);
        onSuccess(json.token);
      } else {
        setError(json.error ?? "Login failed");
      }
    } catch {
      setError("Cannot reach server");
    } finally {
      setLoading(false);
      setPassword("");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <form
        className="w-full max-w-sm p-6 bg-surface-2/60 rounded-xl border border-border space-y-4"
        onSubmit={submit}
      >
        <div>
          <h1 className="text-xl font-heading font-bold text-fg">
            Admin Login
          </h1>
          <p className="text-fg-muted/70 text-sm mt-1">
            Enter the admin password to reach the server/Pi terminal.
          </p>
        </div>
        <input
          autoComplete="current-password"
          className="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-fg placeholder:text-fg-muted/50 outline-none focus:border-brand/60"
          disabled={loading}
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <p className="text-xs text-danger">{error}</p>}
        <button
          className="w-full px-3 py-2 text-sm bg-brand/20 hover:bg-brand/30 text-brand rounded-lg border border-brand/30 transition-colors duration-150 ease-standard disabled:opacity-40"
          disabled={loading || !password.trim()}
          type="submit"
        >
          {loading ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}

// ── Terminal (only rendered once authenticated) ────────────────────────────────

function AdminTerminal({
  token,
  onAuthError,
  onLogout,
}: {
  token: string;
  onAuthError: () => void;
  onLogout: () => void;
}) {
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
    const socket = getAdminSocket(token);

    socketRef.current = socket;

    const onConnect = () => {
      setConnected(true);
      push("info", "Connected to server terminal.\n");
    };
    const onDisconnect = () => {
      setConnected(false);
      setIsRunning(false);
      push("info", "Disconnected from server.\n");
    };
    const onConnectError = (err: Error) => {
      setConnected(false);
      // Socket.IO namespace middleware rejection surfaces here — treat any
      // connect_error on the admin namespace as an expired/invalid token.
      push("stderr", `\nAuthentication failed: ${err.message}\n`);
      onAuthError();
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
    socket.on("connect_error", onConnectError);
    socket.on("terminal_output", onOutput);
    socket.on("pi_status", onPiStatus);
    socket.on("pi_status_all", onPiStatusAll);

    if (socket.connected) {
      setConnected(true);
      push("info", "Connected to server terminal.\n");
    }

    return () => {
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("connect_error", onConnectError);
      socket.off("terminal_output", onOutput);
      socket.off("pi_status", onPiStatus);
      socket.off("pi_status_all", onPiStatusAll);
    };
  }, [token]);

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
        return "text-brand";
      case "stderr":
        return "text-danger";
      case "exit":
        return "text-warning/60";
      case "info":
        return "text-info";
      default:
        return "text-success";
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
            <h1 className="text-2xl font-heading font-bold text-fg">
              Admin Terminal
            </h1>
            <p className="text-fg-muted/70 text-sm mt-1">
              Run shell commands on the server or directly on each Raspberry Pi
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${connected ? "bg-success animate-pulse" : "bg-danger"}`}
              />
              <span className="text-sm text-fg-muted">
                {connected ? "Connected" : "Disconnected"}
              </span>
            </div>
            <button
              className="text-xs px-2 py-1 rounded-lg bg-surface-2/60 hover:bg-surface-2 text-fg-muted hover:text-fg border border-border"
              onClick={onLogout}
            >
              Log out
            </button>
          </div>
        </div>

        {/* Target selector */}
        <div className="p-4 bg-surface-2/60 rounded-xl border border-border">
          <p className="text-[10px] uppercase tracking-widest text-fg-muted/70 mb-3">
            Target Device
          </p>
          <div className="flex gap-3">
            {TARGETS.map((t) => {
              const online = isTargetOnline(t.id);
              const active = target === t.id;

              return (
                <button
                  key={t.id}
                  className={`flex-1 flex flex-col items-start gap-1 px-4 py-3 rounded-xl border transition-colors duration-150 ease-standard ${
                    active
                      ? "bg-brand/15 border-brand/60 text-fg"
                      : "bg-surface-2/40 border-border text-fg-muted hover:bg-surface-2/80 hover:text-fg"
                  }`}
                  onClick={() => {
                    setTarget(t.id);
                    push("info", `Switched target to ${t.label}.\n`);
                  }}
                >
                  <div className="flex items-center gap-2 w-full">
                    <span
                      className={`w-2 h-2 rounded-full flex-shrink-0 ${
                        online ? "bg-success animate-pulse" : "bg-danger/60"
                      }`}
                    />
                    <span className="font-semibold text-sm">{t.label}</span>
                    <span
                      className={`ml-auto text-[10px] px-1.5 py-0.5 rounded font-mono ${
                        online
                          ? "bg-success/15 text-success"
                          : "bg-danger/10 text-danger/60"
                      }`}
                    >
                      {online ? "ONLINE" : "OFFLINE"}
                    </span>
                  </div>
                  <span className="text-[11px] text-fg-muted/70 pl-4">
                    {t.desc}
                  </span>
                </button>
              );
            })}
          </div>
          {(target === "pi4" || target === "pi5") && !piStatus[target] && (
            <p className="mt-3 text-xs text-warning/70 pl-1">
              {target} is offline — make sure{" "}
              <code className="font-mono bg-surface-2 px-1 rounded">
                roadsentinel-agent
              </code>{" "}
              service is running on that Pi. The Pi must be able to reach the
              Node service URL.
            </p>
          )}
        </div>

        {/* Quick commands */}
        <div className="p-4 bg-surface-2/60 rounded-xl border border-border">
          <p className="text-[10px] uppercase tracking-widest text-fg-muted/70 mb-3">
            Quick Commands — {TARGETS.find((t) => t.id === target)?.label}
          </p>
          <div className="flex flex-wrap gap-2">
            {visibleCommands.map((qc) => (
              <button
                key={qc.cmd}
                className="px-3 py-1.5 text-xs rounded-lg bg-surface-2 hover:bg-surface-2/80 text-fg-muted hover:text-fg border border-border transition-colors duration-150 ease-standard disabled:opacity-30 disabled:cursor-not-allowed font-mono"
                disabled={isRunning || !connected || !isTargetOnline(target)}
                onClick={() => run(qc.cmd)}
              >
                {qc.label}
              </button>
            ))}
          </div>
        </div>

        {/* Terminal window */}
        {/* Traffic-light dots intentionally stay literal red/yellow/green —
            they mimic a real macOS terminal titlebar, not app severity chrome. */}
        <div className="rounded-xl border border-border overflow-hidden shadow-2xl bg-bg">
          {/* Title bar */}
          <div className="flex items-center gap-2 px-4 py-2.5 bg-surface border-b border-border">
            <span className="w-3 h-3 rounded-full bg-[#ff5f57]" />
            <span className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
            <span className="w-3 h-3 rounded-full bg-[#28c840]" />
            <span className="ml-3 text-xs text-fg-muted/50 font-mono">
              road-sentinel @{" "}
              <span className="text-brand">
                {TARGETS.find((t) => t.id === target)?.label ?? target}
              </span>
            </span>
            {isRunning && (
              <button
                className="ml-auto px-2.5 py-0.5 text-xs bg-danger/20 hover:bg-danger/40 text-danger rounded border border-danger/30 transition-colors duration-150 ease-standard font-mono"
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
                    className="w-1 h-1 bg-fg-muted/40 rounded-full animate-bounce"
                    style={{ animationDelay: `${d}ms` }}
                  />
                ))}
              </span>
            )}
          </div>

          {/* Input row */}
          <div className="flex items-center gap-2 px-4 py-3 border-t border-border bg-bg">
            <span className="text-brand font-mono select-none text-sm">
              [{TARGETS.find((t) => t.id === target)?.label}] $
            </span>
            <input
              ref={inputRef}
              autoComplete="off"
              className="flex-1 bg-transparent text-fg/90 font-mono text-sm outline-none placeholder:text-fg-muted/40"
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
              className="px-3 py-1 text-xs bg-brand/20 hover:bg-brand/30 text-brand rounded border border-brand/30 transition-colors duration-150 ease-standard disabled:opacity-30 disabled:cursor-not-allowed font-mono"
              disabled={isRunning || !connected || !command.trim()}
              onClick={() => run(command)}
            >
              Run ↵
            </button>
          </div>
        </div>

        {/* Tips */}
        <div className="p-4 bg-surface-2/60 rounded-xl border border-border text-xs text-fg-muted/70 space-y-1">
          <p className="text-fg-muted font-medium mb-1">
            How Pi terminal works
          </p>
          <p>
            Each Pi runs a{" "}
            <code className="font-mono bg-surface-2 px-1 rounded">
              roadsentinel-agent
            </code>{" "}
            service that connects back to this Node server&apos;s authenticated{" "}
            <code className="font-mono bg-surface-2 px-1 rounded">/admin</code>{" "}
            namespace using a shared token — no SSH or open ports needed on the
            Pi.
          </p>
          <p className="mt-2 text-fg-muted font-medium">Keyboard shortcuts</p>
          <p>
            ↑ / ↓ — navigate command history &nbsp;·&nbsp; Ctrl+C — kill process
            &nbsp;·&nbsp; Ctrl+L — clear
          </p>
        </div>
      </div>
    </div>
  );
}

// ── Page: gates the terminal behind a login ─────────────────────────────────────

export default function AdminPage() {
  const [token, setToken] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    setToken(getStoredAdminToken());
    setChecked(true);
  }, []);

  const handleLogout = useCallback(() => {
    clearStoredAdminToken();
    setToken(null);
  }, []);

  // Same handler for both: an expired/invalid token forces re-login
  const handleAuthError = handleLogout;

  if (!checked) return null;

  if (!token) {
    return <LoginForm onSuccess={setToken} />;
  }

  return (
    <AdminTerminal
      token={token}
      onAuthError={handleAuthError}
      onLogout={handleLogout}
    />
  );
}
