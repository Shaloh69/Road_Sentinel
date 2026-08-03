// Client-side preference storage. No settings backend exists yet (Phase 2
// scope — see ROAD_SENTINEL_REVAMP_MASTER.md), so these persist to
// localStorage rather than being decorative no-op switches.

const STORAGE_KEY = "rs_settings";

export interface Settings {
  emailNotifications: boolean;
  soundAlerts: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  emailNotifications: true,
  soundAlerts: true,
};

export function getSettings(): Settings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);

    if (!raw) return DEFAULT_SETTINGS;

    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: Settings): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

// Short beep via the Web Audio API -- no external asset needed.
export function playAlertSound(): void {
  if (typeof window === "undefined") return;
  if (!getSettings().soundAlerts) return;
  try {
    const AudioCtx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    const ctx = new AudioCtx();
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();

    oscillator.type = "sine";
    oscillator.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
    oscillator.connect(gain);
    gain.connect(ctx.destination);
    oscillator.start();
    oscillator.stop(ctx.currentTime + 0.4);
    oscillator.onended = () => ctx.close();
  } catch {
    // Audio unsupported/blocked (autoplay policy) -- fail silently.
  }
}
