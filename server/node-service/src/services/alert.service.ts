import axios from "axios";
import { logger } from "../config/logger";
import { Incident } from "../types";

// Generic webhook alert hook (Phase 2 "new feature"). Deliberately provider-
// agnostic — POSTs a plain JSON payload to ALERT_WEBHOOK_URL, which works
// as-is with Slack/Discord incoming webhooks, Zapier, IFTTT, n8n, or a custom
// endpoint. No SMTP/SMS credentials are assumed or required; wiring a
// specific provider (e.g. actually sending email) is a follow-up someone can
// build on top of this once they have credentials for one.

const SEVERITY_RANK: Record<string, number> = {
  low: 0,
  medium: 1,
  high: 2,
  critical: 3,
};

function minSeverityRank(): number {
  const configured = (
    process.env.ALERT_WEBHOOK_MIN_SEVERITY || "critical"
  ).toLowerCase();
  return SEVERITY_RANK[configured] ?? SEVERITY_RANK.critical;
}

export async function notifyIncident(incident: Incident): Promise<void> {
  const webhookUrl = process.env.ALERT_WEBHOOK_URL;
  if (!webhookUrl) return; // not configured — silent no-op, not an error

  const rank = SEVERITY_RANK[incident.severity] ?? 0;
  if (rank < minSeverityRank()) return;

  try {
    await axios.post(
      webhookUrl,
      {
        // Slack/Discord-compatible top-level "text" field, plus the full
        // structured incident for endpoints that want to parse it.
        text: `🚨 [${incident.severity.toUpperCase()}] ${incident.title} — ${incident.camera_id}`,
        incident,
      },
      { timeout: 5000 },
    );
    logger.info(
      `Alert webhook notified for incident #${incident.id} (${incident.severity})`,
    );
  } catch (err) {
    // Never let a broken webhook affect incident logging — log and move on.
    const msg = err instanceof Error ? err.message : String(err);
    logger.warn(`Alert webhook delivery failed: ${msg}`);
  }
}
