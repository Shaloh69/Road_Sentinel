import winston from "winston";
import dotenv from "dotenv";
import path from "path";

dotenv.config();

const logLevel = process.env.LOG_LEVEL || "info";
const isProd = process.env.NODE_ENV === "production";

// LOG_FILE names the combined log; the error log sits alongside it with an
// "error." prefix (e.g. LOG_FILE=./logs/app.log -> ./logs/error.app.log).
const combinedLogFile = process.env.LOG_FILE || "logs/combined.log";
const errorLogFile = path.join(
  path.dirname(combinedLogFile),
  `error.${path.basename(combinedLogFile)}`,
);

// Safe JSON stringify — handles circular references from axios errors etc.
function safeStringify(obj: unknown): string {
  const seen = new Set<unknown>();
  return JSON.stringify(obj, (_, value) => {
    if (typeof value === "object" && value !== null) {
      if (seen.has(value)) return "[Circular]";
      seen.add(value);
    }
    return value;
  });
}

const consoleFormat = winston.format.combine(
  winston.format.colorize(),
  winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
  winston.format.printf(({ timestamp, level, message, ...meta }) => {
    let msg = `${timestamp} [${level}]: ${message}`;
    const keys = Object.keys(meta).filter((k) => k !== "service");
    if (keys.length > 0) {
      const slim = Object.fromEntries(keys.map((k) => [k, meta[k]]));
      try {
        msg += ` ${safeStringify(slim)}`;
      } catch {
        msg += " [unserializable meta]";
      }
    }
    return msg;
  }),
);

const fileFormat = winston.format.combine(
  winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
  winston.format.errors({ stack: true }),
  winston.format.splat(),
  winston.format.printf((info) => {
    try {
      return safeStringify(info);
    } catch {
      return JSON.stringify({
        level: info.level,
        message: String(info.message),
      });
    }
  }),
);

const transports: winston.transport[] = [];

if (!isProd) {
  transports.push(new winston.transports.Console({ format: consoleFormat }));
} else {
  transports.push(
    new winston.transports.Console({ format: consoleFormat }),
    new winston.transports.File({
      filename: errorLogFile,
      level: "error",
      maxsize: 5242880,
      maxFiles: 5,
      format: fileFormat,
    }),
    new winston.transports.File({
      filename: combinedLogFile,
      maxsize: 5242880,
      maxFiles: 5,
      format: fileFormat,
    }),
  );
}

export const logger = winston.createLogger({
  level: logLevel,
  defaultMeta: { service: "road-sentinel-node" },
  transports,
});

export default logger;
