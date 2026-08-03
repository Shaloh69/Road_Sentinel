// Client-side export helpers — no backend dependency, no new packages.
// CSV: build a Blob and trigger a download.
// PDF: open a print-formatted window and invoke the browser's native
// "Save as PDF" via window.print() — a real, working PDF export without
// pulling in a PDF-generation library for what's still a Phase 1 fix.

function csvEscape(value: string | number | null | undefined): string {
  const s = value === null || value === undefined ? "" : String(value);

  if (/[",\n]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }

  return s;
}

export function downloadCsv(
  filename: string,
  headers: string[],
  rows: (string | number | null | undefined)[][],
): void {
  const lines = [headers, ...rows].map((row) => row.map(csvEscape).join(","));
  const csv = lines.join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");

  a.href = url;
  a.download = filename.endsWith(".csv") ? filename : `${filename}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export interface PrintSection {
  heading: string;
  rows: [string, string | number][];
}

export function printPdfReport(title: string, sections: PrintSection[]): void {
  const win = window.open("", "_blank", "width=800,height=1000");

  if (!win) return; // popup blocked — nothing more we can do client-side

  const sectionsHtml = sections
    .map(
      (s) => `
        <h2>${s.heading}</h2>
        <table>
          ${s.rows
            .map(
              ([k, v]) =>
                `<tr><td class="label">${k}</td><td class="value">${v}</td></tr>`,
            )
            .join("")}
        </table>
      `,
    )
    .join("");

  win.document.write(`
    <!DOCTYPE html>
    <html>
      <head>
        <title>${title}</title>
        <style>
          body { font-family: -apple-system, Segoe UI, sans-serif; color: #12151C; padding: 32px; }
          h1 { font-size: 20px; margin-bottom: 4px; border-bottom: 3px solid #F2B33D; padding-bottom: 8px; }
          .meta { color: #666; font-size: 12px; margin-bottom: 24px; margin-top: 8px; }
          h2 { font-size: 14px; margin-top: 24px; margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
          table { width: 100%; border-collapse: collapse; font-size: 13px; }
          td { padding: 4px 0; }
          .label { color: #555; }
          .value { text-align: right; font-weight: 600; }
        </style>
      </head>
      <body>
        <h1>Road Sentinel — ${title}</h1>
        <div class="meta">Generated ${new Date().toLocaleString()}</div>
        ${sectionsHtml}
      </body>
    </html>
  `);
  win.document.close();
  win.focus();
  // Give the new document a tick to finish rendering before invoking print.
  setTimeout(() => win.print(), 250);
}
