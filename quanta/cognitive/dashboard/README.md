# Quanta Cognitive Cockpit — Web Dashboard

**Interactive Biomorphic Decision & Contextual Plasticity Dashboard for Quanta SDK**

The **Cognitive Cockpit** is a modern, responsive, dark-themed Web UI for monitoring the Quanta Cognitive Engine in real time. It visualizes the dual-tier biomorphic memory hierarchy (Permanent Core Anchors vs. Ephemeral Decaying Decisions), Quantum Zeno Pinning arbitration, microglial synaptic pruning audits, and closed-loop feedback verification.

---

## 1. Architectural Principles

- **100% Self-Contained (Zero-CDN)**:
  - No external fonts (`fonts.googleapis.com`), CSS stylesheets, or JavaScript frameworks are loaded over HTTP/HTTPS.
  - Gauge visualizations are rendered using native inline SVG vectors.
  - Typography utilizes system font stacks (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, ...`).
  - Completely operational offline and inside restricted/air-gapped network perimeters.

- **Dual-Mode Data Ingestion (Immune to `file://` CORS)**:
  1. **HTML5 Drag & Drop + File Input (`FileReader`)**: Drag and drop `quanta_cognitive_state.json` directly onto the dashboard or click **"Dosya Seç (JSON)"**. The file is parsed instantly via the browser's native `FileReader` API without triggering cross-origin (`CORS`) restrictions.
  2. **HTTP Live Polling**: When served over HTTP/HTTPS (e.g. `python -m http.server 8000` or via SDK server), the cockpit polls `quanta_cognitive_state.json` every 1s, 2s, 3s, 5s, or 10s with countdown and pause/resume controls.
  3. **Embedded Initial Snapshot**: The HTML file embeds an initial state snapshot (`window.EMBEDDED_INITIAL_STATE`) so the dashboard displays live, rich metrics immediately upon opening even before any file is uploaded or fetched.

---

## 2. Core Visual Components

| # | Component | Visual Elements | Cognitive Subsystem |
|---|---|---|---|
| 1 | **Top KPI Grid** | 5 Cards: Turn Count, Active Core Anchors, Active Transient Decisions, Total Pruned, Mean CSF Shield ($\kappa_{\text{csf}} = 1/6250$) | State Aggregation & PreInvocation |
| 2 | **Zeno Pinning Gauge** | 180° SVG Arc ($P_{\text{zeno}}$ vs Anti-Zeno), dynamic needle, percentage readout, status badges | Quantum Zeno Arbitration |
| 3 | **Bilinçaltı Kural Muhafızlığı** | Fidelity progress bars (`PRISTINE`, `ACTIVE`, `DECAYING`), SWR replay timestamps, dopaminergic salience tags | SWR Replay & CSF Dielectric Shield |
| 4 | **Dokunulmaz Çekirdek Çıpalar** | Glowing CSF shield badges (`🛡️ CSF KORUMALI`), confidence %, drift status, real-time search filter | Permanent Invariants ($\text{salience} \ge 2.0$) |
| 5 | **Aktif Geçici Kararlar** | **Dual Progress Bars**: (1) Fidelity Bar, (2) Turn Lifetime / Age Bar, Pruning hazard warnings | Contextual Plasticity ($\text{salience} \in [0.2, 0.8]$) |
| 6 | **Mikroglial Budama & Geri Bildirim** | Mini-KPIs (Total evaluations, Success rate, Consecutive streak, Drift status) + Pruning audit stream | Closed-Loop Outcome Feedback & Active Forgetting |
| 7 | **Kuantum Kararları & Sıralama** | 6-Qubit quantum rankings table with $\text{Tr}(\rho \Pi)$ Hilbert projections, winner indicators, and tensor latencies | Quantum Arbiter |

---

## 3. Usage & Execution

### A. Open Directly in Browser (Local File Mode)
You can open the dashboard directly using your default web browser:
```bash
open dashboard.html
# or
open quanta/cognitive/dashboard/index.html
```
- The dashboard loads with the embedded state snapshot immediately.
- Drag and drop your project's `quanta_cognitive_state.json` onto the dashboard banner anytime to view the latest state.

### B. Generate from CLI
Update or generate the dashboard using the Quanta CLI:
```bash
# Generate dashboard
quanta monitor --dashboard

# Generate to a custom output path
quanta monitor --dashboard --output my_dashboard.html
```

### C. Live HTTP Server Mode
Serve the dashboard locally with continuous HTTP polling:
```bash
python3 -m http.server 8000
```
Open `http://localhost:8000/dashboard.html` in your browser. The dashboard automatically polls `quanta_cognitive_state.json` every 3 seconds (configurable in the UI).

---

## 4. File Locations

- Canonical Package Asset: `quanta/cognitive/dashboard/index.html`
- Workspace Root Convenience Copy: `dashboard.html`
- MkDocs Documentation Mirror: `site/cognitive_cockpit.html`
- Generator Backend: `quanta/cognitive/telemetry.py:generate_dashboard_html()`
