# NetDiag

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Android-lightgrey.svg)]()

**High-performance network diagnostics CLI tool for latency monitoring, packet loss analysis, and connection troubleshooting.**

Differentiates between local network (WLAN/LAN) issues and ISP problems through parallel multi-target ping analysis with real-time statistics.

## Features

- **Multi-target monitoring** – Simultaneous gateway + external target pings
- **Parallel execution** – Concurrent pings via ThreadPoolExecutor
- **Real-time statistics** – Rolling min/max/avg, jitter (RFC 3550), packet loss
- **Multiple output formats** – CSV, JSON, JSON Lines (JSONL)
- **Cross-platform** – Windows, Linux, macOS, **Android (Pydroid/Termux)**
- **Zero dependencies** – Pure Python stdlib (optional: `rich` for enhanced output)
- **Configurable thresholds** – Custom latency limits, packet sizes, intervals
- **Mobile version** – TCP-based ping without root requirements

## Available Versions

| Script | Platform | Method | Root Required |
|--------|----------|--------|---------------|
| `netdiag.py` | Windows, Linux, macOS | ICMP Ping | No (Windows/macOS), sometimes (Linux) |
| `netdiag_mobile.py` | Android (Pydroid, Termux), iOS | TCP Connect | **No** |

## Installation

```bash
# Clone repository
git clone https://github.com/m0h1nd4/NetDiag.git
cd netdiag

# Optional: Install rich for enhanced terminal output (desktop only)
pip install rich

# Make executable (Linux/macOS)
chmod +x netdiag.py netdiag_mobile.py
```

### Requirements

- Python 3.8+
- No external dependencies (stdlib only)
- Optional: `rich` for colored tables and progress indicators (desktop)

### Files

| File | Description |
|------|-------------|
| `netdiag.py` | Desktop version (ICMP ping) |
| `netdiag_mobile.py` | Mobile version (TCP connect, no root) |
| `README.md` | This documentation |
| `LICENSE` | MIT License |
| `requirements.txt` | Optional dependencies |

## Quick Start

### Desktop (Windows/Linux/macOS)

```bash
# Basic monitoring (default: 1s interval, CSV output)
python netdiag.py

# Custom targets with JSON Lines output
python netdiag.py -t 8.8.8.8 1.1.1.1 9.9.9.9 -f jsonl -o network.jsonl

# High-frequency monitoring (500ms) for 1 hour
python netdiag.py -i 0.5 -d 3600 -o session.csv

# Strict threshold with statistics export
python netdiag.py --threshold 50 --export-stats stats.json -o log.csv
```

### Mobile (Android - Pydroid/Termux)

```bash
# Basic test (external targets only)
python netdiag_mobile.py

# With gateway test (get IP from WLAN settings)
python netdiag_mobile.py -g 192.168.178.1    # Fritz!Box
python netdiag_mobile.py -g 192.168.1.1      # Standard router
python netdiag_mobile.py -g 192.168.0.1      # Some ISP routers

# Log to Download folder
python netdiag_mobile.py -g 192.168.178.1 -o /storage/emulated/0/Download/wlan_log.csv

# 5 minute test session
python netdiag_mobile.py -g 192.168.1.1 -d 300 -o network_test.csv
```

## Usage

### Desktop Version (`netdiag.py`)

```
usage: netdiag [-h] [-V] [-t IP [IP ...]] [-g IP] [-i SEC] [--timeout SEC]
               [-d SEC] [--threshold MS] [-s BYTES] [-o FILE]
               [-f {csv,json,jsonl}] [-q] [--no-color] [--no-parallel]
               [--detect-gateway] [--export-stats FILE]

Network Diagnostics CLI Tool - Monitor WLAN and ISP connection quality

Targets:
  -t, --targets IP [IP ...]   External targets to ping (default: 8.8.8.8 1.1.1.1)
  -g, --gateway IP            Override auto-detected gateway IP

Timing:
  -i, --interval SEC          Interval between measurements (default: 1.0)
  --timeout SEC               Ping timeout in seconds (default: 2)
  -d, --duration SEC          Run for specified duration (default: infinite)

Thresholds:
  --threshold MS              Latency threshold for WLAN_WEAK (default: 100ms)
  -s, --packet-size BYTES     Ping packet size (default: 64)

Output:
  -o, --output FILE           Output file path
  -f, --format {csv,json,jsonl}  Output format (default: csv)
  -q, --quiet                 Suppress console output
  --no-color                  Disable colored output

Performance:
  --no-parallel               Disable parallel pings (sequential mode)

Commands:
  --detect-gateway            Detect and print gateway IP, then exit
  --export-stats FILE         Export final statistics to JSON file
```

### Mobile Version (`netdiag_mobile.py`)

```
usage: netdiag_mobile.py [-h] [-V] [-g GATEWAY] [--gateway-port GATEWAY_PORT]
                         [-i INTERVAL] [-t TIMEOUT] [--threshold THRESHOLD]
                         [-o OUTPUT] [-f {csv,jsonl}] [-d DURATION]

Network Diagnostics for Android/Pydroid (no root required)

Options:
  -g, --gateway GATEWAY       Gateway/Router IP (from WLAN settings)
  --gateway-port PORT         Gateway port to test (default: 80)
  -i, --interval SEC          Test interval in seconds (default: 1.0)
  -t, --timeout SEC           Connection timeout (default: 2.0)
  --threshold MS              Latency threshold in ms (default: 100)
  -o, --output FILE           Output file path
  -f, --format {csv,jsonl}    Output format (default: csv)
  -d, --duration SEC          Run duration in seconds
```

## Status Codes

| Code | Status | Description | Root Cause |
|------|--------|-------------|------------|
| `0` | `OK` | All connections nominal | – |
| `1` | `WLAN_WEAK` | Gateway latency > threshold | Local interference, distance, congestion |
| `2` | `WLAN_DOWN` | Gateway unreachable | AP down, DHCP issue, local network failure |
| `3` | `ISP_DOWN` | Gateway OK, external targets unreachable | ISP outage, DNS, upstream routing |

## Output Formats

### CSV (Default)

Semicolon-delimited, Excel-compatible:

```csv
timestamp;gateway_ip;gateway_latency_ms;gateway_ttl;8.8.8.8_latency_ms;8.8.8.8_ttl;1.1.1.1_latency_ms;1.1.1.1_ttl;status;status_code
2024-01-15T14:30:00.123456;192.168.1.1;2.5;64;12.3;117;11.8;57;OK;0
2024-01-15T14:30:01.234567;192.168.1.1;TIMEOUT;;TIMEOUT;;TIMEOUT;;WLAN_DOWN;2
```

### JSON Lines (JSONL)

One JSON object per line – ideal for streaming/log aggregation:

```json
{"timestamp":"2024-01-15T14:30:00.123456","gateway_ip":"192.168.1.1","gateway_latency_ms":2.5,"gateway_ttl":64,"targets":{"8.8.8.8":{"latency_ms":12.3,"ttl":117,"success":true,"error":null}},"status":"OK","status_code":0}
```

### JSON

Complete session as single JSON object with metadata:

```json
{
  "metadata": {
    "version": "1.0.0",
    "generated_at": "2024-01-15T15:00:00.000000",
    "total_samples": 3600
  },
  "results": [...]
}
```

## Statistics Export

Use `--export-stats` for detailed analysis data:

```json
{
  "metadata": {
    "version": "1.0.0",
    "generated_at": "2024-01-15T15:00:00.000000",
    "gateway": "192.168.1.1",
    "targets": ["8.8.8.8", "1.1.1.1"],
    "threshold_ms": 100
  },
  "statistics": {
    "gateway": {
      "target": "192.168.1.1",
      "samples": 3600,
      "successes": 3580,
      "failures": 20,
      "packet_loss_pct": 0.56,
      "min_latency_ms": 1.2,
      "max_latency_ms": 245.8,
      "avg_latency_ms": 3.4,
      "jitter_ms": 2.1
    }
  }
}
```

## Advanced Examples

### Long-term Monitoring with Rotation

```bash
# 24h capture with hourly files
for i in {0..23}; do
  python netdiag.py -d 3600 -o "log_$(date +%Y%m%d)_${i}.csv" -q
done
```

### CI/CD Network Health Check

```bash
# Exit code reflects worst status in session
python netdiag.py -d 60 -q --threshold 50 -f json -o /dev/null
echo "Max status code: check statistics"
```

### Multiple ISP Comparison

```bash
# Test multiple upstream providers
python netdiag.py -t 8.8.8.8 1.1.1.1 9.9.9.9 208.67.222.222 \
  -i 0.5 -f jsonl -o multi_isp.jsonl --export-stats comparison.json
```

### Systemd Service

```ini
# /etc/systemd/system/netdiag.service
[Unit]
Description=Network Diagnostics Monitor
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/netdiag/netdiag.py -o /var/log/netdiag/netdiag.csv -q
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Data Analysis

### Analysis Prompt for LLM/AI

Use this prompt with Claude, GPT-4, or similar for professional network analysis:

```
You are a senior network engineer performing root cause analysis on connection quality data.

## Input Data Format

The data is from `netdiag`, a multi-target ping monitoring tool. Format details:

- `timestamp`: ISO 8601 measurement time
- `gateway_ip`: Local router/AP address
- `gateway_latency_ms`: RTT to gateway (WLAN/LAN quality indicator)
- `gateway_ttl`: Time-to-live from gateway response
- `{target}_latency_ms`: RTT to external target (ISP path quality)
- `{target}_ttl`: TTL from target (can indicate routing changes)
- `status`: Diagnostic classification (OK, WLAN_WEAK, WLAN_DOWN, ISP_DOWN)
- `status_code`: Numeric status (0=OK, 1=WLAN_WEAK, 2=WLAN_DOWN, 3=ISP_DOWN)
- `TIMEOUT`: No response within timeout period

## Diagnostic Logic

1. **WLAN_DOWN (code 2)**: Gateway unreachable → Local network failure
2. **WLAN_WEAK (code 1)**: Gateway latency > threshold → Local congestion/interference
3. **ISP_DOWN (code 3)**: Gateway OK + all external targets timeout → Upstream failure
4. **OK (code 0)**: All targets responding within thresholds

## Analysis Requirements

Provide a structured analysis covering:

### 1. Executive Summary
- Overall connection stability (uptime percentage)
- Primary failure mode identification
- Confidence level in root cause determination

### 2. Quantitative Metrics
- Packet loss rate (gateway vs external)
- Latency distribution (P50, P95, P99 if calculable)
- Jitter analysis
- Correlation between gateway and external failures

### 3. Temporal Pattern Analysis
- Time-of-day patterns (if data spans multiple hours)
- Burst vs distributed failure patterns
- Duration of outage events
- Mean time between failures (MTBF)

### 4. Root Cause Classification
Categorize as one of:
- **L1/Physical**: Cable, antenna, hardware failure
- **L2/Data Link**: WLAN interference, AP overload, driver issues
- **L3/Network**: DHCP, ARP, routing, ISP peering
- **L4+/Transport**: DNS, congestion, throttling

### 5. Actionable Recommendations
Prioritized list with:
- Immediate mitigations
- Diagnostic commands to run
- Evidence to collect for ISP support ticket
- Hardware/configuration changes to consider

### 6. ISP Support Script
If ISP is implicated, provide a technical summary suitable for L2/L3 support escalation, including:
- Specific timestamps of outages
- Evidence that local network was stable during failures
- Request for line quality check / upstream trace

## Data

```
[PASTE DATA HERE]
```
```

### Quick Analysis with jq (JSONL)

```bash
# Count status occurrences
jq -s 'group_by(.status) | map({status: .[0].status, count: length})' log.jsonl

# Extract failure timestamps
jq -r 'select(.status_code > 0) | "\(.timestamp) \(.status)"' log.jsonl

# Calculate average gateway latency
jq -s '[.[].gateway_latency_ms | select(. != null)] | add/length' log.jsonl

# Find worst latency spikes
jq -s 'sort_by(.gateway_latency_ms) | reverse | .[0:10]' log.jsonl
```

### Python Analysis Script

```python
import pandas as pd
import json

# Load CSV
df = pd.read_csv('log.csv', sep=';', parse_dates=['timestamp'])

# Basic stats
print(f"Total samples: {len(df)}")
print(f"Uptime: {(df['status_code'] == 0).mean() * 100:.2f}%")
print(f"WLAN issues: {(df['status_code'].isin([1,2])).sum()}")
print(f"ISP issues: {(df['status_code'] == 3).sum()}")

# Latency percentiles
gateway_lat = pd.to_numeric(df['gateway_latency_ms'], errors='coerce')
print(f"\nGateway Latency P50: {gateway_lat.quantile(0.5):.1f}ms")
print(f"Gateway Latency P95: {gateway_lat.quantile(0.95):.1f}ms")
print(f"Gateway Latency P99: {gateway_lat.quantile(0.99):.1f}ms")

# Failure windows
df['is_failure'] = df['status_code'] > 0
df['failure_group'] = (df['is_failure'] != df['is_failure'].shift()).cumsum()
failures = df[df['is_failure']].groupby('failure_group').agg({
    'timestamp': ['min', 'max', 'count']
})
print(f"\nFailure events: {len(failures)}")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        NetDiag                              │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface (argparse)                                   │
│    ├── Target Configuration                                 │
│    ├── Timing Parameters                                    │
│    └── Output Selection                                     │
├─────────────────────────────────────────────────────────────┤
│  Diagnostic Engine                                          │
│    ├── NetworkUtils (platform-specific ping/gateway detect) │
│    ├── ThreadPoolExecutor (parallel ping execution)         │
│    └── Statistics (rolling metrics, jitter calculation)     │
├─────────────────────────────────────────────────────────────┤
│  Output Handlers                                            │
│    ├── CSVOutputHandler                                     │
│    ├── JSONOutputHandler                                    │
│    └── JSONLOutputHandler                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Mobile Version Details

### Why a Separate Mobile Version?

Android and iOS restrict raw socket access (ICMP ping) to root/system apps. The mobile version uses **TCP Connect** instead, which:

- ✅ Works without root privileges
- ✅ Works in Pydroid, Termux, Pythonista
- ✅ Measures real connection latency
- ⚠️ Latencies are ~1-5ms higher than ICMP (TCP handshake overhead)
- ⚠️ Gateway IP must be set manually (no auto-detection)

### How TCP Ping Works

Instead of sending ICMP echo requests, the mobile version:

1. Opens a TCP socket to the target
2. Measures time to complete the TCP handshake (SYN → SYN-ACK)
3. Closes the connection immediately

This accurately reflects network latency and availability.

### Default Test Targets

| Name | Host | Port | Purpose |
|------|------|------|---------|
| `google` | 8.8.8.8 | 53 | Google DNS (UDP/TCP) |
| `cloudflare` | 1.1.1.1 | 53 | Cloudflare DNS |
| `google_web` | 142.250.185.78 | 443 | Google HTTPS |
| `cloudflare_web` | 104.16.132.229 | 443 | Cloudflare HTTPS |

### Finding Your Gateway IP (Android)

1. **Settings** → **WLAN** / **Wi-Fi**
2. Tap your connected network name
3. Look for **Gateway**, **Router**, or **Default Gateway**

Common gateway addresses:
- `192.168.178.1` – Fritz!Box
- `192.168.1.1` – Most routers (TP-Link, Netgear, ASUS)
- `192.168.0.1` – Some ISP routers (Vodafone, Unity Media)
- `192.168.2.1` – Some Telekom routers
- `10.0.0.1` – Some enterprise/mesh networks

### Pydroid Setup

1. Install **Pydroid 3** from Play Store
2. Copy `netdiag_mobile.py` to your device
3. Open in Pydroid and run
4. For file output, use paths like:
   - `/storage/emulated/0/Download/log.csv`
   - `/sdcard/Download/log.csv`

### Termux Setup

```bash
pkg install python
python netdiag_mobile.py -g 192.168.1.1
```

### Comparison: Desktop vs Mobile

| Feature | Desktop (`netdiag.py`) | Mobile (`netdiag_mobile.py`) |
|---------|------------------------|------------------------------|
| Ping Method | ICMP | TCP Connect |
| Root Required | No* | **No** |
| Gateway Detection | Automatic | Manual (`-g` flag) |
| Custom Targets | Yes (`-t`) | Predefined set |
| Packet Size | Configurable | N/A (TCP) |
| TTL Reporting | Yes | No |
| Jitter Calculation | RFC 3550 | Basic |
| Output Formats | CSV, JSON, JSONL | CSV, JSONL |
| Statistics Export | JSON file | Console only |
| Rich Terminal | Optional | No |

\* Linux may require `cap_net_raw` capability or root for ICMP

---

## Platform Notes

### Windows
- Uses PowerShell for gateway detection
- Ping output parsing handles German/English locales
- Requires no elevation for basic operation

### Linux
- Uses `ip route` for gateway detection
- Standard ping command with POSIX options

### macOS
- Uses `route -n get default` for gateway detection
- BSD ping syntax

### Android (Pydroid/Termux)
- Use `netdiag_mobile.py` (TCP-based)
- No root required
- Gateway must be specified manually
- File paths: `/storage/emulated/0/` or `/sdcard/`

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License – see [LICENSE](LICENSE) for details.

## Acknowledgments

- Jitter calculation based on RFC 3550 (RTP)
- Inspired by `mtr`, `smokeping`, and `ping-exporter`
