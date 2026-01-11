#!/usr/bin/env python3
"""
NetDiag - Network Diagnostics CLI Tool
Professional WLAN/Network latency monitoring and analysis tool.

Author: [Your Name]
License: MIT
"""

import argparse
import csv
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: Rich for better terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

__version__ = "1.0.0"

# --- Data Classes ---

@dataclass
class PingResult:
    """Single ping measurement result."""
    timestamp: str
    target: str
    target_name: str
    latency_ms: Optional[float]
    ttl: Optional[int]
    packet_size: int
    success: bool
    error: Optional[str] = None

@dataclass
class DiagnosticResult:
    """Complete diagnostic cycle result."""
    timestamp: str
    gateway_ip: str
    gateway_latency_ms: Optional[float]
    gateway_ttl: Optional[int]
    targets: Dict[str, Dict[str, Any]]
    status: str
    status_code: int  # 0=OK, 1=WLAN_WEAK, 2=WLAN_DOWN, 3=ISP_DOWN
    
@dataclass
class Statistics:
    """Rolling statistics for a target."""
    target: str
    samples: int = 0
    successes: int = 0
    failures: int = 0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    avg_latency: float = 0.0
    jitter: float = 0.0
    packet_loss_pct: float = 0.0
    _latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    _last_latency: float = 0.0
    _jitter_sum: float = 0.0
    
    def update(self, latency: Optional[float]):
        self.samples += 1
        if latency is not None:
            self.successes += 1
            self._latencies.append(latency)
            self.min_latency = min(self.min_latency, latency)
            self.max_latency = max(self.max_latency, latency)
            self.avg_latency = sum(self._latencies) / len(self._latencies)
            
            # RFC 3550 jitter calculation
            if self._last_latency > 0:
                diff = abs(latency - self._last_latency)
                self._jitter_sum += (diff - self._jitter_sum) / 16
                self.jitter = self._jitter_sum
            self._last_latency = latency
        else:
            self.failures += 1
        
        self.packet_loss_pct = (self.failures / self.samples) * 100 if self.samples > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "samples": self.samples,
            "successes": self.successes,
            "failures": self.failures,
            "packet_loss_pct": round(self.packet_loss_pct, 2),
            "min_latency_ms": round(self.min_latency, 2) if self.min_latency != float('inf') else None,
            "max_latency_ms": round(self.max_latency, 2) if self.max_latency > 0 else None,
            "avg_latency_ms": round(self.avg_latency, 2) if self.avg_latency > 0 else None,
            "jitter_ms": round(self.jitter, 2)
        }


# --- Network Utilities ---

class NetworkUtils:
    """Platform-independent network utilities."""
    
    @staticmethod
    def get_default_gateway() -> Optional[str]:
        """Detect default gateway IP address."""
        system = platform.system().lower()
        
        try:
            if system == "windows":
                output = subprocess.check_output(
                    ["powershell", "-Command", 
                     "(Get-NetRoute | Where-Object { $_.DestinationPrefix -eq '0.0.0.0/0' -and $_.NextHop -ne '0.0.0.0' } | Select-Object -First 1).NextHop"],
                    text=True, stderr=subprocess.DEVNULL
                ).strip()
                return output if output else None
                
            elif system == "linux":
                output = subprocess.check_output(
                    ["ip", "route", "show", "default"],
                    text=True, stderr=subprocess.DEVNULL
                )
                match = re.search(r'default via (\d+\.\d+\.\d+\.\d+)', output)
                return match.group(1) if match else None
                
            elif system == "darwin":  # macOS
                output = subprocess.check_output(
                    ["route", "-n", "get", "default"],
                    text=True, stderr=subprocess.DEVNULL
                )
                match = re.search(r'gateway: (\d+\.\d+\.\d+\.\d+)', output)
                return match.group(1) if match else None
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return None
    
    @staticmethod
    def ping(target: str, timeout: int = 2, packet_size: int = 64) -> PingResult:
        """Execute ping and parse results."""
        system = platform.system().lower()
        timestamp = datetime.now().isoformat()
        
        try:
            if system == "windows":
                cmd = ["ping", "-n", "1", "-w", str(timeout * 1000), "-l", str(packet_size), target]
                pattern_time = r'Zeit[=<](\d+)ms|time[=<](\d+)ms'
                pattern_ttl = r'TTL=(\d+)'
            else:
                cmd = ["ping", "-c", "1", "-W", str(timeout), "-s", str(packet_size), target]
                pattern_time = r'time=(\d+\.?\d*)\s*ms'
                pattern_ttl = r'ttl=(\d+)'
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout + 1
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                # Extract latency
                time_match = re.search(pattern_time, output, re.IGNORECASE)
                if time_match:
                    latency = float(time_match.group(1) or time_match.group(2))
                else:
                    latency = None
                
                # Extract TTL
                ttl_match = re.search(pattern_ttl, output, re.IGNORECASE)
                ttl = int(ttl_match.group(1)) if ttl_match else None
                
                return PingResult(
                    timestamp=timestamp,
                    target=target,
                    target_name=target,
                    latency_ms=latency,
                    ttl=ttl,
                    packet_size=packet_size,
                    success=True
                )
            else:
                return PingResult(
                    timestamp=timestamp,
                    target=target,
                    target_name=target,
                    latency_ms=None,
                    ttl=None,
                    packet_size=packet_size,
                    success=False,
                    error="No response"
                )
                
        except subprocess.TimeoutExpired:
            return PingResult(
                timestamp=timestamp,
                target=target,
                target_name=target,
                latency_ms=None,
                ttl=None,
                packet_size=packet_size,
                success=False,
                error="Timeout"
            )
        except Exception as e:
            return PingResult(
                timestamp=timestamp,
                target=target,
                target_name=target,
                latency_ms=None,
                ttl=None,
                packet_size=packet_size,
                success=False,
                error=str(e)
            )
    
    @staticmethod
    def resolve_hostname(hostname: str) -> Optional[str]:
        """Resolve hostname to IP address."""
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            return None


# --- Output Handlers ---

class OutputHandler:
    """Base class for output handlers."""
    
    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = filepath
        self.file = None
    
    def open(self):
        if self.filepath:
            self.file = open(self.filepath, 'a', encoding='utf-8', newline='')
    
    def close(self):
        if self.file:
            self.file.close()
    
    def write(self, result: DiagnosticResult):
        raise NotImplementedError
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()


class CSVOutputHandler(OutputHandler):
    """CSV output handler."""
    
    def __init__(self, filepath: Optional[Path] = None, targets: List[str] = None):
        super().__init__(filepath)
        self.targets = targets or []
        self.writer = None
        self.header_written = False
    
    def open(self):
        super().open()
        if self.file:
            self.writer = csv.writer(self.file, delimiter=';')
            # Check if file is empty (new file)
            if self.filepath and self.filepath.stat().st_size == 0:
                self._write_header()
    
    def _write_header(self):
        headers = ["timestamp", "gateway_ip", "gateway_latency_ms", "gateway_ttl"]
        for target in self.targets:
            headers.extend([f"{target}_latency_ms", f"{target}_ttl"])
        headers.extend(["status", "status_code"])
        self.writer.writerow(headers)
        self.file.flush()
        self.header_written = True
    
    def write(self, result: DiagnosticResult):
        if not self.writer:
            return
        
        if not self.header_written:
            self._write_header()
        
        row = [
            result.timestamp,
            result.gateway_ip,
            result.gateway_latency_ms if result.gateway_latency_ms else "TIMEOUT",
            result.gateway_ttl or ""
        ]
        
        for target in self.targets:
            if target in result.targets:
                t = result.targets[target]
                row.append(t.get("latency_ms") if t.get("latency_ms") else "TIMEOUT")
                row.append(t.get("ttl", ""))
            else:
                row.extend(["", ""])
        
        row.extend([result.status, result.status_code])
        self.writer.writerow(row)
        self.file.flush()


class JSONLOutputHandler(OutputHandler):
    """JSON Lines output handler."""
    
    def write(self, result: DiagnosticResult):
        if not self.file:
            return
        self.file.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
        self.file.flush()


class JSONOutputHandler(OutputHandler):
    """JSON array output handler (keeps all results in memory)."""
    
    def __init__(self, filepath: Optional[Path] = None):
        super().__init__(filepath)
        self.results: List[dict] = []
    
    def write(self, result: DiagnosticResult):
        self.results.append(asdict(result))
    
    def close(self):
        if self.filepath and self.results:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "version": __version__,
                        "generated_at": datetime.now().isoformat(),
                        "total_samples": len(self.results)
                    },
                    "results": self.results
                }, f, indent=2, ensure_ascii=False)
        super().close()


# --- Main Diagnostic Engine ---

class NetDiag:
    """Main network diagnostic engine."""
    
    STATUS_OK = 0
    STATUS_WLAN_WEAK = 1
    STATUS_WLAN_DOWN = 2
    STATUS_ISP_DOWN = 3
    
    STATUS_MESSAGES = {
        STATUS_OK: "OK",
        STATUS_WLAN_WEAK: "WLAN_WEAK",
        STATUS_WLAN_DOWN: "WLAN_DOWN",
        STATUS_ISP_DOWN: "ISP_DOWN"
    }
    
    def __init__(
        self,
        targets: List[str] = None,
        interval: float = 1.0,
        timeout: int = 2,
        packet_size: int = 64,
        latency_threshold: int = 100,
        output_format: str = "csv",
        output_file: Optional[Path] = None,
        quiet: bool = False,
        no_color: bool = False,
        parallel: bool = True
    ):
        self.targets = targets or ["8.8.8.8", "1.1.1.1"]
        self.interval = interval
        self.timeout = timeout
        self.packet_size = packet_size
        self.latency_threshold = latency_threshold
        self.output_format = output_format
        self.output_file = output_file
        self.quiet = quiet
        self.no_color = no_color
        self.parallel = parallel
        
        self.gateway = NetworkUtils.get_default_gateway()
        self.running = False
        self.stats: Dict[str, Statistics] = {}
        self.output_handler: Optional[OutputHandler] = None
        
        # Initialize statistics
        if self.gateway:
            self.stats["gateway"] = Statistics(target=self.gateway)
        for target in self.targets:
            self.stats[target] = Statistics(target=target)
    
    def _create_output_handler(self) -> OutputHandler:
        """Create appropriate output handler based on format."""
        if self.output_format == "csv":
            return CSVOutputHandler(self.output_file, self.targets)
        elif self.output_format == "jsonl":
            return JSONLOutputHandler(self.output_file)
        elif self.output_format == "json":
            return JSONOutputHandler(self.output_file)
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")
    
    def _diagnose_status(
        self, 
        gateway_result: Optional[PingResult], 
        target_results: Dict[str, PingResult]
    ) -> tuple[str, int]:
        """Determine overall connection status."""
        
        # Check gateway first
        if gateway_result is None or not gateway_result.success:
            return self.STATUS_MESSAGES[self.STATUS_WLAN_DOWN], self.STATUS_WLAN_DOWN
        
        if gateway_result.latency_ms and gateway_result.latency_ms > self.latency_threshold:
            return self.STATUS_MESSAGES[self.STATUS_WLAN_WEAK], self.STATUS_WLAN_WEAK
        
        # Check external targets
        all_failed = all(not r.success for r in target_results.values())
        if all_failed and target_results:
            return self.STATUS_MESSAGES[self.STATUS_ISP_DOWN], self.STATUS_ISP_DOWN
        
        return self.STATUS_MESSAGES[self.STATUS_OK], self.STATUS_OK
    
    def _run_cycle(self) -> DiagnosticResult:
        """Run one diagnostic cycle."""
        timestamp = datetime.now().isoformat()
        
        # Ping gateway
        gateway_result = None
        if self.gateway:
            gateway_result = NetworkUtils.ping(
                self.gateway, self.timeout, self.packet_size
            )
            self.stats["gateway"].update(gateway_result.latency_ms)
        
        # Ping targets (parallel or sequential)
        target_results: Dict[str, PingResult] = {}
        
        if self.parallel and len(self.targets) > 1:
            with ThreadPoolExecutor(max_workers=len(self.targets)) as executor:
                futures = {
                    executor.submit(
                        NetworkUtils.ping, target, self.timeout, self.packet_size
                    ): target for target in self.targets
                }
                for future in as_completed(futures):
                    target = futures[future]
                    result = future.result()
                    target_results[target] = result
                    self.stats[target].update(result.latency_ms)
        else:
            for target in self.targets:
                result = NetworkUtils.ping(target, self.timeout, self.packet_size)
                target_results[target] = result
                self.stats[target].update(result.latency_ms)
        
        # Determine status
        status, status_code = self._diagnose_status(gateway_result, target_results)
        
        # Build result
        targets_dict = {}
        for target, result in target_results.items():
            targets_dict[target] = {
                "latency_ms": result.latency_ms,
                "ttl": result.ttl,
                "success": result.success,
                "error": result.error
            }
        
        return DiagnosticResult(
            timestamp=timestamp,
            gateway_ip=self.gateway or "N/A",
            gateway_latency_ms=gateway_result.latency_ms if gateway_result else None,
            gateway_ttl=gateway_result.ttl if gateway_result else None,
            targets=targets_dict,
            status=status,
            status_code=status_code
        )
    
    def _print_result(self, result: DiagnosticResult):
        """Print result to console."""
        if self.quiet:
            return
        
        # Color codes
        if not self.no_color and sys.stdout.isatty():
            GREEN = "\033[92m"
            RED = "\033[91m"
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            BOLD = "\033[1m"
        else:
            GREEN = RED = YELLOW = RESET = BOLD = ""
        
        # Status color
        if result.status_code == self.STATUS_OK:
            status_color = GREEN
        elif result.status_code == self.STATUS_WLAN_WEAK:
            status_color = YELLOW
        else:
            status_color = RED
        
        # Format gateway
        gw_str = f"{result.gateway_latency_ms:.1f}ms" if result.gateway_latency_ms else "TIMEOUT"
        
        # Format targets
        targets_str = " | ".join([
            f"{t}: {d['latency_ms']:.1f}ms" if d['latency_ms'] else f"{t}: TIMEOUT"
            for t, d in result.targets.items()
        ])
        
        ts = result.timestamp.split('T')[1].split('.')[0]  # Just time
        print(f"{ts} | GW: {gw_str} | {targets_str} | {status_color}{BOLD}{result.status}{RESET}")
    
    def _print_header(self):
        """Print startup header."""
        if self.quiet:
            return
        
        print(f"\n{'='*60}")
        print(f"  NetDiag v{__version__} - Network Diagnostics Tool")
        print(f"{'='*60}")
        print(f"  Gateway:    {self.gateway or 'Not detected'}")
        print(f"  Targets:    {', '.join(self.targets)}")
        print(f"  Interval:   {self.interval}s")
        print(f"  Threshold:  {self.latency_threshold}ms")
        print(f"  Output:     {self.output_file or 'stdout only'}")
        print(f"  Format:     {self.output_format}")
        print(f"{'='*60}")
        print("  Press Ctrl+C to stop and show statistics\n")
    
    def _print_statistics(self):
        """Print final statistics."""
        if self.quiet:
            return
        
        print(f"\n{'='*60}")
        print("  STATISTICS")
        print(f"{'='*60}\n")
        
        if RICH_AVAILABLE and not self.no_color:
            table = Table(title="Connection Statistics")
            table.add_column("Target", style="cyan")
            table.add_column("Samples", justify="right")
            table.add_column("Loss %", justify="right")
            table.add_column("Min ms", justify="right")
            table.add_column("Avg ms", justify="right")
            table.add_column("Max ms", justify="right")
            table.add_column("Jitter ms", justify="right")
            
            for name, stat in self.stats.items():
                s = stat.to_dict()
                loss_style = "red" if s["packet_loss_pct"] > 5 else "green"
                table.add_row(
                    name,
                    str(s["samples"]),
                    f"[{loss_style}]{s['packet_loss_pct']}%[/]",
                    str(s["min_latency_ms"] or "-"),
                    str(s["avg_latency_ms"] or "-"),
                    str(s["max_latency_ms"] or "-"),
                    str(s["jitter_ms"])
                )
            
            console.print(table)
        else:
            print(f"{'Target':<20} {'Samples':>8} {'Loss%':>8} {'Min':>8} {'Avg':>8} {'Max':>8} {'Jitter':>8}")
            print("-" * 76)
            for name, stat in self.stats.items():
                s = stat.to_dict()
                print(f"{name:<20} {s['samples']:>8} {s['packet_loss_pct']:>7}% "
                      f"{s['min_latency_ms'] or '-':>8} {s['avg_latency_ms'] or '-':>8} "
                      f"{s['max_latency_ms'] or '-':>8} {s['jitter_ms']:>8}")
        
        print()
    
    def get_statistics_dict(self) -> dict:
        """Get statistics as dictionary (for JSON export)."""
        return {name: stat.to_dict() for name, stat in self.stats.items()}
    
    def run(self, duration: Optional[int] = None):
        """Run diagnostic loop."""
        self.running = True
        self._print_header()
        
        start_time = time.time()
        
        with self._create_output_handler() as handler:
            self.output_handler = handler
            
            try:
                while self.running:
                    cycle_start = time.time()
                    
                    result = self._run_cycle()
                    handler.write(result)
                    self._print_result(result)
                    
                    # Check duration limit
                    if duration and (time.time() - start_time) >= duration:
                        break
                    
                    # Sleep for remaining interval
                    elapsed = time.time() - cycle_start
                    sleep_time = max(0, self.interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
            except KeyboardInterrupt:
                pass
            finally:
                self.running = False
                self._print_statistics()
    
    def stop(self):
        """Stop the diagnostic loop."""
        self.running = False


# --- CLI Interface ---

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="netdiag",
        description="Network Diagnostics CLI Tool - Monitor WLAN and ISP connection quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Basic monitoring with defaults
  %(prog)s -t 1.1.1.1 9.9.9.9           # Custom targets
  %(prog)s -i 0.5 -o log.csv            # 500ms interval, save to CSV
  %(prog)s -f jsonl -o log.jsonl        # JSON Lines output
  %(prog)s --duration 3600              # Run for 1 hour
  %(prog)s --threshold 50 --no-parallel # Strict threshold, sequential pings

Status Codes:
  0 = OK           All connections working normally
  1 = WLAN_WEAK    Gateway latency above threshold
  2 = WLAN_DOWN    Cannot reach gateway (local network issue)
  3 = ISP_DOWN     Gateway OK but cannot reach external targets
        """
    )
    
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    # Target configuration
    target_group = parser.add_argument_group("Targets")
    target_group.add_argument(
        "-t", "--targets",
        nargs="+",
        default=["8.8.8.8", "1.1.1.1"],
        metavar="IP",
        help="External targets to ping (default: 8.8.8.8 1.1.1.1)"
    )
    target_group.add_argument(
        "-g", "--gateway",
        metavar="IP",
        help="Override auto-detected gateway IP"
    )
    
    # Timing configuration
    timing_group = parser.add_argument_group("Timing")
    timing_group.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Interval between measurements in seconds (default: 1.0)"
    )
    timing_group.add_argument(
        "--timeout",
        type=int,
        default=2,
        metavar="SEC",
        help="Ping timeout in seconds (default: 2)"
    )
    timing_group.add_argument(
        "-d", "--duration",
        type=int,
        metavar="SEC",
        help="Run for specified duration (default: infinite)"
    )
    
    # Thresholds
    threshold_group = parser.add_argument_group("Thresholds")
    threshold_group.add_argument(
        "--threshold",
        type=int,
        default=100,
        metavar="MS",
        help="Latency threshold for WLAN_WEAK status (default: 100ms)"
    )
    threshold_group.add_argument(
        "-s", "--packet-size",
        type=int,
        default=64,
        metavar="BYTES",
        help="Ping packet size (default: 64)"
    )
    
    # Output configuration
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output",
        type=Path,
        metavar="FILE",
        help="Output file path"
    )
    output_group.add_argument(
        "-f", "--format",
        choices=["csv", "json", "jsonl"],
        default="csv",
        help="Output format (default: csv)"
    )
    output_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress console output (except errors)"
    )
    output_group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    # Performance
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel pings (sequential mode)"
    )
    
    # Special commands
    cmd_group = parser.add_argument_group("Commands")
    cmd_group.add_argument(
        "--detect-gateway",
        action="store_true",
        help="Detect and print gateway IP, then exit"
    )
    cmd_group.add_argument(
        "--export-stats",
        type=Path,
        metavar="FILE",
        help="Export final statistics to JSON file"
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle special commands
    if args.detect_gateway:
        gateway = NetworkUtils.get_default_gateway()
        if gateway:
            print(gateway)
            sys.exit(0)
        else:
            print("Could not detect gateway", file=sys.stderr)
            sys.exit(1)
    
    # Create and configure diagnostics
    diag = NetDiag(
        targets=args.targets,
        interval=args.interval,
        timeout=args.timeout,
        packet_size=args.packet_size,
        latency_threshold=args.threshold,
        output_format=args.format,
        output_file=args.output,
        quiet=args.quiet,
        no_color=args.no_color,
        parallel=not args.no_parallel
    )
    
    # Override gateway if specified
    if args.gateway:
        diag.gateway = args.gateway
        diag.stats["gateway"] = Statistics(target=args.gateway)
    
    # Check if gateway was detected
    if not diag.gateway:
        print("Warning: Could not detect gateway. Local network tests will be skipped.", 
              file=sys.stderr)
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        diag.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run diagnostics
    try:
        diag.run(duration=args.duration)
    finally:
        # Export statistics if requested
        if args.export_stats:
            stats = diag.get_statistics_dict()
            with open(args.export_stats, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "version": __version__,
                        "generated_at": datetime.now().isoformat(),
                        "gateway": diag.gateway,
                        "targets": diag.targets,
                        "threshold_ms": diag.latency_threshold
                    },
                    "statistics": stats
                }, f, indent=2)
            print(f"\nStatistics exported to: {args.export_stats}")


if __name__ == "__main__":
    main()
