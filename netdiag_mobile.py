#!/usr/bin/env python3
"""
NetDiag Mobile - Network Diagnostics for Pydroid/Android
Uses TCP connect instead of ICMP ping (no root required).

Author: m0h1nd4
License: MIT
"""

import argparse
import csv
import json
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

__version__ = "1.1.0-mobile"

# --- Data Classes ---

@dataclass
class PingResult:
    timestamp: str
    target: str
    port: int
    latency_ms: Optional[float]
    success: bool
    error: Optional[str] = None

@dataclass 
class Statistics:
    target: str
    samples: int = 0
    successes: int = 0
    failures: int = 0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    avg_latency: float = 0.0
    packet_loss_pct: float = 0.0
    _latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, latency: Optional[float]):
        self.samples += 1
        if latency is not None:
            self.successes += 1
            self._latencies.append(latency)
            self.min_latency = min(self.min_latency, latency)
            self.max_latency = max(self.max_latency, latency)
            self.avg_latency = sum(self._latencies) / len(self._latencies)
        else:
            self.failures += 1
        self.packet_loss_pct = (self.failures / self.samples) * 100 if self.samples > 0 else 0.0

# --- TCP Ping (No Root Required) ---

def tcp_ping(host: str, port: int = 443, timeout: float = 2.0) -> PingResult:
    """
    TCP connect ping - measures time to establish TCP connection.
    Works without root privileges on Android/Pydroid.
    """
    timestamp = datetime.now().isoformat()
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        start = time.perf_counter()
        sock.connect((host, port))
        end = time.perf_counter()
        
        sock.close()
        
        latency = (end - start) * 1000  # ms
        
        return PingResult(
            timestamp=timestamp,
            target=host,
            port=port,
            latency_ms=round(latency, 2),
            success=True
        )
        
    except socket.timeout:
        return PingResult(
            timestamp=timestamp,
            target=host,
            port=port,
            latency_ms=None,
            success=False,
            error="Timeout"
        )
    except socket.error as e:
        return PingResult(
            timestamp=timestamp,
            target=host,
            port=port,
            latency_ms=None,
            success=False,
            error=str(e)
        )
    except Exception as e:
        return PingResult(
            timestamp=timestamp,
            target=host,
            port=port,
            latency_ms=None,
            success=False,
            error=str(e)
        )

# --- Main Class ---

class NetDiagMobile:
    """Mobile-optimized network diagnostics."""
    
    # Default targets with their TCP ports
    DEFAULT_TARGETS = {
        "google": ("8.8.8.8", 53),        # Google DNS
        "cloudflare": ("1.1.1.1", 53),    # Cloudflare DNS  
        "google_web": ("142.250.185.78", 443),  # google.com
        "cloudflare_web": ("104.16.132.229", 443),  # cloudflare.com
    }
    
    STATUS_OK = 0
    STATUS_SLOW = 1
    STATUS_DOWN = 2
    
    def __init__(
        self,
        gateway: Optional[str] = None,
        gateway_port: int = 80,
        targets: Dict[str, tuple] = None,
        interval: float = 1.0,
        timeout: float = 2.0,
        threshold: int = 100,
        output_file: Optional[str] = None,
        output_format: str = "csv"
    ):
        self.gateway = gateway
        self.gateway_port = gateway_port
        self.targets = targets or self.DEFAULT_TARGETS
        self.interval = interval
        self.timeout = timeout
        self.threshold = threshold
        self.output_file = output_file
        self.output_format = output_format
        
        self.running = False
        self.stats: Dict[str, Statistics] = {}
        
        # Init stats
        if self.gateway:
            self.stats["gateway"] = Statistics(target=self.gateway)
        for name in self.targets:
            self.stats[name] = Statistics(target=name)
    
    def _determine_status(
        self, 
        gw_result: Optional[PingResult],
        target_results: Dict[str, PingResult]
    ) -> tuple[str, int]:
        """Determine connection status."""
        
        # Check gateway first (if configured)
        if self.gateway:
            if gw_result is None or not gw_result.success:
                return "WLAN_DOWN", self.STATUS_DOWN
            if gw_result.latency_ms and gw_result.latency_ms > self.threshold:
                return "WLAN_SLOW", self.STATUS_SLOW
        
        # Check external targets
        successful = sum(1 for r in target_results.values() if r.success)
        total = len(target_results)
        
        if successful == 0:
            return "INTERNET_DOWN", self.STATUS_DOWN
        elif successful < total:
            return "PARTIAL", self.STATUS_SLOW
        
        # Check latencies
        high_latency = any(
            r.latency_ms and r.latency_ms > self.threshold 
            for r in target_results.values() if r.success
        )
        if high_latency:
            return "SLOW", self.STATUS_SLOW
            
        return "OK", self.STATUS_OK
    
    def _run_cycle(self) -> dict:
        """Run one measurement cycle."""
        timestamp = datetime.now().isoformat()
        results = {"timestamp": timestamp}
        
        # Test gateway (if configured)
        gw_result = None
        if self.gateway:
            gw_result = tcp_ping(self.gateway, self.gateway_port, self.timeout)
            self.stats["gateway"].update(gw_result.latency_ms)
            results["gateway"] = {
                "ip": self.gateway,
                "port": self.gateway_port,
                "latency_ms": gw_result.latency_ms,
                "success": gw_result.success
            }
        
        # Test external targets (parallel)
        target_results: Dict[str, PingResult] = {}
        
        with ThreadPoolExecutor(max_workers=len(self.targets)) as executor:
            futures = {
                executor.submit(tcp_ping, host, port, self.timeout): name
                for name, (host, port) in self.targets.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                result = future.result()
                target_results[name] = result
                self.stats[name].update(result.latency_ms)
        
        # Add target results
        results["targets"] = {}
        for name, result in target_results.items():
            host, port = self.targets[name]
            results["targets"][name] = {
                "host": host,
                "port": port,
                "latency_ms": result.latency_ms,
                "success": result.success
            }
        
        # Determine status
        status, code = self._determine_status(gw_result, target_results)
        results["status"] = status
        results["status_code"] = code
        
        return results
    
    def _print_result(self, result: dict):
        """Print result to console."""
        ts = result["timestamp"].split("T")[1].split(".")[0]
        
        # Gateway
        if "gateway" in result:
            gw = result["gateway"]
            gw_str = f"{gw['latency_ms']}ms" if gw['latency_ms'] else "TIMEOUT"
        else:
            gw_str = "N/A"
        
        # Targets
        targets_str = " | ".join([
            f"{name}: {data['latency_ms']}ms" if data['latency_ms'] else f"{name}: FAIL"
            for name, data in result["targets"].items()
        ])
        
        # Status with color (ANSI)
        status = result["status"]
        code = result["status_code"]
        
        if code == 0:
            color = "\033[92m"  # Green
        elif code == 1:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[91m"  # Red
        reset = "\033[0m"
        
        print(f"{ts} | GW: {gw_str} | {targets_str} | {color}{status}{reset}")
    
    def _write_csv(self, file, result: dict, write_header: bool = False):
        """Write result to CSV."""
        if write_header:
            headers = ["timestamp"]
            if self.gateway:
                headers.append("gateway_latency_ms")
            for name in self.targets:
                headers.append(f"{name}_latency_ms")
            headers.extend(["status", "status_code"])
            file.write(";".join(headers) + "\n")
        
        row = [result["timestamp"]]
        if self.gateway:
            gw_lat = result.get("gateway", {}).get("latency_ms")
            row.append(str(gw_lat) if gw_lat else "TIMEOUT")
        for name in self.targets:
            lat = result["targets"].get(name, {}).get("latency_ms")
            row.append(str(lat) if lat else "TIMEOUT")
        row.extend([result["status"], str(result["status_code"])])
        file.write(";".join(row) + "\n")
        file.flush()
    
    def _print_stats(self):
        """Print final statistics."""
        print(f"\n{'='*60}")
        print("  STATISTICS")
        print(f"{'='*60}\n")
        
        print(f"{'Target':<20} {'Samples':>8} {'Loss%':>8} {'Min':>8} {'Avg':>8} {'Max':>8}")
        print("-" * 68)
        
        for name, stat in self.stats.items():
            min_l = f"{stat.min_latency:.1f}" if stat.min_latency != float('inf') else "-"
            max_l = f"{stat.max_latency:.1f}" if stat.max_latency > 0 else "-"
            avg_l = f"{stat.avg_latency:.1f}" if stat.avg_latency > 0 else "-"
            
            print(f"{name:<20} {stat.samples:>8} {stat.packet_loss_pct:>7.1f}% {min_l:>8} {avg_l:>8} {max_l:>8}")
        
        print()
    
    def run(self, duration: Optional[int] = None):
        """Run diagnostic loop."""
        self.running = True
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"  NetDiag Mobile v{__version__}")
        print(f"{'='*60}")
        print(f"  Gateway:    {self.gateway or 'Not configured (use -g)'}")
        print(f"  Targets:    {', '.join(self.targets.keys())}")
        print(f"  Interval:   {self.interval}s")
        print(f"  Method:     TCP Connect (no root needed)")
        print(f"{'='*60}")
        print("  Press Ctrl+C to stop\n")
        
        file = None
        write_header = True
        
        if self.output_file:
            file = open(self.output_file, "a", encoding="utf-8")
            write_header = file.tell() == 0
        
        try:
            while self.running:
                cycle_start = time.time()
                
                result = self._run_cycle()
                self._print_result(result)
                
                if file:
                    if self.output_format == "csv":
                        self._write_csv(file, result, write_header)
                        write_header = False
                    else:  # jsonl
                        file.write(json.dumps(result) + "\n")
                        file.flush()
                
                # Duration check
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Sleep
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if file:
                file.close()
            self._print_stats()


def main():
    parser = argparse.ArgumentParser(
        description="NetDiag Mobile - Network diagnostics for Android/Pydroid (no root)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python netdiag_mobile.py                    # Basic test (external only)
  python netdiag_mobile.py -g 192.168.1.1     # With gateway/router test
  python netdiag_mobile.py -g 192.168.178.1   # Fritz!Box gateway
  python netdiag_mobile.py -o log.csv -d 300  # Log for 5 minutes

How to find your gateway IP:
  Android: Settings → WLAN → [Your Network] → Gateway
  Usually: 192.168.1.1, 192.168.0.1, 192.168.178.1 (Fritz!Box)
        """
    )
    
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    
    parser.add_argument(
        "-g", "--gateway",
        help="Gateway/Router IP (find in WLAN settings)"
    )
    parser.add_argument(
        "--gateway-port",
        type=int,
        default=80,
        help="Gateway port to test (default: 80)"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Test interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=2.0,
        help="Connection timeout (default: 2.0)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Latency threshold in ms (default: 100)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        help="Run duration in seconds"
    )
    
    args = parser.parse_args()
    
    diag = NetDiagMobile(
        gateway=args.gateway,
        gateway_port=args.gateway_port,
        interval=args.interval,
        timeout=args.timeout,
        threshold=args.threshold,
        output_file=args.output,
        output_format=args.format
    )
    
    diag.run(duration=args.duration)


if __name__ == "__main__":
    main()

