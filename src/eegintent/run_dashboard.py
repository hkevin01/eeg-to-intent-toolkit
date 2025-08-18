#!/usr/bin/env python3
"""CLI runner for the EEG-to-Intent real-time dashboard."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "realtime" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
    ]
    
    print("Starting EEG-to-Intent Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop")
    
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except (subprocess.SubprocessError, OSError) as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
