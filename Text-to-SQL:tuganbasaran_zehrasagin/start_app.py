#!/usr/bin/env python3
"""
Enhanced launcher script to start the Streamlit app
"""

import os
import sys
import subprocess
import time
import signal
import threading

# Configure environment for improved error handling
os.environ["PYTHONUNBUFFERED"] = "1"

def run_streamlit_app():
    """Run the Streamlit app with proper error handling"""
    print("\n🔄 Starting AI-Powered Smart Data Assistant...\n")
    
    try:
        # Use subprocess to start the streamlit app
        process = subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.port", "8503"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor output in separate threads to prevent blocking
        def monitor_output(pipe, prefix):
            for line in iter(pipe.readline, ''):
                print(f"{prefix}: {line.strip()}")
        
        # Start monitoring threads
        stdout_thread = threading.Thread(target=monitor_output, args=(process.stdout, "INFO"))
        stderr_thread = threading.Thread(target=monitor_output, args=(process.stderr, "ERROR"))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for the process to indicate it's running
        time.sleep(2)
        print("\n🌐 Smart Data Assistant running at http://localhost:8503\n")
        print("✅ AI-powered visualization engine active")
        print("✅ Smart prediction engine enabled")
        print("\n📊 Try making queries like:")
        print("- 'Show monthly revenue as a heatmap'")
        print("- 'Predict sales for next quarter'")
        print("- 'Compare top customers by region'\n")
        print("Press Ctrl+C to stop the server\n")
        
        # Wait for the process to finish or user interrupt
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Smart Data Assistant...")
        try:
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()
        except:
            pass
        print("✅ Application stopped")
    
    except Exception as e:
        print(f"\n❌ Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit_app()