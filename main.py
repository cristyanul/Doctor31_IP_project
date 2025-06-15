"""
Doctor31 Medical Validator - Main Entry Point
This is the main entry point for PyInstaller to create the Windows EXE
"""

import os
import sys
import time
import webbrowser
import threading

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def main():
    """Main entry point for the application"""
    try:
        # Import Flask app after setting up the path
        os.chdir(current_dir)
        
        # Import the web GUI components
        from src.web_gui import app, find_free_port
        
        # Find an available port
        port = find_free_port()
        
        print("🏥 Doctor31 Medical Validator")
        print("=" * 50)
        print(f"🚀 Starting server on port {port}")
        print(f"🌐 Open your browser to: http://127.0.0.1:{port}")
        print("💡 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open(f'http://127.0.0.1:{port}')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start the Flask application
        app.run(host='127.0.0.1', port=port, debug=False)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Press Enter to exit...")
        input()

if __name__ == "__main__":
    main()