"""
A2A HTTP Server for Face Recognition Agent.

This server exposes the Face Recognition Agent over HTTP,
allowing other agents to query and interact with it using the A2A protocol.
"""

import asyncio
import json
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_recognition.face_recognition_agent import FaceRecognitionAgent


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for A2A protocol."""

    agent: FaceRecognitionAgent = None

    def _set_headers(self, status=200, content_type='application/json'):
        """Set response headers."""
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, data: Dict[str, Any], status=200):
        """Send JSON response."""
        self._set_headers(status)
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def do_OPTIONS(self):
        """Handle OPTIONS request for CORS."""
        self._set_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/':
            # Welcome page
            self._send_json({
                "agent": "face-recognition-agent",
                "status": "running",
                "version": "1.0.0",
                "endpoints": {
                    "/": "This help message",
                    "/health": "Health check",
                    "/people": "List all known people",
                    "/person/<name>": "Get person information",
                    "/query": "A2A query endpoint (POST)"
                },
                "supported_operations": [
                    "query_person",
                    "list_people",
                    "request_questions",
                    "search_people"
                ]
            })

        elif path == '/health':
            # Health check
            self._send_json({
                "status": "healthy",
                "agent_id": self.agent.agent_id,
                "people_count": len(self.agent.db.list_people())
            })

        elif path == '/people':
            # List all people
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.agent.handle_a2a_request({"operation": "list_people"})
            )
            self._send_json(result)

        elif path.startswith('/person/'):
            # Get person by name
            name = path.split('/person/', 1)[1]
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.agent.handle_a2a_request({
                    "operation": "query_person",
                    "name": name
                })
            )
            self._send_json(result)

        else:
            self._send_json({
                "error": "Not found",
                "path": path
            }, status=404)

    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            request_data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            self._send_json({
                "error": "Invalid JSON in request body"
            }, status=400)
            return

        if path == '/query':
            # A2A query endpoint
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.agent.handle_a2a_request(request_data)
            )
            self._send_json(result)

        elif path == '/search':
            # Search people by metadata
            query = request_data.get('query', {})
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.agent.handle_a2a_request({
                    "operation": "search_people",
                    "query": query
                })
            )
            self._send_json(result)

        else:
            self._send_json({
                "error": "Not found",
                "path": path
            }, status=404)

    def log_message(self, format, *args):
        """Custom log message format."""
        print(f"[{self.log_date_time_string()}] {format % args}")


def run_server(host='127.0.0.1', port=8080, data_dir='face_recognition/people_data'):
    """
    Run the A2A HTTP server.

    Args:
        host: Host to bind to
        port: Port to bind to
        data_dir: Directory for person data storage
    """
    print("\n" + "=" * 80)
    print("FACE RECOGNITION AGENT - A2A HTTP SERVER")
    print("=" * 80)
    print(f"\nInitializing agent with data directory: {data_dir}")

    # Create agent instance
    agent = FaceRecognitionAgent(data_dir=data_dir)

    # Set agent on handler class
    A2ARequestHandler.agent = agent

    # Create server
    server = HTTPServer((host, port), A2ARequestHandler)

    print(f"\nServer running at http://{host}:{port}/")
    print("\nEndpoints:")
    print(f"  GET  http://{host}:{port}/           - API information")
    print(f"  GET  http://{host}:{port}/health     - Health check")
    print(f"  GET  http://{host}:{port}/people     - List all people")
    print(f"  GET  http://{host}:{port}/person/<name> - Get person info")
    print(f"  POST http://{host}:{port}/query      - A2A query")
    print(f"  POST http://{host}:{port}/search     - Search people")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()
        print("Server stopped.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="A2A HTTP Server for Face Recognition Agent"
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to (default: 8080)'
    )
    parser.add_argument(
        '--data-dir',
        default='face_recognition/people_data',
        help='Directory for person data (default: face_recognition/people_data)'
    )

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, data_dir=args.data_dir)


if __name__ == '__main__':
    main()
