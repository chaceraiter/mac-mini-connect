import socket
import argparse
import sys
import time

def log_message(message):
    """Logs a message with a timestamp."""
    print(f"[{time.time()}] {message}", flush=True)

def run_server(host, port):
    """Starts a simple TCP server."""
    # The host passed from the script is the public IP, but for robust binding,
    # we listen on '0.0.0.0' to accept connections on all available interfaces.
    listen_host = '0.0.0.0'
    log_message(f"Starting server, binding to {listen_host}:{port} (will accept connections for {host})")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((listen_host, port))
            sock.listen(1)
            log_message(f"Server listening on {listen_host}:{port}")
            conn, addr = sock.accept()
            with conn:
                log_message(f"Accepted connection from {addr}")
                data = conn.recv(1024)
                log_message(f"Received: {data.decode()}")
                conn.sendall(b"Hello from server!")
                log_message("Sent response and closing connection.")
        except Exception as e:
            log_message(f"Server error: {e}")
            sys.exit(1)
    log_message("Server finished successfully.")

def run_client(host, port):
    """Starts a simple TCP client."""
    log_message(f"Starting client, attempting to connect to {host}:{port}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect((host, port))
            log_message("Connection successful.")
            sock.sendall(b"Hello from client!")
            log_message("Sent message.")
            data = sock.recv(1024)
            log_message(f"Received: {data.decode()}")
        except Exception as e:
            log_message(f"Client error: {e}")
            sys.exit(1)
    log_message("Client finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple TCP Client/Server")
    subparsers = parser.add_subparsers(dest="role", required=True)

    # Server parser
    server_parser = subparsers.add_parser("server", help="Run as server")
    server_parser.add_argument("host", help="Host IP to bind to")
    server_parser.add_argument("port", type=int, help="Port to listen on")

    # Client parser
    client_parser = subparsers.add_parser("client", help="Run as client")
    client_parser.add_argument("host", help="Host IP to connect to")
    client_parser.add_argument("port", type=int, help="Port to connect to")

    args = parser.parse_args()

    if args.role == "server":
        run_server(args.host, args.port)
    elif args.role == "client":
        run_client(args.host, args.port) 