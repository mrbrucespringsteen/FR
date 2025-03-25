import socket
import json
from enclave_app import handle_request

# VSOCK constants
VMADDR_CID_ANY = 0xFFFFFFFF
VMADDR_PORT = 5000

def main():
    # Create VSOCK socket
    sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((VMADDR_CID_ANY, VMADDR_PORT))
    sock.listen()
    
    print(f"VSOCK server listening on port {VMADDR_PORT}")
    
    while True:
        client, addr = sock.accept()
        print(f"Connection from {addr}")
        
        # Receive data
        data = client.recv(4096).decode('utf-8')
        
        # Process request
        response = handle_request(data)
        
        # Send response
        client.send(response.encode('utf-8'))
        client.close()

if __name__ == "__main__":
    main() 