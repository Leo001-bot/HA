import re

# Read the file
with open('server.py', 'r') as f:
    content = f.read()

# Fix the indentation issues - too many spaces before socketio.emit
content = content.replace(
    '                socketio.emit(',
    '            socketio.emit('
)
content = content.replace(
    '                    socketio.emit(',
    '                socketio.emit('
)

# Remove , namespace='/' from all emits 
content = content.replace(", namespace='/'", '')

# Write back
with open('server.py', 'w') as f:
    f.write(content)

print('Fixed server.py!')
