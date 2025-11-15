import os
import json
import re
import base64
from flask import Flask, render_template_string, request, jsonify
import requests

# ==================== CONFIGURATION ====================
GITHUB_USERNAME = "your-username"  # Replace with your GitHub username
GITHUB_REPO = "your-repo"          # Replace with your repository name
GITHUB_BRANCH = "main"             # Branch to commit to
RAILWAY_APP_URL = "https://your-app.railway.app"  # Replace with your Railway app URL
PORT = 8080

# Environment variables (set these before running)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# ==================== FLASK APP ====================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Gemini Code Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        .header p {
            opacity: 0.9;
        }
        .chat-container {
            padding: 30px;
        }
        .message-area {
            margin-bottom: 20px;
            min-height: 200px;
            max-height: 400px;
            overflow-y: auto;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            background: #f9f9f9;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px 15px;
            border-radius: 8px;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user-message {
            background: #667eea;
            color: white;
            margin-left: 20%;
        }
        .status-message {
            background: #ffd700;
            color: #333;
            font-weight: 500;
        }
        .success-message {
            background: #4caf50;
            color: white;
        }
        .error-message {
            background: #f44336;
            color: white;
        }
        .railway-link {
            background: #2196F3;
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            margin-top: 15px;
        }
        .railway-link a {
            color: white;
            text-decoration: none;
            font-weight: bold;
            font-size: 1.1em;
        }
        .railway-link a:hover {
            text-decoration: underline;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        #userInput {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            resize: vertical;
            min-height: 60px;
        }
        #userInput:focus {
            outline: none;
            border-color: #667eea;
        }
        #sendBtn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        #sendBtn:hover {
            transform: translateY(-2px);
        }
        #sendBtn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .loading {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Gemini Code Generator</h1>
            <p>Describe your app and deploy it to Railway automatically</p>
        </div>
        <div class="chat-container">
            <div class="message-area" id="messageArea"></div>
            <div class="input-area">
                <textarea id="userInput" placeholder="Describe the app you want to build..."></textarea>
                <button id="sendBtn" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        function addMessage(content, type) {
            const messageArea = document.getElementById('messageArea');
            const msg = document.createElement('div');
            msg.className = `message ${type}-message`;
            msg.innerHTML = content;
            messageArea.appendChild(msg);
            messageArea.scrollTop = messageArea.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const btn = document.getElementById('sendBtn');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            btn.disabled = true;
            
            // Add processing message
            addMessage('<span class="loading"></span>Processing your request...', 'status');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('❌ Error: ' + data.error, 'error');
                } else {
                    // Add status updates
                    if (data.gemini_status) {
                        addMessage('✓ Gemini: ' + data.gemini_status, 'status');
                    }
                    if (data.github_status) {
                        addMessage('✓ GitHub: ' + data.github_status, 'status');
                    }
                    
                    // Add success message with Railway link
                    const successMsg = `
                        ✅ Successfully deployed to GitHub! Railway will automatically build and deploy your app.
                        <div class="railway-link">
                            <a href="${data.railway_url}" target="_blank">🚀 View Your App on Railway</a>
                        </div>
                    `;
                    addMessage(successMsg, 'success');
                }
            } catch (error) {
                addMessage('❌ Error: ' + error.message, 'error');
            }
            
            btn.disabled = false;
        }
        
        // Allow Enter to send (Shift+Enter for new line)
        document.getElementById('userInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

# ==================== HELPER FUNCTIONS ====================

def call_gemini_api(user_message):
    """Call Gemini API with the coding agent prompt."""
    system_prompt = (
        "You are a coding agent. The following is a description of the code you are to design. "
        "Respond only with a JSON object containing two keys: 'main.py' and 'requirements.txt'. "
        "In the first key, write the entire code of the project. In the second, write the requirements. "
        "Important: Host a web server on port 8080 and display the output of the program."
    )
    
    full_prompt = f"{system_prompt}\n\nUser request: {user_message}"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": full_prompt
            }]
        }]
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    
    # Extract text from response
    if "candidates" in data and len(data["candidates"]) > 0:
        candidate = data["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            text = candidate["content"]["parts"][0].get("text", "")
            return text
    
    raise Exception("No valid response from Gemini")


def extract_json_from_text(text):
    """Extract JSON object from text that may contain markdown or other content."""
    # Try to find JSON object in the text
    # Look for patterns like ```json ... ``` or just { ... }
    
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find a JSON object
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found a complete JSON object
                json_str = text[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Continue searching
                    start_idx = -1
    
    raise Exception("No valid JSON object found in response")


def update_github_file(filepath, content, commit_message):
    """Update or create a file in the GitHub repository."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{filepath}"
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check if file exists to get its SHA
    response = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    
    sha = None
    if response.status_code == 200:
        sha = response.json().get("sha")
    
    # Encode content to base64
    content_bytes = content.encode('utf-8')
    content_b64 = base64.b64encode(content_bytes).decode('utf-8')
    
    # Prepare payload
    payload = {
        "message": commit_message,
        "content": content_b64,
        "branch": GITHUB_BRANCH
    }
    
    if sha:
        payload["sha"] = sha
    
    # Update or create file
    response = requests.put(url, json=payload, headers=headers)
    response.raise_for_status()
    
    return response.json()


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        if not GEMINI_API_KEY:
            return jsonify({"error": "GEMINI_API_KEY not set"}), 500
        
        if not GITHUB_TOKEN:
            return jsonify({"error": "GITHUB_TOKEN not set"}), 500
        
        # Step 1: Call Gemini API
        gemini_response = call_gemini_api(user_message)
        gemini_status = "Received response from Gemini"
        
        # Step 2: Extract JSON from response
        code_json = extract_json_from_text(gemini_response)
        
        if "main.py" not in code_json or "requirements.txt" not in code_json:
            return jsonify({"error": "Invalid JSON format from Gemini. Missing required keys."}), 500
        
        main_py_content = code_json["main.py"]
        requirements_content = code_json["requirements.txt"]
        
        # Step 3: Push to GitHub
        update_github_file("main.py", main_py_content, "Update main.py via Gemini agent")
        update_github_file("requirements.txt", requirements_content, "Update requirements.txt via Gemini agent")
        
        github_status = f"Successfully pushed to {GITHUB_USERNAME}/{GITHUB_REPO}"
        
        return jsonify({
            "success": True,
            "gemini_status": gemini_status,
            "github_status": github_status,
            "railway_url": RAILWAY_APP_URL
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    print(f"GitHub: {GITHUB_USERNAME}/{GITHUB_REPO}")
    print(f"Railway URL: {RAILWAY_APP_URL}")
    print(f"\nMake sure to set environment variables:")
    print("  - GEMINI_API_KEY")
    print("  - GITHUB_TOKEN")
    app.run(host='0.0.0.0', port=PORT, debug=True)
