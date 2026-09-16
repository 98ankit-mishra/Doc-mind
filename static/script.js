async function uploadPDF() {
    const fileInput = document.getElementById('pdf-upload');
    const statusDiv = document.getElementById('upload-status');
    const uploadBtn = document.getElementById('upload-btn');

    if (!fileInput.files.length) {
        statusDiv.innerHTML = '<span class="error">Please select a PDF file first.</span>';
        return;
    }

    const file = fileInput.files[0];
    if (file.type !== 'application/pdf') {
        statusDiv.innerHTML = '<span class="error">Only PDF files are supported.</span>';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    statusDiv.innerHTML = '<span>Uploading and processing... This may take a minute depending on PDF size.</span>';
    uploadBtn.disabled = true;

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            statusDiv.innerHTML = `<span class="success">Success: ${data.message} (${data.filename})</span>`;
        } else {
            statusDiv.innerHTML = `<span class="error">Error: ${data.detail}</span>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<span class="error">Error connecting to server.</span>`;
    } finally {
        uploadBtn.disabled = false;
    }
}

async function askQuestion() {
    const input = document.getElementById('question-input');
    const chatHistory = document.getElementById('chat-history');
    const askBtn = document.getElementById('ask-btn');
    const question = input.value.trim();

    if (!question) return;

    // Add user message
    appendMessage('user', question);
    input.value = '';
    askBtn.disabled = true;

    // Add loading message
    const loadingId = appendMessage('ai', 'Thinking...');

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        
        // Remove loading
        document.getElementById(loadingId).remove();

        if (response.ok) {
            appendMessage('ai', data.answer, data.sources);
        } else {
            appendMessage('ai', `Error: ${data.detail || 'Failed to get answer'}`);
        }
    } catch (error) {
        document.getElementById(loadingId).remove();
        appendMessage('ai', 'Error connecting to server.');
    } finally {
        askBtn.disabled = false;
        input.focus();
    }
}

async function clearDB() {
    const statusDiv = document.getElementById('upload-status');
    const clearBtn = document.getElementById('clear-btn');
    
    clearBtn.disabled = true;
    try {
        const response = await fetch('/clear', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            statusDiv.innerHTML = `<span class="success">${data.message}</span>`;
        } else {
            statusDiv.innerHTML = `<span class="error">Error: ${data.detail}</span>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<span class="error">Error connecting to server.</span>`;
    } finally {
        clearBtn.disabled = false;
    }
}

function appendMessage(role, text, sources = null) {
    const chatHistory = document.getElementById('chat-history');
    const msgDiv = document.createElement('div');
    const msgId = 'msg-' + Date.now();
    msgDiv.id = msgId;
    msgDiv.className = `message ${role}`;
    
    let innerHTML = `<div>${text}</div>`;
    
    if (sources && sources.length > 0) {
        const sourceHtml = sources.map(s => `Page ${s.page} (${s.source})`).join(', ');
        innerHTML += `<div class="sources"><strong>Sources:</strong> ${sourceHtml}</div>`;
    }
    
    msgDiv.innerHTML = innerHTML;
    chatHistory.appendChild(msgDiv);
    
    // Auto-scroll to bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return msgId;
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        askQuestion();
    }
}
