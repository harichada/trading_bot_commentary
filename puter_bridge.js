#!/usr/bin/env node
/**
 * Puter AI Bridge — called by Python pre_entry_debate.py
 * Usage: node puter_bridge.js "Your prompt here" [model]
 * Returns JSON: {text: "response", model: "claude-haiku-4-5"}
 */

const https = require('https');

const prompt = process.argv[2];
const model = process.argv[3] || 'claude-haiku-4-5';

if (!prompt) {
    console.log(JSON.stringify({error: 'No prompt provided'}));
    process.exit(1);
}

// Puter's public AI endpoint (no auth needed)
const data = JSON.stringify({
    interface: 'puter-chat-completion',
    driver: model.startsWith('gpt') ? 'openai-completion' : 'claude',
    method: 'complete',
    args: {
        messages: [{role: 'user', content: prompt}],
        model: model,
        max_tokens: 200,
        temperature: 0.3,
    }
});

const options = {
    hostname: 'api.puter.com',
    path: '/drivers/call',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length,
    },
    timeout: 15000,
};

const req = https.request(options, (res) => {
    let body = '';
    res.on('data', chunk => body += chunk);
    res.on('end', () => {
        try {
            const json = JSON.parse(body);
            const text = json?.message?.content?.[0]?.text || json?.text || body;
            console.log(JSON.stringify({text, model}));
        } catch(e) {
            console.log(JSON.stringify({text: body, model}));
        }
    });
});

req.on('error', (e) => {
    console.log(JSON.stringify({error: e.message}));
});

req.on('timeout', () => {
    req.destroy();
    console.log(JSON.stringify({error: 'timeout'}));
});

req.write(data);
req.end();
