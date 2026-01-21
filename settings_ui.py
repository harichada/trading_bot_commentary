#!/usr/bin/env python3
"""
Settings UI - Standalone settings management interface
Run alongside the main trading bot for settings control.

Usage:
    python settings_ui.py --bot-port 9000 --ui-port 8001
    Then open http://localhost:8001 in your browser
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import json
import argparse
from pathlib import Path

app = FastAPI(title="Trading Bot Settings")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Main bot URL - will be set from command line args
BOT_PORT = 9000  # Default, overridden by args

SETTINGS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Settings</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0a0a;
            color: #fff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #4a9eff;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        h2 {
            color: #888;
            font-size: 18px;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .status-bar {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #666;
        }
        .status-dot.running { background: #22c55e; }
        .status-dot.stopped { background: #ef4444; }
        .status-dot.live { background: #ef4444; animation: pulse 1.5s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Account Stats Panel */
        .account-stats {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .account-stats h3 {
            margin: 0 0 15px 0;
            color: #4a9eff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        .stat-box {
            background: #0a0a0a;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #222;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 20px;
            font-weight: bold;
            color: #fff;
        }
        .stat-value.positive { color: #22c55e; }
        .stat-value.negative { color: #ef4444; }
        .stat-value.live { color: #ef4444; }
        .stat-value.simulation { color: #4a9eff; }
        .data-source {
            font-size: 10px;
            color: #666;
            margin-top: 5px;
        }
        .data-source.real { color: #22c55e; }
        .data-source.simulated { color: #f59e0b; }

        /* Mode Toggle */
        .mode-toggle-container {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px;
            background: #0a0a0a;
            border-radius: 6px;
            border: 2px solid #333;
        }
        .mode-toggle-container.live-mode {
            border-color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
        }
        .mode-label {
            font-weight: bold;
            font-size: 14px;
        }
        .mode-label.live { color: #ef4444; }
        .mode-label.simulation { color: #4a9eff; }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-primary { background: #4a9eff; color: white; }
        .btn-primary:hover { background: #3182ce; }
        .btn-danger { background: #ef4444; color: white; }
        .btn-danger:hover { background: #dc2626; }
        .btn-success { background: #22c55e; color: white; }
        .btn-success:hover { background: #16a34a; }
        .btn-secondary { background: #666; color: white; }
        .btn-secondary:hover { background: #555; }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        .settings-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
        }
        .settings-card h3 {
            margin-top: 0;
            color: #4a9eff;
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .setting-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #222;
        }
        .setting-row:last-child { border-bottom: none; }
        .setting-label {
            flex: 1;
        }
        .setting-name {
            font-weight: 500;
            margin-bottom: 4px;
        }
        .setting-desc {
            font-size: 12px;
            color: #666;
        }
        .setting-input {
            width: 120px;
            padding: 8px 12px;
            background: #0a0a0a;
            border: 1px solid #333;
            border-radius: 4px;
            color: white;
            font-size: 14px;
        }
        .setting-input:focus {
            outline: none;
            border-color: #4a9eff;
        }
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: #333;
            transition: 0.3s;
            border-radius: 26px;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background: white;
            transition: 0.3s;
            border-radius: 50%;
        }
        input:checked + .toggle-slider {
            background: #4a9eff;
        }
        input:checked + .toggle-slider:before {
            transform: translateX(24px);
        }
        .message {
            padding: 12px 16px;
            border-radius: 5px;
            margin-bottom: 15px;
            display: none;
        }
        .message.success {
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid #22c55e;
            color: #22c55e;
        }
        .message.error {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            color: #ef4444;
        }
        .actions-bar {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            justify-content: flex-end;
        }
        .quick-actions {
            display: flex;
            gap: 10px;
        }
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }
        .tab {
            padding: 10px 20px;
            background: transparent;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 14px;
            border-radius: 5px 5px 0 0;
        }
        .tab.active {
            background: #1a1a1a;
            color: #4a9eff;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚙️ Trading Bot Settings</h1>

        <div id="message" class="message"></div>

        <div class="status-bar">
            <div class="status-indicator">
                <div id="statusDot" class="status-dot"></div>
                <span id="statusText">Checking...</span>
            </div>
            <div class="mode-toggle-container" id="modeContainer">
                <span class="mode-label simulation" id="modeLabel">SIMULATION</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="mode-toggle" onchange="toggleTradingMode()">
                    <span class="toggle-slider"></span>
                </label>
                <span style="font-size: 12px; color: #666;">Live Trading</span>
            </div>
            <div class="quick-actions">
                <button class="btn btn-success" onclick="startBot()">▶ Start</button>
                <button class="btn btn-danger" onclick="stopBot()">⏹ Stop</button>
                <button class="btn btn-secondary" onclick="refreshStatus()">↻ Refresh</button>
            </div>
        </div>

        <!-- Real-time Account Stats -->
        <div class="account-stats">
            <h3>📊 Account Stats <span id="dataSourceBadge" class="data-source simulated">(Simulated Data)</span></h3>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Account Balance</div>
                    <div class="stat-value" id="accountBalance">$0.00</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Daily P&L</div>
                    <div class="stat-value" id="dailyPnl">$0.00</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Buying Power</div>
                    <div class="stat-value" id="buyingPower">$0.00</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Open Positions</div>
                    <div class="stat-value" id="positionCount">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Total P&L</div>
                    <div class="stat-value" id="totalPnl">$0.00</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Schwab Connected</div>
                    <div class="stat-value" id="schwabStatus">No</div>
                </div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('trading')">Trading</button>
            <button class="tab" onclick="showTab('risk')">Risk Management</button>
            <button class="tab" onclick="showTab('strategies')">Strategies</button>
            <button class="tab" onclick="showTab('advanced')">Advanced</button>
        </div>

        <div id="trading" class="tab-content active">
            <div class="settings-grid">
                <div class="settings-card">
                    <h3>📊 Position Sizing</h3>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Max Positions</div>
                            <div class="setting-desc">Maximum number of open positions</div>
                        </div>
                        <input type="number" class="setting-input" id="max_positions" min="1" max="20" value="5">
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Max Position Value ($)</div>
                            <div class="setting-desc">Maximum value per position</div>
                        </div>
                        <input type="number" class="setting-input" id="max_position_value" min="100" max="100000" value="10000">
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Min Position Size</div>
                            <div class="setting-desc">Minimum shares per trade</div>
                        </div>
                        <input type="number" class="setting-input" id="min_position_size" min="1" max="100" value="1">
                    </div>
                </div>

                <div class="settings-card">
                    <h3>💰 Risk Per Trade</h3>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Risk Per Trade (%)</div>
                            <div class="setting-desc">Maximum risk per trade</div>
                        </div>
                        <input type="number" class="setting-input" id="max_risk_per_trade" min="0.1" max="10" step="0.1" value="2">
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Max Daily Loss (%)</div>
                            <div class="setting-desc">Stop trading if daily loss exceeds</div>
                        </div>
                        <input type="number" class="setting-input" id="max_daily_loss" min="1" max="20" step="0.5" value="5">
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Min Risk/Reward Ratio</div>
                            <div class="setting-desc">Minimum reward for risk taken</div>
                        </div>
                        <input type="number" class="setting-input" id="min_risk_reward_ratio" min="1" max="5" step="0.1" value="2">
                    </div>
                </div>
            </div>
        </div>

        <div id="risk" class="tab-content">
            <div class="settings-grid">
                <div class="settings-card">
                    <h3>🛑 Stop Loss & Take Profit</h3>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Default Stop Loss (%)</div>
                            <div class="setting-desc">Default stop loss percentage</div>
                        </div>
                        <input type="number" class="setting-input" id="default_stop_loss" min="0.5" max="10" step="0.1" value="2">
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Default Take Profit (%)</div>
                            <div class="setting-desc">Default take profit percentage</div>
                        </div>
                        <input type="number" class="setting-input" id="default_take_profit" min="1" max="20" step="0.5" value="4">
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Trailing Stop Enabled</div>
                            <div class="setting-desc">Use trailing stops</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="trailing_stop_enabled" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>

                <div class="settings-card">
                    <h3>⚠️ Circuit Breakers</h3>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Circuit Breaker Enabled</div>
                            <div class="setting-desc">Stop trading on consecutive losses</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="circuit_breaker_enabled" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Max Consecutive Losses</div>
                            <div class="setting-desc">Pause after this many losses</div>
                        </div>
                        <input type="number" class="setting-input" id="max_consecutive_losses" min="1" max="10" value="3">
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Emergency Stop Loss (%)</div>
                            <div class="setting-desc">Emergency stop if exceeded</div>
                        </div>
                        <input type="number" class="setting-input" id="emergency_stop_loss" min="5" max="50" value="15">
                    </div>
                </div>
            </div>
        </div>

        <div id="strategies" class="tab-content">
            <div class="settings-grid">
                <div class="settings-card">
                    <h3>📈 Strategy Selection</h3>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Breakout Strategy</div>
                            <div class="setting-desc">Trade price breakouts</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="strategy_breakout" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Mean Reversion</div>
                            <div class="setting-desc">Trade price reversions</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="strategy_mean_reversion" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Momentum Strategy</div>
                            <div class="setting-desc">Trade momentum signals</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="strategy_momentum" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Min Strategy Consensus</div>
                            <div class="setting-desc">Minimum strategies agreeing</div>
                        </div>
                        <input type="number" class="setting-input" id="min_consensus" min="1" max="3" value="2">
                    </div>
                </div>

                <div class="settings-card">
                    <h3>🤖 ML Predictions</h3>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">ML Predictions Enabled</div>
                            <div class="setting-desc">Use machine learning signals</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="ml_prediction_enabled">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">ML Confidence Threshold</div>
                            <div class="setting-desc">Minimum ML confidence</div>
                        </div>
                        <input type="number" class="setting-input" id="ml_confidence_threshold" min="50" max="95" value="60">
                    </div>
                </div>
            </div>
        </div>

        <div id="advanced" class="tab-content">
            <div class="settings-grid">
                <div class="settings-card">
                    <h3>🕐 Trading Hours</h3>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Allow Pre-Market</div>
                            <div class="setting-desc">Trade in pre-market (4-9:30 AM)</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="allow_premarket">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Allow After-Hours</div>
                            <div class="setting-desc">Trade in after-hours (4-8 PM)</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="allow_afterhours">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>

                <div class="settings-card">
                    <h3>📝 Logging & Commentary</h3>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Commentary Enabled</div>
                            <div class="setting-desc">Show trading commentary</div>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="commentary_enabled" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="setting-row">
                        <div class="setting-label">
                            <div class="setting-name">Detail Level</div>
                            <div class="setting-desc">Commentary verbosity</div>
                        </div>
                        <select class="setting-input" id="detail_level">
                            <option value="minimal">Minimal</option>
                            <option value="normal">Normal</option>
                            <option value="verbose" selected>Verbose</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>

        <div class="actions-bar">
            <button class="btn btn-secondary" onclick="resetSettings()">Reset to Defaults</button>
            <button class="btn btn-primary" onclick="saveSettings()">💾 Save Settings</button>
        </div>
    </div>

    <script>
        const BOT_URL = '{{BOT_URL}}';

        // Tab switching
        function showTab(tabId) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tabId}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        // Show message
        function showMessage(text, type) {
            const msg = document.getElementById('message');
            msg.textContent = text;
            msg.className = 'message ' + type;
            msg.style.display = 'block';
            setTimeout(() => msg.style.display = 'none', 3000);
        }

        // Load settings from bot
        async function loadSettings() {
            try {
                const response = await fetch(`${BOT_URL}/api/settings`);
                const data = await response.json();

                if (data.status === 'success') {
                    const settings = data.settings;

                    // Trading settings
                    if (settings.trading) {
                        setVal('max_positions', settings.trading.max_positions);
                        setVal('max_position_value', settings.trading.max_position_value);
                        setVal('min_position_size', settings.trading.min_position_size);
                        setVal('max_risk_per_trade', (settings.trading.max_risk_per_trade || 0.02) * 100);
                        setVal('max_daily_loss', (settings.trading.max_daily_loss || 0.05) * 100);
                        setVal('min_risk_reward_ratio', settings.trading.min_risk_reward_ratio);
                        setVal('max_consecutive_losses', settings.trading.max_consecutive_losses);
                        setCheck('ml_prediction_enabled', settings.trading.ml_prediction_enabled);
                    }

                    // Commentary settings
                    if (settings.commentary) {
                        setCheck('commentary_enabled', settings.commentary.enabled);
                        setVal('detail_level', settings.commentary.detail_level);
                    }

                    // Error recovery settings
                    if (settings.error_recovery) {
                        setCheck('circuit_breaker_enabled', settings.error_recovery.circuit_breaker_enabled);
                        setVal('emergency_stop_loss', (settings.error_recovery.emergency_stop_loss || 0.15) * 100);
                    }
                }
            } catch (e) {
                console.error('Failed to load settings:', e);
            }
        }

        function setVal(id, val) {
            const el = document.getElementById(id);
            if (el && val !== undefined) el.value = val;
        }

        function setCheck(id, val) {
            const el = document.getElementById(id);
            if (el) el.checked = !!val;
        }

        // Save settings
        async function saveSettings() {
            const settings = {
                'trading.max_positions': parseInt(document.getElementById('max_positions').value),
                'trading.max_position_value': parseFloat(document.getElementById('max_position_value').value),
                'trading.min_position_size': parseInt(document.getElementById('min_position_size').value),
                'trading.max_risk_per_trade': parseFloat(document.getElementById('max_risk_per_trade').value) / 100,
                'trading.max_daily_loss': parseFloat(document.getElementById('max_daily_loss').value) / 100,
                'trading.min_risk_reward_ratio': parseFloat(document.getElementById('min_risk_reward_ratio').value),
                'trading.max_consecutive_losses': parseInt(document.getElementById('max_consecutive_losses').value),
                'trading.ml_prediction_enabled': document.getElementById('ml_prediction_enabled').checked,
                'commentary.enabled': document.getElementById('commentary_enabled').checked,
                'commentary.detail_level': document.getElementById('detail_level').value,
                'error_recovery.circuit_breaker_enabled': document.getElementById('circuit_breaker_enabled').checked,
                'error_recovery.emergency_stop_loss': parseFloat(document.getElementById('emergency_stop_loss').value) / 100,
            };

            try {
                const response = await fetch(`${BOT_URL}/api/settings`, {
                    method: 'PUT',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(settings)
                });
                const data = await response.json();

                if (data.status === 'success') {
                    showMessage('Settings saved successfully!', 'success');
                } else {
                    showMessage('Failed to save: ' + data.message, 'error');
                }
            } catch (e) {
                showMessage('Connection error: ' + e.message, 'error');
            }
        }

        // Reset settings
        async function resetSettings() {
            if (!confirm('Reset all settings to defaults?')) return;

            try {
                const response = await fetch(`${BOT_URL}/api/settings/reset`, {method: 'POST'});
                const data = await response.json();

                if (data.status === 'success') {
                    showMessage('Settings reset to defaults', 'success');
                    loadSettings();
                }
            } catch (e) {
                showMessage('Connection error: ' + e.message, 'error');
            }
        }

        // Bot control
        async function refreshStatus() {
            try {
                const response = await fetch(`${BOT_URL}/api/status`);
                const data = await response.json();

                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');
                const modeToggle = document.getElementById('mode-toggle');
                const modeLabel = document.getElementById('modeLabel');
                const modeContainer = document.getElementById('modeContainer');

                if (data.bot_running) {
                    const isLive = data.mode === 'live';
                    dot.className = isLive ? 'status-dot live' : 'status-dot running';
                    text.textContent = `Running (${data.mode.toUpperCase()}) - ${data.positions_count} positions`;

                    // Update mode toggle
                    modeToggle.checked = isLive;
                    modeLabel.textContent = isLive ? 'LIVE TRADING' : 'SIMULATION';
                    modeLabel.className = isLive ? 'mode-label live' : 'mode-label simulation';
                    modeContainer.className = isLive ? 'mode-toggle-container live-mode' : 'mode-toggle-container';

                    // Update Schwab status
                    document.getElementById('schwabStatus').textContent = data.schwab_connected ? 'Yes' : 'No';
                    document.getElementById('schwabStatus').className = data.schwab_connected ? 'stat-value positive' : 'stat-value negative';
                } else {
                    dot.className = 'status-dot stopped';
                    text.textContent = 'Stopped';
                }

                // Fetch detailed account stats
                await refreshAccountStats();
            } catch (e) {
                document.getElementById('statusDot').className = 'status-dot';
                document.getElementById('statusText').textContent = 'Cannot connect to bot';
            }
        }

        async function refreshAccountStats() {
            try {
                // Try to get detailed stats
                const response = await fetch(`${BOT_URL}/api/account-stats`);
                const data = await response.json();

                if (data.status === 'success') {
                    const stats = data.stats;
                    const isReal = data.source === 'schwab';

                    // Update balance
                    document.getElementById('accountBalance').textContent = `$${(stats.balance || 0).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

                    // Update daily P&L
                    const pnl = stats.daily_pnl || 0;
                    const pnlEl = document.getElementById('dailyPnl');
                    pnlEl.textContent = `$${pnl.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                    pnlEl.className = pnl >= 0 ? 'stat-value positive' : 'stat-value negative';

                    // Update buying power
                    document.getElementById('buyingPower').textContent = `$${(stats.buying_power || 0).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

                    // Update position count
                    document.getElementById('positionCount').textContent = stats.position_count || 0;

                    // Update total P&L
                    const totalPnl = stats.total_pnl || 0;
                    const totalPnlEl = document.getElementById('totalPnl');
                    totalPnlEl.textContent = `$${totalPnl.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                    totalPnlEl.className = totalPnl >= 0 ? 'stat-value positive' : 'stat-value negative';

                    // Update data source badge
                    const badge = document.getElementById('dataSourceBadge');
                    badge.textContent = isReal ? '(Live Schwab Data)' : '(Simulated Data)';
                    badge.className = isReal ? 'data-source real' : 'data-source simulated';
                }
            } catch (e) {
                console.error('Failed to fetch account stats:', e);
            }
        }

        async function toggleTradingMode() {
            const toggle = document.getElementById('mode-toggle');
            const modeLabel = document.getElementById('modeLabel');
            const modeContainer = document.getElementById('modeContainer');
            const isLive = toggle.checked;

            // Confirm before switching to live
            if (isLive) {
                if (!confirm('WARNING: You are about to switch to LIVE TRADING mode.\\n\\nThis will use real money from your Schwab account.\\n\\nAre you sure?')) {
                    toggle.checked = false;
                    return;
                }
            }

            try {
                const response = await fetch(`${BOT_URL}/api/toggle-mode`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({mode: isLive ? 'live' : 'simulation'})
                });
                const data = await response.json();

                if (data.status === 'success') {
                    modeLabel.textContent = isLive ? 'LIVE TRADING' : 'SIMULATION';
                    modeLabel.className = isLive ? 'mode-label live' : 'mode-label simulation';
                    modeContainer.className = isLive ? 'mode-toggle-container live-mode' : 'mode-toggle-container';
                    showMessage(`Switched to ${isLive ? 'LIVE TRADING' : 'SIMULATION'} mode`, 'success');

                    // Refresh stats immediately
                    setTimeout(refreshAccountStats, 500);
                } else {
                    toggle.checked = !isLive;  // Revert toggle
                    showMessage(data.message || 'Failed to switch mode', 'error');
                }
            } catch (e) {
                toggle.checked = !isLive;  // Revert toggle
                showMessage('Connection error: ' + e.message, 'error');
            }
        }

        async function startBot() {
            try {
                const response = await fetch(`${BOT_URL}/api/start`, {method: 'POST'});
                const data = await response.json();
                showMessage(data.message || 'Bot started', 'success');
                setTimeout(refreshStatus, 1000);
            } catch (e) {
                showMessage(`Cannot connect to main bot. Make sure trading_bot_commentary_updated.py is running on port ${BOT_URL.split(':').pop()}`, 'error');
            }
        }

        async function stopBot() {
            try {
                const response = await fetch(`${BOT_URL}/api/stop`, {method: 'POST'});
                const data = await response.json();
                showMessage(data.message || 'Bot stopped', 'success');
                setTimeout(refreshStatus, 1000);
            } catch (e) {
                showMessage(`Cannot connect to main bot. Make sure trading_bot_commentary_updated.py is running on port ${BOT_URL.split(':').pop()}`, 'error');
            }
        }

        // Initialize
        loadSettings();
        refreshStatus();
        setInterval(refreshStatus, 5000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def settings_page():
    # Replace the placeholder with actual bot URL
    bot_url = f"http://localhost:{BOT_PORT}"
    html = SETTINGS_HTML.replace("{{BOT_URL}}", bot_url)
    return html

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    global BOT_PORT

    parser = argparse.ArgumentParser(description="Trading Bot Settings UI")
    parser.add_argument("--bot-port", type=int, default=9000,
                        help="Port where the main trading bot is running (default: 9000)")
    parser.add_argument("--ui-port", type=int, default=8001,
                        help="Port for the settings UI (default: 8001)")
    args = parser.parse_args()

    BOT_PORT = args.bot_port
    ui_port = args.ui_port

    print("\n" + "="*50)
    print("  Trading Bot Settings UI")
    print("="*50)
    print(f"\n  Settings UI: http://localhost:{ui_port}")
    print(f"  Connecting to bot on port: {args.bot_port}")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=ui_port)

if __name__ == "__main__":
    main()
