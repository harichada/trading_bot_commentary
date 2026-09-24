#!/usr/bin/env python3
"""Generate a self-contained Activity page demo HTML with fixture data pre-rendered.

This reads the actual templates/activity.html and injects fixture data as pre-rendered HTML,
ensuring the demo exactly matches what the template would produce.
"""
import json
from pathlib import Path


def format_currency(val):
    if val is None:
        return '-'
    sign = '+' if val >= 0 else ''
    return f"{sign}${abs(val):.2f}"


def format_percent(val):
    if val is None:
        return '-'
    return f"{val:.1f}%"


def format_r(val):
    if val is None:
        return 'n/a'
    sign = '+' if val >= 0 else ''
    return f"{sign}{val:.2f}R"


def format_price(val):
    if val is None:
        return '-'
    return f"${val:.2f}"


def format_hold_time(minutes):
    if minutes is None:
        return '-'
    if minutes < 60:
        return f"{minutes}m"
    h = minutes // 60
    m = minutes % 60
    return f"{h}h {m}m"


def format_sl_tp(sl, tp):
    if sl is None and tp is None:
        return '-'
    sl_str = f"${int(sl)}" if sl is not None else '-'
    tp_str = f"${int(tp)}" if tp is not None else '-'
    return f'<span class="sl">{sl_str}</span>/<span class="tp">{tp_str}</span>'


def get_exit_reason_class(reason):
    if not reason:
        return ''
    if 'take_profit' in reason or 'target' in reason:
        return 'take_profit'
    if 'stop_loss' in reason or 'stop' in reason:
        return 'stop_loss'
    if 'proactive' in reason:
        return 'proactive'
    return ''


def get_reasoning_preview(trade):
    r = trade.get('reasoning', {})
    if trade.get('is_orphan'):
        return 'Broker fill—no snapshot'
    if not r.get('setup_type') and not r.get('regime'):
        return 'No reasoning'
    parts = []
    if r.get('setup_type'):
        parts.append(r['setup_type'])
    if r.get('regime'):
        parts.append(r['regime'])
    if r.get('score') is not None:
        parts.append(f"score {r['score']:.1f}")
    return ' · '.join(parts)


def build_narrative(trade):
    r = trade.get('reasoning', {})
    parts = []
    
    if r.get('setup_type'):
        parts.append(f"<strong>Setup:</strong> {r['setup_type']}")
    
    if r.get('regime'):
        desc = f"Market regime was <strong>{r['regime']}</strong>"
        if r.get('spy_slope_pct') is not None:
            sign = '+' if r['spy_slope_pct'] > 0 else ''
            desc += f" (SPY slope {sign}{r['spy_slope_pct']*100:.2f}%)"
        parts.append(desc)
    
    if r.get('rs_vs_spy') is not None:
        sign = '+' if r['rs_vs_spy'] > 0 else ''
        basis = f" ({r['rs_basis']})" if r.get('rs_basis') else ''
        parts.append(f"RS vs SPY: <strong>{sign}{r['rs_vs_spy']*100:.1f}%</strong>{basis}")
    
    if r.get('rsi') is not None:
        rsi_val = r['rsi'] * 100 if r['rsi'] < 1 else r['rsi']
        parts.append(f"RSI: <strong>{rsi_val:.0f}</strong>")
    
    if r.get('volume_ratio') is not None:
        parts.append(f"Volume ratio: <strong>{r['volume_ratio']:.1f}x</strong>")
    
    if r.get('score') is not None:
        parts.append(f"Signal-strength score: <strong>{r['score']:.2f}</strong> (heuristic, not probability)")
    
    if r.get('gates_passed'):
        gates = ', '.join(r['gates_passed']) if isinstance(r['gates_passed'], list) else r['gates_passed']
        parts.append(f"Gates passed: {gates}")
    
    if r.get('shadow_gates'):
        sg = ', '.join(r['shadow_gates']) if isinstance(r['shadow_gates'], list) else r['shadow_gates']
        parts.append(f"Shadow gates (would skip): {sg}")
    
    if r.get('is_day_trade'):
        parts.append(f"Day trade with flatten hour {r.get('flatten_hour', 15)}:00 ET")
    
    if r.get('exit_rationale'):
        parts.append(f"<strong>Exit:</strong> {r['exit_rationale']}")
    
    if not parts:
        if trade.get('is_orphan'):
            return '<em>Reasoning unavailable - broker fill without matching decision snapshot</em>'
        return '<em>No detailed reasoning recorded</em>'
    
    return '. '.join(parts) + '.'


def render_trade_row(trade, idx, expanded=False):
    pnl = trade.get('pnl', 0)
    r_mult = trade.get('r_multiple')
    
    pnl_class = 'positive' if pnl > 0 else ('negative' if pnl < 0 else 'neutral')
    r_class = ('positive' if r_mult > 0 else ('negative' if r_mult < 0 else 'neutral')) if r_mult is not None else 'neutral'
    side_class = 'side-long' if trade['side'] == 'long' else 'side-short'
    exit_class = get_exit_reason_class(trade.get('exit_reason', ''))
    
    badges = ''
    if trade.get('is_hands_off'):
        badges += '<span class="badge badge-hands-off">H</span>'
    if trade.get('is_external'):
        badges += '<span class="badge badge-external">E</span>'
    if trade.get('is_orphan'):
        badges += '<span class="badge badge-orphan">O</span>'
    
    entry_cell = f'''<div class="entry-exit-cell">
        <span class="time font-mono">{trade.get('entry_time_et', '-')}</span>
        <span class="price font-mono tabular-nums">{format_price(trade.get('entry_price'))}</span>
    </div>'''
    
    exit_cell = f'''<div class="entry-exit-cell">
        <span class="time font-mono">{trade.get('exit_time_et', '-')}</span>
        <span class="price font-mono tabular-nums">{format_price(trade.get('exit_price'))}</span>
    </div>'''
    
    sl_tp_cell = f'<div class="sl-tp-compact font-mono tabular-nums">{format_sl_tp(trade.get("stop_loss"), trade.get("take_profit"))}</div>'
    
    r = trade.get('reasoning', {})
    reasoning_grid = ''
    if r.get('setup_type'):
        reasoning_grid += f'<div class="reasoning-item"><label>Setup</label><span>{r["setup_type"]}</span></div>'
    if r.get('regime'):
        reasoning_grid += f'<div class="reasoning-item"><label>Regime</label><span>{r["regime"]}</span></div>'
    if r.get('rs_vs_spy') is not None:
        reasoning_grid += f'<div class="reasoning-item"><label>RS vs SPY</label><span>{r["rs_vs_spy"]*100:.1f}%</span></div>'
    if r.get('rsi') is not None:
        rsi_val = r['rsi'] * 100 if r['rsi'] < 1 else r['rsi']
        reasoning_grid += f'<div class="reasoning-item"><label>RSI</label><span>{rsi_val:.0f}</span></div>'
    if r.get('volume_ratio') is not None:
        reasoning_grid += f'<div class="reasoning-item"><label>Vol Ratio</label><span>{r["volume_ratio"]:.1f}x</span></div>'
    if r.get('score') is not None:
        reasoning_grid += f'<div class="reasoning-item"><label>Signal</label><span>{r["score"]:.2f}</span></div>'
    if trade.get('confidence') is not None:
        reasoning_grid += f'<div class="reasoning-item"><label>Conf</label><span>{trade["confidence"]*100:.0f}%</span></div>'
    if trade.get('meta_proba') is not None:
        reasoning_grid += f'<div class="reasoning-item"><label>Meta</label><span>{trade["meta_proba"]*100:.0f}%</span></div>'
    
    expanded_class = ' expanded' if expanded else ''
    visible_class = ' visible' if expanded else ''
    
    strategy = trade.get('strategy') or '-'
    if len(strategy) > 12:
        strategy = strategy[:10] + '…'
    
    exit_reason = trade.get('exit_reason') or '-'
    if len(exit_reason) > 10:
        exit_reason = exit_reason[:8] + '…'
    
    return f'''
    <tr class="expandable{expanded_class}" data-idx="{idx}">
        <td></td>
        <td class="symbol-cell">{trade['symbol']}{badges}</td>
        <td class="{side_class}">{trade['side'].upper()}</td>
        <td style="font-size: 0.6875rem;">{strategy}</td>
        <td>{entry_cell}</td>
        <td>{exit_cell}</td>
        <td class="font-mono tabular-nums">{trade['quantity']}</td>
        <td>{sl_tp_cell}</td>
        <td><span class="exit-reason {exit_class}" style="font-size: 0.625rem;">{exit_reason}</span></td>
        <td class="font-mono">{format_hold_time(trade.get('hold_time_min'))}</td>
        <td class="font-mono tabular-nums {pnl_class}">{format_currency(trade.get('pnl'))}</td>
        <td class="font-mono tabular-nums {r_class}">{format_r(trade.get('r_multiple'))}</td>
        <td><span class="reasoning-preview one-line-truncate">{get_reasoning_preview(trade)}</span></td>
    </tr>
    <tr class="reasoning-row{visible_class}" data-idx="{idx}">
        <td colspan="13">
            <div class="reasoning-content">
                <h4>Why This Trade</h4>
                <div class="reasoning-narrative">{build_narrative(trade)}</div>
                <div class="reasoning-grid">{reasoning_grid}</div>
            </div>
        </td>
    </tr>'''


def main():
    workspace = Path(__file__).parent.parent
    
    # Read fixture data
    with open(workspace / 'tests' / 'fixtures' / 'activity_fixture_data.json') as f:
        fixture = json.load(f)
    
    # Read the template
    with open(workspace / 'templates' / 'activity.html') as f:
        template = f.read()
    
    summary = fixture['summary']
    trades = fixture['trades']
    
    # Pre-render all trade rows (CRWV at index 1 is expanded)
    trade_rows = ''
    for idx, trade in enumerate(trades):
        expanded = (idx == 1)  # CRWV
        trade_rows += render_trade_row(trade, idx, expanded)
    
    # Build the summary values
    bot_pnl = summary['bot_pnl']
    bot_pnl_class = 'positive' if bot_pnl >= 0 else 'negative'
    all_pnl = summary['pnl']
    all_pnl_class = 'positive' if all_pnl >= 0 else 'negative'
    win_rate_class = 'positive' if summary['bot_win_rate'] >= 50 else 'negative'
    total_r_class = 'positive' if summary['bot_total_r'] >= 0 else 'negative'
    
    # Generate output HTML
    output = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=1280">
    <title>Activity - FIXTURE DATA Demo (1280px)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
          --bg:           oklch(0.06 0.012 270);
          --fg:           oklch(0.98 0 0);
          --card:         oklch(0.10 0.015 270 / 0.85);
          --card-fg:      oklch(0.98 0 0);
          --secondary:    oklch(0.14 0.015 270);
          --muted:        oklch(0.16 0.01 270);
          --muted-fg:     oklch(0.55 0 0);
          --border:       oklch(0.20 0.018 270 / 0.45);
          --border-soft:  oklch(0.18 0.015 270 / 0.25);
          --accent:       oklch(0.62 0.18 195);
          --accent-fg:    oklch(0.98 0 0);
          --success:      oklch(0.70 0.18 155);
          --success-fg:   oklch(0.10 0.05 155);
          --warning:      oklch(0.78 0.14 85);
          --destructive:  oklch(0.58 0.20 25);
          --destructive-fg: oklch(0.98 0 0);
          --radius: 0.625rem;
          --radius-sm: calc(var(--radius) - 3px);
          --radius-lg: calc(var(--radius) + 4px);
          --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
          --font-mono: 'JetBrains Mono', ui-monospace, monospace;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: var(--font-sans);
            font-feature-settings: "cv02", "cv03", "cv04", "cv11";
            background: var(--bg);
            color: var(--fg);
            -webkit-font-smoothing: antialiased;
            min-height: 100vh;
        }}
        .ambient-blobs {{ position: fixed; inset: 0; pointer-events: none; overflow: hidden; z-index: -1; }}
        .ambient-blob {{ position: absolute; border-radius: 9999px; filter: blur(140px); opacity: 0.30; }}
        .ambient-blob-1 {{ top: -5%; left: 20%; width: 500px; height: 500px; background: oklch(0.62 0.18 195 / 0.08); }}
        .ambient-blob-2 {{ bottom: 5%; right: 20%; width: 450px; height: 450px; background: oklch(0.78 0.14 85 / 0.06); filter: blur(120px); }}
        .glass {{
          background: oklch(0.09 0.015 270 / 0.75);
          backdrop-filter: blur(24px) saturate(160%);
          -webkit-backdrop-filter: blur(24px) saturate(160%);
          border: 1px solid oklch(0.25 0.02 270 / 0.25);
          box-shadow: inset 0 1px 0 oklch(0.30 0.02 270 / 0.15), 0 1px 3px oklch(0 0 0 / 0.3);
        }}
        .tabular-nums {{ font-variant-numeric: tabular-nums; }}
        .font-mono {{ font-family: var(--font-mono); }}
        ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: oklch(0.3 0.02 270 / 0.5); border-radius: 4px; }}
        .container {{ max-width: 1280px; margin: 0 auto; padding: 12px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding: 12px 16px; border-radius: var(--radius); }}
        .header h1 {{ font-size: 1.25rem; font-weight: 700; color: var(--fg); }}
        .header a {{ color: var(--accent); text-decoration: none; font-size: 0.8125rem; }}
        .fixture-banner {{ background: oklch(0.78 0.14 85 / 0.2); border: 1px solid var(--warning); padding: 8px 12px; border-radius: var(--radius); margin-bottom: 12px; text-align: center; font-weight: 600; color: var(--warning); font-size: 0.75rem; }}
        .filters {{ display: flex; gap: 10px; align-items: center; margin-bottom: 12px; padding: 10px 14px; border-radius: var(--radius); }}
        .filters label {{ font-size: 0.6875rem; color: var(--muted-fg); text-transform: uppercase; letter-spacing: 0.05em; }}
        .filters input[type="date"], .filters select {{
            background: oklch(0.12 0.01 270); border: 1px solid var(--border); color: var(--fg);
            padding: 6px 10px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.8125rem;
        }}
        .summary-strip {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-bottom: 12px; }}
        .summary-card {{ padding: 10px; border-radius: var(--radius); text-align: center; }}
        .summary-card.bot-stats {{ border-left: 3px solid var(--accent); }}
        .summary-value {{ font-size: 1.125rem; font-weight: 700; font-family: var(--font-mono); margin-bottom: 2px; }}
        .summary-label {{ font-size: 0.5625rem; color: var(--muted-fg); text-transform: uppercase; letter-spacing: 0.04em; }}
        .positive {{ color: var(--success); }}
        .negative {{ color: var(--destructive); }}
        .neutral {{ color: var(--muted-fg); }}
        .trades-table-container {{ border-radius: var(--radius); overflow: hidden; }}
        table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
        th {{ text-align: left; padding: 6px 4px; font-weight: 600; font-size: 0.5625rem; color: var(--muted-fg); text-transform: uppercase; letter-spacing: 0.03em; background: oklch(0.08 0.01 270); border-bottom: 1px solid var(--border); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        th:nth-child(1) {{ width: 18px; }}
        th:nth-child(2) {{ width: 70px; }}
        th:nth-child(3) {{ width: 36px; }}
        th:nth-child(4) {{ width: 68px; }}
        th:nth-child(5) {{ width: 80px; }}
        th:nth-child(6) {{ width: 80px; }}
        th:nth-child(7) {{ width: 32px; }}
        th:nth-child(8) {{ width: 58px; }}
        th:nth-child(9) {{ width: 60px; }}
        th:nth-child(10) {{ width: 48px; }}
        th:nth-child(11) {{ width: 62px; }}
        th:nth-child(12) {{ width: 48px; }}
        th:nth-child(13) {{ width: auto; min-width: 80px; }}
        td {{ padding: 5px 4px; font-size: 0.6875rem; border-bottom: 1px solid var(--border-soft); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        tr:hover td {{ background: oklch(0.11 0.015 270 / 0.5); }}
        tr.expandable {{ cursor: pointer; }}
        tr.expandable td:first-child::before {{ content: '+'; display: inline-block; width: 14px; color: var(--accent); font-weight: 600; font-size: 0.75rem; }}
        tr.expandable.expanded td:first-child::before {{ content: '−'; }}
        .symbol-cell {{ font-weight: 600; font-family: var(--font-mono); font-size: 0.6875rem; }}
        .side-long {{ color: var(--success); }}
        .side-short {{ color: var(--destructive); }}
        .badge {{ display: inline-block; padding: 1px 3px; border-radius: 2px; font-size: 0.5rem; font-weight: 600; text-transform: uppercase; margin-left: 3px; }}
        .badge-hands-off {{ background: oklch(0.78 0.14 85 / 0.2); color: var(--warning); }}
        .badge-external {{ background: oklch(0.55 0 0 / 0.2); color: var(--muted-fg); }}
        .badge-orphan {{ background: oklch(0.58 0.20 25 / 0.2); color: var(--destructive); }}
        .exit-reason {{ font-size: 0.5625rem; padding: 1px 4px; border-radius: 2px; background: oklch(0.15 0.01 270); }}
        .exit-reason.take_profit {{ background: oklch(0.70 0.18 155 / 0.15); color: var(--success); }}
        .exit-reason.stop_loss {{ background: oklch(0.58 0.20 25 / 0.15); color: var(--destructive); }}
        .exit-reason.proactive {{ background: oklch(0.78 0.14 85 / 0.15); color: var(--warning); }}
        .reasoning-row {{ display: none; }}
        .reasoning-row.visible {{ display: table-row; }}
        .reasoning-row td {{ background: oklch(0.07 0.01 270); padding: 10px 12px; white-space: normal; overflow: visible; }}
        .reasoning-content {{ font-size: 0.75rem; line-height: 1.5; color: oklch(0.85 0 0); max-width: 100%; overflow-wrap: break-word; }}
        .reasoning-content h4 {{ font-size: 0.625rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }}
        .reasoning-grid {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; max-width: 100%; }}
        .reasoning-item {{ padding: 5px 7px; background: oklch(0.10 0.015 270); border-radius: var(--radius-sm); border: 1px solid var(--border-soft); min-width: 70px; max-width: 120px; flex: 0 1 auto; }}
        .reasoning-item label {{ display: block; font-size: 0.4375rem; color: var(--muted-fg); text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 1px; }}
        .reasoning-item span {{ font-family: var(--font-mono); font-size: 0.625rem; }}
        .reasoning-narrative {{ margin-top: 8px; padding: 8px 10px; background: oklch(0.08 0.01 270); border-radius: var(--radius-sm); border-left: 3px solid var(--accent); font-size: 0.625rem; line-height: 1.4; white-space: normal; overflow-wrap: break-word; }}
        .entry-exit-cell {{ display: flex; flex-direction: column; gap: 0px; line-height: 1.1; }}
        .entry-exit-cell .time {{ font-size: 0.5625rem; color: var(--muted-fg); }}
        .entry-exit-cell .price {{ font-size: 0.625rem; font-weight: 500; }}
        .sl-tp-compact {{ font-size: 0.5625rem; line-height: 1.2; }}
        .sl-tp-compact .sl {{ color: var(--destructive); }}
        .sl-tp-compact .tp {{ color: var(--success); }}
        .reasoning-preview {{ max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.5625rem; color: var(--muted-fg); font-style: italic; }}
        .reasoning-preview.one-line-truncate {{ display: block; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    </style>
</head>
<body>
    <div class="ambient-blobs">
        <div class="ambient-blob ambient-blob-1"></div>
        <div class="ambient-blob ambient-blob-2"></div>
    </div>
    <div class="container">
        <header class="header glass">
            <h1>Activity Log</h1>
            <a href="/">Back to Dashboard</a>
        </header>
        <div class="fixture-banner">FIXTURE DATA - Not live trading data</div>
        <div class="filters glass">
            <label>Date</label>
            <input type="date" value="2026-09-24" />
            <label>Strategy</label>
            <select><option value="">All Strategies</option></select>
            <button style="margin-left: auto; background: var(--accent); color: white; border: none; padding: 5px 10px; border-radius: var(--radius-sm); cursor: pointer; font-weight: 600; font-size: 0.6875rem;">Refresh</button>
        </div>
        <div class="summary-strip">
            <div class="summary-card glass bot-stats">
                <div class="summary-value">{summary['bot_trades']}</div>
                <div class="summary-label">Bot Trades</div>
            </div>
            <div class="summary-card glass bot-stats">
                <div class="summary-value {win_rate_class}">{summary['bot_win_rate']:.1f}%</div>
                <div class="summary-label">Bot Win Rate</div>
            </div>
            <div class="summary-card glass bot-stats">
                <div class="summary-value {bot_pnl_class}">{format_currency(bot_pnl)}</div>
                <div class="summary-label">Bot P&L</div>
            </div>
            <div class="summary-card glass bot-stats">
                <div class="summary-value {total_r_class}">{format_r(summary['bot_total_r'])}</div>
                <div class="summary-label">Bot Total R</div>
            </div>
            <div class="summary-card glass">
                <div class="summary-value">{summary['trades']}</div>
                <div class="summary-label">All Trades</div>
            </div>
            <div class="summary-card glass">
                <div class="summary-value {all_pnl_class}">{format_currency(all_pnl)}</div>
                <div class="summary-label">All P&L</div>
            </div>
        </div>
        <div class="trades-table-container glass">
            <table id="trades-table">
                <thead>
                    <tr>
                        <th></th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Strategy</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>Qty</th>
                        <th>SL/TP</th>
                        <th>Exit</th>
                        <th>Hold</th>
                        <th>P&L</th>
                        <th>R</th>
                        <th>Why</th>
                    </tr>
                </thead>
                <tbody>{trade_rows}
                </tbody>
            </table>
        </div>
    </div>
    <script>
        document.querySelectorAll('tr.expandable').forEach(row => {{
            row.addEventListener('click', () => {{
                const idx = row.dataset.idx;
                const reasoningRow = document.querySelector(`tr.reasoning-row[data-idx="${{idx}}"]`);
                row.classList.toggle('expanded');
                reasoningRow.classList.toggle('visible');
            }});
        }});
    </script>
</body>
</html>'''
    
    output_path = workspace / 'tests' / 'fixtures' / 'activity_screenshot_demo.html'
    with open(output_path, 'w') as f:
        f.write(output)
    print(f"Generated: {output_path}")


if __name__ == '__main__':
    main()
