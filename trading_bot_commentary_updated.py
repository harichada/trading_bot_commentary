<execute_ipython>
import trading_bot_commentary_updated

with open('trading_bot_commentary_updated.py', 'r') as f:
    original_code = f.read()

new_code = original_code.replace(
    """        function updateTickerTape(data) {
            if (!data || data.length === 0) return;
            
            const tickerContent = document.getElementById('ticker-content');
            
            data.forEach(stock => {
                const existingEl = document.getElementById(`ticker-${stock.symbol}`);
                const pnl_percent = stock.pnl_percent || stock.unrealized_pnl / (stock.entry_price * stock.quantity) * 100 || 0;
                const current_price = stock.current_price || stock.last || 0;
                const volatility = stock.volatility || 'N/A';
                const changeClass = pnl_percent >= 0 ? 'positive' : 'negative';
                const changeSymbol = pnl_percent >= 0 ? '+' : '';

                if (existingEl) {
                    // Update existing element
                    existingEl.querySelector('.ticker-price').textContent = `$${current_price.toFixed(2)}`;
                    const changeEl = existingEl.querySelector('.ticker-change');
                    changeEl.textContent = `${changeSymbol}${pnl_percent.toFixed(2)}%`;
                    changeEl.className = `ticker-change ${changeClass}`;

                    // Add animation
                    existingEl.classList.add('price-updated');
                    setTimeout(() => existingEl.classList.remove('price-updated'), 500);
                } else {
                    // Add new element
                    const tickerHTML = `
                        <div class="ticker-item" id="ticker-${stock.symbol}">
                            <span class="ticker-symbol">${stock.symbol}</span>
                            <span class="ticker-price">$${current_price.toFixed(2)}</span>
                            <span class="ticker-change ${changeClass}">${changeSymbol}${pnl_percent.toFixed(2)}%</span>
                            <span class="ticker-volatility">⚡${volatility}%</span>
                        </div>
                    `;
                    tickerContent.insertAdjacentHTML('beforeend', tickerHTML);
                }
            });
        }""",
    """        function updateTickerTape(data) {
            if (!data || data.length === 0) return;
            
            const tickerContent = document.getElementById('ticker-content');
            
            data.forEach(stock => {
                const existingEl = document.getElementById(`ticker-${stock.symbol}`);
                const pnl_percent = stock.pnl_percent !== undefined ? stock.pnl_percent : (stock.unrealized_pnl / (stock.entry_price * stock.quantity) * 100) || 0;
                const current_price = stock.current_price !== undefined ? stock.current_price : stock.last || 0;
                const volatility = stock.volatility !== undefined ? stock.volatility.toFixed(1) : 'N/A';
                const changeClass = pnl_percent >= 0 ? 'positive' : 'negative';
                const changeSymbol = pnl_percent >= 0 ? '+' : '';

                if (existingEl) {
                    // Update existing element
                    existingEl.querySelector('.ticker-price').textContent = `$${current_price.toFixed(2)}`;
                    const changeEl = existingEl.querySelector('.ticker-change');
                    changeEl.textContent = `${changeSymbol}${pnl_percent.toFixed(2)}%`;
                    changeEl.className = `ticker-change ${changeClass}`;

                    // Add animation
                    existingEl.classList.add('price-updated');
                    setTimeout(() => existingEl.classList.remove('price-updated'), 500);
                } else {
                    // Add new element
                    const tickerHTML = `
                        <div class="ticker-item" id="ticker-${stock.symbol}">
                            <span class="ticker-symbol">${stock.symbol}</span>
                            <span class="ticker-price">$${current_price.toFixed(2)}</span>
                            <span class="ticker-change ${changeClass}">${changeSymbol}${pnl_percent.toFixed(2)}%</span>
                            <span class="ticker-volatility">⚡${volatility}%</span>
                        </div>
                    `;
                    tickerContent.insertAdjacentHTML('beforeend', tickerHTML);
                }
            });
        }"""
)

with open('trading_bot_commentary_updated.py', 'w') as f:
    f.write(new_code)
</execute_ipython>
