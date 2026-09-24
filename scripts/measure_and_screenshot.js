#!/usr/bin/env node
/**
 * Measure layout at 1280px viewport and take screenshot.
 * Verifies no horizontal overflow and all critical columns are visible.
 */
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

async function main() {
    const htmlPath = path.resolve(__dirname, '../tests/fixtures/activity_screenshot_demo.html');
    const screenshotPath = '/opt/cursor/artifacts/screenshots/activity_page_fixture.png';
    
    // Ensure screenshot directory exists
    const screenshotDir = path.dirname(screenshotPath);
    if (!fs.existsSync(screenshotDir)) {
        fs.mkdirSync(screenshotDir, { recursive: true });
    }
    
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    
    // Set viewport to exactly 1280px CSS width with DPR 1
    await page.setViewport({
        width: 1280,
        height: 900,
        deviceScaleFactor: 1
    });
    
    // Load the HTML file
    await page.goto(`file://${htmlPath}`, { waitUntil: 'networkidle0' });
    
    // Wait for fonts to load
    await page.evaluate(() => document.fonts.ready);
    await new Promise(r => setTimeout(r, 500));
    
    // Measure layout
    const measurements = await page.evaluate(() => {
        const docScrollWidth = document.documentElement.scrollWidth;
        const docClientWidth = document.documentElement.clientWidth;
        
        const table = document.querySelector('table');
        const tableScrollWidth = table ? table.scrollWidth : 0;
        const tableClientWidth = table ? table.clientWidth : 0;
        const tableContainer = document.querySelector('.trades-table-container');
        const containerScrollWidth = tableContainer ? tableContainer.scrollWidth : 0;
        const containerClientWidth = tableContainer ? tableContainer.clientWidth : 0;
        
        // Get header cells for Hold, P&L, R
        const headers = Array.from(document.querySelectorAll('th'));
        const holdTh = headers.find(th => th.textContent.trim() === 'Hold');
        const pnlTh = headers.find(th => th.textContent.trim() === 'P&L');
        const rTh = headers.find(th => th.textContent.trim() === 'R');
        
        const holdRect = holdTh ? holdTh.getBoundingClientRect() : null;
        const pnlRect = pnlTh ? pnlTh.getBoundingClientRect() : null;
        const rRect = rTh ? rTh.getBoundingClientRect() : null;
        
        // Check expanded reasoning row
        const reasoningRow = document.querySelector('.reasoning-row.visible');
        const reasoningGrid = reasoningRow ? reasoningRow.querySelector('.reasoning-grid') : null;
        const reasoningNarrative = reasoningRow ? reasoningRow.querySelector('.reasoning-narrative') : null;
        
        return {
            docScrollWidth,
            docClientWidth,
            docOverflow: docScrollWidth > 1280,
            tableScrollWidth,
            tableClientWidth,
            tableOverflow: tableScrollWidth > tableClientWidth,
            containerScrollWidth,
            containerClientWidth,
            containerOverflow: containerScrollWidth > containerClientWidth,
            holdRight: holdRect ? holdRect.right : null,
            pnlRight: pnlRect ? pnlRect.right : null,
            rRight: rRect ? rRect.right : null,
            holdVisible: holdRect ? holdRect.right <= 1280 : false,
            pnlVisible: pnlRect ? pnlRect.right <= 1280 : false,
            rVisible: rRect ? rRect.right <= 1280 : false,
            reasoningGridWidth: reasoningGrid ? reasoningGrid.scrollWidth : null,
            reasoningGridClientWidth: reasoningGrid ? reasoningGrid.clientWidth : null,
            reasoningNarrativeWidth: reasoningNarrative ? reasoningNarrative.scrollWidth : null,
            reasoningNarrativeClientWidth: reasoningNarrative ? reasoningNarrative.clientWidth : null,
        };
    });
    
    console.log('=== Layout Measurements at 1280px (DPR 1) ===');
    console.log(`Document scrollWidth: ${measurements.docScrollWidth}`);
    console.log(`Document clientWidth: ${measurements.docClientWidth}`);
    console.log(`Document overflow: ${measurements.docOverflow ? 'YES - FAIL' : 'NO - PASS'}`);
    console.log('');
    console.log(`Table scrollWidth: ${measurements.tableScrollWidth}`);
    console.log(`Table clientWidth: ${measurements.tableClientWidth}`);
    console.log(`Table overflow: ${measurements.tableOverflow ? 'YES - FAIL' : 'NO - PASS'}`);
    console.log('');
    console.log(`Container scrollWidth: ${measurements.containerScrollWidth}`);
    console.log(`Container clientWidth: ${measurements.containerClientWidth}`);
    console.log(`Container overflow: ${measurements.containerOverflow ? 'YES - FAIL' : 'NO - PASS'}`);
    console.log('');
    console.log('=== Critical Column Visibility ===');
    console.log(`Hold header right: ${measurements.holdRight?.toFixed(1)}px (visible: ${measurements.holdVisible ? 'YES' : 'NO'})`);
    console.log(`P&L header right: ${measurements.pnlRight?.toFixed(1)}px (visible: ${measurements.pnlVisible ? 'YES' : 'NO'})`);
    console.log(`R header right: ${measurements.rRight?.toFixed(1)}px (visible: ${measurements.rVisible ? 'YES' : 'NO'})`);
    console.log('');
    console.log('=== Expanded Reasoning Panel ===');
    console.log(`Reasoning grid scrollWidth: ${measurements.reasoningGridWidth}`);
    console.log(`Reasoning grid clientWidth: ${measurements.reasoningGridClientWidth}`);
    console.log(`Reasoning narrative scrollWidth: ${measurements.reasoningNarrativeWidth}`);
    console.log(`Reasoning narrative clientWidth: ${measurements.reasoningNarrativeClientWidth}`);
    
    // Take screenshot
    await page.screenshot({
        path: screenshotPath,
        fullPage: true
    });
    console.log('');
    console.log(`Screenshot saved to: ${screenshotPath}`);
    
    // Get file size
    const stats = fs.statSync(screenshotPath);
    console.log(`Screenshot size: ${(stats.size / 1024).toFixed(1)} KB`);
    
    await browser.close();
    
    // Exit with error if overflow detected
    const hasOverflow = measurements.docOverflow || !measurements.holdVisible || !measurements.pnlVisible || !measurements.rVisible;
    if (hasOverflow) {
        console.log('');
        console.log('ERROR: Layout verification FAILED');
        process.exit(1);
    }
    
    console.log('');
    console.log('SUCCESS: Layout verification PASSED');
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
