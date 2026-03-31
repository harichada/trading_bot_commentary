# Rudra Trading Engine — Terms of Service

**Last Updated: March 31, 2026**

By accessing or using the Rudra Trading Engine ("Service"), you agree to be bound by these Terms of Service ("Terms"). If you do not agree, do not use the Service.

---

## 1. SERVICE DESCRIPTION

Rudra Trading Engine is an automated algorithmic trading platform that executes trades on your brokerage account based on predefined strategies. The Service connects to your brokerage account (e.g., Alpaca, Interactive Brokers) via API and places trades on your behalf according to the strategy configuration you select.

**The Service is a tool, not financial advice.** We do not provide investment advice, tax advice, or recommendations to buy or sell any securities.

---

## 2. RISK DISCLOSURE AND DISCLAIMER

### 2.1 No Guarantee of Profits

**PAST PERFORMANCE IS NOT INDICATIVE OF FUTURE RESULTS.** Backtested results, simulated performance, and historical returns shown on the platform are hypothetical and do not represent actual trading. Actual results may differ materially from backtested results due to, but not limited to:

- Market conditions that differ from historical data
- Slippage, latency, and execution differences
- Broker API outages or failures
- Software bugs or system failures
- Changes in market microstructure, regulations, or liquidity
- Black swan events, circuit breakers, and halted securities

### 2.2 Risk of Loss

**Trading securities involves substantial risk of loss.** You may lose some or all of your invested capital. You should only trade with capital you can afford to lose entirely. Algorithmic trading carries additional risks including but not limited to:

- **Technical failure**: The software may malfunction, lose connectivity, or execute trades incorrectly
- **Stop-loss failure**: Stop orders may not execute at the expected price during fast-moving markets, gaps, or halts
- **Overnight risk**: Positions held overnight are subject to gap risk from after-hours news or events
- **Liquidity risk**: The system may enter positions in stocks with insufficient liquidity to exit at desired prices
- **Concentration risk**: The system may allocate a significant portion of capital to a small number of positions

### 2.3 Not a Registered Investment Advisor

Rudra Trading Engine and its operators are **NOT** registered as an investment advisor, broker-dealer, or financial planner with the Securities and Exchange Commission (SEC), the Financial Industry Regulatory Authority (FINRA), or any state securities regulatory authority. The Service does not provide personalized investment advice.

### 2.4 Your Responsibility

You are solely responsible for:

- Deciding whether to use the Service and which strategies to enable
- Setting appropriate risk parameters for your financial situation
- Monitoring your account and the Service's activity
- Understanding the strategies and their risks before enabling them
- Ensuring your brokerage account has appropriate permissions and margin
- Tax reporting and compliance for all trades executed on your account
- Complying with all applicable securities laws and regulations

---

## 3. ACCOUNT AND ACCESS

### 3.1 Eligibility

You must be at least 18 years old and legally permitted to trade securities in your jurisdiction. You must have a valid brokerage account with a supported broker.

### 3.2 Brokerage Credentials

You provide your brokerage API credentials to enable the Service to trade on your behalf. Your credentials are encrypted at rest using industry-standard encryption. We do not store your brokerage password — only API keys that you generate specifically for this Service.

**You may revoke access at any time** by deleting your API keys from your brokerage account. This immediately stops all trading activity.

### 3.3 Account Security

You are responsible for maintaining the security of your account credentials. You must notify us immediately of any unauthorized access to your account.

---

## 4. SUBSCRIPTION AND BILLING

### 4.1 Tiers

| Tier | Monthly Fee | Monthly Profit Cap |
|------|-------------|-------------------|
| Free | $0 | $200 |
| Starter | $29 | $1,000 |
| Pro | $59 | $5,000 |
| Enterprise | Custom | Unlimited |

### 4.2 Profit Cap

When your monthly realized profit reaches the cap for your tier, the Service will stop entering new positions and close existing positions at end of day. The cap resets on the 1st of each calendar month.

### 4.3 Billing

- Subscription fees are billed monthly in advance via the payment method you provide.
- You may cancel at any time. Cancellation takes effect at the end of the current billing period.
- No refunds are provided for partial months.
- If your account has a net loss in a given month, your subscription fee is **not** refunded for that month. Trading losses are your responsibility regardless of subscription status.

### 4.4 Free Tier

The Free tier provides limited access with a $200 monthly profit cap. No payment information is required. We reserve the right to modify or discontinue the Free tier at any time.

---

## 5. LIMITATION OF LIABILITY

### 5.1 Disclaimer of Warranties

THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT.

We do not warrant that:
- The Service will be uninterrupted, timely, secure, or error-free
- Trading results will be profitable or match backtested performance
- The Service will be compatible with your brokerage account at all times
- Any defects in the Service will be corrected

### 5.2 Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT SHALL RUDRA TRADING ENGINE, ITS OPERATORS, AFFILIATES, OR LICENSORS BE LIABLE FOR:

- Any trading losses incurred through use of the Service
- Any indirect, incidental, special, consequential, or punitive damages
- Any loss of profits, revenue, data, or business opportunities
- Any damages arising from system failures, bugs, or connectivity issues

**OUR TOTAL LIABILITY SHALL NOT EXCEED THE AMOUNT YOU PAID IN SUBSCRIPTION FEES DURING THE TWELVE (12) MONTHS PRECEDING THE CLAIM.**

### 5.3 Indemnification

You agree to indemnify and hold harmless Rudra Trading Engine and its operators from any claims, damages, losses, or expenses arising from your use of the Service, your trading activities, or your violation of these Terms.

---

## 6. ACCEPTABLE USE

You agree not to:

- Use the Service for any illegal purpose
- Attempt to reverse-engineer, decompile, or extract the source code of proprietary strategies
- Share your account access with unauthorized persons
- Use the Service to manipulate markets or engage in wash trading
- Exceed the rate limits or usage quotas of your tier
- Interfere with or disrupt the Service or its infrastructure

---

## 7. DATA AND PRIVACY

### 7.1 Data We Collect

- Account information (email, name, profile picture from OAuth provider)
- Brokerage API credentials (encrypted at rest)
- Trading activity logs (entries, exits, P&L)
- Configuration settings and preferences
- Usage analytics (page views, feature usage)

### 7.2 Data We Do NOT Collect

- Your brokerage account password
- Your personal financial information beyond what is visible via the brokerage API
- Payment card details (processed by Stripe, never stored on our servers)

### 7.3 Data Usage

Your data is used solely to operate the Service. We do not sell your personal data or trading history to third parties. Aggregated, anonymized performance data may be used to improve the Service.

### 7.4 Data Retention

Your trading data is retained for as long as your account is active plus 7 years (for tax and regulatory purposes). You may request data export or deletion by contacting us.

---

## 8. INTELLECTUAL PROPERTY

The Service, including its strategies, algorithms, user interface, and documentation, is the intellectual property of Rudra Trading Engine. Your subscription grants you a limited, non-exclusive, non-transferable license to use the Service for your personal trading purposes.

---

## 9. TERMINATION

### 9.1 By You

You may terminate your account at any time by canceling your subscription and revoking your brokerage API keys. Any open positions will be closed at market.

### 9.2 By Us

We may suspend or terminate your access at any time, with or without cause, including but not limited to:

- Violation of these Terms
- Non-payment of subscription fees
- Suspected fraudulent or illegal activity
- Technical necessity

### 9.3 Effect of Termination

Upon termination, the Service will stop trading on your account. Any open positions at the time of termination are your responsibility to manage.

---

## 10. MODIFICATIONS

We reserve the right to modify these Terms at any time. Material changes will be communicated via email or in-app notification at least 30 days before taking effect. Continued use of the Service after changes take effect constitutes acceptance of the new Terms.

---

## 11. GOVERNING LAW AND DISPUTE RESOLUTION

These Terms shall be governed by the laws of the State of Delaware, United States, without regard to conflict of law principles. Any disputes shall be resolved through binding arbitration under the rules of the American Arbitration Association, with the arbitration taking place in Wilmington, Delaware.

---

## 12. MISCELLANEOUS

- **Severability**: If any provision of these Terms is found unenforceable, the remaining provisions remain in effect.
- **Entire Agreement**: These Terms constitute the entire agreement between you and Rudra Trading Engine regarding the Service.
- **Waiver**: Failure to enforce any provision does not constitute a waiver.
- **Assignment**: You may not assign your rights under these Terms. We may assign our rights without restriction.

---

## 13. CONTACT

For questions about these Terms, contact:

**Rudra Trading Engine**
Email: support@rudra.trading
Website: https://rudra.trading

---

**BY CLICKING "I AGREE" OR USING THE SERVICE, YOU ACKNOWLEDGE THAT YOU HAVE READ, UNDERSTOOD, AND AGREE TO BE BOUND BY THESE TERMS OF SERVICE AND THE RISK DISCLOSURE CONTAINED HEREIN.**
