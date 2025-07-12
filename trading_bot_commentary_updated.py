<execute_ipython>
import trading_bot_commentary_updated

with open('trading_bot_commentary_updated.py', 'r') as f:
    original_code = f.read()

new_code = original_code.replace(
    """        .settings-row {
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
        """,
    """        .settings-row {
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
        }"""
)

with open('trading_bot_commentary_updated.py', 'w') as f:
    f.write(new_code)
</execute_ipython>
