import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import time
import pytz

# Load the data
df = pd.read_csv("/Users/praneshwarkk/Downloads/archive-2/NIFTY BANK_minute_data.csv")

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Drop rows with null or corrupt data
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Set 'date' as the index
df.set_index('date', inplace=True)

# Resample to 15-minute intervals
df = df.resample('15T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Drop the 'volume' column
df.drop(columns='volume', inplace=True)

# Filter rows from Jan 1, 2020 to Mar 31, 2025
df = df.loc['2020-01-01':'2025-03-31']

# Adaptive SuperTrend implementation
def adaptive_supertrend(df, atr_window=5, atr_multiplier=2.5):
    df = df.copy()
    df['hl2'] = (df['high'] + df['low']) / 2
    df['tr'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    df['atr'] = df['tr'].rolling(atr_window).mean()

    # Adaptive volatility calculation
    volatility = df['close'].rolling(window=atr_window).std()
    normalized_volatility = volatility / volatility.rolling(window=atr_window).mean()
    df['adaptive_atr'] = df['atr'] * normalized_volatility

    # Bands calculation
    df['upperband'] = df['hl2'] + atr_multiplier * df['adaptive_atr']
    df['lowerband'] = df['hl2'] - atr_multiplier * df['adaptive_atr']

    # Trend detection logic
    df['in_uptrend'] = True
    for current in range(1, len(df)):
        prev = current - 1

        if df['close'].iloc[current] > df['upperband'].iloc[prev]:
            df['in_uptrend'].iloc[current] = True
        elif df['close'].iloc[current] < df['lowerband'].iloc[prev]:
            df['in_uptrend'].iloc[current] = False
        else:
            df['in_uptrend'].iloc[current] = df['in_uptrend'].iloc[prev]
            if df['in_uptrend'].iloc[current] and df['lowerband'].iloc[current] < df['lowerband'].iloc[prev]:
                df['lowerband'].iloc[current] = df['lowerband'].iloc[prev]
            if not df['in_uptrend'].iloc[current] and df['upperband'].iloc[current] > df['upperband'].iloc[prev]:
                df['upperband'].iloc[current] = df['upperband'].iloc[prev]

    # Generate signals with forward-looking bias prevention
    df['signal'] = np.where(df['in_uptrend'], 1, -1)
    df['signal'] = df['signal'].shift(1).fillna(0)
    
    return df

# Apply strategy
df = adaptive_supertrend(df)

# --- Critical Fix: Proper Timezone Handling ---
if df.index.tz is None:
    # First localize to UTC if coming from naive timestamp
    df.index = df.index.tz_localize('UTC')
# Convert to IST
df.index = df.index.tz_convert('Asia/Kolkata')

# --- Time-Based Trade Filter ---
# Define NSE trading hours boundaries
trade_restriction_mask = (
    (df.index.time >= time(9, 15)) &  # Market open
    (df.index.time < time(9, 45)) |    # First 30 minutes
    (df.index.time >= time(15, 0)) &   # Last 30 minutes start
    (df.index.time <= time(15, 30))    # Market close
)

# Neutralize signals during restricted periods
df.loc[trade_restriction_mask, 'signal'] = 0

# --- Performance Calculation ---
# Clean data before calculations
df = df.dropna(subset=['signal', 'close']).copy()

# Returns calculation
df['return'] = df['close'].pct_change().fillna(0)
df['strategy_return'] = df['return'] * df['signal']

# Portfolio simulation
initial_investment = 1e7  # 1 crore INR
df['cumulative_strategy_value'] = (1 + df['strategy_return']).cumprod() * int(initial_investment / (0.2 * 75 * df['close'].iloc[0])) * (75 * df['close'].iloc[0])
df['cumulative_market_value'] = (1 + df['return']).cumprod() * initial_investment

# --- Risk-Reward Metrics ---
# Clean data before metric calculations
df_metrics = df.dropna(subset=['cumulative_strategy_value', 'cumulative_market_value'])

# Directional accuracy
correct_signals = np.sign(df_metrics['signal']) == np.sign(df_metrics['return'])
direction_accuracy = correct_signals.sum() / len(df_metrics)

# Sharpe ratio (annualized)
trading_periods = 252 * 25  # 15-min intervals in a year
sharpe_ratio = (df_metrics['strategy_return'].mean() / df_metrics['strategy_return'].std()) * np.sqrt(trading_periods)

# Drawdown analysis
df_metrics['cumulative_max'] = df_metrics['cumulative_strategy_value'].cummax()
df_metrics['drawdown'] = df_metrics['cumulative_strategy_value'] / df_metrics['cumulative_max'] - 1
max_drawdown = df_metrics['drawdown'].min()

# Drawdown duration calculation
drawdown_end = df_metrics['drawdown'].idxmin()
drawdown_start = df_metrics['cumulative_strategy_value'][:drawdown_end].idxmax()
drawdown_duration = drawdown_end - drawdown_start

# Calmar ratio
total_days = (df_metrics.index[-1] - df_metrics.index[0]).days
annual_return = (df_metrics['cumulative_strategy_value'].iloc[-1] / initial_investment) ** (1 / (total_days / 365)) - 1
calmar_ratio = annual_return / abs(max_drawdown)

# CAGR (Compounded Annual Growth Rate)
ending_value = df_metrics['cumulative_strategy_value'].iloc[-1]
cagr = (ending_value / initial_investment) ** (1 / (total_days / 365)) - 1

# Sortino Ratio (uses downside deviation instead of total std deviation)
downside_returns = df_metrics.loc[df_metrics['strategy_return'] < 0, 'strategy_return']
downside_std = downside_returns.std()
sortino_ratio = (df_metrics['strategy_return'].mean() / downside_std) * np.sqrt(trading_periods) if downside_std > 0 else np.nan

# Win Rate (percentage of profitable trades)
winning_trades = df_metrics.loc[df_metrics['strategy_return'] > 0]
win_rate = len(winning_trades) / len(df_metrics[df_metrics['signal'] != 0])

# --- Results Display ---
print("\n--- Enhanced Strategy Performance ---")
print(f"Direction Accuracy   : {direction_accuracy * 100:.2f}%")
print(f"Final Strategy Value : ₹{df_metrics['cumulative_strategy_value'].iloc[-1]:,.2f}")
print(f"CAGR                 : {cagr * 100:.2f}%")
print(f"Sharpe Ratio         : {sharpe_ratio:.2f}")
print(f"Sortino Ratio        : {sortino_ratio:.2f}")
print(f"Calmar Ratio         : {calmar_ratio:.2f}")
print(f"Win Rate             : {win_rate * 100:.2f}%")
print(f"Max Drawdown         : {max_drawdown * 100:.2f}%")
print(f"Drawdown Duration    : {drawdown_duration}")

# --- Visualization Fixes ---
plt.figure(figsize=(14, 7))

# Use cleaned data for plotting
plt.plot(df_metrics.index, df_metrics['cumulative_strategy_value'], 
         label='Enhanced Strategy', linewidth=1.5, color='#2ecc71')
plt.plot(df_metrics.index, df_metrics['cumulative_market_value'], 
         label='Buy & Hold', linestyle='--', color='#3498db')

plt.title('Enhanced SuperTrend Strategy vs Buy & Hold (NSE Nifty 50)', pad=20)
plt.xlabel('Date', labelpad=15)
plt.ylabel('Portfolio Value (₹)', labelpad=15)
plt.legend()


plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()
