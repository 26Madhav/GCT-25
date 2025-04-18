import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate




# EMD
#pseudo def, for code running
def snr(data, noise):
    # return 0
    if (len(noise))==0:
        return 0
    return np.sqrt(np.sum(data**2)/np.sum(noise**2))*0.000
# correlation points of upper and lower envelope
def max_min_peaks(data, point_type:str = "emd"):
    assert(point_type in ["emd", "ext", "mid"])
    emd_id = {"emd": 1, "ext": 2, "mid": 3}
    point_type = emd_id[point_type]

    point_num = np.size(data)
    peaks_max = signal.argrelextrema(data, np.greater)[0]
    peaks_min = signal.argrelextrema(data, np.less)[0]

    if point_type > 1:
        peaks_max = np.concatenate(([0], peaks_max, [point_num-1]))
        peaks_min = np.concatenate(([0], peaks_min, [point_num-1]))

    if point_type > 2:
        _tmp = np.sort(np.concatenate(([0], peaks_max, peaks_min, [point_num-1])))
        _tmp = np.delete(_tmp, np.where(_tmp[1:] == _tmp[:-1]))
        mid_point = []
        for i in range(_tmp.shape[0] - 1):
            mid_point.append(int((_tmp[i]+_tmp[i+1]) / 2))

        peaks_max = np.sort(np.concatenate((peaks_max, mid_point)))
        peaks_min = np.sort(np.concatenate((peaks_min, mid_point)))

    peaks_max = np.delete(peaks_max, np.where(peaks_max[1:] == peaks_max[:-1]))
    peaks_min = np.delete(peaks_min, np.where(peaks_min[1:] == peaks_min[:-1]))

    return peaks_max, peaks_min

# cubic interpolation sampling for less than 4 points
def cubic_spline_3pts(x, y, T):
    """
    Apparently scipy.interpolate.interp1d does not support
    cubic spline for less than 4 points.
    """
    x0, x1, x2 = x
    y0, y1, y2 = y

    x1x0, x2x1 = x1 - x0, x2 - x1
    y1y0, y2y1 = y1 - y0, y2 - y1
    _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1

    m11, m12, m13 = 2 * _x1x0, _x1x0, 0
    m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
    m31, m32, m33 = 0, _x2x1, 2.0 * _x2x1

    v1 = 3 * y1y0 * _x1x0 * _x1x0
    v3 = 3 * y2y1 * _x2x1 * _x2x1
    v2 = v1 + v3

    M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    v = np.array([v1, v2, v3]).T
    k = np.array(np.linalg.inv(M).dot(v))

    a1 = k[0] * x1x0 - y1y0
    b1 = -k[1] * x1x0 + y1y0
    a2 = k[1] * x2x1 - y2y1
    b2 = -k[2] * x2x1 + y2y1

    t = T
    t1 = (T[np.r_[T < x1]] - x0) / x1x0
    t2 = (T[np.r_[T >= x1]] - x1) / x2x1
    t11, t22 = 1.0 - t1, 1.0 - t2

    q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
    q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
    q = np.append(q1, q2)

    return t, q

# cubic interpolation sampling for more than 3 points
def envelopes(data, peaks_max, peaks_min):
    point_num = len(data)

    if len(peaks_max) > 3:
        inp_max = interpolate.splrep(peaks_max, data[peaks_max], k=3)
        fit_max = interpolate.splev(np.arange(point_num), inp_max)
    else:
        _, fit_max = cubic_spline_3pts(peaks_max, data[peaks_max], np.arange(len(data)))

    if len(peaks_min) > 3:
        inp_min = interpolate.splrep(peaks_min, data[peaks_min], k=3)
        fit_min = interpolate.splev(np.arange(point_num), inp_min)
    else:
        _, fit_min = cubic_spline_3pts(peaks_min, data[peaks_min], np.arange(len(data)))

    return fit_max, fit_min

# determine if modal decomposition is over
def imf_judge(x: np.array, y: np.array):
    """
    x: The sequence after decomposition.
    y: The sequence before decomposition.
    """
    if (y.max() - y.min()) != 0 and ((x - y)**2).sum() / (y.max() - y.min()) < 0.001:
        return True

    if not np.any(x == 0) and (((x - y) / x)**2).sum() < 0.2:
        return True

    if (y**2).sum() != 0 and ((x - y)**2).sum() / (y**2).sum() < 0.2:
        return True

    return False

# Empirical Mode Decomposition (EMD)
def emd(signal):
    origin_signal = signal.copy()
    # extrema point
    emd_peaks_max, emd_peaks_min = max_min_peaks(signal, "emd")
    # print ('sig ',signal,'max_peak ',emd_peaks_max)

    # emd
    std_continue, old_std = 0, 0.0
    # envelope line
    emd_up_envelopes, emd_down_envelopes = 0, 0
    continue_time = 511

    while True:
        # number of extreme points
        if len(emd_peaks_max) < 3 or len(emd_peaks_min) < 3:
            break

        fit_max, fit_min = envelopes(signal, emd_peaks_max, emd_peaks_min)
        emd_up_envelopes, emd_down_envelopes = emd_up_envelopes + fit_max, emd_down_envelopes + fit_min
        signal_old = signal.copy()
        signal = signal - (fit_max + fit_min) / 2

        emd_peaks_max, emd_peaks_min = max_min_peaks(signal, "emd")
        pass_zero = np.sum(signal[:-1] * signal[1:] < 0)

        std = abs((fit_max + fit_min) / 2 / origin_signal).mean()
        std_continue = (std_continue << 1) & continue_time
        std_continue += 1 if abs(old_std - std) < 1e-6 else 0
        old_std = std

        if (abs(pass_zero - len(emd_peaks_max) - len(emd_peaks_min)) < 2) or imf_judge(signal, signal_old) or std_continue == continue_time:
            break

    if isinstance(emd_up_envelopes, int) and isinstance(emd_down_envelopes, int):
        return signal, signal
    return emd_up_envelopes, emd_down_envelopes

# Extrema Empirical Mode Decomposition (extemd)
def extemd(signal):
    origin_signal = signal.copy()
    # extrema point
    emd_peaks_max, emd_peaks_min = max_min_peaks(signal, "ext")

    # emd
    std_continue, old_std = 0, 0.0
    # envelope line
    emd_up_envelopes, emd_down_envelopes = 0, 0
    continue_time = 511

    while True:
        # number of extreme points
        if len(emd_peaks_max) < 3 or len(emd_peaks_min) < 3:
            break

        fit_max, fit_min = envelopes(signal, emd_peaks_max, emd_peaks_min)
        emd_up_envelopes, emd_down_envelopes = emd_up_envelopes + fit_max, emd_down_envelopes + fit_min
        signal_old = signal.copy()
        signal = signal - (fit_max + fit_min) / 2

        emd_peaks_max, emd_peaks_min = max_min_peaks(signal, "ext")
        pass_zero = np.sum(signal[:-1] * signal[1:] < 0)

        std = abs((fit_max + fit_min) / 2 / origin_signal).mean()
        std_continue = (std_continue << 1) & continue_time
        std_continue += 1 if abs(old_std - std) < 1e-6 else 0
        old_std = std

        if (abs(pass_zero - len(emd_peaks_max) - len(emd_peaks_min)) < 2) or imf_judge(signal, signal_old) or std_continue == continue_time:
            break

    if isinstance(emd_up_envelopes, int) and isinstance(emd_down_envelopes, int):
        return signal, signal
    return emd_up_envelopes, emd_down_envelopes

# Aliased Complete Ensemble Empirical Mode Decomposition (ACEEMD)
def aceemd(extsignal, midsignal, alpha = 0.5):
    origin_signal = extsignal.copy()
    # extrema point
    ext_peaks_max, ext_peaks_min = max_min_peaks(extsignal, "ext")
    mid_peaks_max, mid_peaks_min = max_min_peaks(midsignal, "mid")

    # emd
    std_continue, old_std = 0, 0.0
    # envelope line
    ext_up_envelopes, ext_down_envelopes = 0, 0
    mid_up_envelopes, mid_down_envelopes = 0, 0
    continue_time = 511

    while True:
        # 极值点个数 //to be decoded when you're free :)
        if len(ext_peaks_max) < 3 or len(ext_peaks_min) < 3:
            break

        # 中点点集 //to be decoded when you're free :)
        fit_max, fit_min = envelopes(midsignal, mid_peaks_max, mid_peaks_min)
        mid_up_envelopes, mid_down_envelopes = mid_up_envelopes + fit_max, mid_down_envelopes + fit_min
        midsignal = midsignal - (fit_max + fit_min) / 2

        mid_peaks_max, mid_peaks_min = max_min_peaks(midsignal, "mid")

        # 端点点集 //to be decoded when you're free :)
        fit_max, fit_min = envelopes(extsignal, ext_peaks_max, ext_peaks_min)
        ext_up_envelopes, ext_down_envelopes = ext_up_envelopes + fit_max, ext_down_envelopes + fit_min
        extsignal_old = extsignal.copy()
        extsignal = extsignal - (fit_max + fit_min) / 2

        ext_peaks_max, ext_peaks_min = max_min_peaks(extsignal, "ext")

        # 判断循环 //to be decoded when you're free :)
        pass_zero = np.sum(extsignal[:-1] * extsignal[1:] < 0)

        std = abs((fit_max + fit_min) / 2 / origin_signal).mean()
        std_continue = (std_continue << 1) & continue_time
        std_continue += 1 if abs(old_std - std) < 1e-6 else 0
        old_std = std

        if (abs(pass_zero - len(ext_peaks_max) - len(ext_peaks_min)) < 2) or imf_judge(extsignal, extsignal_old) or std_continue == continue_time:
            break

    if isinstance(ext_up_envelopes, int) and isinstance(mid_up_envelopes, int) and isinstance(ext_down_envelopes, int) and isinstance(mid_down_envelopes, int):
        return signal, signal
    return ext_up_envelopes * (1-alpha) + mid_up_envelopes * alpha, ext_down_envelopes * (1-alpha) + mid_down_envelopes * alpha







class EMD_dealt:
    def __init__(self, source_data: np.array, emd_type: int=3, imf_times: int=10):
        '''
        :param source_data: three-dimensional data, respectively, batch size, number of days, data per stock
        :param emd_type: the emd type, 1->iceemd, 2->eceemd, 3->aceemd (default)
        '''
        #  Gaussian noise
        noise_list = []
        win_len = source_data.shape[-2]
        for _ in range(imf_times // 2):
            noise = np.random.randn(win_len)
            n_up_envelopes, n_down_envelopes = emd(noise)
            noise_list.append((n_up_envelopes + n_down_envelopes) / 2 / np.std(noise))

            n_up_envelopes, n_down_envelopes = emd(-noise)
            noise_list.append((n_up_envelopes + n_down_envelopes) / 2 / np.std(-noise))

        # emd process
        emd_result = []
        for s in range(len(source_data)):
            emd_tmp = []
            for d in range(len(source_data[s].T)):
                _data = (source_data[s].T)[d]
                # envelope line
                up_list, down_list = [], []
                # iceemd & eceemd
                if emd_type == 1 or emd_type == 2:
                    for noise in noise_list:
                        _emd_data = _data.copy() + noise * snr(_data, noise)
                        up, down = emd(_emd_data) if emd_type == 1 else extemd(_emd_data)

                        up_list.append(up)
                        down_list.append(down)
                # aceemd
                else:
                    for i in range(imf_times // 2):
                        _exemd_data = _data.copy() + noise_list[2*i] * snr(_data, noise_list[2*i])
                        _acemd_data = _data.copy() + noise_list[2*i+1] * snr(_data, noise_list[2*i+1])
                        up, down = aceemd(_exemd_data, _acemd_data, 0.3)

                        up_list.append(up)
                        down_list.append(down)
                # denoise
                emd_tmp.append((np.array(up_list).mean(axis=0) + np.array(down_list).mean(axis=0)) / 2)
            emd_result.append(np.array(emd_tmp).T)
        self.emd_result = np.array(emd_result)
        # self.emd_result=source_data  #This line removes cleaning

    def getEmdResult(self):
        return self.emd_result


# End EMD



df=pd.read_csv('/Users/madhav/Downloads/FINANCE/GTC-Kailasa/data/NIFTY 50_daily_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= pd.to_datetime('2020-01-01')]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df.reset_index(drop=True, inplace=True)







#DEFINING COLUMNS



df['lips'] = 0
df['teeth'] = 0
df['jaw'] = 0
df['signal']=0
df['return']=0
df["strategy_return"]=0
df["Portfolio"]=0
df["Transaction"]=0
df["Transaction Cost"]=0
df["Portfolio Before T"]=0
df["Portfolio After T"]=0
df['S_Normal_Return']=0
df['Sreturn']=0
df['volatility']=0
df['Drawdown']=0

df = df.astype({
    'signal': 'int',
    'return': 'float',
    'strategy_return': 'float',
    'Portfolio': 'float',
    'Transaction': 'int',
    'Transaction Cost': 'float',
    'Portfolio Before T': 'float',
    'Portfolio After T': 'float'
})


# REMOVING NOISE FROM THE CLOSING DATA THROUGH WINDOWS TO      PREVENT FORWARD BIASES
prices = df['close'].values
n = len(prices)
cleaned = np.full(n, np.nan)
# minimum look‑back window before you trust the EMD result
min_win = 50  
for t in range(min_win-1, n):
    window = prices[:t+1]                  # only data ≤ t
    # reshape for EMD_dealt: (batch=1, window_length, features=1)
    emd_proc = EMD_dealt(window.reshape(1, -1, 1), emd_type=3, imf_times=10)
    cleansed_window = emd_proc.getEmdResult().reshape(-1)
    cleaned[t] = cleansed_window[-1]       # only take today’s cleaned value
df['close_cleaned'] = cleaned





# ALLIGATOR STRATEGY
df['lips'] = df['close_cleaned'].ewm(span=5, adjust=False).mean()
df['teeth'] = df['close_cleaned'].ewm(span=8, adjust=False).mean()
df['jaw'] = df['close_cleaned'].ewm(span=13, adjust=False).mean()


for i in range (0,len(df)):
    if(df.loc[i,'close_cleaned']>df.loc[i,'lips'] and df.loc[i,'lips']>df.loc[i,'teeth'] and df.loc[i,'teeth']>df.loc[i,'jaw']):
        df.loc[i,'signal']=1
    else:
        df.loc[i,'signal']=0

# FRACTALS SIGNAL
for i in range (0,len(df)):
    if(df.loc[i,'lips']<df.loc[i,'teeth'] and df.loc[i,'teeth']<df.loc[i,'jaw']and df.loc[i,'close_cleaned']>df.loc[i,'teeth']):
        df.loc[i,'signal']=1


# VOLATILITY FILTER

for i in range (0,len(df)-1):
    df.loc[i,'return']=float((df.loc[i+1,'close']-df.loc[i,'close'])/df.loc[i,'close'])
df["Sreturn"]=df["return"].shift(1)
df['volatility'] = (df['Sreturn'].rolling(window=30).std())*15.8745
for i in range (38,len(df)):
    if(df.loc[i,'volatility']>= 0.4   ):
        df.loc[i,'signal']=0


#  RSI
def calculate_rsi(df, window=7):
    delta = df['close_cleaned'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df)


# take trades when RSI is below 15 (overbought)
df.loc[df['rsi'] < 15, 'signal'] = 1


#  transaction
for i in range(1,len(df)):
    df.loc[i,'Transaction']=df.loc[i,'signal']-df.loc[i-1,'signal']

df["strategy_return"]=df['signal']*df['return']





df.loc[0,'Portfolio Before T']=float(df.loc[0,'close'])
df.loc[0,'Portfolio After T']=float(df.loc[0,'close'])
df.loc[0,'Portfolio']=float(df.loc[0,'close'])

for i in range (1,len(df)):
    df.loc[i,'Portfolio']=(df.loc[i-1,'Portfolio']*(1+df.loc[i-1,'strategy_return']))


for i in range (1,len(df)):

    df.loc[i,'Portfolio Before T']=df.loc[i-1,'Portfolio After T']*(1+df.loc[i-1,'strategy_return'])
    df.loc[i,'Portfolio After T']=df.loc[i,'Portfolio Before T']-0.0001*df.loc[i,'Portfolio Before T']*abs(df.loc[i,'Transaction'])
    df.iloc[i,14]=float(0.0001*df.loc[i,'Portfolio Before T']*abs(df.loc[i,'Transaction']))


       #  PLOT
plt.figure(figsize=(10, 5))


plt.plot(df['date'], df['close']/float(df.loc[0,'close'])*10000000, label="Buy and Hold ", color='blue')
plt.plot(df['date'], df['Portfolio After T']/float(df.loc[0,'close'])*10000000, label="Portfolio after Transaction Charges", color='orange')
plt.xlabel("date")
plt.ylabel("Portfolio Value")
plt.title("Comparison of Two Strategies")
plt.legend()
plt.grid(True)

plt.show()



df.to_csv('samoutput.csv')




df['S_Normal_Return']=0
for i in range(0,len(df)-1):
    df.iloc[i,17]=(df.iloc[i+1,16]-df.loc[i,'Portfolio After T'])/df.loc[i,'Portfolio After T']


r=df["S_Normal_Return"]


#sharpe Ratio
sharpe_ratio=(r.mean()/r.std())*15.8745

#Sortino 
downside_returns = [r if r<0 else 0 for r in r]
downside_deviation = np.std(downside_returns)
sortino_ratio = (r.mean()) / downside_deviation*15.8745

print("\n \nSharpe ratio is \n \n ",sharpe_ratio)
print(" \n \n Sortino Ratio is \n \n ",sortino_ratio)



s=[]
s1=pd.Series(s)


for i in range(0,len(df)):

    s2=pd.Series(float(df.loc[i,'Portfolio After T']))

    s1=pd.concat([s1,s2],ignore_index=True)

    max1=s1.max()
    drawdown=(float(max1)-float(df.loc[i,'Portfolio After T']))/float(max1)*100
    df.loc[i,'Drawdown']=drawdown
df.to_csv("samoutput.csv")





# Calculate Annualized Return
initial_value = df.loc[0, 'Portfolio After T']
final_value = df.loc[len(df)-1, 'Portfolio After T']
years = len(df) / 252  # Assuming 252 trading days in a year

annualized_return = (final_value / initial_value) ** (1 / years) - 1

# Calculate Maximum Drawdown
max_drawdown = df['Drawdown'].max()

# Calculate Calmar Ratio
calmar_ratio = annualized_return / (max_drawdown / 100)

print(f"Calmar Ratio: {calmar_ratio}")



# Initialize variables to store trade results
trade_results = []
open_price = None

# Iterate through the rows to track trades
for i in range(1, len(df)):
    if df.loc[i, 'signal'] == 1 and df.loc[i-1, 'signal'] == 0:  # Position opened
        open_price = df.loc[i, 'close_cleaned']
    elif df.loc[i, 'signal'] == 0 and df.loc[i-1, 'signal'] == 1:  # Position closed
        if open_price is not None:
            close_price = df.loc[i, 'close_cleaned']
            trade_return = (close_price - open_price) / open_price  # Calculate return
            trade_results.append(trade_return)  # Store trade return
            open_price = None  # Reset open price

# Calculate win rate
wins = sum(1 for r in trade_results if r > 0)
total_trades = len(trade_results)
win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

print(f"Win Rate: {win_rate:.2f}%")

