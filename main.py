import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from moexalgo import Ticker, session
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns
import gudhi
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------
# 1. Загрузка данных с MOEX
# ------------------------------
load_dotenv()

username = os.getenv('MOEX_USERNAME')
password = os.getenv('MOEX_PASSWORD')

if not username or not password:
    raise ValueError("Не удалось загрузить username или password из .env файла")

session.authorize(username, password)

tickers = [
    'SBER', 'GAZP', 'LKOH', 'YNDX', 'ROSN',
    'NVTK', 'GMKN', 'VTBR', 'TATN', 'CHMF',
    'MOEX', 'RUAL', 'AFKS'
]

start_date = '2019-01-01'
end_date = '2025-12-31'
all_data = []

print("Начинаем загрузку данных по акциям...")
for ticker_symbol in tickers:
    print(f"  Загружаем {ticker_symbol}...")
    try:
        ticker = Ticker(ticker_symbol)
        data = ticker.candles(start=start_date, end=end_date, period='1d')
        if data is not None and not data.empty:
            df = pd.DataFrame({
                'date': pd.to_datetime(data['begin']).dt.date,
                ticker_symbol: data['close'].values
            })
            all_data.append(df)
            print(f"    Загружено {len(df)} записей")
        else:
            print(f"    Данные для {ticker_symbol} не получены")
    except Exception as e:
        print(f"    Ошибка при загрузке {ticker_symbol}: {e}")

if not all_data:
    raise ValueError("Не удалось загрузить данные ни для одного тикера")

merged_data = all_data[0]
for df in all_data[1:]:
    merged_data = pd.merge(merged_data, df, on='date', how='outer')
merged_data = merged_data.sort_values('date')

dates = merged_data['date'].copy()
merged_data.set_index('date', inplace=True)

merged_data.to_csv('russian_stocks_daily_close.csv', index=True)
print(f"\nЦены закрытия сохранены в 'russian_stocks_daily_close.csv'")
print(f"   Период: с {dates.iloc[0]} по {dates.iloc[-1]}")
print(f"   Всего дней: {len(merged_data)}")

# ------------------------------
# 2. Расчёт логарифмических доходностей
# ------------------------------
print("\nРасчёт логарифмических доходностей...")
log_returns = np.log(merged_data / merged_data.shift(1))
log_returns = log_returns.dropna()
print(f"   Доходности рассчитаны для {len(log_returns)} дней")
log_returns.to_csv('russian_stocks_log_returns.csv', index=True)

# ------------------------------
# 3. Загрузка RVI
# ------------------------------
print("\nЗагрузка индекса волатильности RVI...")
try:
    rvi_ticker = Ticker('RVI')
    rvi_data = rvi_ticker.candles(start=start_date, end=end_date, period='1d')
    if rvi_data is not None and not rvi_data.empty:
        rvi_df = pd.DataFrame({
            'date': pd.to_datetime(rvi_data['begin']).dt.date,
            'RVI': rvi_data['close'].values
        }).set_index('date')
        print(f"   Загружено {len(rvi_df)} записей RVI")
    else:
        rvi_df = pd.DataFrame()
        print("   Данные RVI не получены")
except Exception as e:
    print(f"   Ошибка при загрузке RVI: {e}")
    rvi_df = pd.DataFrame()

# Загрузка нефти Brent (тикер BRJ4 или используем MOEX фьючерс)
print("\nЗагрузка цены нефти Brent...")
try:
    oil_ticker = Ticker('BRJ4')
    oil_data = oil_ticker.candles(start=start_date, end=end_date, period='1d')
    if oil_data is not None and not oil_data.empty:
        oil_df = pd.DataFrame({
            'date': pd.to_datetime(oil_data['begin']).dt.date,
            'OIL': oil_data['close'].values
        }).set_index('date')
        print(f"   Загружено {len(oil_df)} записей нефти")
    else:
        oil_df = pd.DataFrame()
        print("   Данные нефти не получены")
except Exception as e:
    print(f"   Ошибка при загрузке нефти: {e}")
    oil_df = pd.DataFrame()

# ------------------------------
# 4. Скользящие корреляционные матрицы + TDA
# ------------------------------
window = 60

rolling_correlations = []
avg_correlations = []
tda_results = []

print(f"\nРасчёт скользящих корреляций и TDA (окно {window} дней)...")

for i in range(window, len(log_returns) + 1):
    window_data = log_returns.iloc[i - window:i]
    if window_data.isnull().any().any():
        continue

    corr_matrix = window_data.corr()
    date = log_returns.index[i - 1]

    # Средняя корреляция
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    avg_corr = upper.stack().mean()

    rolling_correlations.append({'date': date, 'correlation_matrix': corr_matrix})
    avg_correlations.append({'date': date, 'avg_correlation': avg_corr})

    # --- TDA: корреляция → расстояние → Rips-комплекс → персистентность ---
    # Шаг 3: расстояние = 1 - corr
    dist_matrix = (1 - corr_matrix).values.copy()
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)  # убираем отрицательные
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  # симметризация

    # Шаг 4: Rips-комплекс
    rips = gudhi.RipsComplex(distance_matrix=dist_matrix, max_edge_length=2.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()

    # Извлекаем персистентные интервалы H1
    persistence_h1 = simplex_tree.persistence_intervals_in_dimension(1)

    if len(persistence_h1) > 0:
        lifetimes = persistence_h1[:, 1] - persistence_h1[:, 0]
        # Убираем бесконечные
        lifetimes = lifetimes[np.isfinite(lifetimes)]

        if len(lifetimes) > 0:
            # Максимальная персистентность
            max_persistence = np.max(lifetimes)
            # Полная персистентность (сумма)
            total_persistence = np.sum(lifetimes)
            # Энтропия персистентности
            probs = lifetimes / total_persistence
            entropy = -np.sum(probs * np.log(probs + 1e-15))
        else:
            max_persistence = 0
            total_persistence = 0
            entropy = 0
    else:
        max_persistence = 0
        total_persistence = 0
        entropy = 0

    tda_results.append({
        'date': date,
        'max_persistence': max_persistence,
        'total_persistence': total_persistence,
        'entropy': entropy
    })

avg_corr_df = pd.DataFrame(avg_correlations).set_index('date')
avg_corr_df.to_csv('rolling_avg_correlation.csv')

tda_df = pd.DataFrame(tda_results).set_index('date')
tda_df.index = pd.to_datetime(tda_df.index.astype(str))
tda_df.to_csv('tda_indicators.csv')

print(f"   TDA рассчитан для {len(tda_df)} точек")

# ============================================================
# 5. ВИЗУАЛИЗАЦИИ
# ============================================================

last_corr = rolling_correlations[-1]['correlation_matrix'].copy()
labels = [t + '.ME' for t in last_corr.columns]

# --- ГРАФИК 1: Корреляционная матрица (heatmap) ---
print("\nПостроение корреляционной матрицы...")
fig, ax = plt.subplots(figsize=(12, 10))

corr_show = last_corr.copy()
corr_show.index = labels
corr_show.columns = labels

sns.heatmap(corr_show, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Корреляция Пирсона', 'shrink': 0.8},
            annot_kws={'size': 9}, ax=ax)
ax.set_title(f'Корреляционная матрица (последние {window} дней)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('1_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# --- ГРАФИК 2: Кластеризация с дендрограммой ---
print("Построение кластерной карты...")

g = sns.clustermap(corr_show, method='average', cmap='YlOrRd_r',
                   vmin=-1, vmax=1, annot=True, fmt='.2f',
                   figsize=(13, 11), linewidths=0.5,
                   annot_kws={'size': 9},
                   dendrogram_ratio=(0.15, 0.15),
                   cbar_pos=(0.02, 0.8, 0.03, 0.15))
g.fig.suptitle('Кластеризация российского рынка акций', fontsize=15, fontweight='bold', y=1.01)
plt.savefig('2_clustermap.png', dpi=150, bbox_inches='tight')
plt.show()

# --- ГРАФИК 3: Распределение корреляций ---
print("Построение распределения корреляций...")

# Извлекаем верхний треугольник без диагонали
n = last_corr.shape[0]
corr_vals = []
for i_row in range(n):
    for j_col in range(i_row + 1, n):
        val = last_corr.iloc[i_row, j_col]
        if not np.isnan(val):
            corr_vals.append(val)

corr_vals = np.array(corr_vals)
print(f"   Количество парных корреляций: {len(corr_vals)}")
print(f"   Мин: {corr_vals.min():.3f}, Макс: {corr_vals.max():.3f}")

mean_c = float(np.mean(corr_vals))
median_c = float(np.median(corr_vals))
print(f"   Средняя: {mean_c:.3f}, Медиана: {median_c:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

# Гистограмма
ax1.hist(corr_vals, bins=15, edgecolor='black', alpha=0.7, color='steelblue', linewidth=0.8)
ax1.axvline(mean_c, color='green', linestyle='--', linewidth=2, label=f'Средняя: {mean_c:.3f}')
ax1.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Корреляция', fontsize=12)
ax1.set_ylabel('Частота', fontsize=12)
ax1.set_title(f'Распределение корреляций на российском рынке\n'
              f'Средняя: {mean_c:.3f}, Медиана: {median_c:.3f}',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_xlim(-0.2, 1.1)
ax1.grid(axis='y', alpha=0.3)

# Boxplot
bp = ax2.boxplot(corr_vals, vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_linewidth(1.5)
bp['medians'][0].set_color('orange')
bp['medians'][0].set_linewidth(2)
for whisker in bp['whiskers']:
    whisker.set_linewidth(1.5)
for cap in bp['caps']:
    cap.set_linewidth(1.5)
ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_ylabel('Корреляция', fontsize=12)
ax2.set_title('Box plot распределения', fontsize=13, fontweight='bold')
ax2.set_ylim(-0.2, 1.1)
ax2.set_xticklabels([''])
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('3_correlation_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# --- ГРАФИК 4: Динамика парных корреляций ---
print("Построение динамики парных корреляций...")

pairs = [('SBER', 'GAZP'), ('GAZP', 'ROSN'), ('SBER', 'LKOH')]
available = list(log_returns.columns)
valid_pairs = [(a, b) for a, b in pairs if a in available and b in available]

fig, ax = plt.subplots(figsize=(16, 6))
for t1, t2 in valid_pairs:
    rc = log_returns[t1].rolling(window).corr(log_returns[t2]).dropna()
    dates_plot = pd.to_datetime(pd.Series(rc.index).astype(str))
    ax.plot(dates_plot.values, rc.values, label=f'{t1}.ME - {t2}.ME', linewidth=1.2)

ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axhline(-0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_xlabel('Дата', fontsize=12)
ax.set_ylabel('Корреляция', fontsize=12)
ax.set_title(f'Динамика корреляций (окно = {window} дней)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('4_rolling_pairwise_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# --- ГРАФИК 5: Персистентная диаграмма (последнее окно) ---
print("Построение персистентной диаграммы...")

dist_last = (1 - last_corr).values.copy()
np.fill_diagonal(dist_last, 0)
dist_last = np.maximum(dist_last, 0)
dist_last = (dist_last + dist_last.T) / 2

rips = gudhi.RipsComplex(distance_matrix=dist_last, max_edge_length=2.0)
st = rips.create_simplex_tree(max_dimension=2)
st.compute_persistence()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

gudhi.plot_persistence_diagram(st.persistence(), axes=ax1)
ax1.set_title('Персистентная диаграмма (последнее окно)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Рождение', fontsize=11)
ax1.set_ylabel('Смерть', fontsize=11)

gudhi.plot_persistence_barcode(st.persistence(), axes=ax2)
ax2.set_title('Баркод персистентности', fontsize=13, fontweight='bold')
ax2.set_xlabel('Расстояние', fontsize=11)

plt.tight_layout()
plt.savefig('5_persistence_diagram.png', dpi=150, bbox_inches='tight')
plt.show()

# --- ГРАФИК 6: TDA-индикаторы + кризисные даты ---
print("Построение TDA-индикаторов...")

crisis_dates = {
    'COVID-19\n(март 2020)': pd.Timestamp('2020-03-01'),
    'Начало СВО\n(февраль 2022)': pd.Timestamp('2022-02-24'),
}

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

titles_y = [
    ('entropy', 'Энтропия\nперсистентности', 'blue'),
    ('max_persistence', 'Максимальная\nперсистентность', 'green'),
    ('total_persistence', 'Полная\nперсистентность', 'purple'),
]

for idx, (col, ylabel, color) in enumerate(titles_y):
    axes[idx].plot(tda_df.index, tda_df[col], color=color, linewidth=1)
    axes[idx].set_ylabel(ylabel, fontsize=11)
    axes[idx].grid(alpha=0.3)
    for label, d in crisis_dates.items():
        axes[idx].axvline(d, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        if idx == 0:
            axes[idx].text(d, axes[idx].get_ylim()[1] * 0.9, label, fontsize=8,
                           color='red', ha='center', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

axes[0].set_title('TDA-индикаторы российского рынка акций', fontsize=15, fontweight='bold')
axes[2].set_xlabel('Дата', fontsize=12)
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('6_tda_indicators.png', dpi=150, bbox_inches='tight')
plt.show()

# --- ГРАФИК 7: Сравнение TDA с RVI и нефтью ---
print("Построение сравнения TDA с RVI и нефтью...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

# Энтропия
axes[0].plot(tda_df.index, tda_df['entropy'], color='blue', linewidth=1)
axes[0].set_ylabel('Энтропия\nперсистентности', fontsize=10)
axes[0].set_title('Сравнение TDA-индикаторов с RVI и ценой нефти', fontsize=14, fontweight='bold')
for d in crisis_dates.values():
    axes[0].axvline(d, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].grid(alpha=0.3)

# Максимальная персистентность
axes[1].plot(tda_df.index, tda_df['max_persistence'], color='green', linewidth=1)
axes[1].set_ylabel('Макс.\nперсистентность', fontsize=10)
for d in crisis_dates.values():
    axes[1].axvline(d, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].grid(alpha=0.3)

# RVI
if not rvi_df.empty:
    rvi_plot = rvi_df.copy()
    rvi_plot.index = pd.to_datetime(rvi_plot.index.astype(str))
    axes[2].plot(rvi_plot.index, rvi_plot['RVI'], color='orange', linewidth=1)
    for d in crisis_dates.values():
        axes[2].axvline(d, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
else:
    axes[2].text(0.5, 0.5, 'Данные RVI не доступны', transform=axes[2].transAxes,
                 ha='center', fontsize=14)
axes[2].set_ylabel('RVI', fontsize=10)
axes[2].grid(alpha=0.3)

# Нефть
if not oil_df.empty:
    oil_plot = oil_df.copy()
    oil_plot.index = pd.to_datetime(oil_plot.index.astype(str))
    axes[3].plot(oil_plot.index, oil_plot['OIL'], color='black', linewidth=1)
    for d in crisis_dates.values():
        axes[3].axvline(d, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
else:
    axes[3].text(0.5, 0.5, 'Данные нефти не доступны', transform=axes[3].transAxes,
                 ha='center', fontsize=14)
axes[3].set_ylabel('Нефть Brent', fontsize=10)
axes[3].grid(alpha=0.3)

axes[3].set_xlabel('Дата', fontsize=12)
axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('7_tda_vs_rvi_oil.png', dpi=150, bbox_inches='tight')
plt.show()

# --- ГРАФИК 8: Средняя корреляция + TDA ---
print("Построение средней корреляции и TDA...")

avg_corr_df.index = pd.to_datetime(avg_corr_df.index.astype(str))

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

axes[0].plot(avg_corr_df.index, avg_corr_df['avg_correlation'], color='teal', linewidth=1)
axes[0].set_ylabel('Средняя корреляция', fontsize=11)
axes[0].set_title('Средняя корреляция и полная персистентность', fontsize=14, fontweight='bold')
for d in crisis_dates.values():
    axes[0].axvline(d, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].grid(alpha=0.3)

axes[1].plot(tda_df.index, tda_df['total_persistence'], color='purple', linewidth=1)
axes[1].set_ylabel('Полная персистентность', fontsize=11)
axes[1].set_xlabel('Дата', fontsize=12)
for d in crisis_dates.values():
    axes[1].axvline(d, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].grid(alpha=0.3)

axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig('8_avg_corr_vs_tda.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Итоги
# ============================================================
print("\n" + "=" * 60)
print("Все визуализации построены:")
print("  1. Корреляционная матрица (heatmap)")
print("  2. Кластерная карта с дендрограммой")
print("  3. Распределение корреляций (гистограмма + boxplot)")
print("  4. Динамика парных корреляций")
print("  5. Персистентная диаграмма и баркод")
print("  6. TDA-индикаторы с кризисными датами")
print("  7. Сравнение TDA с RVI и нефтью")
print("  8. Средняя корреляция vs полная персистентность")
print("=" * 60)
print("\nCSV файлы:")
print("  - russian_stocks_daily_close.csv")
print("  - russian_stocks_log_returns.csv")
print("  - rolling_avg_correlation.csv")
print("  - tda_indicators.csv")
