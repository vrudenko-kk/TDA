import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from moexalgo import Ticker, session
import matplotlib.pyplot as plt
import gudhi
import warnings
warnings.filterwarnings('ignore')

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
        data = ticker.candles(
            start=start_date,
            end=end_date,
            period='1d'
        )
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
print("\n📊 Расчёт логарифмических доходностей...")
log_returns = np.log(merged_data / merged_data.shift(1))
log_returns = log_returns.dropna()
print(f"   Доходности рассчитаны для {len(log_returns)} дней")
log_returns.to_csv('russian_stocks_log_returns.csv', index=True)

# ------------------------------
# 3. Загрузка дополнительных данных (RVI)
# ------------------------------
print("\n📈 Загрузка индекса волатильности RVI...")
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

# ------------------------------
# 4. Скользящие корреляционные матрицы (окно 60 дней)
# ------------------------------
window = 60
rolling_correlations = []
avg_correlations = []

print("\n🔗 Расчёт скользящих корреляционных матриц...")
for i in range(window, len(log_returns) + 1):
    window_data = log_returns.iloc[i - window:i]
    # Пропуск, если в окне есть пропуски
    if window_data.isnull().any().any():
        continue
    corr_matrix = window_data.corr()
    # Средняя корреляция (без учёта диагонали)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    avg_corr = upper_triangle.stack().mean()
    date = log_returns.index[i - 1]
    rolling_correlations.append({
        'date': date,
        'correlation_matrix': corr_matrix
    })
    avg_correlations.append({
        'date': date,
        'avg_correlation': avg_corr
    })

avg_corr_df = pd.DataFrame(avg_correlations).set_index('date')
avg_corr_df.to_csv('rolling_avg_correlation.csv')
print(f"   Рассчитано {len(rolling_correlations)} корреляционных матриц")
print(f"   Первые 5 значений средней корреляции:\n{avg_corr_df.head()}")

# ------------------------------
# 5. Вычисление топологических дескрипторов для каждого окна
# ------------------------------
persistence_results = []

print("\n🧬 Вычисление персистентных дескрипторов...")
for item in rolling_correlations:
    date = item['date']
    corr_matrix = item['correlation_matrix'].values

    # Преобразование корреляций в расстояния
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Построение Rips-комплекса
    rips = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=2.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()

    # Извлечение lifetimes для H1 (циклы)
    h1_lifetimes = []
    for dim, (birth, death) in persistence:
        if dim == 1 and death != float('inf'):
            lifetime = death - birth
            if lifetime > 0:
                h1_lifetimes.append(lifetime)

    total_lifetime = sum(h1_lifetimes)
    if total_lifetime > 0:
        probs = [lt / total_lifetime for lt in h1_lifetimes]
        entropy = -sum(p * np.log(p) for p in probs)
        max_persistence = max(h1_lifetimes)
        total_persistence = total_lifetime
    else:
        entropy = 0.0
        max_persistence = 0.0
        total_persistence = 0.0

    persistence_results.append({
        'date': date,
        'entropy': entropy,
        'max_persistence': max_persistence,
        'total_persistence': total_persistence
    })

persistence_df = pd.DataFrame(persistence_results).set_index('date')
persistence_df.to_csv('persistence_indicators.csv')
print(f"   Сохранено {len(persistence_df)} записей")
print(f"   Первые 5 записей персистентности:\n{persistence_df.head()}")

# ------------------------------
# 6. Сбор всех данных для построения графиков
# ------------------------------
combined = persistence_df.join(avg_corr_df, how='outer')
if not rvi_df.empty:
    combined = combined.join(rvi_df, how='outer')

combined.sort_index(inplace=True)

# Заполняем пропуски (вперед), чтобы графики были непрерывны
combined.ffill(inplace=True)
# Если первые значения NaN, заполняем назад
combined.bfill(inplace=True)

print(f"\n📊 Итоговый DataFrame для построения графиков:")
print(f"   Период: с {combined.index.min()} по {combined.index.max()}")
print(f"   Количество записей: {len(combined)}")
print(f"   Колонки: {list(combined.columns)}")

# ------------------------------
# 7. Построение графиков
# ------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# 1) Энтропия персистентности
axes[0].plot(combined.index, combined['entropy'], color='blue', linewidth=1.5)
axes[0].set_ylabel('Энтропия персистентности')
axes[0].grid(True, alpha=0.3)

# 2) Максимальная персистентность
axes[1].plot(combined.index, combined['max_persistence'], color='green', linewidth=1.5)
axes[1].set_ylabel('Максимальная персистентность')
axes[1].grid(True, alpha=0.3)

# 3) Полная персистентность
axes[2].plot(combined.index, combined['total_persistence'], color='red', linewidth=1.5)
axes[2].set_ylabel('Полная персистентность')
axes[2].grid(True, alpha=0.3)

# 4) Средняя корреляция, RVI
axes[3].plot(combined.index, combined['avg_correlation'], color='gray', linewidth=1.5, label='Средняя корреляция')
if 'RVI' in combined.columns:
    axes[3].plot(combined.index, combined['RVI'], color='purple', linewidth=1.5, label='RVI')
axes[3].set_ylabel('Значения')
axes[3].legend(loc='upper left')
axes[3].grid(True, alpha=0.3)

# Отметки кризисных дат
crisis_dates = ['2020-03-01', '2022-02-24']
for ax in axes:
    for crisis in crisis_dates:
        ax.axvline(pd.to_datetime(crisis), color='black', linestyle='--', alpha=0.7, linewidth=1.2)

axes[-1].set_xlabel('Дата')
plt.suptitle('Топологические дескрипторы российского фондового рынка (окно 60 дней)', fontsize=14)
plt.tight_layout()
plt.savefig('persistence_indicators.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ График сохранён как 'persistence_indicators.png'")
