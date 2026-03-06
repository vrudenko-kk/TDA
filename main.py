import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from moexalgo import Ticker, session

load_dotenv()

username = os.getenv('MOEX_USERNAME')
password = os.getenv('MOEX_PASSWORD')

if not username or not password:
    raise ValueError("Не удалось загрузить username или password из .env файла")

session.authorize(username, password)

tickers = ['SBER', 'GAZP', 'YNDX']
# tickers = [
#     'SBER', 'GAZP', 'LKOH', 'YNDX', 'ROSN',
#     'NVTK', 'GMKN', 'VTBR', 'TATN', 'CHMF'
# ]
all_data = []

print("Начинаем загрузку данных...")

for ticker_symbol in tickers:
    print(f"  Загружаем {ticker_symbol}...")
    try:
        ticker = Ticker(ticker_symbol)

        data = ticker.candles(
            start='2020-01-01',
            end='2020-12-31',
            period='1d'  # дневные свечи
        )

        if data is not None and not data.empty:
            df = pd.DataFrame({
                'date': pd.to_datetime(data['begin']).dt.date,
                ticker_symbol: data['close'].values
            })

            all_data.append(df)
            print(f"    ✓ Загружено {len(df)} записей")
        else:
            print(f"    ✗ Данные для {ticker_symbol} не получены")

    except Exception as e:
        print(f"    ✗ Ошибка при загрузке {ticker_symbol}: {e}")

if all_data:
    # Объединяем данные
    merged_data = all_data[0]
    for df in all_data[1:]:
        merged_data = pd.merge(merged_data, df, on='date', how='outer')

    # Сортируем по дате
    merged_data = merged_data.sort_values('date')

    # Сохраняем даты отдельно для вывода
    dates = merged_data['date'].copy()

    # Устанавливаем дату как индекс
    merged_data.set_index('date', inplace=True)

    # Сохраняем в CSV
    output_file = 'russian_stocks_daily_close.csv'
    merged_data.to_csv(output_file, index=True)  # index=True сохраняет дату как колонку в CSV

    print(f"\n✅ Данные сохранены в файл: {output_file}")
    print(f"   Период: с {dates.iloc[0]} по {dates.iloc[-1]}")
    print(f"   Всего дней: {len(merged_data)}")
else:
    print("❌ Не удалось загрузить данные ни для одного тикера")
    exit()

# ================== 2. РАСЧЁТ ЛОГАРИФМИЧЕСКИХ ДОХОДНОСТЕЙ ==================
print("\n📊 Расчёт логарифмических доходностей...")

# Логарифмические доходности: log(P_t / P_{t-1})
log_returns = np.log(merged_data / merged_data.shift(1))

# Удаляем первую строку с NaN
log_returns = log_returns.dropna()

print(f"   Доходности рассчитаны для {len(log_returns)} дней")
print(f"   Период доходностей: с {log_returns.index[0]} по {log_returns.index[-1]}")

# Сохраняем доходности
log_returns.to_csv('russian_stocks_log_returns.csv', index=True)
print("   Доходности сохранены в 'russian_stocks_log_returns.csv'")

# Статистика доходностей
print("\n📈 Статистика доходностей:")
stats = pd.DataFrame({
    'Среднее': log_returns.mean(),
    'Ст. отклонение': log_returns.std(),
    'Мин': log_returns.min(),
    'Макс': log_returns.max()
})
print(stats.round(6))

# ================== 3. СКОЛЬЗЯЩАЯ КОРРЕЛЯЦИОННАЯ МАТРИЦА ==================

window = 60  # размер окна в днях
rolling_correlations = []
avg_correlations = []

for i in range(window, len(log_returns) + 1):
    # Берём окно данных
    window_data = log_returns.iloc[i - window:i]

    # Рассчитываем корреляционную матрицу
    corr_matrix = window_data.corr()

    # Рассчитываем среднюю корреляцию (без учёта диагонали)
    # Берём только верхний треугольник
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    avg_corr = upper_triangle.stack().mean()

    # Сохраняем результаты
    date = log_returns.index[i - 1]
    rolling_correlations.append({
        'date': date,
        'correlation_matrix': corr_matrix
    })
    avg_correlations.append({
        'date': date,
        'avg_correlation': avg_corr
    })

print(f"   Рассчитано {len(rolling_correlations)} корреляционных матриц")

# Создаём DataFrame со средними корреляциями
avg_corr_df = pd.DataFrame(avg_correlations)
avg_corr_df.set_index('date', inplace=True)

# Сохраняем средние корреляции
avg_corr_df.to_csv('rolling_avg_correlation.csv')
print("   Средние корреляции сохранены в 'rolling_avg_correlation.csv'")

