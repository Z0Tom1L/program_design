import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report, mean_absolute_error

# 读取消息文件
def load_message_file(path):
    cols = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction']
    df = pd.read_csv(path, header=None, names=cols)
    df['Price'] = df['Price'] / 10000  # 转换实际价格
    return df

# 读取订单簿文件（假设LEVEL=10）
def load_orderbook_file(path, levels=4):
    cols = []
    for i in range(0, levels+1):
        cols.extend([f'AskPrice_{i}', f'AskSize_{i}', f'BidPrice_{i}', f'BidSize_{i}'])
    df = pd.read_csv(path, header=None, names=cols)
    df = df / 10000  # 转换价格列
    return df

def calculate_basic_features(df):
    # 价差和中间价
    df['Spread'] = df['AskPrice_1'] - df['BidPrice_1']
    df['MidPrice'] = (df['AskPrice_1'] + df['BidPrice_1']) / 2

    # 订单簿不平衡度
    df['OrderImbalance'] = (df['BidSize_1'] - df['AskSize_1']) / (df['BidSize_1'] + df['AskSize_1'])

    # 深度加权价格
    for i in [1, 2, 3]:
        df[f'WeightedAsk_{i}'] = df[f'AskPrice_{i}'] * df[f'AskSize_{i}']
        df[f'WeightedBid_{i}'] = df[f'BidPrice_{i}'] * df[f'BidSize_{i}']

    return df


# 计算订单流不平衡（过去N个事件）
def order_flow_imbalance(df, window=100):
    df['BuyOrderFlow'] = ((df['Type'] == 4) & (df['Direction'] == -1)) | \
                         ((df['Type'] == 5) & (df['Direction'] == -1)) | \
                         ((df['Type'] == 1) & (df['Direction'] == 1))

    df['SellOrderFlow'] = ((df['Type'] == 4) & (df['Direction'] == 1)) | \
                          ((df['Type'] == 5) & (df['Direction'] == 1)) | \
                          ((df['Type'] == 1) & (df['Direction'] == -1))

    df['OFI'] = df['BuyOrderFlow'].rolling(window).sum() - df['SellOrderFlow'].rolling(window).sum()
    return df

# 示例加载
msg_df = load_message_file('./message.csv')
orderbook_df = load_orderbook_file('./orderbook.csv')

# 确保时间戳对齐
assert len(msg_df) == len(orderbook_df), "数据长度不匹配"
combined_df = pd.concat([msg_df, orderbook_df], axis=1)

# 处理交易暂停事件（Type=7）
halt_indices = combined_df[combined_df['Type'] == 7].index #因为7表示交易暂停，属于异常事件
combined_df.drop(halt_indices, inplace=True)  # 根据需求决定是否保留

# 使用示例
combined_df = calculate_basic_features(combined_df)

combined_df = order_flow_imbalance(combined_df)

# 移动平均和波动率
combined_df['MA_10'] = combined_df['MidPrice'].rolling(10).mean()
combined_df['Volatility'] = combined_df['MidPrice'].rolling(100).std()

# 未来k步中间价变化
k = 10  # 预测未来10个事件后的价格
combined_df['FutureMidPrice'] = combined_df['MidPrice'].shift(-k)
combined_df['PriceChange'] = (combined_df['FutureMidPrice'] - combined_df['MidPrice']).apply(
    lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

future_window = 100
combined_df['FutureVolatility'] = combined_df['MidPrice'].rolling(future_window).std().shift(-future_window)

features = ['Spread', 'OrderImbalance', 'OFI', 'MA_10', 'Volatility',
            'AskSize_1', 'BidSize_1', 'WeightedAsk_1', 'WeightedBid_1']
target = 'PriceChange'  # 或 'FutureVolatility'

X = combined_df[features].dropna()
y = combined_df[target].loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = lgb.LGBMClassifier(
    max_depth=7,
    num_leaves=20,
    min_child_samples=10,
    learning_rate=0.01,
    n_estimators=200,
    reg_alpha=0.2,
    reg_lambda=0.2,
)
print(model.get_params())
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)

print(len(combined_df['AskPrice_1']))
# 方向预测评估
print(classification_report(y_test, model.predict(X_test)))

# 波动率预测评估
print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))






