import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터 로드
print("📂 데이터 로딩 중...")
df = pd.read_csv('train.csv')
df = df.sort_values(['game_id', 'game_episode', 'action_id'])

# 2. 기본 피처 엔지니어링
print("⚙️ 피처 생성 중...")
groupby_episode = df.groupby('game_episode')

df['prev_end_x'] = groupby_episode['end_x'].shift(1)
df['prev_end_y'] = groupby_episode['end_y'].shift(1)
df['prev_time'] = groupby_episode['time_seconds'].shift(1)
df['prev_type'] = groupby_episode['type_name'].shift(1)
df['prev_result'] = groupby_episode['result_name'].shift(1)

df['prev_end_x'] = df['prev_end_x'].fillna(df['start_x'])
df['prev_end_y'] = df['prev_end_y'].fillna(df['start_y'])
df['prev_time'] = df['prev_time'].fillna(df['time_seconds'])
df['prev_type'] = df['prev_type'].fillna('Start')
df['prev_result'] = df['prev_result'].fillna('None')

last_actions = df.groupby('game_episode').tail(1).copy()

# 타겟: 이동량
last_actions['diff_x'] = last_actions['end_x'] - last_actions['start_x']
last_actions['diff_y'] = last_actions['end_y'] - last_actions['start_y']

# 물리학
dx = last_actions['start_x'] - last_actions['prev_end_x']
dy = last_actions['start_y'] - last_actions['prev_end_y']
last_actions['dist_prev'] = np.sqrt(dx**2 + dy**2)
last_actions['angle_prev'] = np.arctan2(dy, dx)
dt = last_actions['time_seconds'] - last_actions['prev_time']
last_actions['speed'] = np.where(dt > 0, last_actions['dist_prev'] / dt, 0)
last_actions['dist_goal'] = np.sqrt((105 - last_actions['start_x'])**2 + (34 - last_actions['start_y'])**2)

# --------------------------------------------------------------------------------
# 💡 [신기술 1] K-Means Clustering (공간을 10개 구역으로 나누기)
# 운동장을 10개의 구역(Zone)으로 나눠서 "어느 구역에 있는지" 알려줌
print("🗺️ 공간 클러스터링(Zone 나누기) 중...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
# (x, y) 좌표를 넣으면 0~9번 구역 번호를 줌
last_actions['zone_id'] = kmeans.fit_predict(last_actions[['start_x', 'start_y']])

# --------------------------------------------------------------------------------
# 💡 [신기술 2] K-Fold Target Encoding (과적합 없이 능력치 수치화)
# AI가 자기 정답을 못 베끼게 "5등분" 해서 서로 채점해줌
print("📊 K-Fold 타겟 인코딩 중...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 결과를 담을 빈 컬럼 생성
last_actions['player_mean_x'] = np.nan
last_actions['player_mean_y'] = np.nan
last_actions['team_mean_x'] = np.nan
last_actions['team_mean_y'] = np.nan

# K-Fold 돌면서 채우기 (Out-of-Fold)
for train_idx, val_idx in kf.split(last_actions):
    # 훈련용 데이터(4/5)와 검증용 데이터(1/5) 나누기
    X_tr, X_val = last_actions.iloc[train_idx], last_actions.iloc[val_idx]
    
    # 훈련용 데이터로만 통계 내기 (정답 유출 방지!)
    p_stats = X_tr.groupby('player_id')[['diff_x', 'diff_y']].mean()
    t_stats = X_tr.groupby('team_id')[['diff_x', 'diff_y']].mean()
    
    # 검증용 데이터에 맵핑
    last_actions.loc[last_actions.index[val_idx], 'player_mean_x'] = X_val['player_id'].map(p_stats['diff_x'])
    last_actions.loc[last_actions.index[val_idx], 'player_mean_y'] = X_val['player_id'].map(p_stats['diff_y'])
    last_actions.loc[last_actions.index[val_idx], 'team_mean_x'] = X_val['team_id'].map(t_stats['diff_x'])
    last_actions.loc[last_actions.index[val_idx], 'team_mean_y'] = X_val['team_id'].map(t_stats['diff_y'])

# 혹시 결측치(새로운 선수 등)가 있으면 전체 평균으로 채움
global_mean_x = last_actions['diff_x'].mean()
global_mean_y = last_actions['diff_y'].mean()
last_actions = last_actions.fillna({
    'player_mean_x': global_mean_x, 'player_mean_y': global_mean_y,
    'team_mean_x': global_mean_x, 'team_mean_y': global_mean_y
})

# 나중에 test.csv 예측할 때 쓸 "전체 통계표"도 따로 만들어둠 (Global Statistics)
final_player_stats = last_actions.groupby('player_id')[['diff_x', 'diff_y']].mean().reset_index()
final_team_stats = last_actions.groupby('team_id')[['diff_x', 'diff_y']].mean().reset_index()
# --------------------------------------------------------------------------------

# 인코딩 (기본)
le_type = LabelEncoder()
all_types = list(df['type_name'].unique()) + ['Start']
le_type.fit(all_types)
le_result = LabelEncoder()
all_results = list(df['result_name'].astype(str).unique()) + ['None']
le_result.fit(all_results)

last_actions['type_enc'] = le_type.transform(last_actions['type_name'])
last_actions['prev_type_enc'] = le_type.transform(last_actions['prev_type'])
last_actions['prev_result_enc'] = le_result.transform(last_actions['prev_result'].astype(str))

features = [
    'start_x', 'start_y', 'time_seconds',
    'dist_prev', 'angle_prev', 'speed', 'dist_goal',
    'prev_end_x', 'prev_end_y',
    'period_id', 'action_id', 'zone_id', # Zone 추가
    'type_enc', 'prev_type_enc', 'prev_result_enc',
    'player_mean_x', 'player_mean_y', # K-Fold로 계산된 안전한 통계치
    'team_mean_x', 'team_mean_y'
]
targets = ['diff_x', 'diff_y']

X = last_actions[features]
y = last_actions[targets]

# 3. 앙상블 학습 (Stacking 대신 Voting으로 회귀 - 안정성 우선)
print(f"🚀 학습 시작 (Advanced Features)")
model1 = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=5, random_state=41, early_stopping=True)
model2 = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.1, max_depth=8, random_state=42, early_stopping=True)
model3 = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=6, random_state=43, early_stopping=True)

voting_model = VotingRegressor(estimators=[('m1', model1), ('m2', model2), ('m3', model3)])
final_model = MultiOutputRegressor(voting_model)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)

# 검증
val_pred_diff = final_model.predict(X_val)
true_end_x = X_val['start_x'].values + y_val['diff_x'].values
true_end_y = X_val['start_y'].values + y_val['diff_y'].values
pred_end_x = X_val['start_x'].values + val_pred_diff[:, 0]
pred_end_y = X_val['start_y'].values + val_pred_diff[:, 1]

mse_x = np.mean((true_end_x - pred_end_x)**2)
mse_y = np.mean((true_end_y - pred_end_y)**2)
rmse = np.sqrt((mse_x + mse_y) / 2)

print(f"✅ 학습 완료! RMSE: {rmse:.4f}")
print(f"   (13.16보다 낮으면 성공입니다!)")

joblib.dump(final_model, "model_adv.pkl")
joblib.dump(le_type, "le_type_adv.pkl")
joblib.dump(le_result, "le_result_adv.pkl")
joblib.dump(kmeans, "kmeans_adv.pkl") # 구역 나누는 기계도 저장
# 통계 파일 저장
final_player_stats.to_csv('player_stats_adv.csv', index=False)
final_team_stats.to_csv('team_stats_adv.csv', index=False)