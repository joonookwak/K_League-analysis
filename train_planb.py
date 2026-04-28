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
print("📂 [Plan B] 데이터 로딩 및 극좌표계 변환...")
df = pd.read_csv('train.csv')
df = df.sort_values(['game_id', 'game_episode', 'action_id'])

# 2. 피처 엔지니어링
groupby_episode = df.groupby('game_episode')
df['prev_end_x'] = groupby_episode['end_x'].shift(1).fillna(df['start_x'])
df['prev_end_y'] = groupby_episode['end_y'].shift(1).fillna(df['start_y'])
df['prev_time'] = groupby_episode['time_seconds'].shift(1).fillna(df['time_seconds'])
df['prev_type'] = groupby_episode['type_name'].shift(1).fillna('Start')
df['prev_result'] = groupby_episode['result_name'].shift(1).fillna('None')

last_actions = df.groupby('game_episode').tail(1).copy()

# ------------------------------------------------------------------
# 💡 [핵심 변경] 타겟을 (x, y)가 아니라 (거리, 각도)로 변경!
# ------------------------------------------------------------------
diff_x = last_actions['end_x'] - last_actions['start_x']
diff_y = last_actions['end_y'] - last_actions['start_y']

# 타겟 1: 이동 거리 (Distance)
last_actions['target_dist'] = np.sqrt(diff_x**2 + diff_y**2)
# 타겟 2: 이동 각도 (Angle) - (-pi ~ pi 라디안 값)
last_actions['target_angle'] = np.arctan2(diff_y, diff_x)

# 입력용 물리학 피처
dx = last_actions['start_x'] - last_actions['prev_end_x']
dy = last_actions['start_y'] - last_actions['prev_end_y']
last_actions['dist_prev'] = np.sqrt(dx**2 + dy**2)
last_actions['angle_prev'] = np.arctan2(dy, dx)
dt = last_actions['time_seconds'] - last_actions['prev_time']
last_actions['speed'] = np.where(dt > 0, last_actions['dist_prev'] / dt, 0)
last_actions['dist_goal'] = np.sqrt((105 - last_actions['start_x'])**2 + (34 - last_actions['start_y'])**2)

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
last_actions['zone_id'] = kmeans.fit_predict(last_actions[['start_x', 'start_y']])

# K-Fold 타겟 인코딩 (거리, 각도 평균 계산)
print("📊 Polar Target Encoding 중...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

last_actions['p_mean_dist'] = np.nan
last_actions['p_mean_angle'] = np.nan
last_actions['t_mean_dist'] = np.nan
last_actions['t_mean_angle'] = np.nan

for train_idx, val_idx in kf.split(last_actions):
    X_tr, X_val = last_actions.iloc[train_idx], last_actions.iloc[val_idx]
    
    # 선수/팀별 평균 거리와 평균 각도를 계산
    p_stats = X_tr.groupby('player_id')[['target_dist', 'target_angle']].mean()
    t_stats = X_tr.groupby('team_id')[['target_dist', 'target_angle']].mean()
    
    last_actions.loc[last_actions.index[val_idx], 'p_mean_dist'] = X_val['player_id'].map(p_stats['target_dist'])
    last_actions.loc[last_actions.index[val_idx], 'p_mean_angle'] = X_val['player_id'].map(p_stats['target_angle'])
    last_actions.loc[last_actions.index[val_idx], 't_mean_dist'] = X_val['team_id'].map(t_stats['target_dist'])
    last_actions.loc[last_actions.index[val_idx], 't_mean_angle'] = X_val['team_id'].map(t_stats['target_angle'])

# 결측치 채우기
g_dist = last_actions['target_dist'].mean()
g_angle = last_actions['target_angle'].mean()
last_actions = last_actions.fillna({
    'p_mean_dist': g_dist, 'p_mean_angle': g_angle,
    't_mean_dist': g_dist, 't_mean_angle': g_angle
})

# 통계표 저장
final_p_stats = last_actions.groupby('player_id')[['target_dist', 'target_angle']].mean().reset_index()
final_t_stats = last_actions.groupby('team_id')[['target_dist', 'target_angle']].mean().reset_index()

# 인코딩
le_type = LabelEncoder()
le_type.fit(list(df['type_name'].unique()) + ['Start'])
le_result = LabelEncoder()
le_result.fit(list(df['result_name'].astype(str).unique()) + ['None'])

last_actions['type_enc'] = le_type.transform(last_actions['type_name'])
last_actions['prev_type_enc'] = le_type.transform(last_actions['prev_type'])
last_actions['prev_result_enc'] = le_result.transform(last_actions['prev_result'].astype(str))

features = [
    'start_x', 'start_y', 'time_seconds',
    'dist_prev', 'angle_prev', 'speed', 'dist_goal',
    'prev_end_x', 'prev_end_y',
    'period_id', 'action_id', 'zone_id',
    'type_enc', 'prev_type_enc', 'prev_result_enc',
    'p_mean_dist', 'p_mean_angle', 
    't_mean_dist', 't_mean_angle'
]
targets = ['target_dist', 'target_angle']

X = last_actions[features]
y = last_actions[targets]

# 3. 학습
print(f"🚀 Polar Model 학습 시작")
model1 = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.03, max_depth=6, random_state=41)
model2 = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.05, max_depth=8, random_state=42)
model3 = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.03, max_depth=7, random_state=43)

voting_model = VotingRegressor(estimators=[('m1', model1), ('m2', model2), ('m3', model3)])
final_model = MultiOutputRegressor(voting_model)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)

# -------------------------------------------------------------------------
# [중요] 검증 및 RMSE 계산 (Polar -> Cartesian 변환 후 계산)
# -------------------------------------------------------------------------
print("📊 검증 데이터로 RMSE 계산 중...")
val_pred = final_model.predict(X_val) # 모델은 [거리, 각도]를 뱉음

# 1. 예측값 변환: (거리, 각도) -> (x, y)
pred_dist = val_pred[:, 0]
pred_angle = val_pred[:, 1]
pred_diff_x = pred_dist * np.cos(pred_angle)
pred_diff_y = pred_dist * np.sin(pred_angle)

# 2. 실제값 변환: (거리, 각도) -> (x, y)
# 학습 때 y_val은 [dist, angle]이므로 다시 좌표로 복원해서 비교
true_dist = y_val['target_dist'].values
true_angle = y_val['target_angle'].values
true_diff_x = true_dist * np.cos(true_angle)
true_diff_y = true_dist * np.sin(true_angle)

# 3. RMSE 계산
mse_x = np.mean((true_diff_x - pred_diff_x)**2)
mse_y = np.mean((true_diff_y - pred_diff_y)**2)
rmse = np.sqrt((mse_x + mse_y) / 2)

print(f"✅ 학습 완료! (Polar Coordinates)")
print(f"📉 변환 후 RMSE 점수: {rmse:.4f}")
print("-" * 30)

if rmse < 13.5:
    print("🌟 대박! 극좌표계가 먹혔습니다. 무조건 제출하세요.")
elif rmse < 14.5:
    print("🤔 흠... 비슷하네요. 그래도 리더보드 점수는 다를 수 있습니다.")
else:
    print("😭 점수 하락... AI가 '각도' 계산을 어려워합니다.")

# 저장
joblib.dump(final_model, "model_polar.pkl")
joblib.dump(le_type, "le_type_polar.pkl")
joblib.dump(le_result, "le_result_polar.pkl")
joblib.dump(kmeans, "kmeans_polar.pkl")
final_p_stats.to_csv('p_stats_polar.csv', index=False)
final_t_stats.to_csv('t_stats_polar.csv', index=False)