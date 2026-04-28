import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

print("📂 Polar Model 로딩 중...")
# 1. 모델 및 도구 로드
model = joblib.load("model_polar.pkl")
le_type = joblib.load("le_type_polar.pkl")
le_result = joblib.load("le_result_polar.pkl")
kmeans = joblib.load("kmeans_polar.pkl")

# 2. 통계표 로드 (거리/각도 통계)
p_stats = pd.read_csv('p_stats_polar.csv')
t_stats = pd.read_csv('t_stats_polar.csv')

# 글로벌 평균 (데이터 없을 때 대타)
g_dist = p_stats['target_dist'].mean()
g_angle = p_stats['target_angle'].mean()

def safe_transform(encoder, value):
    try: return encoder.transform([str(value)])[0]
    except: return 0

test_df = pd.read_csv('test.csv')
print(f"🚀 총 {len(test_df)}개의 문제 풀이 시작!")

results = []

for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
    game_episode = row['game_episode']
    file_path = row['path']
    
    if os.path.exists(file_path):
        ep_df = pd.read_csv(file_path)
        curr = ep_df.iloc[-1]
        
        # --- 1. 과거 데이터 처리 ---
        if len(ep_df) > 1:
            prev = ep_df.iloc[-2]
            prev_end_x = prev['end_x'] if not np.isnan(prev['end_x']) else curr['start_x']
            prev_end_y = prev['end_y'] if not np.isnan(prev['end_y']) else curr['start_y']
            prev_time = prev['time_seconds']
            prev_type = prev['type_name']
            prev_result = prev['result_name']
        else:
            prev_end_x = curr['start_x']
            prev_end_y = curr['start_y']
            prev_time = curr['time_seconds']
            prev_type = 'Start'
            prev_result = 'None'
            
        # --- 2. 물리학 피처 ---
        dx = curr['start_x'] - prev_end_x
        dy = curr['start_y'] - prev_end_y
        dist_prev = np.sqrt(dx**2 + dy**2)
        angle_prev = np.arctan2(dy, dx)
        dt = curr['time_seconds'] - prev_time
        speed = dist_prev / dt if dt > 0 else 0
        dist_goal = np.sqrt((105 - curr['start_x'])**2 + (34 - curr['start_y'])**2)
        
        zone_id = kmeans.predict([[curr['start_x'], curr['start_y']]])[0]
        type_enc = safe_transform(le_type, curr['type_name'])
        prev_type_enc = safe_transform(le_type, prev_type)
        prev_result_enc = safe_transform(le_result, prev_result)
        
        # --- 3. [Plan B] 통계 매핑 (거리, 각도) ---
        pid = curr['player_id']
        tid = curr['team_id']
        
        # 선수별 평균 (거리, 각도)
        p_match = p_stats[p_stats['player_id'] == pid]
        if not p_match.empty:
            p_mean_d = p_match['target_dist'].values[0]
            p_mean_a = p_match['target_angle'].values[0]
        else:
            p_mean_d, p_mean_a = g_dist, g_angle
            
        # 팀별 평균 (거리, 각도)
        t_match = t_stats[t_stats['team_id'] == tid]
        if not t_match.empty:
            t_mean_d = t_match['target_dist'].values[0]
            t_mean_a = t_match['target_angle'].values[0]
        else:
            t_mean_d, t_mean_a = g_dist, g_angle
            
        # 입력 피처
        features = [[
            curr['start_x'], curr['start_y'], curr['time_seconds'],
            dist_prev, angle_prev, speed, dist_goal,
            prev_end_x, prev_end_y,
            curr['period_id'], curr['action_id'], zone_id,
            type_enc, prev_type_enc, prev_result_enc,
            p_mean_d, p_mean_a,
            t_mean_d, t_mean_a
        ]]
        
        # --- 4. 예측 및 변환 ---
        pred_polar = model.predict(features)
        pred_dist = pred_polar[0][0]
        pred_angle = pred_polar[0][1]
        
        # 💡 [핵심] 극좌표(r, theta) -> 직교좌표(x, y) 변환
        pred_diff_x = pred_dist * np.cos(pred_angle)
        pred_diff_y = pred_dist * np.sin(pred_angle)
        
        results.append({
            'game_episode': game_episode,
            'end_x': curr['start_x'] + pred_diff_x,
            'end_y': curr['start_y'] + pred_diff_y
        })
    else:
        results.append({'game_episode': game_episode, 'end_x': 52.5, 'end_y': 34.0})

submission = pd.DataFrame(results)
submission.to_csv('submission_polar.csv', index=False)
print("\n🎉 Polar 예측 완료! 'submission_polar.csv' 생성됨.")