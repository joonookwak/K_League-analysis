import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

print("📂 모델 및 통계 로딩 중...")
model = joblib.load("model_adv.pkl")
le_type = joblib.load("le_type_adv.pkl")
le_result = joblib.load("le_result_adv.pkl")
kmeans = joblib.load("kmeans_adv.pkl") # 구역 나누는 기계

player_stats = pd.read_csv('player_stats_adv.csv')
team_stats = pd.read_csv('team_stats_adv.csv')

# 글로벌 평균 (모르는 선수용)
global_mean_x = player_stats['diff_x'].mean()
global_mean_y = player_stats['diff_y'].mean()

def safe_transform(encoder, value):
    try: return encoder.transform([str(value)])[0]
    except: return 0

test_df = pd.read_csv('test.csv')
results = []

for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
    game_episode = row['game_episode']
    file_path = row['path']
    
    if os.path.exists(file_path):
        ep_df = pd.read_csv(file_path)
        curr = ep_df.iloc[-1]
        
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
            
        dx = curr['start_x'] - prev_end_x
        dy = curr['start_y'] - prev_end_y
        dist_prev = np.sqrt(dx**2 + dy**2)
        angle_prev = np.arctan2(dy, dx)
        dt = curr['time_seconds'] - prev_time
        speed = dist_prev / dt if dt > 0 else 0
        dist_goal = np.sqrt((105 - curr['start_x'])**2 + (34 - curr['start_y'])**2)
        
        # Zone ID 예측 (KMeans)
        zone_id = kmeans.predict([[curr['start_x'], curr['start_y']]])[0]
        
        type_enc = safe_transform(le_type, curr['type_name'])
        prev_type_enc = safe_transform(le_type, prev_type)
        prev_result_enc = safe_transform(le_result, prev_result)
        
        # 통계 매핑 (Target Encoding)
        pid = curr['player_id']
        tid = curr['team_id']
        
        p_stat = player_stats[player_stats['player_id'] == pid]
        if not p_stat.empty:
            p_mean_x, p_mean_y = p_stat['diff_x'].values[0], p_stat['diff_y'].values[0]
        else:
            p_mean_x, p_mean_y = global_mean_x, global_mean_y
            
        t_stat = team_stats[team_stats['team_id'] == tid]
        if not t_stat.empty:
            t_mean_x, t_mean_y = t_stat['diff_x'].values[0], t_stat['diff_y'].values[0]
        else:
            t_mean_x, t_mean_y = global_mean_x, global_mean_y
        
        features = [[
            curr['start_x'], curr['start_y'], curr['time_seconds'],
            dist_prev, angle_prev, speed, dist_goal,
            prev_end_x, prev_end_y,
            curr['period_id'], curr['action_id'], zone_id,
            type_enc, prev_type_enc, prev_result_enc,
            p_mean_x, p_mean_y,
            t_mean_x, t_mean_y
        ]]
        
        pred_diff = model.predict(features)
        
        results.append({
            'game_episode': game_episode,
            'end_x': curr['start_x'] + pred_diff[0][0],
            'end_y': curr['start_y'] + pred_diff[0][1]
        })
    else:
        results.append({'game_episode': game_episode, 'end_x': 52.5, 'end_y': 34.0})

submission = pd.DataFrame(results)
submission.to_csv('submission_adv.csv', index=False)
print("\n🎉 최종 진화 예측 파일 생성 완료!")