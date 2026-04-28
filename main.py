import pandas as pd
import os

# 1. 파일 경로 설정
train_path = 'train.csv'
test_folder = 'test'  # 압축 푼 폴더 이름

print("=" * 30)
print("📂 데이터를 확인하는 중입니다...")

# 2. train.csv 확인
if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    print(f"✅ train.csv 로딩 성공!")
    print(f"   - 데이터 개수: {len(train_df)}행")
    print(f"   - 컬럼 목록: {list(train_df.columns)[:5]} ... 등등")
else:
    print(f"❌ [오류] train.csv가 없습니다. 폴더에 파일을 넣었는지 확인하세요!")

# 3. test 폴더 확인
if os.path.exists(test_folder):
    # 폴더 안에 파일 개수 세기
    files = []
    for root, dirs, filenames in os.walk(test_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                files.append(filename)
    
    file_count = len(files)
    
    if file_count > 1000:
        print(f"✅ test 폴더 확인 완료!")
        print(f"   - CSV 파일 개수: {file_count}개 (정상입니다)")
    else:
        print(f"⚠️ test 폴더는 있는데 파일이 너무 적습니다 ({file_count}개). 압축을 제대로 푼 게 맞나요?")
else:
    print(f"❌ [오류] 'test' 폴더가 안 보입니다. 압축을 풀어서 K_League 폴더 안에 넣어주세요.")

print("=" * 30)