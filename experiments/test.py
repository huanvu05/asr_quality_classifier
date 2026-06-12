import pandas as pd
import numpy as np

def run_eda_text_only(csv_path):
    print(f"[*] Đang tải dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. TIỀN XỬ LÝ NHẸ
    df['target'] = df['label_text'].apply(lambda x: 1 if str(x).strip().lower() == 'usable' else 0)
    
    print("\n=========================================")
    print("PHẦN 1: TỔNG QUAN DỮ LIỆU (OVERVIEW)")
    print("=========================================")
    print(f"- Tổng số mẫu audio: {len(df)}")
    print(f"- Tổng số người đánh nhãn (Annotators): {df['username'].nunique()} người")
    print(f"- Tổng số câu text (Transcripts) độc lập: {df['transcript'].nunique()} câu")
    print(f"- Tỷ lệ Usable (Nhãn 1): {(df['target'].mean() * 100):.2f}%")
    print(f"- Tỷ lệ Unusable (Nhãn 0): {((1 - df['target'].mean()) * 100):.2f}%")

    print("\n=========================================")
    print("PHẦN 2: PHÂN TÍCH HÀNH VI NGƯỜI ĐÁNH NHÃN")
    print("=========================================")
    user_stats = df.groupby('username').agg(
        total_tasks=('target', 'count'),
        usable_count=('target', 'sum'),
    ).reset_index()
    user_stats['unusable_count'] = user_stats['total_tasks'] - user_stats['usable_count']
    user_stats['acceptance_rate(%)'] = (user_stats['usable_count'] / user_stats['total_tasks'] * 100).round(2)
    user_stats['rejection_rate(%)'] = 100 - user_stats['acceptance_rate(%)']
    
    user_stats = user_stats.sort_values('rejection_rate(%)', ascending=False)
    print(user_stats.to_string(index=False))

    print("\n=========================================")
    print("PHẦN 3: PHÂN TÍCH ĐỘ NHIỄU (TRANSCRIPT CONFLICTS)")
    print("=========================================")
    transcript_stats = df.groupby('transcript').agg(
        audio_versions=('file_name', 'count'),
        usable_votes=('target', 'sum')
    ).reset_index()
    
    transcript_stats['usable_ratio'] = transcript_stats['usable_votes'] / transcript_stats['audio_versions']
    
    perfect_agreement = transcript_stats[(transcript_stats['usable_ratio'] == 1.0) | (transcript_stats['usable_ratio'] == 0.0)]
    conflicted_transcripts = transcript_stats[(transcript_stats['usable_ratio'] > 0.0) & (transcript_stats['usable_ratio'] < 1.0)].copy()
    
    print(f"- Số transcript ĐỒNG THUẬN 100%: {len(perfect_agreement)} ({len(perfect_agreement)/len(transcript_stats)*100:.1f}%)")
    print(f"- Số transcript BỊ TRANH CÃI (Mix nhãn 0 và 1): {len(conflicted_transcripts)} ({len(conflicted_transcripts)/len(transcript_stats)*100:.1f}%)")

    print("\n=========================================")
    print("PHẦN 4: ĐỘ SÂU CỦA NHIỄU (AUDIO-LEVEL NOISE)")
    print("=========================================")
    # Tính số audio thực tế bị ảnh hưởng
    conflicted_audio_df = df[df['transcript'].isin(conflicted_transcripts['transcript'])]
    total_conflicted_audio = len(conflicted_audio_df)
    total_audio = len(df)
    
    print(f"- Tổng số audio dính líu đến mâu thuẫn: {total_conflicted_audio} / {total_audio} file")
    print(f"- TỶ LỆ NHIỄU THỰC TẾ (Audio Conflict Rate): {(total_conflicted_audio / total_audio * 100):.2f}%\n")
    
    # Phân loại tính chất mâu thuẫn
    conflicted_transcripts.loc[:, 'unusable_votes'] = conflicted_transcripts['audio_versions'] - conflicted_transcripts['usable_votes']
    
    def categorize_conflict(row):
        u_votes = row['usable_votes']
        un_votes = row['unusable_votes']
        if u_votes == un_votes:
            return "50/50 (Tranh cãi cân bằng, vd 1-1, 2-2)"
        elif min(u_votes, un_votes) == 1 and max(u_votes, un_votes) >= 3:
            return "Minority Dissent (Lệch hẳn, vd 1-3, 1-4, 1-5)"
        else:
            return "Mix (Lệch nhẹ, vd 1-2, 2-3)"

    conflicted_transcripts.loc[:, 'conflict_type'] = conflicted_transcripts.apply(categorize_conflict, axis=1)
    
    print("[*] Phân loại tính chất của các transcript bị mâu thuẫn:")
    type_counts = conflicted_transcripts['conflict_type'].value_counts()
    for c_type, count in type_counts.items():
        print(f"  + {c_type}: {count} transcript ({count/len(conflicted_transcripts)*100:.1f}%)")

    minority_cases = conflicted_transcripts[conflicted_transcripts['conflict_type'].str.contains("Minority Dissent")]
    if len(minority_cases) > 0:
        print(f"\n[!] CƠ HỘI SỬA NHIỄU: Có {len(minority_cases)} transcript có 1 người đi ngược lại số đông.")
        print("    -> Nên dùng thuật toán Hard Voting/Majority Voting để lật lại nhãn thiểu số.")

# === CÁCH CHẠY ===
csv_file_path = '/Users/admin/Documents/AI_ThucChien/asr_quality_classifier/data/transcripts/training.csv'
run_eda_text_only(csv_file_path)