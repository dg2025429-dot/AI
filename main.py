from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI를 쓰면 성적이 오를까?", page_icon="📈", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ai_student_impact_dataset.csv"

HOUR_BINS = [-0.1, 2, 5, 10, 15, 20, 41]
HOUR_LABELS = ["0~2시간", "2~5시간", "5~10시간", "10~15시간", "15~20시간", "20시간 이상"]

USE_CASE_KR = {
    "Debugging/Troubleshooting": "코드 디버깅/문제 해결",
    "Ideation": "아이디어 구상",
    "Copywriting/Drafting": "글쓰기/초안 작성",
    "Summarizing_Reading": "읽기 자료 요약",
    "Direct_Answer_Generation": "정답 바로 받기",
}
SKILL_KR = {"Beginner": "초급", "Intermediate": "중급", "Advanced": "고급"}
MAJOR_KR = {"STEM": "이공계", "Business": "경영", "Humanities": "인문", "Medical": "의료", "Arts": "예술"}
YEAR_KR = {"Freshman": "1학년", "Sophomore": "2학년", "Junior": "3학년", "Senior": "4학년", "Graduate": "대학원"}
POLICY_KR = {
    "Strict_Ban": "사용 금지",
    "Allowed_With_Citation": "출처 표기 시 허용",
    "Actively_Encouraged": "적극 권장",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["GPA_Change"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]
    df["Hour_Group"] = pd.cut(df["Weekly_GenAI_Hours"], bins=HOUR_BINS, labels=HOUR_LABELS)
    return df


if not DATA_PATH.exists():
    st.error(f"데이터 파일을 찾을 수 없어요: {DATA_PATH}")
    st.stop()

df = load_data()

# ---------------------------------------------------------------- 사이드바 필터
st.sidebar.header("🔎 학생 집단 고르기")
majors = st.sidebar.multiselect("전공", list(MAJOR_KR), default=list(MAJOR_KR), format_func=MAJOR_KR.get)
years = st.sidebar.multiselect("학년", list(YEAR_KR), default=list(YEAR_KR), format_func=YEAR_KR.get)
policies = st.sidebar.multiselect("학교의 AI 사용 정책", list(POLICY_KR), default=list(POLICY_KR), format_func=POLICY_KR.get)

view = df[df["Major_Category"].isin(majors) & df["Year_of_Study"].isin(years) & df["Institutional_Policy"].isin(policies)]
st.sidebar.caption(f"선택된 학생: {len(view):,}명 / 전체 {len(df):,}명")

# ---------------------------------------------------------------- 머리말
st.title("📈 AI를 쓰면 성적이 오를까?")
st.write(
    "대학생 5만 명의 **학기 전·후 GPA**와 **생성형 AI 사용 습관**이 담긴 데이터로, "
    "'AI를 많이 쓰면 성적이 더 오른다'는 말이 맞는지 직접 확인해 봅니다."
)

if view.empty:
    st.warning("선택한 조건에 맞는 학생이 없어요. 왼쪽에서 조건을 넓혀 주세요.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("학생 수", f"{len(view):,}명")
c2.metric("학기 전 평균 GPA", f"{view['Pre_Semester_GPA'].mean():.2f}")
c3.metric("학기 후 평균 GPA", f"{view['Post_Semester_GPA'].mean():.2f}", f"{view['GPA_Change'].mean():+.2f}")
c4.metric("주당 평균 AI 사용", f"{view['Weekly_GenAI_Hours'].mean():.1f}시간")

tab_data, tab_hours, tab_how, tab_other, tab_try = st.tabs(
    ["📋 데이터 보기", "⏱️ 사용 시간별", "🧠 쓰는 방법별", "⚖️ 다른 요인과 비교", "🧪 직접 비교해 보기"]
)

# ---------------------------------------------------------------- 1. 데이터 보기
with tab_data:
    st.subheader("이 데이터는 어떻게 생겼나요?")
    st.write("학생 한 명이 한 줄이고, 학기 전 GPA(`Pre_Semester_GPA`)와 학기 후 GPA(`Post_Semester_GPA`)가 모두 있어요.")
    st.dataframe(view.drop(columns=["Hour_Group"]).head(20), use_container_width=True)

    st.subheader("GPA는 학기 동안 얼마나 변했나요?")
    st.caption("변화량 = 학기 후 GPA − 학기 전 GPA (0.1 단위로 묶어서 학생 수를 센 그림)")
    change_bins = (view["GPA_Change"] / 0.1).round() * 0.1
    hist = change_bins.round(1).value_counts().sort_index().rename("학생 수")
    st.bar_chart(hist)
    up_ratio = (view["GPA_Change"] > 0).mean() * 100
    st.info(f"선택한 학생 중 **{up_ratio:.1f}%** 는 GPA가 올랐어요. AI를 거의 안 쓴 학생들도 올랐는지는 다음 탭에서 확인해요.")

# ---------------------------------------------------------------- 2. 사용 시간별
with tab_hours:
    st.subheader("AI를 오래 쓸수록 GPA가 더 오를까?")
    by_hours = (
        view.groupby("Hour_Group", observed=True)
        .agg(학생수=("GPA_Change", "size"), GPA변화=("GPA_Change", "mean"), 기억점수=("Skill_Retention_Score", "mean"))
        .round(3)
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**주당 AI 사용 시간 구간별 평균 GPA 변화**")
        st.bar_chart(by_hours["GPA변화"])
    with right:
        st.markdown("**같은 구간의 평균 '기술 기억 점수'** (배운 내용을 얼마나 기억하는지)")
        st.bar_chart(by_hours["기억점수"])
    st.dataframe(by_hours, use_container_width=True)
    st.info(
        "읽는 법: AI를 거의 안 쓴 학생도 GPA가 올랐고, 사용 시간이 늘어도 변화는 크게 달라지지 않아요. "
        "아주 많이 쓰는 구간(20시간 이상)에서는 오히려 GPA 상승폭과 기억 점수가 함께 낮아졌어요."
    )

# ---------------------------------------------------------------- 3. 쓰는 방법별
with tab_how:
    st.subheader("얼마나 쓰느냐보다 어떻게 쓰느냐가 중요할까?")
    left, right = st.columns(2)
    with left:
        st.markdown("**주된 사용 목적별 평균 GPA 변화**")
        by_use = view.groupby("Primary_Use_Case")["GPA_Change"].mean().rename(index=USE_CASE_KR).sort_values()
        st.bar_chart(by_use.rename("GPA 변화"))
    with right:
        st.markdown("**프롬프트 작성 실력별 평균 GPA 변화**")
        by_skill = view.groupby("Prompt_Engineering_Skill")["GPA_Change"].mean().reindex(["Beginner", "Intermediate", "Advanced"])
        st.bar_chart(by_skill.rename(index=SKILL_KR).rename("GPA 변화"))
    st.info("정답을 바로 받아 쓰는 학생은 상승폭이 가장 작았고, 디버깅처럼 AI와 함께 문제를 풀어 가는 쓰임에서 가장 컸어요.")

# ---------------------------------------------------------------- 4. 다른 요인과 비교
with tab_other:
    st.subheader("GPA 변화와 가장 관계 깊은 요인은?")
    cols = {
        "Weekly_GenAI_Hours": "주당 AI 사용 시간",
        "Traditional_Study_Hours": "AI 없이 공부한 시간",
        "Tool_Diversity": "쓰는 AI 도구 수",
        "Perceived_AI_Dependency": "AI 의존도(체감)",
        "Anxiety_Level_During_Exams": "시험 불안 수준",
        "Skill_Retention_Score": "기술 기억 점수",
        "Pre_Semester_GPA": "학기 전 GPA",
    }
    corr = view[list(cols)].corrwith(view["GPA_Change"]).rename(index=cols).sort_values().round(3)
    st.bar_chart(corr.rename("GPA 변화와의 상관계수"))
    st.caption("상관계수는 -1~1 사이 값이에요. 0에 가까울수록 관계가 약하고, 양수면 함께 오르는 경향이에요.")
    st.info(
        "AI 사용 시간과 GPA 변화의 상관은 0에 가깝지만, AI 없이 공부한 시간과는 뚜렷한 양의 관계가 있어요. "
        "단, 상관은 '원인'을 뜻하지 않아요. 공부를 많이 하는 학생이 원래 성실해서일 수도 있어요."
    )

# ---------------------------------------------------------------- 5. 직접 비교
with tab_try:
    st.subheader("두 집단을 직접 비교해 보기")
    st.write("주당 AI 사용 시간으로 학생들을 나누고, 집단 간 GPA 변화를 비교해 보세요.")
    cut_low, cut_high = st.slider("AI 사용 시간 기준 (시간/주)", 0.0, 40.0, (3.0, 12.0), step=0.5)
    low = view[view["Weekly_GenAI_Hours"] <= cut_low]
    high = view[view["Weekly_GenAI_Hours"] >= cut_high]
    if low.empty or high.empty:
        st.warning("한쪽 집단에 학생이 없어요. 기준을 조정해 주세요.")
    else:
        a, b = st.columns(2)
        a.metric(f"적게 쓴 집단 (≤{cut_low}시간, {len(low):,}명)", f"{low['GPA_Change'].mean():+.3f}", "평균 GPA 변화", delta_color="off")
        b.metric(f"많이 쓴 집단 (≥{cut_high}시간, {len(high):,}명)", f"{high['GPA_Change'].mean():+.3f}", "평균 GPA 변화", delta_color="off")
        diff = high["GPA_Change"].mean() - low["GPA_Change"].mean()
        st.write(f"두 집단의 평균 차이: **{diff:+.3f}점**")

st.divider()
st.caption("데이터: ai_student_impact_dataset.csv (학생 5만 명, 16개 변수) · 이 앱은 수업 실습용이며, 보이는 관계는 인과관계가 아닐 수 있어요.")
