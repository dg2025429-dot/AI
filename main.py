import altair as alt
import pandas as pd
import streamlit as st

alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="AI를 쓰면 성적이 오를까?", page_icon="🎓", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/dg2025429-dot/AI/refs/heads/main/"
    "ai_student_impact_dataset%20(1).csv"
)

YEAR_ORDER = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
SKILL_ORDER = ["Beginner", "Intermediate", "Advanced"]
BURNOUT_ORDER = ["Low", "Medium", "High"]
HOURS_LABELS = ["0~2h", "2~5h", "5~10h", "10~20h", "20h 이상"]

COL_KR = {
    "Major_Category": "전공 계열",
    "Year_of_Study": "학년",
    "Primary_Use_Case": "주 사용 목적",
    "Prompt_Engineering_Skill": "프롬프트 숙련도",
    "Institutional_Policy": "학교 정책",
    "Paid_Subscription": "유료 구독 여부",
    "Burnout_Risk_Level": "번아웃 위험",
    "AI_Hours_Group": "주간 AI 사용 시간대",
}

NUM_KR = {
    "Pre_Semester_GPA": "학기 전 GPA",
    "Post_Semester_GPA": "학기 후 GPA",
    "GPA_Change": "GPA 변화",
    "Weekly_GenAI_Hours": "주간 AI 사용시간",
    "Traditional_Study_Hours": "전통 학습시간",
    "Tool_Diversity": "AI 도구 다양성",
    "Perceived_AI_Dependency": "AI 의존도(체감)",
    "Anxiety_Level_During_Exams": "시험 불안",
    "Skill_Retention_Score": "기술 유지 점수",
}


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    # 열 이름 앞뒤 공백·BOM 제거
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    required = [
        "Major_Category", "Year_of_Study", "Pre_Semester_GPA", "Weekly_GenAI_Hours",
        "Primary_Use_Case", "Prompt_Engineering_Skill", "Tool_Diversity",
        "Paid_Subscription", "Traditional_Study_Hours", "Perceived_AI_Dependency",
        "Institutional_Policy", "Anxiety_Level_During_Exams", "Post_Semester_GPA",
        "Skill_Retention_Score", "Burnout_Risk_Level",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV에서 다음 열을 찾지 못했습니다: {missing}")
        st.write("실제로 읽힌 열 이름:", list(df.columns))
        st.write("읽힌 데이터 미리보기:")
        st.dataframe(df.head())
        st.stop()

    df["GPA_Change"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]
    df["AI_Hours_Group"] = pd.cut(
        df["Weekly_GenAI_Hours"],
        bins=[-0.01, 2, 5, 10, 20, 40.01],
        labels=HOURS_LABELS,
    )
    # True/False(bool)이든 "True"/"False"(문자열)이든 모두 처리
    df["Paid_Subscription"] = (
        df["Paid_Subscription"].astype(str).str.strip().str.lower()
        .map({"true": "유료", "false": "무료", "1": "유료", "0": "무료"})
        .fillna("알 수 없음")
    )
    return df


df = load_data()

# ---------------------------------------------------------------- 사이드바
st.sidebar.header("🔎 필터")
majors = st.sidebar.multiselect(
    "전공 계열", sorted(df["Major_Category"].unique()), default=sorted(df["Major_Category"].unique())
)
years = st.sidebar.multiselect("학년", YEAR_ORDER, default=YEAR_ORDER)
policies = st.sidebar.multiselect(
    "학교 정책",
    sorted(df["Institutional_Policy"].unique()),
    default=sorted(df["Institutional_Policy"].unique()),
)
paid = st.sidebar.radio("유료 구독", ["전체", "유료", "무료"], horizontal=True)
hours_min, hours_max = st.sidebar.slider("주간 AI 사용시간 (h)", 0.0, 40.0, (0.0, 40.0))

f = df[
    df["Major_Category"].isin(majors)
    & df["Year_of_Study"].isin(years)
    & df["Institutional_Policy"].isin(policies)
    & df["Weekly_GenAI_Hours"].between(hours_min, hours_max)
]
if paid != "전체":
    f = f[f["Paid_Subscription"] == paid]

st.title("🎓 AI를 쓰면 성적이 오를까?")
st.caption("AI 도구 사용이 학생에게 미친 영향 데이터셋 시각화")

if f.empty:
    st.warning("선택한 조건에 맞는 학생이 없습니다. 필터를 조정해 주세요.")
    st.stop()

# ---------------------------------------------------------------- KPI
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("학생 수", f"{len(f):,}명")
c2.metric("학기 전 평균 GPA", f"{f['Pre_Semester_GPA'].mean():.2f}")
c3.metric("학기 후 평균 GPA", f"{f['Post_Semester_GPA'].mean():.2f}")
c4.metric("평균 GPA 변화", f"{f['GPA_Change'].mean():+.3f}")
c5.metric("성적이 오른 학생 비율", f"{(f['GPA_Change'] > 0).mean() * 100:.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 데이터 개요", "📈 AI 사용시간과 성적", "👥 집단별 비교", "😮‍💨 번아웃·불안·기술 유지", "🔗 상관관계"]
)

# ---------------------------------------------------------------- 탭1
with tab1:
    st.subheader("데이터 미리보기")
    st.dataframe(f.drop(columns=["AI_Hours_Group"]).head(50), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("기초 통계")
        st.dataframe(f[list(NUM_KR)].rename(columns=NUM_KR).describe().T.round(3), use_container_width=True)
    with right:
        st.subheader("결측치")
        miss = f.isna().sum().rename("결측 수").to_frame()
        st.dataframe(miss, use_container_width=True)

    st.subheader("GPA 변화 분포")
    hist = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("GPA_Change:Q", bin=alt.Bin(maxbins=40), title="GPA 변화 (학기 후 − 학기 전)"),
            y=alt.Y("count()", title="학생 수"),
        )
        .properties(height=300)
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="red", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(hist + zero, use_container_width=True)

# ---------------------------------------------------------------- 탭2
with tab2:
    st.subheader("주간 AI 사용시간과 GPA 변화")
    pts = (
        alt.Chart(f)
        .mark_circle(opacity=0.35, size=30)
        .encode(
            x=alt.X("Weekly_GenAI_Hours:Q", title="주간 AI 사용시간 (h)"),
            y=alt.Y("GPA_Change:Q", title="GPA 변화"),
            color=alt.Color("Major_Category:N", title="전공 계열"),
            tooltip=["Major_Category", "Year_of_Study", "Weekly_GenAI_Hours", "GPA_Change"],
        )
    )
    trend = pts.transform_regression("Weekly_GenAI_Hours", "GPA_Change").mark_line(color="black", size=3)
    st.altair_chart((pts + trend).properties(height=380), use_container_width=True)

    st.subheader("사용 시간대별 평균 GPA 변화")
    g = (
        f.groupby("AI_Hours_Group", observed=True)["GPA_Change"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "평균 GPA 변화", "count": "학생 수"})
    )
    bar = (
        alt.Chart(g)
        .mark_bar()
        .encode(
            x=alt.X("AI_Hours_Group:N", sort=HOURS_LABELS, title="주간 AI 사용 시간대"),
            y=alt.Y("평균 GPA 변화:Q"),
            tooltip=["AI_Hours_Group", "평균 GPA 변화", "학생 수"],
        )
        .properties(height=320)
    )
    st.altair_chart(bar, use_container_width=True)
    st.caption(
        "⚠️ Post GPA는 4.0이 상한이라 학기 전 GPA가 높은 학생은 '오를 여지'가 작습니다(천장 효과). "
        "상관관계는 인과관계가 아닙니다."
    )

# ---------------------------------------------------------------- 탭3
with tab3:
    st.subheader("집단별 평균 GPA 변화")
    group_col = st.selectbox(
        "비교할 기준", list(COL_KR), format_func=lambda c: COL_KR[c], index=0
    )
    order = {
        "Year_of_Study": YEAR_ORDER,
        "Prompt_Engineering_Skill": SKILL_ORDER,
        "Burnout_Risk_Level": BURNOUT_ORDER,
        "AI_Hours_Group": HOURS_LABELS,
    }.get(group_col)

    gg = (
        f.groupby(group_col, observed=True)
        .agg(평균_GPA_변화=("GPA_Change", "mean"), 학생_수=("GPA_Change", "size"),
             평균_AI시간=("Weekly_GenAI_Hours", "mean"))
        .reset_index()
    )
    chart = (
        alt.Chart(gg)
        .mark_bar()
        .encode(
            x=alt.X(f"{group_col}:N", sort=order, title=COL_KR[group_col]),
            y=alt.Y("평균_GPA_변화:Q", title="평균 GPA 변화"),
            color=alt.Color(f"{group_col}:N", legend=None),
            tooltip=[group_col, alt.Tooltip("평균_GPA_변화:Q", format=".3f"), "학생_수",
                     alt.Tooltip("평균_AI시간:Q", format=".1f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("GPA 변화 분포 (상자그림)")
    box = (
        alt.Chart(f)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X(f"{group_col}:N", sort=order, title=COL_KR[group_col]),
            y=alt.Y("GPA_Change:Q", title="GPA 변화"),
            color=alt.Color(f"{group_col}:N", legend=None),
        )
        .properties(height=340)
    )
    st.altair_chart(box, use_container_width=True)

# ---------------------------------------------------------------- 탭4
with tab4:
    a, b = st.columns(2)
    with a:
        st.subheader("사용 시간대별 번아웃 위험 비율")
        stack = (
            alt.Chart(f)
            .mark_bar()
            .encode(
                x=alt.X("AI_Hours_Group:N", sort=HOURS_LABELS, title="주간 AI 사용 시간대"),
                y=alt.Y("count()", stack="normalize", title="비율"),
                color=alt.Color(
                    "Burnout_Risk_Level:N",
                    sort=BURNOUT_ORDER,
                    scale=alt.Scale(domain=BURNOUT_ORDER, range=["#4CAF50", "#FFC107", "#F44336"]),
                    title="번아웃 위험",
                ),
                tooltip=["AI_Hours_Group", "Burnout_Risk_Level", "count()"],
            )
            .properties(height=340)
        )
        st.altair_chart(stack, use_container_width=True)
    with b:
        st.subheader("AI 의존도와 시험 불안")
        heat = (
            alt.Chart(f)
            .mark_rect()
            .encode(
                x=alt.X("Perceived_AI_Dependency:O", title="AI 의존도(체감, 1~10)"),
                y=alt.Y("Anxiety_Level_During_Exams:O", title="시험 불안(1~10)", sort="descending"),
                color=alt.Color("count()", title="학생 수", scale=alt.Scale(scheme="blues")),
            )
            .properties(height=340)
        )
        st.altair_chart(heat, use_container_width=True)

    st.subheader("AI 사용시간과 기술 유지 점수")
    ret = (
        alt.Chart(f)
        .mark_circle(opacity=0.35, size=30)
        .encode(
            x=alt.X("Weekly_GenAI_Hours:Q", title="주간 AI 사용시간 (h)"),
            y=alt.Y("Skill_Retention_Score:Q", title="기술 유지 점수"),
            color=alt.Color("Primary_Use_Case:N", title="주 사용 목적"),
            tooltip=["Primary_Use_Case", "Weekly_GenAI_Hours", "Skill_Retention_Score"],
        )
    )
    ret_line = ret.transform_regression("Weekly_GenAI_Hours", "Skill_Retention_Score").mark_line(
        color="black", size=3
    )
    st.altair_chart((ret + ret_line).properties(height=360), use_container_width=True)

# ---------------------------------------------------------------- 탭5
with tab5:
    st.subheader("수치형 변수 상관계수")
    corr = f[list(NUM_KR)].rename(columns=NUM_KR).corr().round(2)
    long = corr.stack().reset_index()
    long.columns = ["변수 A", "변수 B", "상관계수"]
    base = alt.Chart(long).encode(
        x=alt.X("변수 A:N", title=None), y=alt.Y("변수 B:N", title=None)
    )
    rect = base.mark_rect().encode(
        color=alt.Color("상관계수:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]))
    )
    text = base.mark_text(fontSize=11).encode(text=alt.Text("상관계수:Q", format=".2f"))
    st.altair_chart((rect + text).properties(height=520), use_container_width=True)

    target = st.selectbox("GPA 변화와의 상관이 큰 순서로 보기", ["GPA 변화"], index=0)
    st.dataframe(
        corr[target].drop(target).sort_values(key=abs, ascending=False).rename("상관계수").to_frame(),
        use_container_width=True,
    )
