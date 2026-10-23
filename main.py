import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

alt.data_transformers.disable_max_rows()
st.set_page_config(page_title="AI를 쓰면 성적이 오를까?", page_icon="🎓", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/dg2025429-dot/AI/refs/heads/main/"
    "ai_student_impact_dataset%20(1).csv"
)

# 꼭 있어야 하는 핵심 열 (이 4개만 있어도 앱이 동작합니다)
CORE = ["Pre_Semester_GPA", "Weekly_GenAI_Hours", "Traditional_Study_Hours", "Post_Semester_GPA"]

HOURS_BINS = [-0.01, 2, 5, 10, 20, 40.01]
HOURS_LABELS = ["0~2h", "2~5h", "5~10h", "10~20h", "20h 이상"]
STUDY_BINS = [-0.01, 5, 10, 15, 100]
STUDY_LABELS = ["5h 이하", "5~10h", "10~15h", "15h 이상"]

# 있을 때만 쓰는 선택 열 (16개 열 버전 데이터용)
CAT_KR = {
    "Major_Category": "전공 계열",
    "Year_of_Study": "학년",
    "Primary_Use_Case": "주 사용 목적",
    "Prompt_Engineering_Skill": "프롬프트 숙련도",
    "Institutional_Policy": "학교 정책",
    "Paid_Subscription": "유료 구독 여부",
    "Burnout_Risk_Level": "번아웃 위험",
}
CAT_ORDER = {
    "Year_of_Study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
    "Burnout_Risk_Level": ["Low", "Medium", "High"],
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
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df


raw = load_data()

missing = [c for c in CORE if c not in raw.columns]
if missing:
    st.error(f"필수 열을 찾지 못했습니다: {missing}")
    st.write("실제로 읽힌 열 이름:", list(raw.columns))
    st.dataframe(raw.head())
    st.stop()

df = raw.copy()
df["GPA_Change"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]
df["AI_Hours_Group"] = pd.cut(df["Weekly_GenAI_Hours"], HOURS_BINS, labels=HOURS_LABELS)
df["Study_Hours_Group"] = pd.cut(df["Traditional_Study_Hours"], STUDY_BINS, labels=STUDY_LABELS)
if "Paid_Subscription" in df.columns:
    df["Paid_Subscription"] = (
        df["Paid_Subscription"].astype(str).str.strip().str.lower()
        .map({"true": "유료", "false": "무료", "1": "유료", "0": "무료"})
        .fillna("알 수 없음")
    )

CAT_COLS = [c for c in CAT_KR if c in df.columns]
NUM_COLS = [c for c in NUM_KR if c in df.columns]

# ------------------------------------------------------------ 사이드바
st.sidebar.header("🔎 필터")
h_lo, h_hi = st.sidebar.slider("주간 AI 사용시간 (h)", 0.0, 40.0, (0.0, 40.0))
s_max = float(np.ceil(df["Traditional_Study_Hours"].max()))
s_lo, s_hi = st.sidebar.slider("전통 학습시간 (h)", 0.0, s_max, (0.0, s_max))
g_lo, g_hi = st.sidebar.slider("학기 전 GPA", 1.0, 4.0, (1.0, 4.0), step=0.1)

mask = (
    df["Weekly_GenAI_Hours"].between(h_lo, h_hi)
    & df["Traditional_Study_Hours"].between(s_lo, s_hi)
    & df["Pre_Semester_GPA"].between(g_lo, g_hi)
)
for col in CAT_COLS:  # 선택 열이 있으면 필터도 추가
    opts = CAT_ORDER.get(col, sorted(df[col].dropna().unique()))
    chosen = st.sidebar.multiselect(CAT_KR[col], opts, default=opts)
    mask &= df[col].isin(chosen)
f = df[mask]

st.title("🎓 AI를 쓰면 성적이 오를까?")
st.caption(f"AI 도구 사용이 학생에게 미친 영향 · 전체 {len(df):,}명 중 {len(f):,}명 선택")

if f.empty:
    st.warning("선택한 조건에 맞는 학생이 없습니다. 필터를 조정해 주세요.")
    st.stop()

# ------------------------------------------------------------ KPI
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("학생 수", f"{len(f):,}명")
c2.metric("학기 전 평균 GPA", f"{f['Pre_Semester_GPA'].mean():.2f}")
c3.metric("학기 후 평균 GPA", f"{f['Post_Semester_GPA'].mean():.2f}")
c4.metric("평균 GPA 변화", f"{f['GPA_Change'].mean():+.3f}")
c5.metric("성적이 오른 학생 비율", f"{(f['GPA_Change'] > 0).mean() * 100:.1f}%")

tab_names = ["📋 데이터 개요", "📈 AI 사용시간과 성적", "📚 AI × 학습시간"]
if CAT_COLS:
    tab_names.append("👥 집단별 비교")
tab_names.append("🔗 상관관계")
tabs = dict(zip(tab_names, st.tabs(tab_names)))


def sample(d: pd.DataFrame, n: int = 3000) -> pd.DataFrame:
    return d.sample(n, random_state=0) if len(d) > n else d


# ------------------------------------------------------------ 탭: 개요
with tabs["📋 데이터 개요"]:
    st.subheader("데이터 미리보기")
    st.dataframe(f.drop(columns=["AI_Hours_Group", "Study_Hours_Group"]).head(50), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("기초 통계")
        st.dataframe(f[NUM_COLS].rename(columns=NUM_KR).describe().T.round(3), use_container_width=True)
    with right:
        st.subheader("데이터 점검")
        chk = pd.DataFrame(
            {
                "값": [
                    f"{raw.shape[0]:,}행 × {raw.shape[1]}열",
                    int(raw.isna().sum().sum()),
                    f"{(df['Post_Semester_GPA'] >= 4.0).mean() * 100:.1f}%",
                    f"{(df['Weekly_GenAI_Hours'] >= 40).mean() * 100:.1f}%",
                ]
            },
            index=["데이터 크기", "결측치 합계", "학기 후 GPA가 4.0(상한)인 비율", "주간 AI 사용시간이 40h(상한)인 비율"],
        )
        st.dataframe(chk, use_container_width=True)

    st.subheader("GPA 변화 분포")
    hist = (
        alt.Chart(f[["GPA_Change"]])
        .mark_bar()
        .encode(
            x=alt.X("GPA_Change:Q", bin=alt.Bin(maxbins=50), title="GPA 변화 (학기 후 − 학기 전)"),
            y=alt.Y("count()", title="학생 수"),
        )
        .properties(height=300)
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="red", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(hist + zero, use_container_width=True)

# ------------------------------------------------------------ 탭: AI 사용시간
with tabs["📈 AI 사용시간과 성적"]:
    st.subheader("주간 AI 사용시간별 평균 GPA 변화 (2시간 구간)")
    tmp = f.assign(h_bin=(np.floor(f["Weekly_GenAI_Hours"] / 2) * 2).clip(upper=38) + 1)
    agg = (
        tmp.groupby("h_bin")["GPA_Change"]
        .agg(mean="mean", sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0, n="size")
        .reset_index()
    )
    agg = agg[agg["n"] >= 20]
    agg["lo"] = agg["mean"] - 1.96 * agg["sem"]
    agg["hi"] = agg["mean"] + 1.96 * agg["sem"]
    band = alt.Chart(agg).mark_area(opacity=0.25).encode(
        x=alt.X("h_bin:Q", title="주간 AI 사용시간 (h)"), y=alt.Y("lo:Q", title="평균 GPA 변화"), y2="hi:Q"
    )
    line = alt.Chart(agg).mark_line(point=True).encode(
        x="h_bin:Q",
        y="mean:Q",
        tooltip=["h_bin", alt.Tooltip("mean:Q", format=".3f"), "n"],
    )
    zero_h = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((band + line + zero_h).properties(height=360), use_container_width=True)
    st.caption("선은 구간별 평균, 옅은 띠는 95% 신뢰구간입니다. 구간별 학생이 20명 미만이면 제외했습니다.")

    a, b = st.columns(2)
    with a:
        st.subheader("사용 시간대별 평균 GPA 변화")
        g = (
            f.groupby("AI_Hours_Group", observed=True)["GPA_Change"]
            .agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "평균 GPA 변화", "count": "학생 수"})
        )
        bar = alt.Chart(g).mark_bar().encode(
            x=alt.X("AI_Hours_Group:N", sort=HOURS_LABELS, title="주간 AI 사용 시간대"),
            y=alt.Y("평균 GPA 변화:Q"),
            tooltip=["AI_Hours_Group", alt.Tooltip("평균 GPA 변화:Q", format=".3f"), "학생 수"],
        ).properties(height=320)
        st.altair_chart(bar, use_container_width=True)
    with b:
        st.subheader("GPA 변화 분포 (시간대별)")
        box = alt.Chart(sample(f, 10000)).mark_boxplot(extent="min-max").encode(
            x=alt.X("AI_Hours_Group:N", sort=HOURS_LABELS, title="주간 AI 사용 시간대"),
            y=alt.Y("GPA_Change:Q", title="GPA 변화"),
            color=alt.Color("AI_Hours_Group:N", legend=None),
        ).properties(height=320)
        st.altair_chart(box, use_container_width=True)

    st.subheader("개별 학생 산점도 (무작위 3,000명)")
    pts = alt.Chart(sample(f)).mark_circle(opacity=0.3, size=25).encode(
        x=alt.X("Weekly_GenAI_Hours:Q", title="주간 AI 사용시간 (h)"),
        y=alt.Y("GPA_Change:Q", title="GPA 변화"),
        tooltip=["Weekly_GenAI_Hours", "Traditional_Study_Hours", "GPA_Change"],
    )
    st.altair_chart(pts.properties(height=360), use_container_width=True)
    st.caption(
        "⚠️ 학기 후 GPA는 4.0이 상한이라 학기 전 GPA가 높은 학생은 오를 여지가 작습니다(천장 효과). "
        "또한 이 그래프는 상관관계이지 AI가 성적을 바꿨다는 인과관계의 증거는 아닙니다."
    )

# ------------------------------------------------------------ 탭: AI × 학습시간
with tabs["📚 AI × 학습시간"]:
    st.subheader("AI 사용시간 × 전통 학습시간별 평균 GPA 변화")
    hm = (
        f.groupby(["AI_Hours_Group", "Study_Hours_Group"], observed=True)["GPA_Change"]
        .agg(["mean", "count"]).reset_index()
    )
    base = alt.Chart(hm).encode(
        x=alt.X("AI_Hours_Group:N", sort=HOURS_LABELS, title="주간 AI 사용 시간대"),
        y=alt.Y("Study_Hours_Group:N", sort=STUDY_LABELS[::-1], title="전통 학습시간"),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("mean:Q", title="평균 GPA 변화", scale=alt.Scale(scheme="redyellowgreen", domainMid=0)),
        tooltip=["AI_Hours_Group", "Study_Hours_Group", alt.Tooltip("mean:Q", format=".3f"), "count"],
    )
    text = base.mark_text(fontSize=13).encode(text=alt.Text("mean:Q", format="+.2f"))
    st.altair_chart((rect + text).properties(height=340), use_container_width=True)
    st.caption("AI를 많이 써도 전통 학습시간이 충분하면 성적 변화가 어떻게 달라지는지 비교해 볼 수 있습니다.")

    st.subheader("전통 학습시간별 평균 GPA 변화")
    sg = (
        f.groupby("Study_Hours_Group", observed=True)["GPA_Change"]
        .agg(["mean", "count"]).reset_index().rename(columns={"mean": "평균 GPA 변화", "count": "학생 수"})
    )
    sbar = alt.Chart(sg).mark_bar().encode(
        x=alt.X("Study_Hours_Group:N", sort=STUDY_LABELS, title="전통 학습시간"),
        y="평균 GPA 변화:Q",
        tooltip=["Study_Hours_Group", alt.Tooltip("평균 GPA 변화:Q", format=".3f"), "학생 수"],
    ).properties(height=300)
    st.altair_chart(sbar, use_container_width=True)

# ------------------------------------------------------------ 탭: 집단별 비교 (선택 열이 있을 때만)
if CAT_COLS:
    with tabs["👥 집단별 비교"]:
        group_col = st.selectbox("비교할 기준", CAT_COLS, format_func=lambda c: CAT_KR[c])
        order = CAT_ORDER.get(group_col)
        gg = (
            f.groupby(group_col, observed=True)["GPA_Change"]
            .agg(["mean", "count"]).reset_index().rename(columns={"mean": "평균 GPA 변화", "count": "학생 수"})
        )
        chart = alt.Chart(gg).mark_bar().encode(
            x=alt.X(f"{group_col}:N", sort=order, title=CAT_KR[group_col]),
            y="평균 GPA 변화:Q",
            color=alt.Color(f"{group_col}:N", legend=None),
            tooltip=[group_col, alt.Tooltip("평균 GPA 변화:Q", format=".3f"), "학생 수"],
        ).properties(height=340)
        st.altair_chart(chart, use_container_width=True)

        if "Burnout_Risk_Level" in f.columns:
            st.subheader("사용 시간대별 번아웃 위험 비율")
            stack = alt.Chart(f).mark_bar().encode(
                x=alt.X("AI_Hours_Group:N", sort=HOURS_LABELS, title="주간 AI 사용 시간대"),
                y=alt.Y("count()", stack="normalize", title="비율"),
                color=alt.Color(
                    "Burnout_Risk_Level:N",
                    scale=alt.Scale(domain=["Low", "Medium", "High"], range=["#4CAF50", "#FFC107", "#F44336"]),
                    title="번아웃 위험",
                ),
            ).properties(height=320)
            st.altair_chart(stack, use_container_width=True)

# ------------------------------------------------------------ 탭: 상관관계
with tabs["🔗 상관관계"]:
    st.subheader("수치형 변수 상관계수")
    corr = f[NUM_COLS].rename(columns=NUM_KR).corr().round(2)
    long = corr.stack().reset_index()
    long.columns = ["변수 A", "변수 B", "상관계수"]
    base = alt.Chart(long).encode(x=alt.X("변수 A:N", title=None), y=alt.Y("변수 B:N", title=None))
    rect = base.mark_rect().encode(
        color=alt.Color("상관계수:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]))
    )
    text = base.mark_text(fontSize=12).encode(text=alt.Text("상관계수:Q", format=".2f"))
    st.altair_chart((rect + text).properties(height=420), use_container_width=True)

    st.subheader("GPA 변화와의 상관계수 (절댓값 큰 순)")
    st.dataframe(
        corr["GPA 변화"].drop("GPA 변화").sort_values(key=abs, ascending=False).rename("상관계수").to_frame(),
        use_container_width=True,
    )
