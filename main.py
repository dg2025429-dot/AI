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
GPA_BINS = [0, 2.5, 3.0, 3.5, 4.01]
GPA_LABELS = ["2.5 미만", "2.5~3.0", "3.0~3.5", "3.5 이상"]

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
df["Pre_GPA_Group"] = pd.cut(df["Pre_Semester_GPA"], GPA_BINS, labels=GPA_LABELS)
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

tab_names = ["📋 데이터 개요", "📈 AI 사용시간과 성적", "🎯 학기 전·후 GPA", "📚 AI × 학습시간"]
if CAT_COLS:
    tab_names.append("👥 집단별 비교")
tab_names.append("🔗 상관관계")
tabs = dict(zip(tab_names, st.tabs(tab_names)))


def sample(d: pd.DataFrame, n: int = 3000) -> pd.DataFrame:
    return d.sample(n, random_state=0) if len(d) > n else d


# ------------------------------------------------------------ 탭: 개요
with tabs["📋 데이터 개요"]:
    st.subheader("데이터 미리보기")
    st.dataframe(f.drop(columns=["AI_Hours_Group", "Study_Hours_Group", "Pre_GPA_Group"]).head(50), use_container_width=True)

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
GROUP_OPTIONS = {"나누지 않음": None, "학기 전 GPA 수준": "Pre_GPA_Group", "전통 학습시간 수준": "Study_Hours_Group"}
GROUP_OPTIONS.update({CAT_KR[c]: c for c in CAT_COLS})
GROUP_ORDER = {"Pre_GPA_Group": GPA_LABELS, "Study_Hours_Group": STUDY_LABELS, **CAT_ORDER}


def binned(d: pd.DataFrame, width: float, group: str | None) -> pd.DataFrame:
    """주간 AI 사용시간을 width 간격 구간으로 나눠 GPA 변화 통계를 계산합니다."""
    start_ = (np.floor(d["Weekly_GenAI_Hours"] / width) * width).clip(upper=40 - width)
    t = d.assign(h_mid=start_ + width / 2)
    keys = ["h_mid"] + ([group] if group else [])
    out = (
        t.groupby(keys, observed=True)["GPA_Change"]
        .agg(
            mean="mean",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            sd="std",
            n="size",
            up=lambda x: (x > 0).mean(),
        )
        .reset_index()
    )
    out["sem"] = (out["sd"] / np.sqrt(out["n"])).fillna(0)
    out["lo"] = out["mean"] - 1.96 * out["sem"]
    out["hi"] = out["mean"] + 1.96 * out["sem"]
    return out


with tabs["📈 AI 사용시간과 성적"]:
    st.subheader("자세히 보기 설정")
    o1, o2, o3 = st.columns(3)
    width = o1.select_slider("사용시간 구간 폭 (h)", options=[1, 2, 4, 5], value=2)
    min_n = o2.slider("구간별 최소 학생 수", 5, 200, 30, help="학생이 이보다 적은 구간은 그래프에서 뺍니다.")
    group_label = o3.selectbox("집단으로 나눠서 보기", list(GROUP_OPTIONS))
    group = GROUP_OPTIONS[group_label]

    st_all = binned(f, width, None)
    st_all = st_all[st_all["n"] >= min_n]
    st_grp = binned(f, width, group) if group else st_all
    st_grp = st_grp[st_grp["n"] >= min_n]
    color = (
        alt.Color(f"{group}:N", sort=GROUP_ORDER.get(group), title=group_label) if group else alt.value("#1f77b4")
    )
    xax = alt.X("h_mid:Q", title="주간 AI 사용시간 (h)", scale=alt.Scale(domain=[0, 40]))

    if not st_all.empty:
        best = st_all.loc[st_all["mean"].idxmax()]
        worst = st_all.loc[st_all["mean"].idxmin()]
        k1, k2, k3 = st.columns(3)
        k1.metric("평균 GPA 변화가 가장 큰 구간", f"{best['h_mid'] - width / 2:.0f}~{best['h_mid'] + width / 2:.0f}h",
                  f"{best['mean']:+.3f}")
        k2.metric("가장 작은 구간", f"{worst['h_mid'] - width / 2:.0f}~{worst['h_mid'] + width / 2:.0f}h",
                  f"{worst['mean']:+.3f}")
        k3.metric("두 구간의 차이", f"{best['mean'] - worst['mean']:.3f} GPA")

    # ---- 1) 평균 + 95% 신뢰구간
    st.subheader("① 구간별 평균 GPA 변화 (95% 신뢰구간)")
    band = alt.Chart(st_grp).mark_area(opacity=0.18).encode(
        x=xax, y=alt.Y("lo:Q", title="평균 GPA 변화"), y2="hi:Q", color=color
    )
    line = alt.Chart(st_grp).mark_line(point=True).encode(
        x=xax, y="mean:Q", color=color,
        tooltip=[alt.Tooltip("h_mid:Q", title="구간 중앙(h)"), alt.Tooltip("mean:Q", format=".3f", title="평균"),
                 alt.Tooltip("n:Q", title="학생 수")] + ([f"{group}:N"] if group else []),
    )
    zero_h = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((band + line + zero_h).properties(height=360), use_container_width=True)
    st.caption("띠가 좁을수록 평균이 믿을 만합니다. 띠가 겹치는 구간끼리는 차이가 우연일 수 있습니다.")

    # ---- 2) 중앙값 + 사분위 범위
    st.subheader("② 학생들의 GPA 변화는 얼마나 퍼져 있을까? (중앙값과 사분위 범위)")
    iqr = alt.Chart(st_grp).mark_area(opacity=0.18).encode(
        x=xax, y=alt.Y("q25:Q", title="GPA 변화"), y2="q75:Q", color=color
    )
    med = alt.Chart(st_grp).mark_line(point=True).encode(
        x=xax, y="median:Q", color=color,
        tooltip=[alt.Tooltip("h_mid:Q", title="구간 중앙(h)"), alt.Tooltip("median:Q", format=".3f", title="중앙값"),
                 alt.Tooltip("q25:Q", format=".3f", title="하위 25%"), alt.Tooltip("q75:Q", format=".3f", title="상위 25%")],
    )
    st.altair_chart((iqr + med + zero_h).properties(height=340), use_container_width=True)
    st.caption("선은 중앙값, 띠는 가운데 50% 학생의 범위입니다. 띠가 넓으면 같은 시간을 써도 결과 차이가 크다는 뜻입니다.")

    # ---- 3) 성적이 오른 학생 비율
    st.subheader("③ 성적이 오른 학생의 비율")
    upc = alt.Chart(st_grp).mark_line(point=True).encode(
        x=xax, y=alt.Y("up:Q", title="성적이 오른 학생 비율", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=color,
        tooltip=[alt.Tooltip("h_mid:Q", title="구간 중앙(h)"), alt.Tooltip("up:Q", format=".1%", title="오른 비율"),
                 alt.Tooltip("n:Q", title="학생 수")],
    )
    half = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((upc + half).properties(height=300), use_container_width=True)

    # ---- 4) 밀도 히트맵
    st.subheader("④ 학생들이 어디에 몰려 있나? (사용시간 × GPA 변화 밀도)")
    hx = pd.cut(f["Weekly_GenAI_Hours"], np.arange(0, 41, 2), include_lowest=True)
    gy = pd.cut(f["GPA_Change"].clip(-1, 1), np.linspace(-1, 1, 21), include_lowest=True)
    dens = f.groupby([hx, gy], observed=True).size().rename("학생 수").reset_index()
    dens["x0"] = dens["Weekly_GenAI_Hours"].apply(lambda i: i.left)
    dens["x1"] = dens["Weekly_GenAI_Hours"].apply(lambda i: i.right)
    dens["y0"] = dens["GPA_Change"].apply(lambda i: i.left)
    dens["y1"] = dens["GPA_Change"].apply(lambda i: i.right)
    heat = alt.Chart(dens).mark_rect().encode(
        x=alt.X("x0:Q", title="주간 AI 사용시간 (h)", scale=alt.Scale(domain=[0, 40])), x2="x1:Q",
        y=alt.Y("y0:Q", title="GPA 변화 (±1로 자름)"), y2="y1:Q",
        color=alt.Color("학생 수:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["x0", "x1", "y0", "y1", "학생 수"],
    ).properties(height=340)
    st.altair_chart(heat + zero_h, use_container_width=True)

    # ---- 5) 수치표
    st.subheader("⑤ 구간별 수치표")
    show = st_grp.copy()
    show.insert(0, "구간", show["h_mid"].map(lambda m: f"{m - width / 2:.0f}~{m + width / 2:.0f}h"))
    cols = ["구간"] + ([group] if group else []) + ["n", "mean", "median", "q25", "q75", "up", "lo", "hi"]
    show = show[cols].rename(columns={
        "n": "학생 수", "mean": "평균", "median": "중앙값", "q25": "하위25%", "q75": "상위25%",
        "up": "오른 비율", "lo": "95%CI 하한", "hi": "95%CI 상한"})
    st.dataframe(show.round(3), use_container_width=True, hide_index=True)
    st.download_button("표 CSV로 내려받기", show.round(4).to_csv(index=False).encode("utf-8-sig"),
                       file_name="gpa_change_by_ai_hours.csv", mime="text/csv")

    # ---- 6) 개별 학생 산점도
    st.subheader("⑥ 개별 학생 산점도 (무작위 3,000명)")
    pts = alt.Chart(sample(f)).mark_circle(opacity=0.3, size=25).encode(
        x=alt.X("Weekly_GenAI_Hours:Q", title="주간 AI 사용시간 (h)"),
        y=alt.Y("GPA_Change:Q", title="GPA 변화"),
        tooltip=["Weekly_GenAI_Hours", "Traditional_Study_Hours", "GPA_Change"],
    )
    st.altair_chart(pts.properties(height=360), use_container_width=True)
    st.caption(
        "⚠️ 학기 후 GPA는 4.0이 상한이라 학기 전 GPA가 높은 학생은 오를 여지가 작습니다(천장 효과). "
        "'집단으로 나눠서 보기'에서 학기 전 GPA 수준을 고르면 이 영향을 줄여서 볼 수 있습니다. "
        "이 그래프는 상관관계이지 AI가 성적을 바꿨다는 인과관계의 증거는 아닙니다."
    )

# ------------------------------------------------------------ 탭: 학기 전·후 GPA
DIR_ORDER = ["오름", "변화 없음", "내림"]
DIR_COLORS = ["#2e7d32", "#9e9e9e", "#c62828"]

with tabs["🎯 학기 전·후 GPA"]:
    st.subheader("학기 전 GPA 대비 학기 후 GPA")
    p1, p2, p3 = st.columns(3)
    n_pts = p1.slider("산점도에 그릴 학생 수", 1000, 10000, 3000, step=500, key="pp_npts")
    tol = p2.slider("'변화 없음'으로 볼 허용폭 (± GPA)", 0.0, 0.10, 0.0, step=0.01, key="pp_tol")
    color_by = p3.radio("점 색 기준", ["오름/내림", "주간 AI 사용시간대"], horizontal=True, key="pp_color")

    d = f.copy()
    d["Direction"] = np.select([d["GPA_Change"] > tol, d["GPA_Change"] < -tol], ["오름", "내림"], "변화 없음")
    cnt = d["Direction"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("오른 학생", f"{cnt.get('오름', 0):,}명", f"{cnt.get('오름', 0) / len(d) * 100:.1f}%")
    m2.metric("내린 학생", f"{cnt.get('내림', 0):,}명", f"{cnt.get('내림', 0) / len(d) * 100:.1f}%", delta_color="inverse")
    m3.metric("변화 없음", f"{cnt.get('변화 없음', 0):,}명")
    m4.metric("학기 후 GPA가 4.0(상한)인 학생", f"{(d['Post_Semester_GPA'] >= 3.999).mean() * 100:.1f}%")

    pts = sample(d, n_pts)
    if color_by == "오름/내림":
        col = alt.Color("Direction:N", scale=alt.Scale(domain=DIR_ORDER, range=DIR_COLORS), title="변화")
    else:
        col = alt.Color("AI_Hours_Group:N", sort=HOURS_LABELS, scale=alt.Scale(scheme="viridis"), title="주간 AI 사용시간")
    scale_xy = alt.Scale(domain=[1, 4.05])
    sc = alt.Chart(pts).mark_circle(size=22, opacity=0.4).encode(
        x=alt.X("Pre_Semester_GPA:Q", scale=scale_xy, title="학기 전 GPA"),
        y=alt.Y("Post_Semester_GPA:Q", scale=scale_xy, title="학기 후 GPA"),
        color=col,
        tooltip=[
            alt.Tooltip("Pre_Semester_GPA:Q", format=".2f", title="학기 전"),
            alt.Tooltip("Post_Semester_GPA:Q", format=".2f", title="학기 후"),
            alt.Tooltip("GPA_Change:Q", format="+.3f", title="변화"),
            alt.Tooltip("Weekly_GenAI_Hours:Q", format=".1f", title="AI 사용(h)"),
        ],
    )
    diag = alt.Chart(pd.DataFrame({"x": [1, 4], "y": [1, 4]})).mark_line(color="black", strokeDash=[6, 3]).encode(
        x="x:Q", y="y:Q"
    )
    ceil_ = alt.Chart(pd.DataFrame({"y": [4.0]})).mark_rule(color="orange", strokeDash=[2, 2]).encode(y="y:Q")
    st.altair_chart((sc + diag + ceil_).properties(height=500), use_container_width=True)
    st.caption(
        "검은 점선은 y = x입니다. 선 **위쪽** 점은 학기 후 GPA가 올랐고, **아래쪽** 점은 내렸습니다. "
        "주황색 점선은 4.0 상한입니다. 점이 이 선에 붙어 있는 학생은 더 오르고 싶어도 오를 수 없었습니다."
    )

    # ------------------------------------------------ 학기 전 GPA 구간별
    st.subheader("학기 전 GPA 구간별로 나눠 보기")
    st.caption("출발 학점이 높을수록 4.0까지 남은 여지가 작아서, 구간을 나눠 비교해야 공정합니다.")
    q1, q2 = st.columns(2)
    w = q1.select_slider("학기 전 GPA 구간 폭", options=[0.25, 0.5], value=0.5, key="pp_w")
    min_n2 = q2.slider("구간별 최소 학생 수", 5, 200, 30, key="pp_minn")

    edges = np.round(np.arange(1.0, 4.0 + 1e-9, w), 2)
    labels = [f"{a:.2f}~{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]
    d["Pre_Bin"] = pd.cut(d["Pre_Semester_GPA"], edges, labels=labels, right=False)

    def share(label):
        return lambda x: (x == label).mean()

    sb = (
        d.groupby("Pre_Bin", observed=True)
        .agg(
            n=("GPA_Change", "size"),
            mean=("GPA_Change", "mean"),
            sd=("GPA_Change", "std"),
            오름=("Direction", share("오름")),
            변화없음=("Direction", share("변화 없음")),
            내림=("Direction", share("내림")),
            at4=("Post_Semester_GPA", lambda x: (x >= 3.999).mean()),
            headroom=("Pre_Semester_GPA", lambda x: (4 - x).mean()),
        )
        .reset_index()
    )
    sb = sb[sb["n"] >= min_n2].copy()
    sb["sem"] = (sb["sd"] / np.sqrt(sb["n"])).fillna(0)
    sb["lo"] = sb["mean"] - 1.96 * sb["sem"]
    sb["hi"] = sb["mean"] + 1.96 * sb["sem"]
    sb["use"] = sb["mean"] / sb["headroom"]
    xb = alt.X("Pre_Bin:N", sort=labels, title="학기 전 GPA 구간")

    a, b = st.columns(2)
    with a:
        st.markdown("**① 구간별 평균 GPA 변화 (95% 신뢰구간)**")
        bar = alt.Chart(sb).mark_bar().encode(
            x=xb, y=alt.Y("mean:Q", title="평균 GPA 변화"),
            tooltip=["Pre_Bin", alt.Tooltip("mean:Q", format="+.3f"), "n"],
        )
        err = alt.Chart(sb).mark_rule(color="black").encode(x=xb, y="lo:Q", y2="hi:Q")
        zr = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((bar + err + zr).properties(height=320), use_container_width=True)
    with b:
        st.markdown("**② 구간별 오른/내린 학생 비율**")
        long = sb.melt(id_vars=["Pre_Bin", "n"], value_vars=["오름", "변화없음", "내림"],
                       var_name="변화", value_name="비율")
        long["변화"] = long["변화"].replace({"변화없음": "변화 없음"})
        stack = alt.Chart(long).mark_bar().encode(
            x=xb, y=alt.Y("비율:Q", axis=alt.Axis(format="%")),
            color=alt.Color("변화:N", scale=alt.Scale(domain=DIR_ORDER, range=DIR_COLORS)),
            tooltip=["Pre_Bin", "변화", alt.Tooltip("비율:Q", format=".1%"), "n"],
        )
        st.altair_chart(stack.properties(height=320), use_container_width=True)

    c, e = st.columns(2)
    with c:
        st.markdown("**③ 학기 후 GPA가 4.0(상한)인 학생 비율**")
        at4 = alt.Chart(sb).mark_bar(color="#ef8f00").encode(
            x=xb, y=alt.Y("at4:Q", title="비율", axis=alt.Axis(format="%")),
            tooltip=["Pre_Bin", alt.Tooltip("at4:Q", format=".1%"), "n"],
        )
        st.altair_chart(at4.properties(height=300), use_container_width=True)
    with e:
        st.markdown("**④ 오를 수 있는 여지 중 실제로 오른 몫**")
        us = alt.Chart(sb).mark_bar(color="#5c6bc0").encode(
            x=xb, y=alt.Y("use:Q", title="평균 변화 ÷ 평균 여지(4.0 − 학기 전)", axis=alt.Axis(format="%")),
            tooltip=["Pre_Bin", alt.Tooltip("use:Q", format=".1%"), alt.Tooltip("headroom:Q", format=".2f", title="평균 여지"), "n"],
        )
        st.altair_chart(us.properties(height=300), use_container_width=True)
    st.caption(
        "③은 상한에 막힌 학생이 어느 구간에 몰려 있는지, ④는 구간별 '여지'를 감안한 개선 정도입니다. "
        "반대로 학기 전 GPA가 아주 낮은 학생은 원래 평균 쪽으로 돌아오는 경향(평균으로의 회귀) 때문에 "
        "AI와 상관없이 오르기 쉽다는 점도 함께 고려해야 합니다."
    )

    # ------------------------------------------------ 학기 전 GPA × AI 사용시간대
    st.subheader("⑤ 학기 전 GPA 구간 × AI 사용시간대별 평균 GPA 변화")
    hm2 = (
        d.groupby(["Pre_Bin", "AI_Hours_Group"], observed=True)["GPA_Change"]
        .agg(["mean", "count"]).reset_index()
    )
    hm2 = hm2[hm2["count"] >= min_n2]
    base2 = alt.Chart(hm2).encode(
        x=alt.X("AI_Hours_Group:N", sort=HOURS_LABELS, title="주간 AI 사용 시간대"),
        y=alt.Y("Pre_Bin:N", sort=labels[::-1], title="학기 전 GPA 구간"),
    )
    rect2 = base2.mark_rect().encode(
        color=alt.Color("mean:Q", title="평균 GPA 변화", scale=alt.Scale(scheme="redyellowgreen", domainMid=0)),
        tooltip=["Pre_Bin", "AI_Hours_Group", alt.Tooltip("mean:Q", format="+.3f"), alt.Tooltip("count:Q", title="학생 수")],
    )
    txt2 = base2.mark_text(fontSize=12).encode(text=alt.Text("mean:Q", format="+.2f"))
    st.altair_chart((rect2 + txt2).properties(height=320), use_container_width=True)
    st.caption("같은 출발 학점끼리 비교하면 AI 사용시간에 따른 차이를 더 공정하게 볼 수 있습니다. 칸의 학생 수가 최소 기준보다 적으면 비워 둡니다.")

    # ------------------------------------------------ 수치표
    st.subheader("⑥ 구간별 수치표")
    tbl = sb[["Pre_Bin", "n", "mean", "lo", "hi", "오름", "변화없음", "내림", "at4", "headroom", "use"]].rename(columns={
        "Pre_Bin": "학기 전 GPA", "n": "학생 수", "mean": "평균 변화", "lo": "95%CI 하한", "hi": "95%CI 상한",
        "변화없음": "변화 없음", "at4": "후 GPA 4.0 비율", "headroom": "평균 여지", "use": "여지 대비 개선"})
    st.dataframe(tbl.round(3), use_container_width=True, hide_index=True)
    st.download_button("표 CSV로 내려받기", tbl.round(4).to_csv(index=False).encode("utf-8-sig"),
                       file_name="gpa_change_by_pre_gpa.csv", mime="text/csv", key="pp_dl")

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
