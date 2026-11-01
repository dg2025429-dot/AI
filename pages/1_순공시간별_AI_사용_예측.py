import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="순공시간별 AI 사용 예측", page_icon="🧮", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/dg2025429-dot/AI/refs/heads/main/"
    "ai_student_impact_dataset%20(1).csv"
)
CORE = ["Pre_Semester_GPA", "Weekly_GenAI_Hours", "Traditional_Study_Hours", "Post_Semester_GPA"]
GPA_MAX = 4.0
PT = 100 / GPA_MAX  # GPA 1.0 = 25점 (단순 비례 환산)

STUDY_BINS = [-np.inf, 5, 10, 15, np.inf]
STUDY_LABELS = ["5h 이하", "5~10h", "10~15h", "15h 이상"]


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    raw = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    raw.columns = raw.columns.str.replace("\ufeff", "", regex=False).str.strip()
    d = raw[CORE].dropna().rename(columns={
        "Pre_Semester_GPA": "pre", "Weekly_GenAI_Hours": "ai",
        "Traditional_Study_Hours": "study", "Post_Semester_GPA": "post"})
    d["delta"] = d["post"] - d["pre"]
    d["group"] = pd.cut(d["study"], STUDY_BINS, labels=STUDY_LABELS).astype(str)
    return d


# ------------------------------------------------------------------
# 모델: 2차 다항 회귀 + 릿지 (numpy만 사용 → 새 라이브러리 설치 필요 없음)
# ------------------------------------------------------------------
def design(X, mu, sd):
    Z = (X - mu) / sd
    cols = [np.ones(len(Z))] + [Z[:, i] for i in range(3)]
    for i in range(3):
        for j in range(i, 3):
            cols.append(Z[:, i] * Z[:, j])
    return np.column_stack(cols)


@st.cache_data(show_spinner="모델을 학습하는 중...")
def train(d: pd.DataFrame):
    X = d[["ai", "study", "pre"]].to_numpy(float)
    y = d["delta"].to_numpy(float)
    idx = np.random.default_rng(42).permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    A = design(X[tr], mu, sd)
    P = np.eye(A.shape[1]) * 1.0
    P[0, 0] = 0
    beta = np.linalg.solve(A.T @ A + P, A.T @ y[tr])
    pred = design(X[te], mu, sd) @ beta
    resid = y[te] - pred
    ss_res, ss_tot = (resid**2).sum(), ((y[te] - y[te].mean()) ** 2).sum()
    return {
        "beta": beta, "mu": mu, "sd": sd,
        "r2": 1 - ss_res / ss_tot,
        "mae": float(np.abs(resid).mean()),
        "base_mae": float(np.abs(y[te] - y[tr].mean()).mean()),
        "resid_sd": float(resid.std()),
        "n_tr": len(tr), "n_te": len(te),
    }


def predict(m, ai, study, pre):
    ai = np.atleast_1d(np.asarray(ai, float))
    X = np.column_stack([ai, np.full_like(ai, study), np.full_like(ai, pre)])
    return design(X, m["mu"], m["sd"]) @ m["beta"]


# ------------------------------------------------------------------
# 화면
# ------------------------------------------------------------------
st.title("🧮 순공시간별 AI 사용 예측")
st.caption("전통 학습시간(순공시간)과 현재 점수에 따라, 주당 AI 사용시간이 GPA 변화와 어떤 관계인지 예측해 보는 페이지")

st.warning(
    "⚠️ **이 사이트의 결과를 너무 믿지 마세요.**\n\n"
    "- 이 페이지는 **연습·참고용**이에요. 공부 계획이나 AI 사용 습관을 이 결과만 보고 정하지 마세요.\n"
    "- 데이터는 **대학생 GPA(4.0 만점)** 기준이에요. 초·중·고 학생에게 그대로 맞는지는 확인되지 않았어요.\n"
    "- 보이는 것은 **'같이 움직인다'는 관계**일 뿐, **AI를 써서 성적이 올랐다는 인과관계가 아니에요.**\n"
    "- 데이터의 출처와 수집 방식을 아직 검증하지 못했어요. 값이 상한(GPA 4.0, AI 40시간)에 몰려 있는 것은 "
    "실제 조사 데이터가 아닐 수도 있다는 신호예요.\n"
    "- 100점 환산 점수는 단순 비례 계산이라 **실제 학교 성적과 다를 수 있어요.**"
)

full = load_data()

with st.expander("데이터 설정 / 천장 효과(점수 상한) 확인"):
    n4 = int((full["post"] >= GPA_MAX - 0.001).sum())
    st.write(
        f"전체 **{len(full):,}명** 중 학기 후 GPA가 상한(4.0)인 학생이 **{n4:,}명 ({n4 / len(full):.1%})** 이에요. "
        "원래 점수가 높은 학생은 더 오를 자리가 없어서 상승폭이 실제보다 작게 보일 수 있어요."
    )
    reduce = st.checkbox("천장 효과 줄이기: 학기 전 GPA 3.5 미만 학생만 사용", value=False)
df = full[full["pre"] < 3.5] if reduce else full
st.caption(f"분석에 사용 중인 학생 수: {len(df):,}명")

info = train(df[["ai", "study", "pre", "delta"]])

st.subheader("내 조건 입력")
c1, c2, c3 = st.columns(3)
study_h = c1.slider("주당 전통 학습시간 (h)", float(df["study"].min()), float(df["study"].max()), 10.0, 0.5)
score_now = c2.number_input("현재 점수 (100점 만점)", 40, 100, 80, 1,
                            help="100점 만점 점수를 GPA(4.0)로 환산해서 모델에 넣어요.")
ai_h = c3.slider("주당 AI 사용시간 (h)", 0.0, 40.0, 5.0, 0.5)

pre_gpa = float(np.clip(score_now / PT, df["pre"].min(), GPA_MAX))
st.caption(f"환산: {score_now}점 → GPA {score_now / PT:.2f} (GPA 1.0 = 25점, 단순 비례)")

grid = np.arange(0, 40.01, 0.5)
curve = predict(info, grid, study_h, pre_gpa)
here = float(predict(info, ai_h, study_h, pre_gpa)[0])
gain = curve - curve[0]  # AI 0시간 대비
gain_here = here - float(curve[0])

pts = here * PT
score_pred = float(np.clip(score_now + pts, 0, 100))

m1, m2, m3, m4 = st.columns(4)
m1.metric("예상 GPA 변화", f"{here:+.3f}")
m2.metric("예상 점수 변화 (참고용)", f"{pts:+.1f}점")
m3.metric("예상 점수 (100점 만점)", f"{score_pred:.0f}점")
m4.metric("AI 0시간 대비", f"{gain_here * PT:+.1f}점")

noise = info["resid_sd"]
st.info(
    f"같은 조건의 학생들도 결과가 크게 흩어져요. 모델이 못 맞히는 개인차는 대략 "
    f"**±{noise:.2f} GPA (±{noise * PT:.0f}점)** 정도예요. 위 숫자는 '평균적인 경향'일 뿐이에요."
)

# ---- ① 내 조건의 곡선
st.subheader("① 내 조건에서의 AI 사용시간별 예상 점수 변화")
cdf = pd.DataFrame({"ai": grid, "점수 변화": curve * PT})
line = alt.Chart(cdf).mark_line(color="#1f77b4", strokeWidth=3).encode(
    x=alt.X("ai:Q", title="주당 AI 사용시간 (h)", scale=alt.Scale(domain=[0, 40])),
    y=alt.Y("점수 변화:Q", title="예상 점수 변화 (100점 환산, 점)"),
    tooltip=[alt.Tooltip("ai:Q", title="AI(h)"), alt.Tooltip("점수 변화:Q", format="+.2f")],
)
mark = alt.Chart(pd.DataFrame({"ai": [ai_h], "점수 변화": [pts]})).mark_point(
    color="crimson", size=140, filled=True).encode(x="ai:Q", y="점수 변화:Q")
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")
layers = line + mark + zero

if gain.max() > 0.02:  # 약 0.5점 이상 오르는 구간이 있을 때만
    near = grid[gain >= 0.9 * gain.max()]
    best_ai = float(grid[int(np.argmax(gain))])
    zone = alt.Chart(pd.DataFrame({"x0": [near.min()], "x1": [near.max()]})).mark_rect(
        color="green", opacity=0.15).encode(x="x0:Q", x2="x1:Q")
    layers = zone + layers
    st.success(
        f"이 조건(학습 {study_h:g}h, {score_now}점)에서 예측 효과가 가장 큰 AI 사용시간은 **주 {best_ai:g}시간 근처**, "
        f"최대 효과의 90% 이상인 구간은 **{near.min():g}~{near.max():g}시간**이에요 "
        f"(AI 0시간 대비 최대 {gain.max() * PT:+.1f}점)."
    )
else:
    st.success("이 조건에서는 AI 사용시간을 늘려도 뚜렷하게 오르는 경향이 모델에서 보이지 않아요.")
st.altair_chart(layers.properties(height=360), use_container_width=True)

# ---- ② 전통 학습시간 수준별 비교
st.subheader("② 전통 학습시간 수준별로 비교하기")
st.caption(f"현재 점수 {score_now}점 기준. 각 집단의 대표값(중앙값)을 학습시간으로 사용했고, AI 0시간일 때와의 차이로 그렸어요.")
rows, summ = [], []
for g in STUDY_LABELS:
    sub = df[df["group"] == g]
    if len(sub) < 30:
        continue
    s_med = float(sub["study"].median())
    cv = predict(info, grid, s_med, pre_gpa)
    gn = (cv - cv[0]) * PT
    rows.append(pd.DataFrame({"ai": grid, "점수 변화": gn, "전통 학습시간": g}))
    if gn.max() > 0.5:
        nr = grid[gn >= 0.9 * gn.max()]
        summ.append({"전통 학습시간": g, "대표 학습시간(h)": round(s_med, 1),
                     "효과가 가장 큰 AI(h)": float(grid[int(np.argmax(gn))]),
                     "효과 90% 이상 구간(h)": f"{nr.min():g}~{nr.max():g}",
                     "AI 0h 대비 최대 효과(점)": round(float(gn.max()), 1)})
    else:
        summ.append({"전통 학습시간": g, "대표 학습시간(h)": round(s_med, 1),
                     "효과가 가장 큰 AI(h)": None, "효과 90% 이상 구간(h)": "뚜렷한 효과 없음",
                     "AI 0h 대비 최대 효과(점)": round(float(gn.max()), 1)})
if rows:
    gdf = pd.concat(rows)
    ch = alt.Chart(gdf).mark_line(strokeWidth=2.5).encode(
        x=alt.X("ai:Q", title="주당 AI 사용시간 (h)", scale=alt.Scale(domain=[0, 40])),
        y=alt.Y("점수 변화:Q", title="AI 0시간 대비 예상 점수 변화 (점)"),
        color=alt.Color("전통 학습시간:N", sort=STUDY_LABELS),
        tooltip=["전통 학습시간", alt.Tooltip("ai:Q", title="AI(h)"), alt.Tooltip("점수 변화:Q", format="+.2f")],
    )
    st.altair_chart((ch + zero).properties(height=360), use_container_width=True)
    st.dataframe(pd.DataFrame(summ), hide_index=True, use_container_width=True)

# ---- ③ 모델 신뢰도
st.subheader("③ 이 모델은 얼마나 믿을 만할까?")
k1, k2, k3 = st.columns(3)
k1.metric("결정계수 R² (테스트)", f"{info['r2']:.3f}",
          help="1에 가까울수록 잘 맞고, 0에 가까우면 거의 못 맞힌다는 뜻이에요.")
k2.metric("평균 오차 MAE (GPA)", f"{info['mae']:.3f}")
k3.metric("그냥 평균으로 찍었을 때 MAE", f"{info['base_mae']:.3f}",
          help="모델 오차가 이 값보다 확실히 작아야 쓸모가 있어요.")
st.caption(
    f"학습 {info['n_tr']:,}명 / 테스트 {info['n_te']:,}명 (무작위 8:2). 모델은 학기 전 GPA·전통 학습시간·AI 사용시간으로 "
    "GPA 변화를 예측하는 2차 다항 회귀(릿지)예요. R²가 낮거나 오차가 '평균으로 찍기'와 비슷하면, "
    "개인의 성적은 거의 예측하지 못한다는 뜻이에요."
)

# ---- ④ 환산표
st.subheader("④ GPA ↔ 100점 만점 환산 (참고용)")
st.write("GPA(4.0 만점)에 **25를 곱해서** 100점 만점으로 바꿨어요. 그래서 GPA 변화 **+0.1 = +2.5점**이에요.")
conv = pd.DataFrame({"GPA (4.0 만점)": [4.0, 3.5, 3.0, 2.5, 2.0, 1.5]})
conv["100점 만점 환산"] = (conv["GPA (4.0 만점)"] * PT).round(1)
st.dataframe(conv, hide_index=True, use_container_width=True)
st.caption("실제 학교는 과목·학년·평가 방식(상대/절대평가)에 따라 점수 체계가 달라서, '대략 이 정도 크기'를 느끼는 참고용이에요.")

with st.expander("🔍 이 결과를 믿기 전에 꼭 확인할 것들"):
    st.markdown(
        """
1. **인과관계가 아니에요.** AI를 많이 쓰는 학생의 원래 특징(전공, 학년, 의욕 등)은 이 분석에서 빠져 있어요.
2. **천장 효과**: GPA 4.0 상한 때문에 점수가 높은 학생의 상승폭이 잘려 있어요.
3. **평균으로의 회귀**: 처음에 낮았던 학생은 저절로 오르기 쉬워요. 그래서 모델에 학기 전 점수를 넣었어요.
4. **사용시간이 길수록 학생 수가 적어요.** 곡선 오른쪽 끝은 우연의 영향이 커요.
5. **대학생 데이터**를 초·중·고에 적용한 것이고, 데이터 출처·수집 방식·이용 조건은 아직 확인되지 않았어요.
6. 이 페이지는 **교육용 실습**이에요. 학습 계획·입시 같은 중요한 결정에는 쓰지 마세요.
        """
    )
