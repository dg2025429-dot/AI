import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="순공시간별 AI 사용 예측", page_icon="🧮", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/dg2025429-dot/AI/"
    "102027983bf9103b236f53776cd6e2dfa34b73f7/%ED%95%99%EC%A0%90.csv"
)
LOCAL_PATH = "data/학점.csv"  # 인터넷이 안 될 때 쓸 예비 경로
CORE = ["Pre_Semester_GPA", "Weekly_GenAI_Hours", "Traditional_Study_Hours", "Post_Semester_GPA"]
CATS = ["Year_of_Study", "Primary_Use_Case", "Prompt_Engineering_Skill"]
GPA_MAX = 4.0  # 데이터의 GPA 만점

YEAR = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
USE = ["Debugging/Troubleshooting", "Copywriting/Drafting", "Ideation",
       "Summarizing_Reading", "Direct_Answer_Generation"]
SKILL = ["Beginner", "Intermediate", "Advanced"]
YEAR_KR = {"Freshman": "대학 1학년", "Sophomore": "대학 2학년", "Junior": "대학 3학년",
           "Senior": "대학 4학년", "Graduate": "대학원"}
USE_KR = {"Debugging/Troubleshooting": "코드·문제 해결(디버깅)", "Copywriting/Drafting": "글쓰기·초안 작성",
          "Ideation": "아이디어 얻기", "Summarizing_Reading": "요약·읽기 도우미",
          "Direct_Answer_Generation": "답 바로 받기"}
SKILL_KR = {"Beginner": "초급", "Intermediate": "중급", "Advanced": "고급"}
AVG = "잘 모르겠어요 (평균)"

STUDY_BINS = [-np.inf, 5, 10, 15, np.inf]
STUDY_LABELS = ["5h 이하", "5~10h", "10~15h", "15h 이상"]

# 점수 방식: 데이터(4.0 만점)를 단순 비례로 바꿔서 보여줍니다.
MODES = {
    "100점 만점 (초·중·고)": {"max": 100.0, "unit": "점"},
    "학점 4.0 만점": {"max": 4.0, "unit": "GPA"},
    "학점 4.5 만점": {"max": 4.5, "unit": "GPA"},
}


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    try:
        raw = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    except Exception:
        raw = pd.read_csv(LOCAL_PATH, encoding="utf-8-sig")
    raw.columns = raw.columns.str.replace("\ufeff", "", regex=False).str.strip()
    missing = [c for c in CORE + CATS if c not in raw.columns]
    if missing:
        raise ValueError(f"필수 열이 없어요: {missing}")
    d = raw[CORE + CATS].dropna().rename(columns={
        "Pre_Semester_GPA": "pre", "Weekly_GenAI_Hours": "ai",
        "Traditional_Study_Hours": "study", "Post_Semester_GPA": "post",
        "Year_of_Study": "year", "Primary_Use_Case": "use", "Prompt_Engineering_Skill": "skill"})
    d = d[d["year"].isin(YEAR) & d["use"].isin(USE) & d["skill"].isin(SKILL)].copy()
    d["delta"] = d["post"] - d["pre"]
    d["group"] = pd.cut(d["study"], STUDY_BINS, labels=STUDY_LABELS).astype(str)
    return d


# ------------------------------------------------------------------
# 모델: 2차 다항 회귀 + 릿지 (numpy만 사용 → 새 라이브러리 설치 필요 없음)
#  - 숫자: 학기 전 GPA, 전통 학습시간, AI 사용시간(2차항·곱 포함)
#  - 범주: 학년, AI 주 사용 목적, 프롬프트 숙련도 (+ AI 사용시간과의 상호작용)
# ------------------------------------------------------------------
def dummies(y, u, s):
    cols = [(y == c) for c in YEAR[1:]] + [(u == c) for c in USE[1:]] + [(s == c) for c in SKILL[1:]]
    return np.column_stack(cols).astype(float)


def design(X, D, mu, sd):
    Z = (X - mu) / sd
    cols = [np.ones(len(Z))] + [Z[:, i] for i in range(3)]
    for i in range(3):
        for j in range(i, 3):
            cols.append(Z[:, i] * Z[:, j])
    base = np.column_stack(cols)
    za = Z[:, [0]]  # AI 사용시간(표준화)
    return np.column_stack([base, D, D * za, D * za**2])


@st.cache_data(show_spinner="모델을 학습하는 중...")
def train(d: pd.DataFrame):
    X = d[["ai", "study", "pre"]].to_numpy(float)
    D = dummies(d["year"].to_numpy(), d["use"].to_numpy(), d["skill"].to_numpy())
    y = d["delta"].to_numpy(float)
    idx = np.random.default_rng(42).permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    A = design(X[tr], D[tr], mu, sd)
    P = np.eye(A.shape[1]) * 1.0
    P[0, 0] = 0
    beta = np.linalg.solve(A.T @ A + P, A.T @ y[tr])
    pred = design(X[te], D[te], mu, sd) @ beta
    resid = y[te] - pred
    y_tr, y_te = y[tr], y[te]
    ss_tot = ((y_te - y_te.mean()) ** 2).sum()

    def score(p):
        e = y_te - p
        return {"mae": float(np.abs(e).mean()), "r2": float(1 - (e**2).sum() / ss_tot)}

    # ---- 베이스라인: 같은 학습/테스트 분할에서 '쉬운 방법'의 점수를 재 둡니다.
    sg = pd.cut(d["study"], STUDY_BINS, labels=False).to_numpy()
    gmean = {g: float(y_tr[sg[tr] == g].mean()) for g in range(4) if (sg[tr] == g).any()}
    lin_A = np.column_stack([np.ones(len(X)), X])
    lin_b = np.linalg.lstsq(lin_A[tr], y_tr, rcond=None)[0]
    baselines = [
        ("전체 평균으로 찍기", score(np.full(len(te), y_tr.mean()))),
        ("전통 학습시간 4집단 평균", score(np.array([gmean.get(g, y_tr.mean()) for g in sg[te]]))),
        ("선형 회귀 (숫자 3개)", score(lin_A[te] @ lin_b)),
    ]
    share = lambda col, cats: [float((d[col] == c).mean()) for c in cats]  # noqa: E731
    return {
        "beta": beta, "mu": mu, "sd": sd,
        "r2": float(1 - (resid**2).sum() / ss_tot),
        "mae": float(np.abs(resid).mean()),
        "baselines": baselines,
        "resid_sd": float(resid.std()),
        "n_tr": len(tr), "n_te": len(te),
        "w_year": share("year", YEAR), "w_use": share("use", USE), "w_skill": share("skill", SKILL),
    }


def predict(m, ai, study, pre, year=None, use=None, skill=None):
    """year/use/skill이 None이면 데이터 속 비율로 평균을 냅니다."""
    ai = np.atleast_1d(np.asarray(ai, float))
    n = len(ai)
    X = np.column_stack([ai, np.full(n, study), np.full(n, pre)])

    def opts(val, cats, w):
        return [(val, 1.0)] if val is not None else list(zip(cats, w))

    out = np.zeros(n)
    for y, wy in opts(year, YEAR, m["w_year"]):
        for u, wu in opts(use, USE, m["w_use"]):
            for s, ws in opts(skill, SKILL, m["w_skill"]):
                D = dummies(np.full(n, y), np.full(n, u), np.full(n, s))
                out += wy * wu * ws * (design(X, D, m["mu"], m["sd"]) @ m["beta"])
    return out


# ------------------------------------------------------------------
# 화면
# ------------------------------------------------------------------
st.title("🧮 순공시간별 AI 사용 예측")
st.caption("전통 학습시간(순공시간)과 현재 성적, AI를 쓰는 목적에 따라 주당 AI 사용시간이 성적 변화와 어떤 관계인지 예측해 보는 페이지")

st.warning(
    "⚠️ **이 사이트의 결과를 너무 믿지 마세요.**\n\n"
    "- 이 페이지는 **연습·참고용**이에요. 공부 계획이나 AI 사용 습관을 이 결과만 보고 정하지 마세요.\n"
    "- 데이터는 **대학생 GPA(4.0 만점)** 기준이에요. 초·중·고 학생에게 그대로 맞는지는 확인되지 않았어요.\n"
    "- 보이는 것은 **'같이 움직인다'는 관계**일 뿐, **AI를 써서 성적이 올랐다는 인과관계가 아니에요.**\n"
    "- 데이터의 출처와 수집 방식을 아직 검증하지 못했어요. 값이 상한(GPA 4.0, AI 40시간)에 몰려 있는 것은 "
    "실제 조사 데이터가 아닐 수도 있다는 신호예요.\n"
    "- 100점·4.5 만점 환산은 단순 비례 계산이라 **실제 학교 성적과 다를 수 있어요.**"
)

try:
    full = load_data()
except Exception as e:  # noqa: BLE001
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

with st.expander("데이터 설정 / 천장 효과(점수 상한) 확인"):
    n4 = int((full["post"] >= GPA_MAX - 0.001).sum())
    st.write(
        f"전체 **{len(full):,}명** 중 학기 후 GPA가 상한(4.0)인 학생이 **{n4:,}명 ({n4 / len(full):.1%})** 이에요. "
        "원래 점수가 높은 학생은 더 오를 자리가 없어서 상승폭이 실제보다 작게 보일 수 있어요."
    )
    reduce = st.checkbox("천장 효과 줄이기: 학기 전 GPA 3.5 미만 학생만 사용", value=False)
df = full[full["pre"] < 3.5] if reduce else full
st.caption(f"분석에 사용 중인 학생 수: {len(df):,}명")

info = train(df[["ai", "study", "pre", "delta", "year", "use", "skill"]])

# ---- 점수 방식 선택 (100점 / 학점 4.0 / 학점 4.5)
st.subheader("내 조건 입력")
mode = st.radio("성적을 어떤 방식으로 볼까요?", list(MODES), horizontal=True)
SMAX, UNIT_NAME = MODES[mode]["max"], MODES[mode]["unit"]
UNIT = SMAX / GPA_MAX  # 데이터 GPA 1.0이 이 방식에서 몇 점/학점인지
IS100 = SMAX == 100.0


def fmt_val(v):  # 점수/학점 값
    return f"{v:.0f}점" if IS100 else f"{v:.2f}"


def fmt_chg(c):  # 변화량
    return f"{c:+.1f}점" if IS100 else f"{c:+.3f}"


def pick(label, cats, kr):
    choice = st.selectbox(label, [AVG] + [kr[c] for c in cats])
    if choice == AVG:
        return None
    return next(c for c in cats if kr[c] == choice)


c1, c2, c3 = st.columns(3)
study_h = c1.slider("주당 전통 학습시간 (h)", float(df["study"].min()), float(df["study"].max()), 10.0, 0.5)
if IS100:
    now = c2.number_input("현재 점수 (100점 만점)", 40, 100, 80, 1)
else:
    now = c2.number_input(f"현재 학점 ({SMAX:g} 만점)", 1.0, SMAX, round(3.0 * UNIT, 2), 0.05)
ai_h = c3.slider("주당 AI 사용시간 (h)", 0.0, 40.0, 5.0, 0.5)

d1, d2, d3 = st.columns(3)
with d1:
    if IS100:  # 초·중·고 방식: 대학 학년은 맞지 않아서 쓰지 않고 평균으로 계산
        year_sel = None
        st.caption("학년은 대학 기준이라, 100점(초·중·고) 방식에서는 쓰지 않고 평균으로 계산해요.")
    else:
        year_sel = pick("학년 (대학)", YEAR, YEAR_KR)
with d2:
    use_sel = pick("AI를 주로 쓰는 목적", USE, USE_KR)
with d3:
    skill_sel = pick("AI에게 질문하는(프롬프트) 숙련도", SKILL, SKILL_KR)

gpa4_raw = now / UNIT
pre_gpa = float(np.clip(gpa4_raw, df["pre"].min(), GPA_MAX))
st.caption(
    f"모델은 4.0 만점 GPA로 계산해요: {fmt_val(now)} → GPA {gpa4_raw:.2f} (4.0 기준, 단순 비례 ×{1 / UNIT:.3f})"
    + (" · 데이터에 없는 낮은 점수라서 가장 가까운 값으로 계산했어요." if abs(pre_gpa - gpa4_raw) > 1e-9 else "")
    + " · '평균'을 고르면 데이터 속 비율대로 평균을 내요."
)

grid = np.arange(0, 40.01, 0.5)
curve = predict(info, grid, study_h, pre_gpa, year_sel, use_sel, skill_sel)
here = float(predict(info, ai_h, study_h, pre_gpa, year_sel, use_sel, skill_sel)[0])
gain = curve - curve[0]  # AI 0시간 대비 (GPA 4.0 기준)
gain_here = here - float(curve[0])

chg = here * UNIT
pred_val = float(np.clip(now + chg, 0, SMAX))

m1, m2, m3, m4 = st.columns(4)
m1.metric("예상 변화", fmt_chg(chg))
m2.metric("예상 점수" if IS100 else "예상 학점", fmt_val(pred_val))
m3.metric("AI 0시간 대비", fmt_chg(gain_here * UNIT))
if IS100:
    m4.metric("예상 GPA 변화 (4.0 기준)", f"{here:+.3f}")
else:
    m4.metric("100점 환산 변화 (참고)", f"{here * 25:+.1f}점")

noise = info["resid_sd"]
noise_txt = f"±{noise * UNIT:.0f}점" if IS100 else f"±{noise * UNIT:.2f}"
st.info(
    f"같은 조건의 학생들도 결과가 크게 흩어져요. 모델이 못 맞히는 개인차는 대략 **{noise_txt}** 정도예요. "
    "위 숫자는 '평균적인 경향'일 뿐이에요."
)

zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")


def best_zone(gn):
    """AI 0시간 대비 효과(gn, 표시 단위)에서 최대 효과의 90% 이상인 구간."""
    if gn.max() <= 0.02 * UNIT:
        return None
    near = grid[gn >= 0.9 * gn.max()]
    return float(grid[int(np.argmax(gn))]), float(near.min()), float(near.max())


# ---- ① 내 조건의 곡선
st.subheader(f"① 내 조건에서의 AI 사용시간별 예상 {'점수' if IS100 else '학점'} 변화")
y_title = f"예상 {'점수' if IS100 else '학점'} 변화 ({UNIT_NAME})"
cdf = pd.DataFrame({"ai": grid, "변화": curve * UNIT})
line = alt.Chart(cdf).mark_line(color="#1f77b4", strokeWidth=3).encode(
    x=alt.X("ai:Q", title="주당 AI 사용시간 (h)", scale=alt.Scale(domain=[0, 40])),
    y=alt.Y("변화:Q", title=y_title),
    tooltip=[alt.Tooltip("ai:Q", title="AI(h)"), alt.Tooltip("변화:Q", format="+.3f")],
)
mark = alt.Chart(pd.DataFrame({"ai": [ai_h], "변화": [chg]})).mark_point(
    color="crimson", size=140, filled=True).encode(x="ai:Q", y="변화:Q")
layers = line + mark + zero
bz = best_zone(gain * UNIT)
if bz:
    best_ai, lo_h, hi_h = bz
    zone = alt.Chart(pd.DataFrame({"x0": [lo_h], "x1": [hi_h]})).mark_rect(
        color="green", opacity=0.15).encode(x="x0:Q", x2="x1:Q")
    layers = zone + layers
    st.success(
        f"이 조건(학습 {study_h:g}h, 현재 {fmt_val(now)})에서 예측 효과가 가장 큰 AI 사용시간은 **주 {best_ai:g}시간 근처**, "
        f"최대 효과의 90% 이상인 구간은 **{lo_h:g}~{hi_h:g}시간**이에요 "
        f"(AI 0시간 대비 최대 {fmt_chg(gain.max() * UNIT)})."
    )
else:
    st.success("이 조건에서는 AI 사용시간을 늘려도 뚜렷하게 오르는 경향이 모델에서 보이지 않아요.")
st.altair_chart(layers.properties(height=360), use_container_width=True)


def compare(pairs, fn, key_col):
    """pairs: [(표시이름, 키)], fn(키) → GPA 4.0 기준 예측 곡선. 그래프와 요약표를 그려요."""
    rows, summ = [], []
    eff_col = f"AI 0h 대비 최대 효과({UNIT_NAME})"
    for label, key in pairs:
        cv = fn(key)
        gn = (cv - cv[0]) * UNIT
        rows.append(pd.DataFrame({"ai": grid, "변화": gn, key_col: label}))
        z = best_zone(gn)
        summ.append({
            key_col: label,
            "효과가 가장 큰 AI(h)": z[0] if z else None,
            "효과 90% 이상 구간(h)": f"{z[1]:g}~{z[2]:g}" if z else "뚜렷한 효과 없음",
            eff_col: round(float(gn.max()), 3),
            "AI 40h일 때 변화": round(float(gn[-1]), 3),
        })
    gdf = pd.concat(rows)
    ch = alt.Chart(gdf).mark_line(strokeWidth=2.5).encode(
        x=alt.X("ai:Q", title="주당 AI 사용시간 (h)", scale=alt.Scale(domain=[0, 40])),
        y=alt.Y("변화:Q", title=f"AI 0시간 대비 예상 변화 ({UNIT_NAME})"),
        color=alt.Color(f"{key_col}:N", sort=[p[0] for p in pairs]),
        tooltip=[key_col, alt.Tooltip("ai:Q", title="AI(h)"), alt.Tooltip("변화:Q", format="+.3f")],
    )
    st.altair_chart((ch + zero).properties(height=360), use_container_width=True)
    st.dataframe(pd.DataFrame(summ), hide_index=True, use_container_width=True)


# ---- ② 전통 학습시간 수준별 비교
st.subheader("② 전통 학습시간 수준별로 비교하기")
st.caption(f"현재 {fmt_val(now)} 기준이고, 위에서 고른 학년·목적·숙련도를 그대로 적용했어요. "
           "각 집단의 대표값(중앙값)을 학습시간으로 썼고, AI 0시간일 때와의 차이로 그렸어요.")
med = {g: float(df.loc[df["group"] == g, "study"].median()) for g in STUDY_LABELS if (df["group"] == g).sum() >= 30}
compare([(f"{g} (대표 {med[g]:.1f}h)", g) for g in med],
        lambda g: predict(info, grid, med[g], pre_gpa, year_sel, use_sel, skill_sel), "전통 학습시간")

# ---- ③ AI 사용 목적별 비교
st.subheader("③ AI를 쓰는 목적별로 비교하기")
st.caption(f"학습시간 {study_h:g}h, 현재 {fmt_val(now)} 기준이에요. 같은 시간을 써도 목적에 따라 곡선이 얼마나 달라지는지 보세요. "
           "(위에서 고른 '목적'은 이 비교에서는 무시돼요.)")
compare([(USE_KR[u], u) for u in USE],
        lambda u: predict(info, grid, study_h, pre_gpa, year_sel, u, skill_sel), "AI 사용 목적")

# ---- ④ 모델 신뢰도 (베이스라인과 비교)
st.subheader("④ 이 모델은 얼마나 믿을 만할까? (베이스라인과 비교)")
st.write(
    "**베이스라인**은 '복잡한 모델 없이 쉬운 방법만 써도 이만큼은 나온다'는 기준 점수예요. "
    "우리 모델이 이 기준보다 확실히 좋아야 모델을 만든 보람이 있어요. 모두 같은 학습/테스트 학생으로 쟀어요."
)
model_row = ("우리 모델: 다항+릿지 (학년·목적·숙련도 포함)", {"mae": info["mae"], "r2": info["r2"]})
rows_cmp = [(f"베이스라인 {i + 1}: {n}", sc) for i, (n, sc) in enumerate(info["baselines"])] + [model_row]
mean_mae = info["baselines"][0][1]["mae"]
best_base = min(info["baselines"], key=lambda t: t[1]["mae"])
cmp_df = pd.DataFrame({
    "방법": [n for n, _ in rows_cmp],
    f"평균 오차 MAE ({UNIT_NAME})": [round(sc["mae"] * UNIT, 3) for _, sc in rows_cmp],
    "R² (설명력)": [round(sc["r2"], 3) for _, sc in rows_cmp],
    "'전체 평균 찍기' 대비 오차 감소": [f"{(mean_mae - sc['mae']) / mean_mae:.1%}" for _, sc in rows_cmp],
})
st.dataframe(cmp_df, hide_index=True, use_container_width=True)

bar = alt.Chart(cmp_df.assign(우리모델=[n.startswith("우리") for n in cmp_df["방법"]])).mark_bar().encode(
    y=alt.Y("방법:N", sort=None, title=None, axis=alt.Axis(labelLimit=380)),
    x=alt.X(f"평균 오차 MAE ({UNIT_NAME}):Q", title=f"평균 오차 (작을수록 좋아요, {UNIT_NAME})"),
    color=alt.Color("우리모델:N", legend=None, scale=alt.Scale(domain=[False, True], range=["#b0b7c3", "#1f77b4"])),
    tooltip=["방법", f"평균 오차 MAE ({UNIT_NAME}):Q", "R² (설명력):Q"],
)
st.altair_chart(bar.properties(height=200), use_container_width=True)

gain_vs_mean = (mean_mae - info["mae"]) / mean_mae
gain_vs_best = (best_base[1]["mae"] - info["mae"]) / best_base[1]["mae"]
b1, b2, b3 = st.columns(3)
b1.metric("우리 모델 R² (테스트)", f"{info['r2']:.3f}",
          help="성적 변화의 흩어짐 중 몇 %를 설명했는지예요. 1에 가까울수록 좋아요.")
b2.metric("'전체 평균 찍기'보다 오차 감소", f"{gain_vs_mean:.1%}")
b3.metric(f"가장 좋은 베이스라인({best_base[0]})보다 오차 감소", f"{gain_vs_best:.1%}")
if gain_vs_best < 0.05:
    st.error("우리 모델이 가장 좋은 베이스라인과 거의 차이가 없어요. 복잡한 모델을 쓸 이유가 약하다는 뜻이에요.")
elif info["r2"] < 0.5:
    st.warning(
        f"베이스라인보다는 확실히 좋아졌지만(오차 {gain_vs_best:.0%} 감소), R²가 {info['r2']:.2f}라서 "
        "아직 개인의 성적을 정확히 맞히는 수준은 아니에요. 평균적인 경향을 보는 용도로만 쓰세요."
    )
else:
    st.success("베이스라인보다 확실히 좋아졌어요.")

share_up = float((df["delta"] > 0).mean())
with st.expander("🧭 베이스라인이 왜 필요할까요?"):
    st.markdown(
        f"""
1. **점수 하나만으로는 좋은지 나쁜지 알 수 없어요.** R²가 0.35라고 해도 기준이 없으면 높은 건지 낮은 건지 몰라요. 쉬운 방법의 점수를 같이 놓으면 비로소 의미가 생겨요.
2. **'아무것도 안 한 것'보다 나아졌는지 확인해요.** 가장 단순한 베이스라인인 '전체 평균으로 찍기'는 R²가 0이에요. 모델이 여기서 얼마나 멀어졌는지가 곧 모델이 배운 양이에요.
3. **복잡한 모델이 정말 필요한지 알려줘요.** 학습시간 집단 평균이나 직선 같은 단순한 방법과 거의 비슷하다면, 복잡한 모델은 설명만 어려워지고 이득이 없어요.
4. **이 데이터는 한쪽으로 치우쳐 있어서 더 중요해요.** 학생의 **{share_up:.1%}** 가 성적이 올랐어요. 그래서 '오른다'고만 찍어도 꽤 맞아 보여요. 기준선이 없으면 모델이 대단해 보이는 착각을 하기 쉬워요.
5. **단계를 밟아 올라가는 모습이 보여요.** 평균 찍기 → 학습시간별 평균 → 직선 → 곡선 모델 순서로 오차가 줄어드는 걸 보여주면, 어디서 얼마나 좋아졌는지 설명할 수 있어요.

**주의:** 베이스라인과 비교해도 '이 점수가 실제 학생에게도 나온다'는 보장은 없어요. 이 데이터 안에서의 비교일 뿐이에요.
        """
    )
st.caption(
    f"학습 {info['n_tr']:,}명 / 테스트 {info['n_te']:,}명 (무작위 8:2). 모델은 학기 전 GPA·전통 학습시간·AI 사용시간·학년·목적·숙련도로 "
    "GPA 변화를 예측하는 2차 다항 회귀(릿지)예요. 베이스라인 ②는 전통 학습시간을 5·10·15시간 기준 4집단으로 나눠 집단 평균을 쓰는 방법, "
    "③은 숫자 3개(학기 전 GPA·학습시간·AI 시간)만 쓰는 직선식이에요."
)

# ---- ⑤ 환산표
st.subheader("⑤ 4.0 학점 ↔ 4.5 학점 ↔ 100점 환산 (참고용)")
st.write("데이터의 GPA(4.0 만점)에 **비례**해서 바꿨어요. 4.5 만점은 ×1.125, 100점 만점은 ×25예요.")
conv = pd.DataFrame({"학점 (4.0 만점)": [4.0, 3.5, 3.0, 2.5, 2.0, 1.5]})
conv["학점 (4.5 만점)"] = (conv["학점 (4.0 만점)"] * 4.5 / 4.0).round(2)
conv["100점 만점"] = (conv["학점 (4.0 만점)"] * 25).round(1)
st.dataframe(conv, hide_index=True, use_container_width=True)
st.caption("실제 학교는 학교·과목·평가 방식(상대/절대평가)에 따라 점수 체계가 달라서, '대략 이 정도 크기'를 느끼는 참고용이에요.")

with st.expander("🔍 이 결과를 믿기 전에 꼭 확인할 것들"):
    st.markdown(
        """
1. **인과관계가 아니에요.** AI를 많이 쓰는 학생의 원래 특징(전공, 의욕 등)은 이 분석에서 빠져 있어요.
2. **천장 효과**: GPA 4.0 상한 때문에 점수가 높은 학생의 상승폭이 잘려 있어요.
3. **평균으로의 회귀**: 처음에 낮았던 학생은 저절로 오르기 쉬워요. 그래서 모델에 학기 전 점수를 넣었어요.
4. **사용시간이 길수록 학생 수가 적어요.** 곡선 오른쪽 끝은 우연의 영향이 커요.
5. **대학생 데이터**를 초·중·고에 적용한 것이에요. 그래서 100점(초·중·고) 방식에서는 학년을 쓰지 않아요. 목적·숙련도도 대학생이 답한 값이에요. 데이터 출처·수집 방식·이용 조건은 아직 확인되지 않았어요.
6. 이 페이지는 **교육용 실습**이에요. 학습 계획·입시 같은 중요한 결정에는 쓰지 마세요.
        """
    )
