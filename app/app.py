import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import joblib, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Academic Stress | IIT Students", layout="wide",
                   initial_sidebar_state="expanded")

P1,P2,P3 = "#2D3A8C","#4F6FD4","#8FA8E8"
ACC,BG,WHITE,TEXT,MUTED,GRID = "#E8522A","#F4F6FB","#FFFFFF","#1C1F3B","#6B7280","#F0F1F5"
LOW_C,MED_C,HIGH_C = "#34D399","#FBBF24","#F87171"
SMAP = {'Low':LOW_C,'Medium':MED_C,'High':HIGH_C}

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
*,[class*="css"]{{font-family:'Inter',sans-serif!important}}
.stApp{{background:{BG}}}
section[data-testid="stSidebar"]{{background:linear-gradient(160deg,{P1} 0%,#1a2260 100%)}}
section[data-testid="stSidebar"] *{{color:white!important}}
section[data-testid="stSidebar"] .stRadio>label{{display:none}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{{
  display:flex;padding:9px 12px;border-radius:8px;font-size:.88rem;font-weight:500;
  margin-bottom:3px;cursor:pointer;color:rgba(255,255,255,.7)!important;transition:all .2s}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{{
  background:rgba(255,255,255,.1);color:white!important}}
.hero{{background:linear-gradient(135deg,{P1} 0%,{P2} 65%,#7BA7F5 100%);
  border-radius:18px;padding:2.5rem 3rem;color:white;margin-bottom:1.5rem}}
.hero-tag{{display:inline-block;background:rgba(255,255,255,.15);border-radius:99px;
  padding:3px 12px;font-size:.72rem;font-weight:600;letter-spacing:.07em;
  text-transform:uppercase;margin-bottom:1rem}}
.hero-title{{font-size:2rem;font-weight:800;line-height:1.2;margin:0 0 .6rem}}
.hero-sub{{font-size:.88rem;opacity:.82;max-width:500px;line-height:1.65;margin:0}}
.kpi{{background:{WHITE};border-radius:12px;padding:1.1rem 1.3rem;
  box-shadow:0 1px 6px rgba(45,58,140,.08);border-top:3px solid {P2}}}
.kpi-v{{font-size:1.75rem;font-weight:800;color:{P1};margin:0;line-height:1}}
.kpi-l{{font-size:.7rem;color:{MUTED};text-transform:uppercase;
  letter-spacing:.07em;margin:5px 0 0;font-weight:500}}
.sec{{font-size:.9rem;font-weight:700;color:{TEXT};margin:1.2rem 0 .4rem}}
.divider{{border:none;border-top:1px solid #E5E7EB;margin:1.1rem 0}}
.insight{{background:#EEF2FF;border-left:4px solid {P2};border-radius:0 8px 8px 0;
  padding:.75rem 1rem;font-size:.82rem;color:{TEXT};line-height:1.6;margin-top:.8rem}}
.iblock{{background:{WHITE};border-radius:10px;padding:.9rem 1rem .5rem;
  margin-bottom:.7rem;box-shadow:0 1px 4px rgba(45,58,140,.06)}}
.ilabel{{font-size:.85rem;font-weight:600;color:{TEXT};margin-bottom:.1rem}}
.ihint{{font-size:.74rem;color:{MUTED};margin-bottom:.4rem;line-height:1.5}}
.res-box{{border-radius:12px;padding:1.6rem 1rem;text-align:center;
  font-size:1.3rem;font-weight:800;margin-bottom:.8rem}}
.res-high{{background:#FEE2E2;color:#991B1B;border:1.5px solid {HIGH_C}}}
.res-medium{{background:#FEF3C7;color:#92400E;border:1.5px solid {MED_C}}}
.res-low{{background:#D1FAE5;color:#065F46;border:1.5px solid {LOW_C}}}
.stButton>button{{background:{P1};color:white;border:none;border-radius:8px;
  padding:.6rem 1.5rem;font-weight:600;font-size:.88rem;width:100%;transition:background .2s}}
.stButton>button:hover{{background:{P2}}}
footer,header,#MainMenu{{visibility:hidden}}
</style>""", unsafe_allow_html=True)

# ── Data & Model ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, 'cleaned_stress_survey.csv'))
    df['Sleep_Hours'] = df['Sleep_Hours'].replace('Option 4','Less than 4 hours')
    df['Stress_Level'] = df['Overall_Stress'].apply(
        lambda x: 'Low' if x<=3 else('Medium' if x<=6 else 'High'))
    df['GAD_Score'] = (df['GAD_Nervous']+df['GAD_Irritable'])/2
    df['Counsellor_Group'] = df['Counsellor_Need'].map({
        'No':'No need felt',
        'Yes, but I did not seek help':"Needed, didn't seek",
        'Yes, and I did seek help':'Sought help'})
    return df

@st.cache_resource
def load_model():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(base, 'best_model.pkl'))

df  = load_data()
mdl = load_model()

# ── Chart helper ──────────────────────────────────────────────────
# NOTE: never pass showlegend inside extra={} — it conflicts with legend param
def chart(fig, h=340, legend=False, xtitle='', ytitle='', angle=0, mb=50, extra=None):
    layout = dict(
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family='Inter',color=TEXT,size=12),
        margin=dict(t=40,b=mb,l=10,r=10),
        height=h, showlegend=legend,
        xaxis_title=xtitle, yaxis_title=ytitle)
    if extra:
        layout.update(extra)
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True,
                     tickangle=angle, tickfont=dict(color=TEXT,size=11))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(color=TEXT,size=11))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

def short(series, long_vals, short_vals):
    m = dict(zip(long_vals, short_vals))
    return series.map(lambda x: m.get(x, x))

# ── Category orders ───────────────────────────────────────────────
SLP   = ['Less than 4 hours','4 - 6 hours','6 - 8 hours','More than 8 hours']
SLP_S = ['< 4 hrs','4–6 hrs','6–8 hrs','> 8 hrs']
SCR   = ['Less than 1 hour','1 - 3 hours','3 - 5 hours','More than 5 hours']
SCR_S = ['< 1 hr','1–3 hrs','3–5 hrs','> 5 hrs']
EXE   = ['Never','1 - 2 times a week','3 - 4 times a week','Daily']
EXE_S = ['Never','1–2×/wk','3–4×/wk','Daily']
DIT   = ['Very poor (mostly junk/irregular)','Poor','Average','Good','Very good (balanced meals)']
DIT_S = ['Very Poor','Poor','Average','Good','Very Good']
STU   = ['Less than 2 hours','2 - 4 hours','4 - 6 hours','More than 6 hours']
STU_S = ['< 2 hrs','2–4 hrs','4–6 hrs','> 6 hrs']
CGA   = ['Below 6.0','6.0 - 7.0','7.0 - 8.0','8.0 - 9.0','Above 9.0']
GRP   = ['No need felt',"Needed, didn't seek",'Sought help']
GCOL  = {'No need felt':LOW_C,"Needed, didn't seek":HIGH_C,'Sought help':MED_C}
GAD_LBL = {1:"Not at all",2:"Several days",3:"More than half the days",4:"Nearly every day"}

INPUTS = [
    ("sleep",   "🛌 Sleep per Night",        "Hours you sleep each night",                        SLP, '6 - 8 hours'),
    ("screen",  "📱 Screen Time (non-acad.)","Social media / entertainment — not study-related",  SCR, '1 - 3 hours'),
    ("exercise","🏃 Exercise Frequency",     "Times per week you exercise or play sport",          EXE, '1 - 2 times a week'),
    ("diet",    "🥗 Diet Quality",           "Very Poor=junk/irregular · Very Good=balanced meals",DIT, 'Average'),
    ("study",   "📚 Self-study Hours/Day",   "Hours studying outside scheduled classes",           STU, '2 - 4 hours'),
    ("cgpa",    "🎓 CGPA Range",             "Cumulative GPA bracket on 10-point scale",          CGA, '7.0 - 8.0'),
]

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='padding:.5rem .5rem 1.2rem'>
        <p style='font-size:1rem;font-weight:700;margin:0'>Stress Analytics</p>
        <p style='font-size:.72rem;opacity:.55;margin:3px 0 0'>IIT Student Survey</p></div>
        <hr style='border-color:rgba(255,255,255,.15);margin:0 0 .8rem'>""",
        unsafe_allow_html=True)
    page = st.radio("nav",["Overview","EDA","Model Results","Stress Predictor"],
                    label_visibility="collapsed")
    st.markdown("""<hr style='border-color:rgba(255,255,255,.15);margin:1rem 0'>
        <p style='font-size:.7rem;opacity:.45;line-height:1.8'>
        102 respondents · Primary survey<br>Python · Sklearn · Streamlit</p>""",
        unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "Overview":
    high_pct = round((df['Stress_Level']=='High').sum()/len(df)*100,1)
    mean_s   = round(df['Overall_Stress'].mean(),1)

    st.markdown(f"""<div class="hero">
        <span class="hero-tag">Data Analytics · Academic Research</span>
        <p class="hero-title">Academic Stress Among IIT Students</p>
        <p class="hero-sub">A primary survey of 102 IIT students examining how lifestyle habits,
        GAD anxiety indicators, and academic factors relate to and predict academic stress.</p>
    </div>""", unsafe_allow_html=True)

    for col,(v,l) in zip(st.columns(4),[
        ("102","Respondents"),( f"{mean_s}/10","Mean Stress Score"),
        (f"{high_pct}%","High Stress"),("71.4%","Best Model Accuracy")]):
        with col:
            st.markdown(f'<div class="kpi"><p class="kpi-v">{v}</p><p class="kpi-l">{l}</p></div>',
                        unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec">Stress Level Distribution</p>', unsafe_allow_html=True)
        sc = df['Stress_Level'].value_counts().reindex(['Low','Medium','High']).reset_index()
        sc.columns=['Level','Count']
        fig = px.bar(sc,x='Level',y='Count',text='Count',color='Level',color_discrete_map=SMAP)
        fig.update_traces(textposition='outside',marker_line_width=0,width=0.5)
        chart(fig, ytitle='Students', xtitle='Stress Level')

    with c2:
        st.markdown('<p class="sec">Stress Score Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(df,x='Overall_Stress',nbins=10,color_discrete_sequence=[P2])
        fig.add_vline(x=mean_s,line_dash='dash',line_color=ACC,
                      annotation_text=f'Mean {mean_s}',annotation_position='top right',
                      annotation_font=dict(color=ACC,size=12))
        fig.update_traces(marker_line_width=0)
        chart(fig, extra=dict(bargap=0.12), xtitle='Stress Score (1–10)', ytitle='Count')

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec">Respondents by Year</p>', unsafe_allow_html=True)
        yr = df['Year'].value_counts().reindex(['1st Yr','2nd Yr','3rd Yr','4th Yr']).reset_index()
        yr.columns=['Year','Count']
        fig = px.funnel(yr,x='Count',y='Year',color_discrete_sequence=[P1,P2,P3,'#B8CCF4'])
        fig.update_layout(plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=280,showlegend=False,
                          font=dict(family='Inter',color=TEXT,size=12),
                          margin=dict(t=30,b=20,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

    with c2:
        st.markdown('<p class="sec">Branch Distribution</p>', unsafe_allow_html=True)
        br = df['Branch'].value_counts().reset_index(); br.columns=['Branch','Count']
        fig = px.pie(br,values='Count',names='Branch',hole=0.48,
                     color_discrete_sequence=[P1,P2,P3,'#6B8EE8','#A8BEF0','#34D399','#FBBF24'])
        fig.update_traces(textposition='outside',textinfo='label+percent',
                          textfont=dict(color=TEXT,size=10))
        fig.update_layout(plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=300,showlegend=False,
                          font=dict(family='Inter',color=TEXT,size=11),
                          margin=dict(t=30,b=30,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

    st.markdown('<div class="insight">41% of IIT students report High stress (7–10). Exams, '
                'assignments and career anxiety are the top factors. 74 of 102 respondents '
                'are 2nd-year students.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# EDA — Q1–Q10 matching eda.ipynb
# ════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.markdown('<p style="font-size:1.5rem;font-weight:800;color:#1C1F3B;margin-bottom:.2rem">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.84rem;color:#6B7280;margin-bottom:1rem">Use filters to explore how lifestyle and academic factors relate to stress.</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:.5rem 0'>",
                    unsafe_allow_html=True)
        year_f   = st.multiselect("Year",  df['Year'].unique().tolist(),  df['Year'].unique().tolist())
        branch_f = st.multiselect("Branch",df['Branch'].unique().tolist(),df['Branch'].unique().tolist())

    fdf = df[df['Year'].isin(year_f)&df['Branch'].isin(branch_f)]
    st.caption(f"{len(fdf)} of {len(df)} respondents selected")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q1
    st.markdown('<p class="sec">Q1 — Gender and Branch profile</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        gc = fdf['Gender'].value_counts().reset_index(); gc.columns=['Gender','Count']
        fig = px.bar(gc,x='Gender',y='Count',text='Count',color='Gender',
                     color_discrete_sequence=[P1,P2,P3])
        fig.update_traces(textposition='outside',marker_line_width=0,width=0.45)
        chart(fig, ytitle='Students', xtitle='Gender')
    with c2:
        bc = fdf['Branch'].value_counts().reset_index(); bc.columns=['Branch','Count']
        fig = px.bar(bc,x='Branch',y='Count',text='Count',color='Branch',
                     color_discrete_sequence=[P1,P2,P3,'#6B8EE8','#A8BEF0','#34D399','#FBBF24'])
        fig.update_traces(textposition='outside',marker_line_width=0,width=0.6)
        chart(fig, ytitle='Students', xtitle='Branch', angle=30, mb=80)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q2
    st.markdown('<p class="sec">Q2 — Which year dominates?</p>', unsafe_allow_html=True)
    yr = fdf['Year'].value_counts().reindex(['1st Yr','2nd Yr','3rd Yr','4th Yr']).reset_index()
    yr.columns=['Year','Count']
    fig = px.bar(yr,x='Year',y='Count',text='Count',color='Year',
                 color_discrete_sequence=[P1,P2,P3,'#B8CCF4'])
    fig.update_traces(textposition='outside',marker_line_width=0,width=0.45)
    chart(fig, h=300, ytitle='Students', xtitle='Year')
    st.markdown('<div class="insight">74 of 102 are 2nd Year — representative of sophomore stress at IIT.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q3
    st.markdown('<p class="sec">Q3 — Overall stress distribution</p>', unsafe_allow_html=True)
    fig = px.histogram(fdf,x='Overall_Stress',nbins=10,color_discrete_sequence=[P2])
    fig.add_vline(x=fdf['Overall_Stress'].mean(),line_dash='dash',line_color=ACC,
                  annotation_text=f"Mean {fdf['Overall_Stress'].mean():.1f}",
                  annotation_position='top right',annotation_font=dict(color=ACC,size=12))
    fig.update_traces(marker_line_width=0)
    chart(fig, extra=dict(bargap=0.12), xtitle='Stress Score (1–10)', ytitle='Count')
    st.markdown('<div class="insight">42 students (41%) in High stress (7–10), 34 Medium, 25 Low.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q4
    st.markdown('<p class="sec">Q4 — GAD anxiety score by stress level</p>', unsafe_allow_html=True)
    st.caption("GAD Score = average of Nervous + Irritable ratings (scale 1–4)")
    fig = px.violin(fdf,x='Stress_Level',y='GAD_Score',
                    category_orders={'Stress_Level':['Low','Medium','High']},
                    color='Stress_Level',box=True,points='all',color_discrete_map=SMAP)
    fig.update_traces(meanline_visible=True,pointpos=0,jitter=0.3,marker=dict(size=4,opacity=0.5))
    chart(fig, legend=True, ytitle='GAD Score (1–4)', xtitle='Stress Level')
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q5
    st.markdown('<p class="sec">Q5 — Sleep duration vs stress</p>', unsafe_allow_html=True)
    tmp = fdf.copy(); tmp['x'] = short(tmp['Sleep_Hours'],SLP,SLP_S)
    fig = px.box(tmp,x='x',y='Overall_Stress',category_orders={'x':SLP_S},
                 color='x',points='all',color_discrete_sequence=[P1,P2,P3,'#B8CCF4'])
    fig.update_traces(marker=dict(size=4,opacity=0.5))
    chart(fig, ytitle='Stress Score (1–10)', xtitle='Sleep per Night')
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q6
    st.markdown('<p class="sec">Q6 — Branch vs stress</p>', unsafe_allow_html=True)
    fig = px.box(fdf,x='Branch',y='Overall_Stress',color='Branch',points='all',
                 color_discrete_sequence=[P1,P2,P3,'#6B8EE8','#A8BEF0','#34D399','#FBBF24'])
    fig.update_traces(marker=dict(size=4,opacity=0.5))
    chart(fig, ytitle='Stress Score (1–10)', xtitle='Branch', angle=30, mb=80)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q7
    st.markdown('<p class="sec">Q7 — CGPA vs stress</p>', unsafe_allow_html=True)
    fig = px.box(fdf,x='CGPA',y='Overall_Stress',category_orders={'CGPA':CGA},
                 color='CGPA',points='all',
                 color_discrete_sequence=[P1,P2,P3,'#8FA8E8','#B8CCF4'])
    fig.update_traces(marker=dict(size=4,opacity=0.5))
    chart(fig, ytitle='Stress Score (1–10)', xtitle='CGPA Range')
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q8
    st.markdown('<p class="sec">Q8 — Most common stress factors</p>', unsafe_allow_html=True)
    all_f = [x.strip() for row in fdf['Stress_Factors'].dropna() for x in row.split(',')]
    fc = pd.DataFrame(Counter(all_f).items(),columns=['Factor','Count']).sort_values('Count')
    fig = px.bar(fc,x='Count',y='Factor',orientation='h',text='Count',color='Count',
                 color_continuous_scale=[[0,'#B8CCF4'],[1,P1]])
    fig.update_traces(textposition='outside',marker_line_width=0,textfont=dict(color=TEXT))
    chart(fig, h=400, xtitle='Students', extra=dict(coloraxis_showscale=False))
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q9 — 4 lifestyle plots
    st.markdown('<p class="sec">Q9 — Lifestyle factors vs stress</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.caption("Q9a — Screen time (non-academic)")
        tmp = fdf.copy(); tmp['x'] = short(tmp['Screen_Time'],SCR,SCR_S)
        fig = px.box(tmp,x='x',y='Overall_Stress',category_orders={'x':SCR_S},
                     color='x',points='all',color_discrete_sequence=[P1,P2,P3,'#B8CCF4'])
        fig.update_traces(marker=dict(size=4,opacity=0.5))
        chart(fig, ytitle='Stress (1–10)', xtitle='Daily Screen Time')
    with c2:
        st.caption("Q9b — Exercise frequency")
        tmp = fdf.copy(); tmp['x'] = short(tmp['Exercise_Freq'],EXE,EXE_S)
        fig = px.violin(tmp,x='x',y='Overall_Stress',category_orders={'x':EXE_S},
                        color='x',box=True,color_discrete_sequence=[P1,P2,P3,'#B8CCF4'])
        fig.update_traces(meanline_visible=True,points='all',jitter=0.3,
                          marker=dict(size=3,opacity=0.4))
        chart(fig, ytitle='Stress (1–10)', xtitle='Exercise per Week')

    c1,c2 = st.columns(2)
    with c1:
        st.caption("Q9c — Diet quality")
        tmp = fdf.copy(); tmp['x'] = short(tmp['Diet_Quality'],DIT,DIT_S)
        fig = px.box(tmp,x='x',y='Overall_Stress',category_orders={'x':DIT_S},
                     color='x',points='all',
                     color_discrete_sequence=[P1,P2,P3,'#8FA8E8','#B8CCF4'])
        fig.update_traces(marker=dict(size=4,opacity=0.5))
        chart(fig, ytitle='Stress (1–10)', xtitle='Diet Quality')
    with c2:
        st.caption("Q9d — Daily study hours")
        tmp = fdf.copy(); tmp['x'] = short(tmp['Study_Hours'],STU,STU_S)
        fig = px.box(tmp,x='x',y='Overall_Stress',category_orders={'x':STU_S},
                     color='x',points='all',color_discrete_sequence=[P1,P2,P3,'#B8CCF4'])
        fig.update_traces(marker=dict(size=4,opacity=0.5))
        chart(fig, ytitle='Stress (1–10)', xtitle='Study Hours/Day')
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Q10
    st.markdown('<p class="sec">Q10 — Counselling need vs action</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = px.box(fdf,x='Counsellor_Group',y='Overall_Stress',
                     category_orders={'Counsellor_Group':GRP},
                     color='Counsellor_Group',points='all',color_discrete_map=GCOL)
        fig.update_traces(marker=dict(size=4,opacity=0.5))
        chart(fig, ytitle='Stress (1–10)', xtitle='Counselling Group')
    with c2:
        cnt = fdf['Counsellor_Group'].value_counts().reindex(GRP).reset_index()
        cnt.columns=['Group','Count']
        fig = px.bar(cnt,x='Group',y='Count',text='Count',color='Group',
                     color_discrete_map=GCOL)
        fig.update_traces(textposition='outside',marker_line_width=0,width=0.5)
        chart(fig, ytitle='Students', xtitle='Counselling Group')
    st.markdown('<div class="insight">Students who needed help but did not seek it report the '
                'highest stress. Mental health support is a key protective factor.</div>',
                unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# MODEL RESULTS
# ════════════════════════════════════════════════════════════════
elif page == "Model Results":
    st.markdown('<p style="font-size:1.5rem;font-weight:800;color:#1C1F3B">Model Results</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.84rem;color:#6B7280;margin-bottom:1rem">Three classifiers trained to predict stress level from lifestyle and GAD features.</p>', unsafe_allow_html=True)

    for col,(v,l) in zip(st.columns(4),[
        ("47.6%","Baseline Accuracy"),("71.4%","Logistic Regression"),
        ("66.7%","Random Forest"),("61.9%","Decision Tree")]):
        with col:
            st.markdown(f'<div class="kpi"><p class="kpi-v">{v}</p><p class="kpi-l">{l}</p></div>',
                        unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec">Accuracy vs Baseline</p>', unsafe_allow_html=True)
        mdf = pd.DataFrame({'Model':['Logistic Reg.','Random Forest','Decision Tree'],
                            'Accuracy':[71.43,66.67,61.90]})
        fig = px.bar(mdf,x='Model',y='Accuracy',text='Accuracy',color='Model',
                     color_discrete_map={'Logistic Reg.':P1,'Random Forest':P2,'Decision Tree':P3})
        fig.add_hline(y=47.62,line_dash='dash',line_color=ACC,
                      annotation_text='Baseline 47.6%',annotation_position='top left',
                      annotation_font=dict(color=ACC,size=12))
        fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside',
                          marker_line_width=0,width=0.45)
        chart(fig, extra=dict(yaxis_range=[0,88]), ytitle='Accuracy (%)', xtitle='Model')

    with c2:
        st.markdown('<p class="sec">Multi-Metric Radar</p>', unsafe_allow_html=True)
        st.caption("Accuracy · Precision · Recall · F1 · Baseline Gap")
        metrics = ['Accuracy','Precision','Recall','F1','Baseline Gap']
        fig = go.Figure()
        for name,scores,color in [
            ('Logistic Reg.',[0.71,0.72,0.74,0.70,0.23],P1),
            ('Random Forest',[0.67,0.64,0.64,0.63,0.19],P2),
            ('Decision Tree',[0.62,0.75,0.63,0.64,0.14],P3)]:
            fig.add_trace(go.Scatterpolar(
                r=scores+[scores[0]],theta=metrics+[metrics[0]],
                fill='toself',name=name,line_color=color,opacity=0.8))
        fig.update_layout(
            polar=dict(bgcolor=WHITE,
                       radialaxis=dict(visible=True,range=[0,1],
                                       tickfont=dict(size=10,color=TEXT),gridcolor=GRID),
                       angularaxis=dict(tickfont=dict(size=12,color=TEXT))),
            showlegend=True,height=340,paper_bgcolor=WHITE,
            font=dict(family='Inter',color=TEXT,size=12),
            margin=dict(t=40,b=30,l=10,r=10),
            legend=dict(orientation='h',y=-0.2,font=dict(color=TEXT,size=11)))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

    st.markdown('<div class="insight">Logistic Regression achieves 71.4% — 23.8 pp above the '
                'baseline. GAD Score and Sleep Hours are the strongest predictors.</div>',
                unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# STRESS PREDICTOR
# ════════════════════════════════════════════════════════════════
elif page == "Stress Predictor":
    st.markdown('<p style="font-size:1.5rem;font-weight:800;color:#1C1F3B">Stress Level Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.84rem;color:#6B7280;margin-bottom:1rem">Enter your daily habits and anxiety indicators. The model predicts your stress category.</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    c_form,_,c_res = st.columns([1.1,0.1,1])

    with c_form:
        st.markdown('<p class="sec">Lifestyle Habits</p>', unsafe_allow_html=True)
        vals = {}
        for key,label,hint,opts,default in INPUTS:
            st.markdown(f'<div class="iblock"><p class="ilabel">{label}</p>'
                        f'<p class="ihint">{hint}</p>', unsafe_allow_html=True)
            vals[key] = st.select_slider(key, opts, value=default, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<p class="sec"> Anxiety Indicators '
                    f'<span style="font-weight:400;color:{MUTED};font-size:.76rem">'
                    f'(past 2 weeks · 1=Not at all → 4=Nearly every day)</span></p>',
                    unsafe_allow_html=True)

        st.markdown(f'<div class="iblock"><p class="ilabel">Feeling Nervous / Anxious</p>'
                    f'<p class="ihint">How often felt nervous, anxious, or on edge?</p>',
                    unsafe_allow_html=True)
        gad_n = st.slider("Nervous",1,4,2,label_visibility="collapsed")
        st.markdown(f'<p style="font-size:.74rem;color:{P2};margin:-4px 0 4px">'
                    f'→ <b>{GAD_LBL[gad_n]}</b></p></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="iblock"><p class="ilabel">Feeling Irritable / Restless</p>'
                    f'<p class="ihint">How often felt irritable, restless, or easily annoyed?</p>',
                    unsafe_allow_html=True)
        gad_i = st.slider("Irritable",1,4,2,label_visibility="collapsed")
        st.markdown(f'<p style="font-size:.74rem;color:{P2};margin:-4px 0 4px">'
                    f'→ <b>{GAD_LBL[gad_i]}</b></p></div>', unsafe_allow_html=True)

        gad_s = (gad_n+gad_i)/2
        st.markdown(f'<p style="font-size:.78rem;color:{MUTED};margin:.4rem 0 .8rem">'
                    f'Combined GAD Score: <b style="color:{P1}">{gad_s:.1f}/4.0</b></p>',
                    unsafe_allow_html=True)
        predict = st.button("Predict My Stress Level")

    with c_res:
        if predict:
            MAPS = {
                'sleep':   dict(zip(SLP,[1,2,3,4])),
                'screen':  dict(zip(SCR,[1,2,3,4])),
                'exercise':dict(zip(EXE,[1,2,3,4])),
                'diet':    dict(zip(DIT,[1,2,3,4,5])),
                'study':   dict(zip(STU,[1,2,3,4])),
                'cgpa':    dict(zip(CGA,[1,2,3,4,5])),
            }
            inp = pd.DataFrame([[
                MAPS['sleep'][vals['sleep']], MAPS['screen'][vals['screen']],
                MAPS['exercise'][vals['exercise']], MAPS['diet'][vals['diet']],
                MAPS['study'][vals['study']], MAPS['cgpa'][vals['cgpa']],
                gad_n, gad_i, gad_s
            ]], columns=['Sleep_Hours','Screen_Time','Exercise_Freq','Diet_Quality',
                         'Study_Hours','CGPA','GAD_Nervous','GAD_Irritable','GAD_Score'])

            pred = mdl.predict(inp)[0]
            cls  = {'High':'res-high','Medium':'res-medium','Low':'res-low'}[pred]
            lbl  = {'High':'🔴 High Stress','Medium':'🟡 Moderate Stress','Low':'🟢 Low Stress'}[pred]
            st.markdown(f'<div class="res-box {cls}">{lbl}</div>', unsafe_allow_html=True)

            raw   = (MAPS['sleep'][vals['sleep']]*(-0.3)+MAPS['screen'][vals['screen']]*0.2+
                     MAPS['exercise'][vals['exercise']]*(-0.2)+gad_n*0.4+gad_i*0.3)+5
            score = round(max(1,min(10,raw)),1)

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                title={'text':'Estimated Stress Score','font':{'size':12,'color':MUTED}},
                number={'font':{'size':28,'color':P1},'suffix':'/10'},
                gauge={'axis':{'range':[0,10],'tickfont':{'size':10,'color':TEXT}},
                       'bar':{'color':P1,'thickness':0.22},'bgcolor':WHITE,'borderwidth':0,
                       'steps':[{'range':[0,3],'color':'#D1FAE5'},
                                 {'range':[3,6],'color':'#FEF3C7'},
                                 {'range':[6,10],'color':'#FEE2E2'}]}))
            fig.update_layout(height=240,paper_bgcolor=WHITE,
                              font=dict(family='Inter',color=TEXT),
                              margin=dict(t=50,b=10,l=30,r=30))
            st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

            st.markdown('<p class="sec">Your Inputs</p>', unsafe_allow_html=True)
            summary = pd.DataFrame(
                [(INPUTS[i][1], vals[INPUTS[i][0]]) for i in range(6)] + [
                    ("Feeling Nervous",  f"{gad_n} — {GAD_LBL[gad_n]}"),
                    ("Feeling Irritable",f"{gad_i} — {GAD_LBL[gad_i]}"),
                    ("GAD Score",        f"{gad_s:.1f}/4.0"),
                ], columns=['Factor','Your Value'])
            st.dataframe(summary, use_container_width=True, hide_index=True)

        else:
            st.markdown(f"""<div style="background:#EEF2FF;border-radius:14px;
                padding:3rem 2rem;text-align:center;margin-top:.5rem">
                <p style="font-size:2rem;margin:0 0 .8rem"></p>
                <p style="font-weight:600;color:{P1};margin:0 0 .4rem;font-size:.95rem">
                    Ready to predict</p>
                <p style="color:{MUTED};font-size:.82rem;margin:0;line-height:1.8">
                    Fill in your habits on the left and click<br>
                    <b>Predict My Stress Level</b></p>
            </div>""", unsafe_allow_html=True)