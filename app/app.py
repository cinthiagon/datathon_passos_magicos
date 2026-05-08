"""
Aplicação Streamlit — Passos Mágicos
Predição de Risco de Defasagem Escolar
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1B4F72, #2980B9);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    background: #f8f9fa;
    border-left: 4px solid #2980B9;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
  }
  .risco-alto {
    background: #fde8e8;
    border-left: 4px solid #E74C3C;
    padding: 1rem;
    border-radius: 8px;
  }
  .risco-baixo {
    background: #e8f8f5;
    border-left: 4px solid #27AE60;
    padding: 1rem;
    border-radius: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ── Carregamento do modelo ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def carregar_modelo():
    modelo  = joblib.load(os.path.join(BASE_DIR, "modelo_risco.pkl"))
    scaler  = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    with open(os.path.join(BASE_DIR, "modelo_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, scaler, meta


modelo, scaler, meta = carregar_modelo()
FEATURES = meta["features"]


@st.cache_data
def carregar_dados():
    caminho = os.path.join(BASE_DIR, "..", "data", "dados_limpos.csv")
    if os.path.exists(caminho):
        return pd.read_csv(caminho, encoding="utf-8-sig")
    return None


df = carregar_dados()

# ── Cabeçalho ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1 style="margin:0; font-size:1.8rem;">🎓 Passos Mágicos</h1>
  <p style="margin:0.3rem 0 0 0; opacity:0.9; font-size:1rem;">
    Sistema Preditivo de Risco de Defasagem Escolar
  </p>
</div>
""", unsafe_allow_html=True)

# ── Abas ────────────────────────────────────────────────────────────────────
aba1, aba2, aba3 = st.tabs([
    "🔍 Previsão Individual",
    "📊 Panorama Geral",
    "ℹ️ Sobre o Modelo",
])

# ════════════════════════════════════════════════════════════════════════════
# ABA 1 — Previsão Individual
# ════════════════════════════════════════════════════════════════════════════
with aba1:
    st.markdown("### Preencha os indicadores do aluno")
    st.caption(
        "Informe os valores dos indicadores coletados nas avaliações. "
        "O modelo estima a probabilidade de o aluno estar em risco de defasagem escolar."
    )

    col_left, col_right = st.columns([2, 1])

    with col_left:
        with st.form("formulario_aluno"):
            st.markdown("#### Indicadores Comportamentais e Psicossociais")
            c1, c2 = st.columns(2)
            with c1:
                iaa = st.slider("IAA — Autoavaliação", 0.0, 10.0, 7.0, 0.1,
                                help="Média das notas de autoavaliação do aluno")
                ieg = st.slider("IEG — Engajamento", 0.0, 10.0, 7.0, 0.1,
                                help="Média das notas de engajamento")
                ips = st.slider("IPS — Psicossocial", 0.0, 10.0, 7.0, 0.1,
                                help="Média das notas psicossociais")
                ipp = st.slider("IPP — Psicopedagógico", 0.0, 10.0, 7.0, 0.1,
                                help="Média das notas psicopedagógicas")
            with c2:
                ida = st.slider("IDA — Aprendizagem", 0.0, 10.0, 7.0, 0.1,
                                help="Média das notas de aprendizagem")
                ipv = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 7.0, 0.1,
                                help="Média das notas do ponto de virada")
                nota_mat  = st.slider("Nota Matemática",  0.0, 10.0, 6.0, 0.1)
                nota_port = st.slider("Nota Português",   0.0, 10.0, 6.0, 0.1)

            st.markdown("#### Contexto do Aluno")
            c3, c4 = st.columns(2)
            with c3:
                nota_ing      = st.slider("Nota Inglês",       0.0, 10.0, 6.0, 0.1)
                num_aval      = st.number_input("Nº de Avaliadores", 1, 6, 3)
            with c4:
                anos_na_pm    = st.number_input("Anos na Passos Mágicos", 0, 15, 2)
                idade         = st.number_input("Idade do Aluno", 6, 24, 14)

            calcular = st.form_submit_button("Calcular Risco", type="primary",
                                              use_container_width=True)

    with col_right:
        st.markdown("#### Guia de Referência")
        st.markdown("""
| Pedra    | INDE          |
|----------|--------------|
| Quartzo  | 2,4 – 5,5   |
| Ágata    | 5,5 – 6,9   |
| Ametista | 6,9 – 8,2   |
| Topázio  | 8,2 – 9,3   |

---
**Como interpretar a probabilidade:**

- **< 30%** — Baixo risco  
- **30–60%** — Risco moderado  
- **> 60%** — Alto risco  

*Recomenda-se encaminhar alunos com probabilidade acima de 50% para avaliação psicopedagógica.*
""")

    if calcular:
        # Monta vetor de features
        ieg_baixo         = 1 if ieg < 5.0 else 0
        divergencia       = iaa - ida
        score_psi         = (ips + ipp) / 2
        nota_mat_baixa    = 1 if nota_mat < 5.0 else 0

        entrada = pd.DataFrame([{
            "iaa": iaa, "ieg": ieg, "ips": ips, "ipp": ipp,
            "ida": ida, "ipv": ipv,
            "nota_mat": nota_mat, "nota_port": nota_port, "nota_ing": nota_ing,
            "num_avaliadores": num_aval, "anos_na_pm": anos_na_pm, "idade": idade,
            "ieg_baixo": ieg_baixo, "divergencia_iaa_ida": divergencia,
            "score_psi": score_psi, "nota_mat_baixa": nota_mat_baixa,
        }])[FEATURES]

        probabilidade = modelo.predict_proba(entrada)[0][1]
        previsao      = int(probabilidade >= 0.5)

        st.markdown("---")
        st.markdown("### Resultado da Análise")

        col_prob, col_gauge = st.columns([1, 1])

        with col_prob:
            if previsao == 1:
                st.markdown(f"""
<div class="risco-alto">
  <h3 style="color:#E74C3C; margin:0;">⚠️ Risco de Defasagem Detectado</h3>
  <p style="font-size:1.8rem; font-weight:bold; color:#E74C3C; margin:0.5rem 0;">
    {probabilidade:.1%}
  </p>
  <p>Probabilidade de estar abaixo do nível ideal para a fase.</p>
  <p><strong>Recomendação:</strong> Encaminhar para avaliação psicopedagógica
  e aumentar acompanhamento individual.</p>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="risco-baixo">
  <h3 style="color:#27AE60; margin:0;">✅ Sem Risco Identificado</h3>
  <p style="font-size:1.8rem; font-weight:bold; color:#27AE60; margin:0.5rem 0;">
    {probabilidade:.1%}
  </p>
  <p>Probabilidade de estar abaixo do nível ideal para a fase.</p>
  <p><strong>Recomendação:</strong> Manter acompanhamento regular do desenvolvimento.</p>
</div>
""", unsafe_allow_html=True)

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probabilidade * 100,
                title={"text": "Prob. de Risco (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#E74C3C" if probabilidade >= 0.5 else "#27AE60"},
                    "steps": [
                        {"range": [0, 30],  "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fef9e7"},
                        {"range": [60, 100],"color": "#fde8e8"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
                number={"suffix": "%", "font": {"size": 32}},
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Radar com os indicadores inseridos
        st.markdown("#### Perfil do Aluno")
        cats = ["IAA", "IEG", "IPS", "IPP", "IDA", "IPV"]
        vals = [iaa, ieg, ips, ipp, ida, ipv]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(41, 128, 185, 0.2)",
            line_color="#2980B9",
            name="Aluno",
        ))
        if df is not None:
            medias = [df["iaa"].mean(), df["ieg"].mean(), df["ips"].mean(),
                      df["ipp"].mean(), df["ida"].mean(), df["ipv"].mean()]
            medias_clean = [m if not np.isnan(m) else 0 for m in medias]
            fig_radar.add_trace(go.Scatterpolar(
                r=medias_clean + [medias_clean[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.1)",
                line_color="#E74C3C",
                line_dash="dash",
                name="Média geral",
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 10])),
            height=380, margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(x=0.8, y=1.1),
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# ABA 2 — Panorama Geral
# ════════════════════════════════════════════════════════════════════════════
with aba2:
    if df is None:
        st.warning("Dataset de análise não encontrado. Execute os notebooks primeiro.")
    else:
        st.markdown("### Visão Geral da Base — PEDE 2022–2024")

        # KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total de Alunos", f"{len(df):,}")
        kpi2.metric("Anos analisados", "2022, 2023 e 2024")
        risco_pct = df["risco_defasagem"].mean() * 100
        kpi3.metric("Em Risco de Defasagem", f"{risco_pct:.1f}%")
        inde_medio = df["inde"].mean()
        kpi4.metric("INDE Médio Geral", f"{inde_medio:.2f}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            # Distribuição por Pedra
            ordem = ["Quartzo", "Ágata", "Ametista", "Topázio"]
            pedra_counts = (
                df[df["pedra_ref"].isin(ordem)]
                .groupby(["ano", "pedra_ref"])
                .size()
                .reset_index(name="n")
            )
            fig_pedra = px.bar(
                pedra_counts,
                x="pedra_ref", y="n", color="ano",
                barmode="group",
                category_orders={"pedra_ref": ordem},
                color_discrete_sequence=["#1B4F72", "#2980B9", "#A9CCE3"],
                title="Distribuição por Classificação (Pedra) — 2022 a 2024",
                labels={"pedra_ref": "Classificação", "n": "Quantidade", "ano": "Ano"},
            )
            fig_pedra.update_layout(plot_bgcolor="white", height=350)
            st.plotly_chart(fig_pedra, use_container_width=True)

        with col_b:
            # Evolução INDE médio por ano
            inde_ano = df.groupby("ano")["inde"].mean().reset_index()
            fig_inde = px.line(
                inde_ano, x="ano", y="inde", markers=True,
                title="Evolução do INDE Médio (2022–2024)",
                labels={"ano": "Ano", "inde": "INDE Médio"},
                color_discrete_sequence=["#1B4F72"],
            )
            fig_inde.update_traces(line_width=3, marker_size=10)
            fig_inde.update_layout(plot_bgcolor="white", height=350,
                                   xaxis=dict(tickvals=[2022, 2023, 2024]))
            st.plotly_chart(fig_inde, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            # % em risco por ano
            risco_ano = df.groupby("ano")["risco_defasagem"].mean().reset_index()
            risco_ano["pct"] = risco_ano["risco_defasagem"] * 100
            fig_risco = px.bar(
                risco_ano, x="ano", y="pct",
                color_discrete_sequence=["#E74C3C"],
                title="% de Alunos em Risco por Ano",
                labels={"ano": "Ano", "pct": "% em Risco"},
                text_auto=".1f",
            )
            fig_risco.update_layout(plot_bgcolor="white", height=320,
                                    xaxis=dict(type="category"))
            st.plotly_chart(fig_risco, use_container_width=True)

        with col_d:
            # Distribuição dos indicadores (box)
            ind_sel = st.selectbox(
                "Selecione o indicador:", ["inde", "iaa", "ieg", "ips", "ida", "ipv", "ipp"],
                format_func=lambda x: {
                    "inde": "INDE (Global)", "iaa": "IAA (Autoavaliação)",
                    "ieg": "IEG (Engajamento)", "ips": "IPS (Psicossocial)",
                    "ida": "IDA (Aprendizagem)", "ipv": "IPV (Ponto de Virada)",
                    "ipp": "IPP (Psicopedagógico)"
                }.get(x, x),
            )
            df_box = df[df["pedra_ref"].isin(["Quartzo", "Ágata", "Ametista", "Topázio"])]
            fig_box = px.box(
                df_box, x="pedra_ref", y=ind_sel, color="pedra_ref",
                category_orders={"pedra_ref": ["Quartzo", "Ágata", "Ametista", "Topázio"]},
                color_discrete_map={
                    "Quartzo": "#5DADE2", "Ágata": "#27AE60",
                    "Ametista": "#8E44AD", "Topázio": "#F39C12"
                },
                title=f"Distribuição de {ind_sel.upper()} por Classificação",
                labels={"pedra_ref": "Classificação", ind_sel: ind_sel.upper()},
            )
            fig_box.update_layout(plot_bgcolor="white", height=320, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        # Tabela resumo
        st.markdown("#### Estatísticas por Ano")
        resumo = df.groupby("ano")[
            ["inde", "iaa", "ieg", "ips", "ida", "ipv", "ipp"]
        ].mean().round(2)
        st.dataframe(resumo.style.background_gradient(cmap="Blues", axis=None),
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# ABA 3 — Sobre o Modelo
# ════════════════════════════════════════════════════════════════════════════
with aba3:
    st.markdown("### Sobre o Modelo Preditivo")

    col_info, col_met = st.columns([1, 1])

    with col_info:
        st.markdown(f"""
**Algoritmo:** {meta['modelo']}

**Problema:** Classificação binária — identificar alunos com risco de defasagem
escolar (abaixo do nível ideal para a fase).

**Definição do target:**  
`risco_defasagem = 1` quando `defasagem < 0`, ou seja, quando o aluno está
matriculado em fase abaixo da que seria ideal para sua idade e histórico.

**Balanceamento:** Técnica SMOTE aplicada ao conjunto de treino para corrigir
o desbalanceamento de classes.

**Divisão dos dados:**  
- {meta['n_treino']:,} registros no treino  
- {meta['n_teste']:,} registros no teste  
- Proporção de risco no treino: {meta['proporcao_risco_treino']:.1%}
""")

    with col_met:
        st.markdown(f"""
**Métricas no conjunto de teste:**

| Métrica         | Valor            |
|-----------------|-----------------|
| ROC-AUC         | **{meta['roc_auc']:.4f}** |
| F1 (em risco)   | **{meta['f1_risco']:.4f}** |
| Threshold       | {meta['threshold']} |
""")

    st.markdown("---")
    st.markdown("#### Features Utilizadas")
    st.markdown("""
Os seguintes indicadores foram utilizados como entrada do modelo.
`ian` e `inde` foram **excluídos** por serem derivados matematicamente da
mesma informação que o target, o que causaria vazamento de dados (*data leakage*).
""")

    feat_desc = {
        "iaa": "Autoavaliação do aluno", "ieg": "Engajamento",
        "ips": "Psicossocial", "ipp": "Psicopedagógico",
        "ida": "Indicador de aprendizagem", "ipv": "Ponto de virada",
        "nota_mat": "Nota em Matemática", "nota_port": "Nota em Português",
        "nota_ing": "Nota em Inglês", "num_avaliadores": "Número de avaliadores",
        "anos_na_pm": "Tempo na Passos Mágicos (anos)", "idade": "Idade do aluno",
        "ieg_baixo": "Flag: engajamento abaixo do quartil 25%",
        "divergencia_iaa_ida": "Diferença IAA − IDA (superestimação)",
        "score_psi": "Score psicossocial composto ((IPS+IPP)/2)",
        "nota_mat_baixa": "Flag: nota de matemática < 5",
    }
    df_feat = pd.DataFrame([
        {"Feature": f, "Tipo": "Engenharia" if f in [
            "ieg_baixo", "divergencia_iaa_ida", "score_psi", "nota_mat_baixa"
        ] else "Original", "Descrição": feat_desc.get(f, "-")}
        for f in meta["features"]
    ])
    st.dataframe(df_feat, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Limitações e Recomendações")
    st.markdown("""
- O modelo foi treinado com dados de 2022–2024. Recomenda-se recalibração anual
  após a coleta de novos dados PEDE.
- A probabilidade estimada não substitui a avaliação presencial da equipe pedagógica;
  ela é uma **ferramenta de triagem e priorização**.
- Alunos com probabilidade entre 40% e 60% estão na zona de incerteza e merecem
  atenção redobrada.
""")

# ── Rodapé ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666; font-size:0.85rem;'>"
    "Projeto desenvolvido no âmbito do Datathon PosTech FIAP · "
    "Associação Passos Mágicos · PEDE 2022–2024"
    "</p>",
    unsafe_allow_html=True,
)
