# Datathon Passos Mágicos — Análise de Desenvolvimento Educacional

Projeto desenvolvido como parte do Datathon da PosTech FIAP, com base nos dados da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE) da Associação Passos Mágicos, referentes aos anos de 2022, 2023 e 2024.

## Contexto

A Associação Passos Mágicos atua há mais de 30 anos na transformação da vida de crianças e jovens de baixa renda no município de Embu-Guaçu (SP), promovendo educação de qualidade, apoio psicológico e ampliação de oportunidades. Este projeto aplica técnicas de análise de dados para responder a dores de negócio reais da organização e construir um modelo preditivo de risco de defasagem escolar.

## Estrutura do Projeto

```
Datathon/
├── database.xlsx                        # Base de dados original (PEDE 2022-2024)
├── requirements.txt                     # Dependências Python
├── README.md                            # Este arquivo
├── data/
│   └── dados_limpos.csv                 # Dataset unificado após limpeza
├── notebooks/
│   ├── 01_limpeza_eda.ipynb             # Limpeza e análise exploratória
│   ├── 02_analise_indicadores.ipynb     # Análise dos indicadores (11 perguntas)
│   └── 03_modelo_preditivo.ipynb        # Modelo preditivo de risco de defasagem
├── app/
│   ├── app.py                           # Aplicação Streamlit
│   ├── modelo_risco.pkl                 # Modelo treinado serializado
│   └── requirements.txt                 # Dependências específicas para deploy
└── figuras/                             # Gráficos exportados para apresentação
```

## Indicadores Analisados

| Sigla | Nome completo | Descrição |
|-------|--------------|-----------|
| INDE  | Índice de Desenvolvimento Educacional | Nota global ponderada |
| IAN   | Adequação ao Nível | Alinhamento do aluno com o nível da fase |
| IDA   | Indicador de Aprendizagem | Desempenho acadêmico médio |
| IEG   | Indicador de Engajamento | Participação e dedicação |
| IAA   | Autoavaliação | Percepção do aluno sobre si mesmo |
| IPS   | Indicador Psicossocial | Aspectos socioemocionais |
| IPP   | Indicador Psicopedagógico | Avaliação psicopedagógica |
| IPV   | Ponto de Virada | Indicador de transformação do aluno |

## Classificação por Pedra (INDE)

| Pedra    | Faixa de INDE |
|----------|--------------|
| Quartzo  | 2,405 – 5,506 |
| Ágata    | 5,506 – 6,868 |
| Ametista | 6,868 – 8,230 |
| Topázio  | 8,230 – 9,294 |

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar os notebooks em ordem

```bash
jupyter lab
```

Execute os notebooks na seguinte ordem:
1. `notebooks/01_limpeza_eda.ipynb`
2. `notebooks/02_analise_indicadores.ipynb`
3. `notebooks/03_modelo_preditivo.ipynb`

### 3. Rodar a aplicação Streamlit localmente

```bash
streamlit run app/app.py
```

## Perguntas de Negócio Respondidas

1. Perfil de defasagem dos alunos (IAN) e evolução ao longo do ano
2. Tendência do desempenho acadêmico (IDA) por fase e ano
3. Relação entre engajamento (IEG) e desempenho (IDA/IPV)
4. Coerência entre autoavaliação (IAA) e indicadores objetivos
5. Padrões psicossociais (IPS) que antecedem quedas de desempenho
6. Convergência entre avaliação psicopedagógica (IPP) e defasagem (IAN)
7. Fatores que influenciam o Ponto de Virada (IPV)
8. Combinações de indicadores que mais elevam o INDE
9. Modelo preditivo de risco de defasagem com Machine Learning
10. Efetividade do programa ao longo das fases (Quartzo → Topázio)
11. Insights adicionais e oportunidades de melhoria

## Modelo Preditivo

O modelo utiliza os indicadores comportamentais e acadêmicos do aluno para estimar a probabilidade de risco de defasagem (estar abaixo do nível ideal para sua idade). Algoritmo selecionado: **Random Forest Classifier** com otimização de hiperparâmetros via GridSearchCV.

## Deploy

A aplicação está disponível em: _(link será adicionado após deploy no Streamlit Community Cloud)_

## Autores

Desenvolvido por alunos da PosTech FIAP — Pós-Graduação em Data Analytics:

- Cinthia Gonçalez da Silva
- Gabriel Huzian
- Karyne Barbosa Silva
