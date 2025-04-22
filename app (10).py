import pandas as pd
import numpy as np
import gradio as gr
from sklearn.tree import DecisionTreeClassifier
from PyPDF2 import PdfReader
import io
import joblib
import re
from collections import Counter

# ============== CONFIGURAÇÃO ==============
CONFIG = {
    "keywords": {
        "Customer Success": {"terms": ["customer success", "cs", "sucesso do cliente"], "weight": 3, "color": "#4ECDC4"},
        "CRM": {"terms": ["crm", "salesforce", "hubspot", "zendesk"], "weight": 2.5, "color": "#FF6B6B"},
        "Atendimento": {"terms": ["atendimento", "suporte", "sac", "serviço ao cliente"], "weight": 2, "color": "#45B7D1"},
        "Métricas": {"terms": ["churn", "nps", "onboarding", "retention"], "weight": 1.5, "color": "#A569BD"},
        "Soft Skills": {"terms": ["comunicação", "empatia", "proatividade"], "weight": 1, "color": "#F4D03F"}
    },
    "thresholds": {
        "Aderente": 75,
        "Potencial": 50
    }
}

# ============== FUNÇÕES ==============
def extract_text_from_pdf(pdf_file):
    """Extrai texto de PDFs de forma robusta"""
    try:
        if isinstance(pdf_file, bytes):
            pdf = PdfReader(io.BytesIO(pdf_file))
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
        return str(pdf_file)
    except Exception as e:
        print(f"Erro ao extrair PDF: {str(e)}")
        return ""

def analyze_resume_content(text):
    """Analisa o currículo com pontuação inteligente"""
    if not text:
        return {"score": 0, "matches": [], "missing_terms": []}
    
    text = text.lower()
    matches = []
    
    # Detecta termos por categoria
    for category, config in CONFIG["keywords"].items():
        for term in config["terms"]:
            count = len(re.findall(rf'\b{term}\b', text))
            if count > 0:
                matches.append({
                    "term": term,
                    "category": category,
                    "count": count,
                    "weight": config["weight"],
                    "score": count * config["weight"]
                })
    
    # Calcula score total (limitado a 10)
    total_score = min(sum(m["score"] for m in matches) / 3, 10)
    
    # Identifica termos faltantes
    found_categories = {m["category"] for m in matches}
    missing_terms = [
        {"category": cat, "suggestion": f"Adicione: {', '.join(CONFIG['keywords'][cat]['terms'][:2])}"}
        for cat in CONFIG["keywords"] if cat not in found_categories
    ]
    
    return {
        "score": round(total_score, 1),
        "matches": matches,
        "missing_terms": missing_terms
    }

# ============== TEMA E INTERFACE ==============
# Tema customizado compatível com Hugging Face
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple"
).set(
    body_background_fill="#f5f7ff",  # Cor de fundo
    block_background_fill="#ffffff",  # Cor dos painéis
    button_primary_background_fill="#6e48aa"  # Cor do botão
)

with gr.Blocks(theme=custom_theme, title="🔍 TalentScope CS") as app:
    # ===== CABEÇALHO =====
    gr.HTML("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px;">
        <h1 style="margin: 0;">TalentScope CS</h1>
        <p style="opacity: 0.9;">Análise Inteligente para Estágio em Customer Success</p>
    </div>
    """)
    
    # ===== LAYOUT PRINCIPAL =====
    with gr.Row():
        # --- COLUNA DE INPUTS ---
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📋 Perfil do Candidato")
                tempo_exp = gr.Slider(0, 24, step=1, label="Experiência (meses)")
                conhecimento_crm = gr.Slider(1, 5, step=1, label="Conhecimento de CRM")
                ingles = gr.Slider(1, 5, step=1, label="Proficiência em Inglês")
                graduacao = gr.Radio(["Cursando", "Completo"], label="Graduação")
            
            with gr.Group():
                gr.Markdown("### 📎 Currículo")
                curriculo = gr.File(label="Envie seu PDF", file_types=[".pdf"])
            
            submit_btn = gr.Button("Analisar", variant="primary")

        # --- COLUNA DE RESULTADOS ---
        with gr.Column(scale=2):
            with gr.Tab("📊 Resultado"):
                with gr.Row():
                    with gr.Column(min_width=150):
                        status = gr.Label(label="Status")
                        prob_final = gr.Number(label="Probabilidade Final (%)")
                    
                    with gr.Column():
                        bonus = gr.Number(label="Bônus do Currículo (%)")
                        prob_base = gr.Number(label="Probabilidade Base (%)")
                
                gr.Markdown("### 🔍 Termos Encontrados")
                with gr.Row():
                    keyword_display = gr.HighlightedText(
                        label="Palavras-chave",
                        show_legend=True,
                        color_map={cat: cfg["color"] for cat, cfg in CONFIG["keywords"].items()}
                    )
                
                with gr.Accordion("💡 Como melhorar", open=False):
                    improvement_tips = gr.JSON(label="Termos Faltantes")

    # ===== LÓGICA DE PREDIÇÃO =====
    def predict_aderencia(tempo_exp, conhecimento_crm, ingles, graduacao, curriculo):
        try:
            # Processa currículo
            resume_text = extract_text_from_pdf(curriculo) if curriculo else ""
            resume_analysis = analyze_resume_content(resume_text)
            
            # Prepara dados para o modelo
            input_data = np.array([[
                tempo_exp,
                conhecimento_crm,
                ingles,
                1 if graduacao == "Cursando" else 0
            ]])
            
            # Predição
            proba = model.predict_proba(input_data)[0][1] * 100
            proba_ajustada = min(100, proba + resume_analysis["score"] * 3)
            
            # Formata termos encontrados
            keyword_data = [
                (f"{m['term']} (x{m['count']})", m["category"])
                for m in resume_analysis["matches"]
            ]
            
            # Determina status
            status_label = (
                "✅ Aderente" if proba_ajustada >= CONFIG["thresholds"]["Aderente"] else
                "🌟 Potencial" if proba_ajustada >= CONFIG["thresholds"]["Potencial"] else
                "❌ Não Aderente"
            )
            
            return {
                status: status_label,
                prob_final: round(proba_ajustada, 1),
                prob_base: round(proba, 1),
                bonus: round(resume_analysis["score"] * 3, 1),
                keyword_display: keyword_data,
                improvement_tips: resume_analysis["missing_terms"]
            }
            
        except Exception as e:
            print(f"Erro: {str(e)}")
            return {status: "⚠️ Erro na análise"}

    # Conecta o botão
    submit_btn.click(
        fn=predict_aderencia,
        inputs=[tempo_exp, conhecimento_crm, ingles, graduacao, curriculo],
        outputs=[status, prob_final, prob_base, bonus, keyword_display, improvement_tips]
    )

# ============== INICIALIZAÇÃO ==============
if __name__ == "__main__":
    # Carrega o modelo (com fallback para teste)
    try:
        model = joblib.load('modelo_estagio_cs.joblib')
    except:
        print("⚠️ Usando modelo dummy para teste")
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(np.array([[0,0,0,0], [1,1,1,1]]), [0, 1])
    
    app.launch(server_port=7860)