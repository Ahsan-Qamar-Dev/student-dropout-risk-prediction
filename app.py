"""Student Dropout Risk Prediction — Gradio entry point."""
import os

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
from prediction import EXAMPLES, Predictor

predictor = Predictor()


def predict_student_risk(*values):
    try:
        result = predictor.predict(values)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return (f"{result['dropout_probability'] * 100:.2f}%",
            result['predicted_class'], result['risk_category'])


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#intro {padding: 26px 30px; border-radius: 18px; background: #102d38; color: white; margin-bottom: 18px;}
#intro h1, #intro p {color: white;}
#intro h1 {font-size: 32px; line-height: 1.2; margin: 8px 0 14px;}
#intro .eyebrow {color: #91e0cd; text-transform: uppercase; font-size: 12px; letter-spacing: 2px;}
#result-panel {border: 1px solid #cbded8; border-radius: 16px; padding: 22px;}
#probability textarea {font-size: 36px !important; font-weight: 700; color: #087e70;}
#predict {min-height: 48px; font-weight: 700;}
"""


def build_app():
    with gr.Blocks(title='Student Dropout Risk Prediction') as demo:
        gr.HTML('''<section id="intro"><div class="eyebrow">Student support · Educational prototype</div>
        <h1>Student Dropout Risk Prediction</h1>
        <p>Explore a student’s estimated dropout risk using academic progress and financial information.</p>
        </section>''')
        gr.Markdown('Use information available **after the second semester**. No student name or ID is needed. '
                    'The remaining 24 model features use saved defaults, so this is a simplified demonstration.')
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown('### 1 · Student background')
                with gr.Row():
                    age = gr.Number(label='Age at enrollment', value=20, minimum=1, maximum=120, step=1)
                    admission = gr.Number(label='Admission grade', info='0–200 scale', value=150, minimum=0, maximum=200)
                    previous = gr.Number(label='Previous qualification (grade)', info='0–200 scale', value=145, minimum=0, maximum=200)
                with gr.Row():
                    tuition = gr.Radio(['Yes', 'No'], label='Tuition fees up to date', value='Yes')
                    debtor = gr.Radio(['Yes', 'No'], label='Debtor', value='No')
                    scholarship = gr.Radio(['Yes', 'No'], label='Scholarship holder', value='Yes')
                gr.Markdown('### 2 · Academic progress')
                with gr.Group():
                    gr.Markdown('**First semester**')
                    with gr.Row():
                        enrolled1 = gr.Number(label='Curricular units 1st sem (enrolled)', value=6, minimum=0, maximum=100, step=1)
                        approved1 = gr.Number(label='Curricular units 1st sem (approved)', value=6, minimum=0, maximum=100, step=1)
                        grade1 = gr.Number(label='Curricular units 1st sem (grade)', info='0–20 scale', value=15, minimum=0, maximum=20)
                with gr.Group():
                    gr.Markdown('**Second semester**')
                    with gr.Row():
                        enrolled2 = gr.Number(label='Curricular units 2nd sem (enrolled)', value=6, minimum=0, maximum=100, step=1)
                        approved2 = gr.Number(label='Curricular units 2nd sem (approved)', value=6, minimum=0, maximum=100, step=1)
                        grade2 = gr.Number(label='Curricular units 2nd sem (grade)', info='0–20 scale', value=15, minimum=0, maximum=20)
                inputs = [age, admission, previous, tuition, debtor, scholarship, enrolled1, approved1, grade1, enrolled2, approved2, grade2]
                button = gr.Button('Predict student risk', variant='primary', elem_id='predict')
            with gr.Column(scale=2, elem_id='result-panel'):
                gr.Markdown('### Prediction summary\nSelect **Predict student risk** to calculate a result.')
                probability = gr.Textbox(label='Dropout probability', value='—', interactive=False, elem_id='probability')
                predicted = gr.Textbox(label='Predicted class', value='—', interactive=False)
                risk = gr.Textbox(label='Risk category', value='—', interactive=False)
                gr.Markdown('**Interpretation bands**\n\n'
                            '- **LOW RISK:** below 30%\n'
                            '- **MEDIUM RISK:** 30% to below 60%\n'
                            '- **HIGH RISK:** 60% or above\n\n'
                            'These are **application-defined interpretation bands**, not thresholds learned by Logistic Regression. '
                            'The predicted class comes directly from the model and may differ from the risk band.')
        button.click(predict_student_risk, inputs, [probability, predicted, risk], api_name='predict')
        with gr.Accordion('Try illustrative student profiles', open=False):
            gr.Examples(examples=EXAMPLES, inputs=inputs, label='Example test cases — not real students', cache_examples=False)
        with gr.Accordion('About this prototype and its limitations', open=False):
            gr.Markdown('This model estimates membership in the dropout class; its probability is not a guarantee. '
                        'Use it to support human review and conversations about academic or financial support. '
                        'Do not use it to automatically determine actions about students.\n\n'
                        'The training cohort excluded students still enrolled. The notebook performed an initial '
                        'standardization before the train/test split, which is a data-leakage limitation of the reported evaluation. '
                        'The app preserves the original model and reproduces its input preprocessing. '
                        'Saved defaults, dataset coverage, and unmeasured fairness/calibration may affect predictions. '
                        'Inputs are processed for inference and are not written to a student database by this app.')
    return demo


demo = build_app()
if __name__ == '__main__':
    demo.launch(server_name=os.getenv('GRADIO_SERVER_NAME', '127.0.0.1'),
                server_port=int(os.getenv('PORT', '7860')), theme=gr.themes.Soft(primary_hue='teal'), css=CSS)
