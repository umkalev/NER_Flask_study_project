from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Загружаем модель NER
model_name = "Gherman/bert-base-NER-Russian"
ner_pipeline = pipeline("token-classification", model=model_name, grouped_entities=True)

@app.route('/', methods=['GET', 'POST'])
def ner():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if not user_input:
            return "Пожалуйста, введите текст."

        # Используем модель для распознавания сущностей
        try:
            entities = ner_pipeline(user_input)
            result = format_entities(entities, user_input)
        except Exception as e:
            result = f"Ошибка при обработке текста: {e}"

        return render_template('ner.html', user_input=user_input, result=result)

    return render_template('ner.html')

def format_entities(entities, text):
    
    #Форматируем в читаемый вид
    
    formatted_result = []
    for entity in entities:
        formatted_result.append(
            f"Сущность: {entity['word']}, Тип: {entity['entity_group']}, Позиция: {entity['start']}-{entity['end']}"
        )
    return "\n".join(formatted_result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)