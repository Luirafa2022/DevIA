import os
import sys
import nltk
from nltk.chat.util import Chat, reflections
from fuzzywuzzy import process
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QListWidget, QListWidgetItem
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QFont, QIcon
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Downloada os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Diretório contendo as pastas e arquivos da documentação
documentation_dir = 'Documentacao'

# Carrega a documentação da linguagem de programação a partir dos arquivos de texto
def load_documentation_text():
    documentation_text = ""
    for root, dirs, files in os.walk(documentation_dir):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documentation_text += content + "\n"
    return documentation_text

# Pré-processa o texto removendo pontuação, stopwords e tornando minúsculo
def preprocess_text(text):
    stop_words = set(stopwords.words('portuguese') + list(punctuation))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Gera uma resposta com base nas informações encontradas na documentação
def generate_response(query, documentation_text):
    preprocessed_query = preprocess_text(query)
    sentences = sent_tokenize(documentation_text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(preprocessed_sentences)
    query_vector = vectorizer.transform([preprocessed_query])
    
    similarities = cosine_similarity(query_vector, sentence_vectors)
    most_similar_sentence_index = similarities.argmax()
    
    return sentences[most_similar_sentence_index]

# Função para encontrar a resposta mais próxima usando fuzzy string matching
def find_closest_match(user_input, patterns):
    closest_match = process.extractOne(user_input, patterns)
    return closest_match[0] if closest_match[1] >= 70 else None

# Pares de padrões e respostas para o chatbot
pairs = [
    [
        r"oi|olá|eae|hey|bom dia|boa tarde|boa noite",
        ["Olá! Como posso ajudar com programação hoje?",
         "Oi! Em que posso ser útil?",
         "Eae! Como posso te ajudar a programar melhor?"]
    ],
    [
        r"tchau|até mais|até logo|xau|adeus",
        ["Até mais! Foi um prazer ajudar.",
         "Tchau! Espero ter sido útil. Volte sempre!",
         "Até logo! Fico à disposição para futuras dúvidas de programação."]
    ]
]

# Cria o chatbot
chatbot = Chat(pairs, reflections)

# Classe da janela principal
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DevIA - IA de Programação")
        self.setWindowIcon(QIcon('LogoDevia.png'))
        self.setGeometry(100, 100, 800, 600)
        self.showMaximized()
        
        # Aplicar estilo de modo claro
        self.setStyleSheet("""
            QWidget {
                background-color: #f6f6f6;
                color: #121212;
            }
            QLineEdit {
                background-color: #ffffff;
                color: #121212;
                border: 1px solid #949494;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton {
                background-color: #e2e2e2;
                color: #121212;
                border: 1px solid #949494;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #949494;
            }
            QListWidget {
                background-color: #ffffff;
                color: #121212;
                border: 1px solid #949494;
                border-radius: 10px;
                padding: 10px;
            }
            QListWidget::item {
                background-color: #f0f0f0;
                border-radius: 10px;
                padding: 5px;
                margin-bottom: 5px;
            }
            QListWidget::item:selected {
                background-color: #949494;
            }
        """)
        
        layout = QVBoxLayout()
        
        self.conversation_list = QListWidget()
        self.conversation_list.setAlternatingRowColors(False)
        self.conversation_list.setWordWrap(True)
        self.conversation_list.setSpacing(5)
        layout.addWidget(self.conversation_list)
        
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.user_input)
        
        self.send_button = QPushButton("Enviar")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setFixedSize(70, 33)
        input_layout.addWidget(self.send_button)
        
        self.copy_button = QPushButton("Copiar")
        self.copy_button.clicked.connect(self.copy_code)
        self.copy_button.setFixedSize(70, 33)
        input_layout.addWidget(self.copy_button)
        
        layout.addLayout(input_layout)
        
        self.setLayout(layout)
    
    def send_message(self):
        user_input = self.user_input.text().strip()
        if user_input:
            user_item = QListWidgetItem(f"Você: {user_input}")
            user_item.setTextAlignment(Qt.AlignRight)
            self.conversation_list.addItem(user_item)
            
            closest_match = find_closest_match(user_input, [pair[0] for pair in pairs])
            if closest_match:
                response = chatbot.respond(closest_match)
            else:
                response = generate_response(user_input, documentation_text)
            
            ai_item = QListWidgetItem()
            ai_item.setTextAlignment(Qt.AlignLeft)
            self.conversation_list.addItem(ai_item)
            
            self.user_input.clear()
            
            self.write_chatbot_response(ai_item, response)
    
    def write_chatbot_response(self, item, response):
        chars = list(response)
        text = ""
        
        def write_char():
            nonlocal text
            if chars:
                char = chars.pop(0)
                text += char
                item.setText(f"DevIA: {text}")
            else:
                timer.stop()
        
        timer = QTimer(self)
        timer.timeout.connect(write_char)
        timer.start(50)
    
    def copy_code(self):
        selected_items = self.conversation_list.selectedItems()
        if selected_items:
            selected_text = "\n".join([item.text() for item in selected_items])
            QApplication.clipboard().setText(selected_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    font = QFont()
    font.setPointSize(12)
    app.setFont(font)
    
    documentation_text = load_documentation_text()
    
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())