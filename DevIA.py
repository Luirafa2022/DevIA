import os
import sys
import nltk
from nltk.chat.util import Chat, reflections
from fuzzywuzzy import process
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QListWidget, QListWidgetItem, QMessageBox, QMenu, QAction, QInputDialog, QFileDialog
from PySide2.QtCore import Qt, QTimer, QSettings, QRegExp
from PySide2.QtGui import QFont, QIcon, QSyntaxHighlighter, QTextCharFormat, QColor
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

# Classe para destacar a sintaxe do código
class CodeHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []
        
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#569CD6"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = ["def", "if", "else", "elif", "for", "while", "try", "except", "finally", "with", "import", "from", "as", "class", "return", "pass", "break", "continue"]
        for keyword in keywords:
            pattern = r"\b" + keyword + r"\b"
            rule = (QRegExp(pattern), keyword_format)
            self.highlighting_rules.append(rule)
        
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6A9955"))
        self.highlighting_rules.append((QRegExp(r"#.*"), comment_format))
        
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))
        self.highlighting_rules.append((QRegExp(r"\".*\""), string_format))
        self.highlighting_rules.append((QRegExp(r"'.*'"), string_format))
    
    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

    def highlight_code(self, item):
        text_widget = self.conversation_list.itemWidget(item)
        if text_widget is None:
            text_widget = QTextEdit()
            text_widget.setReadOnly(True)
            text_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.conversation_list.setItemWidget(item, text_widget)
        text_widget.setText(item.text())
        CodeHighlighter(text_widget.document())

# Classe da janela principal
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DevIA - IA de Programação")
        self.setWindowIcon(QIcon('LogoDevia.png'))
        self.setGeometry(100, 100, 800, 600)
        self.showMaximized()
        
        self.settings = QSettings("DevIA", "DevIA")
        self.load_settings()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
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
        
        self.update_font_size()  # Aplica o tamanho da fonte inicial após a criação de todos os widgets
        
        self.setLayout(layout)
        
        self.create_menu()
    
    def send_message(self):
        user_input = self.user_input.text().strip()
        if user_input:
            user_message = f"Você: {user_input}"
            user_item = QListWidgetItem(user_message)
            self.conversation_list.addItem(user_item)
            
            closest_match = find_closest_match(user_input, [pair[0] for pair in pairs])
            if closest_match:
                response = chatbot.respond(closest_match)
            else:
                try:
                    response = generate_response(user_input, documentation_text)
                except Exception as e:
                    response = f"Desculpe, ocorreu um erro ao gerar a resposta: {str(e)}"
            
            ai_message = f"DevIA: {response}"
            ai_item = QListWidgetItem(ai_message)
            self.conversation_list.addItem(ai_item)
            
            if "```" in response:
                self.highlight_code(ai_item)
            
            self.user_input.clear()
    
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
    
    def create_menu(self):
        menu_bar = self.menuBar()
        
        appearance_menu = menu_bar.addMenu("Aparência")
        dark_mode_action = QAction("Modo Escuro", self, checkable=True)
        dark_mode_action.triggered.connect(self.toggle_dark_mode)
        appearance_menu.addAction(dark_mode_action)
        
        settings_menu = menu_bar.addMenu("Configurações")
        typing_speed_action = QAction("Velocidade de Digitação", self)
        typing_speed_action.triggered.connect(self.change_typing_speed)
        settings_menu.addAction(typing_speed_action)
        
        font_size_action = QAction("Tamanho da Fonte", self)
        font_size_action.triggered.connect(self.change_font_size)
        settings_menu.addAction(font_size_action)
    
    def toggle_dark_mode(self, checked):
        if checked:
            self.apply_dark_mode()
        else:
            self.apply_light_mode()
        self.save_settings()
    
    def apply_dark_mode(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QListWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 10px;
                padding: 10px;
            }
            QListWidget::item {
                background-color: #333333;
                border-radius: 10px;
                padding: 5px;
                margin-bottom: 5px;
            }
            QListWidget::item:selected {
                background-color: #444444;
            }
        """)
    
    def apply_light_mode(self):
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
    
    def change_typing_speed(self):
        speed, ok = QInputDialog.getInt(self, "Velocidade de Digitação", "Digite a velocidade de digitação (em ms):", value=self.settings.value("typing_speed", 50, int), min=10, max=1000, step=10)
        if ok:
            self.settings.setValue("typing_speed", speed)
    
    def change_font_size(self):
        font_size, ok = QInputDialog.getInt(self, "Tamanho da Fonte", "Digite o tamanho da fonte:", self.settings.value("font_size", 12, int), 6, 24, 1)
        if ok:
            self.settings.setValue("font_size", font_size)
            self.update_font_size()  # Chama o método para atualizar o tamanho da fonte imediatamente
            self.save_settings()

    def update_font_size(self):
        font_size = self.settings.value("font_size", 12, int)
        font = self.conversation_list.font()
        font.setPointSize(font_size)
        self.conversation_list.setFont(font)
        self.user_input.setFont(font)
    
    def load_settings(self):
        dark_mode = self.settings.value("dark_mode", False, bool)
        if dark_mode:
            self.apply_dark_mode()
        else:
            self.apply_light_mode()
        
        typing_speed = self.settings.value("typing_speed", 50, int)
        font_size = self.settings.value("font_size", 12, int)
        font = QFont()
        font.setPointSize(font_size)
        QApplication.setFont(font)
    
    def save_settings(self):
        self.settings.setValue("dark_mode", self.styleSheet() == self.apply_dark_mode.__doc__)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    documentation_text = load_documentation_text()
    
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())