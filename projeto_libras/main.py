import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

# Inicialização da captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Inicialização do módulo de detecção de mãos do MediaPipe, limitando a detecção a uma única mão
hands = mp.solutions.hands.Hands(max_num_hands=1)

# Definição das classes que o modelo irá prever
classes = ['A','B','C','D']

# Carregamento do modelo previamente treinado
model = load_model('keras_model.h5')

# Inicialização de um array numpy para armazenar os dados de entrada do modelo
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Loop principal
while True:
    # Captura de um quadro da webcam
    success, img = cap.read()
    
    # Conversão do espaço de cores do quadro de BGR para RGB
    frameRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    # Processamento do quadro para detectar mãos
    results = hands.process(frameRGB)
    
    # Extração dos pontos de referência das mãos detectadas
    handsPoints = results.multi_hand_landmarks
    
    # Obtendo altura e largura do quadro
    h, w, _ = img.shape

    # Verificação se há mãos detectadas
    if handsPoints != None:
        for hand in handsPoints:
            # Inicialização de variáveis para encontrar os limites da mão detectada
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                # Atualização dos limites da mão detectada
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            # Desenho de um retângulo ao redor da mão detectada
            cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

                # Pre-Processamento
            try:
                # Recorte da região da mão detectada
                imgCrop = img[y_min-50:y_max+50,x_min-50:x_max+50]
                # Redimensionamento da imagem recortada para o tamanho esperado pelo modelo
                imgCrop = cv2.resize(imgCrop,(224,224))
                # Conversão da imagem recortada em um array numpy
                imgArray = np.asarray(imgCrop)
                # Normalização dos valores dos pixels da imagem
                normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                # Atualização dos dados de entrada do modelo
                data[0] = normalized_image_array
                # Realização da predição utilizando o modelo
                prediction = model.predict(data)
                # Obtenção do índice da classe com maior probabilidade
                indexVal = np.argmax(prediction)
                # Adição da etiqueta da classe à imagem
                cv2.putText(img,classes[indexVal],(x_min-50,y_min-65),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)

            except:
                continue
                # Pre-Processamento

    # Exibição do quadro com as marcações
    cv2.imshow('Imagem',img)
    # Aguardo pela tecla 'q' ser pressionada para encerrar o programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberação dos recursos utilizados
cap.release()
cv2.destroyAllWindows()
