



import requests
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import ImageFont, ImageDraw, Image


# 2. 模組需要的字詞 Labels
def start():
    actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'check', 'finish', 'give_you',
                    'good', 'i', 'id_card', 'is', 'money', 'saving_book', 'sign', 'taiwan', 'take', 'ten_thousand', 'yes'])


    label_map = {label:num for num, label in enumerate(actions)}
    print(label_map)
    print(actions.shape[0])


    def get_dataset(data_path_trans):
        input_texts = []     
        target_texts = []

        input_characters = set()
        target_characters = set()
        with open(data_path_trans, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines:
            input_text, target_text= line.split('   ')
            # 用tab作用序列的开始，用\n作为序列的结束
            target_text = '\t' + target_text + '\n'

            input_texts.append(input_text)
            target_texts.append(target_text)
            
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        return input_texts,target_texts,input_characters,target_characters


    # data_path_trans = r'C:\Users\heng\Desktop\visual_recognition\Model\EngToChinese_service1_1029.txt'
    data_path_trans = r"./Model/EngToChinese_service1_1029.txt"
    input_texts,target_texts,input_characters,target_characters = get_dataset(data_path_trans)


    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])


    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())


    from tensorflow.keras.models import load_model


    new_model = load_model('./Model/model1_service1_0924.keras')
    new_model_order = load_model("./Model/model2_service1_1029.h5")
    new_model.summary()


    import cv2
    import mediapipe as mp
    from collections import Counter

    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities


    colors = [(245,117,16)] * 24

    # 在影像中繪製模型預測的機率分布條
    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):

            cv2.rectangle(output_frame, (0,60+num*17), (int(prob*100), 90+num*17), colors[num], -1)
        return output_frame


    # 檢測影像中的關鍵點
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Mediapipe 模型只接受 RGB 格式影像，因此需要轉換
        image.flags.writeable = False # 禁止寫入，加速處理
        results = model.process(image) # 返回檢測結果，包括手部、姿態等關鍵點資訊。
        return results

    # 在影像上繪製姿態、左手和右手的關鍵點連接線。
    def draw_styled_landmarks(image, results):
        # Draw pose connections 繪製人體姿態的骨架
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
        # Draw left hand connections 繪製左手的關鍵點與連接線
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        ) 
        # Draw right hand connections 繪製右手的關鍵點與連接線
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        ) 

 # 提取關鍵點座標，僅包括左手和右手（不包括臉部和姿態）
    def extract_keypoints_without_face(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh]) 

    def translate(model_opt):
        # 用零初始化編碼器輸入，形狀是 (1, max_encoder_seq_length, num_encoder_tokens)
        in_encoder = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')

        # 將 model_opt（輸入句子）轉換為 one-hot 編碼，再重新加到 in_encoder
        for t, char in enumerate(model_opt):
            in_encoder[0, t, input_token_index[char]] = 1. # 在對應字符位置填入1，表示該字符
        # 填充剩餘位置為空格字符的 one-hot 編碼
        in_encoder[0, t + 1:, input_token_index[' ']] = 1.

        # 用零初始化解碼器輸入，形狀是 (batch_size, max_decoder_seq_length, num_decoder_tokens)
        in_decoder = np.zeros((len(in_encoder), max_decoder_seq_length, num_decoder_tokens),dtype='float32')
        # 解碼器的第一個輸入是開始符號（"\t"）
        in_decoder[:, 0, target_token_index["\t"]] = 1

        # 生成 decoder 的 output # 預測每個解碼器時間步的輸出並將其作為下一時間步的輸入
        for i in range(max_decoder_seq_length - 1):
            predict = new_model_order.predict([in_encoder, in_decoder])
            predict = predict.argmax(axis=-1)# 從預測結果中選擇概率最大的 token
            predict_ = predict[:, i].ravel().tolist()
            for j, x in enumerate(predict_):
                in_decoder[j, i + 1, x] = 1 # 將每個預測出的 token 設為 decoder 下一個 timestamp 的輸入
        
        seq_index = 0 # 初始化解碼結果
        decoded_sentence = "" # 解碼後的句子
         # 選擇預測的序列中的第一個樣本（通常 batch_size=1）
        output_seq = predict[seq_index, :].ravel().tolist()

        # 把token轉換為字符
        for x in output_seq:
            if reverse_target_char_index[x] == "\n":
                break
            else:
                decoded_sentence+=reverse_target_char_index[x] #將解碼的字符拼接成句子

        return decoded_sentence

    sequence = [] # 儲存最近 30 幀的關鍵點序列，作為模型輸入
    sentence = [] # 儲存當前辨識出的完整手語句子
    predictions = [] # 儲存最近模型預測的結果索引
    threshold = 0.7 # 模型預測的機率閾值，低於此值的結果會被忽略
    alarm_set = False # 標誌是否需要觸發翻譯（防止過於頻繁的翻譯）
    trans_result = "" # 儲存翻譯後的結果

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: # 設定 MediaPipe 的 Model
        while cap.isOpened():
            ret, frame = cap.read() # cap.read(): 從攝像頭捕捉一個影像幀，返回幀的狀態 (ret) 和數據 (frame)
            if not ret:
                break # 如果捕捉失敗 (ret=False)，退出循環

            results = mediapipe_detection(frame, holistic) # 調用 mediapipe_detection 函數，對當前幀進行檢測，返回檢測結果 (results)
            draw_styled_landmarks(frame, results) # 在影像上繪製姿態、左手和右手的關鍵點連接線。
            
            keypoints = extract_keypoints_without_face(results) # 提取關鍵點座標，僅包括左手和右手（不包括臉部和姿態）
            if np.count_nonzero(keypoints) > 30:
                sequence.append(keypoints) # 如果有效關鍵點數量超過 30，將其添加到 sequence 中。
                sequence = sequence[-30:] # 僅保留最近 30 幀的數據。
            
            if len(sequence) == 30:
                res = new_model.predict(np.expand_dims(sequence, axis=0))[0] # 當 sequence 的長度達到 30（滿足模型輸入要求），對序列進行推論，返回機率分布 (res)
                if res[np.argmax(res)] > threshold: 
                    predictions.append(np.argmax(res)) # 如果最高機率的分類超過 threshold，將其索引添加到 predictions

                
            #3. Viz logic
                if Counter(predictions[-10:]).most_common(1)[0][0]==np.argmax(res): # 使用 Counter 計算最近 10 個預測中出現次數最多的結果，與當前結果比較。  
                    if len(sentence) > 0: # 如果 sentence 已有內容
                        if actions[np.argmax(res)] != sentence[-1]: # 且當前預測的動作與最後一個動作不同
                            sentence.append(actions[np.argmax(res)]) # 則添加新的動作到 Sentence。
                            sequence = [] # 則清空 sequence
                            last_updated_time = time.time() # 記錄當前時間作為 last_updated_time
                            alarm_set = True # 表示翻譯觸發條件已滿足
                    else:
                        sentence.append(actions[np.argmax(res)]) # 如果 sentence 為空
                        sequence = [] # 則清空 sequence
                        last_updated_time = time.time() 
                        alarm_set = True

                if len(sentence) > 5: 
                    sentence = sentence[-5:] # 把 sentence 長度限制在 5，僅保留最近的 5 個動作

                # Viz probabilities
                frame = prob_viz(res, actions, frame, colors) # 將模型的預測機率 (res) 視覺化，繪製條形圖到影像 frame 上
                
            current_time = time.time()  
            if alarm_set and current_time - last_updated_time >= 10: # 「需要翻譯」且「時間過10秒」，將 sentence 放入下一個模型進行預測
                trans_result = translate(' '.join(sentence)) # 將 sentence 拼接成一個句子，傳入翻譯函數 translate，生成翻譯結果 trans_result
                print('---result---', trans_result)

                url = 'http://localhost:5000/handlanRes'
                data = {'result': trans_result}  # 將 trans_result 作為結果放入字典中
                response = requests.post(url, data=data) # 將結果發送到後端的 /handlanRes 路由，作為字典 data 的 POST 請求
                return data # Problem: 會導致 start 函數終止

                alarm_set = False # 設為不需翻譯，避免重複翻譯
                sequence = [] # 清空 sequence 資料
                sentence = [] # 清空 sentence 資料
                
            img = np.zeros((40,640,3), dtype='uint8') # 建立一個全黑的空白影像，用於顯示翻譯結果文字。
            fontpath = 'NotoSerifCJKtc-Regular.otf' 
            font = ImageFont.truetype(fontpath, 20) # 加載字型文件並設置字型大小
            imgPil = Image.fromarray(img) # 將 NumPy 陣列 img 轉換為 PIL (Python Imaging Library) 格式的影像
            draw = ImageDraw.Draw(imgPil) # 使用 ImageDraw 在影像上繪製翻譯結果文字
            draw.text((0, 0), trans_result, fill=(255, 255, 255), font=font) # 參數：(文字繪製的起始座標，左上角為 (0, 0), 要顯示的翻譯結果, 文字顏色為白色 (RGB), 指定的字型對象)
            img = np.array(imgPil) # 將繪製了文字的 PIL 影像轉回 NumPy 陣列格式，因為 OpenCV 的後續處理需要 NumPy 格式的影像，img 現在是一個包含翻譯結果文字的 NumPy 陣列影像
            
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1) # 在主影像 frame 的頂部繪製一個橙色矩形背景。
            cv2.putText(frame, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # 在影像的矩形背景上顯示當前的 sentence（辨識出的手語動作）。
            if frame.shape[1] != img.shape[1]: # 確保 frame 和 img 的寬度一致
                # Resize img to have the same number of columns as frame
                img = cv2.resize(img, (frame.shape[1], img.shape[0]))

            # Check if they have the same type
            if frame.dtype != img.dtype:
                # Convert img to the same type as frame
                img = img.astype(frame.dtype)
            outputframe = cv2.vconcat([frame, img])
            # cv2.imshow('SignLanguage', outputframe)
            ret, buffer = cv2.imencode('.jpg', outputframe)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # 這段程式碼的作用是提供用戶一個結束程序的機會（按下 x），然後釋放攝像頭和圖形資源，確保程序正常退出並避免資源洩露。
            if cv2.waitKey(10) & 0xFF == ord('x'):
                break
        cap.release()
        cv2.destroyAllWindows()
    

