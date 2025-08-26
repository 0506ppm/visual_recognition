import requests
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf

# Create a custom LSTM class that ignores the time_major parameter
class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Remove time_major if it exists
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# 定義 SelfAttention 層
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # 正確的 attention 機制實現
        self.W_q = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='query_weight'
        )
        self.W_k = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='key_weight'
        )
        self.W_v = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='value_weight'
        )
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        # 計算 query, key, value
        query = tf.matmul(inputs, self.W_q)
        key = tf.matmul(inputs, self.W_k)
        value = tf.matmul(inputs, self.W_v)
        
        # 計算 attention scores
        attention_scores = tf.matmul(query, key, transpose_b=True)
        
        # 縮放
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        attention_scores = attention_scores / tf.sqrt(d_k)
        
        # 應用 softmax
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # 計算 context
        context = tf.matmul(attention_weights, value)
        
        return context
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        return config

# 主要的手語辨識函數
def start():
    print("🚀 啟動手語辨識系統...")
    
    from tensorflow.keras.models import load_model
    
    # 初始化變數
    new_model = None
    actions = None
    sequence_length = 60
    
    print("📥 嘗試加載模型...")
    
    # 加載模型
    try:
        new_model = load_model("./Model/yu2_3da_atten_0820.keras", 
                              custom_objects={'SelfAttention': SelfAttention}, 
                              compile=False)
        print("✅ 成功加載主模型 yu2_3da_atten_0820.keras")
    except Exception as e:
        print(f"❌ 加載主模型失敗: {e}")
        try:
            new_model = load_model("./Model/j1_0819_noise.keras", compile=False)
            print("✅ 成功加載備用模型 j1_0819_noise.keras")
        except Exception as e2:
            print(f"❌ 加載備用模型失敗: {e2}")
            try:
                new_model = load_model("./Model/j1_noise.keras", compile=False)
                print("✅ 成功加載舊備用模型 j1_noise.keras")
            except Exception as e3:
                print(f"❌ 所有模型加載都失敗: {e3}")
                return
    
    # 檢查模型是否成功加載
    if new_model is None:
        print("❌ 無法加載任何模型，程式終止")
        return
    
    print("✅ 模型加載成功！")
    new_model.summary()
    
    # 分析模型結構
    try:
        model_input_shape = new_model.input.shape
        sequence_length = model_input_shape[1] if model_input_shape[1] is not None else 60
        print(f"🔍 模型期望序列長度: {sequence_length}")
        
        model_output_shape = new_model.output.shape
        num_classes = model_output_shape[-1]
        print(f"🔍 模型輸出類別數: {num_classes}")
        
        # 根據模型輸出設定動作
        if num_classes == 5:
            actions = np.array(['apply_for', 'invest', 'me', 'passbook', 'what'])
            print("🔍 使用5個動作")
        elif num_classes == 12:
            actions = np.array(['complete', 'apply_for', 'invest', 'cover_name', 'me', 'passbook', 'use', 'various', 'want', 'what', 'id_card', 'paper'])
            print("🔍 使用12個動作")
        elif num_classes == 14:
            actions = np.array(['complete', 'apply_for', 'invest', 'cover_name', 'me', 'passbook', 'use', 'various', 'want', 'what', 'id_card', 'paper', 'action_13', 'action_14'])
            print("🔍 使用14個動作（部分名稱需確認）")
        else:
            actions = np.array(['complete', 'apply_for', 'invest', 'cover_name', 'me', 'passbook', 'use', 'various', 'want', 'what', 'id_card', 'paper'])
            print(f"⚠️ 未知類別數 {num_classes}，使用預設12個動作")
    except Exception as e:
        print(f"❌ 分析模型時發生錯誤: {e}")
        sequence_length = 60
        actions = np.array(['complete', 'apply_for', 'invest', 'cover_name', 'me', 'passbook', 'use', 'various', 'want', 'what', 'id_card', 'paper'])
        print("🔍 使用預設設置：60幀，12個動作")
    
    # 建立標籤映射
    label_map = {label: num for num, label in enumerate(actions)}
    print(f"📋 動作標籤: {label_map}")
    print(f"📊 總共 {actions.shape[0]} 個動作")

    import cv2
    import mediapipe as mp
    from collections import Counter

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    colors = [(245, 117, 16)] * len(actions)

    # 在影像中繪製模型預測的機率分布條
    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            if num < len(colors):  # 防止索引超出範圍
                cv2.rectangle(output_frame, (0, 60 + num * 17), (int(prob * 100), 90 + num * 17), colors[num], -1)
        return output_frame

    # 檢測影像中的關鍵點
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        return results

    # 繪製關鍵點
    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        ) 
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        ) 

    # 提取關鍵點座標
    def extract_keypoints_without_face(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh]) 

    # 初始化變數
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7
    alarm_set = False
    trans_result = ""
    last_updated_time = time.time()

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 錯誤：無法打開攝影機！")
        return

    print("🎥 攝影機啟動成功，開始手語辨識...")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(frame, results)
            
            keypoints = extract_keypoints_without_face(results)
            if np.count_nonzero(keypoints) > 30:
                sequence.append(keypoints)
                sequence = sequence[-sequence_length:]
            
            if len(sequence) == sequence_length:
                try:
                    res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
                    if res[np.argmax(res)] > threshold: 
                        predictions.append(np.argmax(res))

                    # 辨識邏輯
                    try:
                        most_common_predictions = Counter(predictions[-10:]).most_common(1)
                        if most_common_predictions and most_common_predictions[0][0] == np.argmax(res):
                            predicted_action_index = np.argmax(res)
                            if predicted_action_index < len(actions):  # 確保索引在範圍內
                                if len(sentence) > 0:
                                    if actions[predicted_action_index] != sentence[-1]:
                                        sentence.append(actions[predicted_action_index])
                                        sequence = []
                                        last_updated_time = time.time()
                                        alarm_set = True
                                else:
                                    sentence.append(actions[predicted_action_index])
                                    sequence = []
                                    last_updated_time = time.time()
                                    alarm_set = True
                    except (IndexError, ValueError) as e:
                        print(f"⚠️ 預測處理警告: {e}")
                        pass

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # 視覺化機率
                    frame = prob_viz(res, actions, frame, colors)
                except Exception as e:
                    print(f"❌ 模型預測時發生錯誤: {e}")
                    continue
                
            current_time = time.time()  
            if alarm_set and current_time - last_updated_time >= 3:
                try:
                    sentence_text = ' '.join(sentence)
                    print(f'---sentence---: {sentence}')
                    print(f'---sentence_text---: {sentence_text}')
                    
                    trans_result = sentence_text
                    print(f'---final result---: {trans_result}')

                    if trans_result:
                        url = 'http://localhost:5000/handlanRes'
                        data = {'result': trans_result}
                        response = requests.post(url, data=data)
                        print(f'---response status---: {response.status_code}')
                    
                    alarm_set = False
                    sequence = []
                    sentence = []
                except Exception as e:
                    print(f'❌ 處理結果時發生錯誤: {e}')
                    alarm_set = False
                    sequence = []
                    sentence = []
                
            img = np.zeros((40, 640, 3), dtype='uint8')
            
            try:
                fontpath = 'NotoSerifCJKtc-Regular.otf' 
                font = ImageFont.truetype(fontpath, 20)
                imgPil = Image.fromarray(img)
                draw = ImageDraw.Draw(imgPil)
                draw.text((0, 0), trans_result, fill=(255, 255, 255), font=font)
                img = np.array(imgPil)
            except Exception as e:
                print(f"⚠️ 字型加載警告: {e}")
                cv2.putText(img, trans_result, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (3, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if frame.shape[1] != img.shape[1]:
                img = cv2.resize(img, (frame.shape[1], img.shape[0]))

            if frame.dtype != img.dtype:
                img = img.astype(frame.dtype)
            
            outputframe = cv2.vconcat([frame, img])
            ret, buffer = cv2.imencode('.jpg', outputframe)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(10) & 0xFF == ord('x'):
                break
        
        cap.release()
        cv2.destroyAllWindows()