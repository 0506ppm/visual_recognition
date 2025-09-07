import requests
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Create a custom LSTM class that ignores the time_major parameter
class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Remove time_major if it exists
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# 定義 SelfAttention 層

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(
            name="att_weight", shape=(d, 1),
            initializer="glorot_uniform", trainable=True
        )
        self.b = self.add_weight(
            name="att_bias", shape=(1,),
            initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (batch, time, feat)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=[[2],[0]]) + self.b)  # (b, t, 1)
        if mask is not None:
            mask = tf.cast(mask, dtype=e.dtype)[:, :, tf.newaxis]          # (b, t, 1)
            e = e - (1.0 - mask) * 1e9
        alpha = tf.nn.softmax(e, axis=1)                                   # (b, t, 1)
        context = x * alpha                                                 # (b, t, f)
        return tf.reduce_sum(context, axis=1)                               # (b, f)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        cfg = super().get_config()
        return cfg

# =========================
#  MediaPipe 關鍵點 → 特徵
# =========================

# 手部節點索引：0=WRIST；拇指 1-4；食指 5-8；中指 9-12；無名指 13-16；小指 17-20
# 每手 23 條骨架邊（指鏈 20 + 手掌橫向 3）
EDGES = [
    # 拇指
    (0,1), (1,2), (2,3), (3,4),
    # 食指
    (0,5), (5,6), (6,7), (7,8),
    # 中指
    (0,9), (9,10), (10,11), (11,12),
    # 無名指
    (0,13), (13,14), (14,15), (15,16),
    # 小指
    (0,17), (17,18), (18,19), (19,20),
    # 手掌橫向
    (5,9), (9,13), (13,17),
]

def _hand_xyz_from_results(hand_landmarks):
    """把 MediaPipe hand landmarks 轉為 21x3 numpy；若缺則全 0。"""
    xyz = np.zeros((21, 3), dtype=np.float32)
    if hand_landmarks:
        for i, lm in enumerate(hand_landmarks.landmark[:21]):
            xyz[i] = [lm.x, lm.y, lm.z]
    return xyz

def _safe_dist(p_i, p_j):
    """若任一點缺(全 0)回傳 0，否則回傳歐氏距離。"""
    if p_i.sum() == 0.0 or p_j.sum() == 0.0:
        return 0.0
    diff = p_i - p_j
    return float(np.sqrt((diff * diff).sum()))

def _hand_scale_ref(hand_xyz):
    """尺度正規化：優先用 0-9（腕→中指MCP）；其次 0-{5,9,13,17} 平均；最後 1.0。"""
    eps = 1e-6
    d = _safe_dist(hand_xyz[0], hand_xyz[9])
    if d > 0:
        return max(d, eps)
    alts = [_safe_dist(hand_xyz[0], hand_xyz[j]) for j in (5,9,13,17)]
    alts = [a for a in alts if a > 0]
    return max(float(np.mean(alts)), eps) if alts else 1.0

def compute_hand_edge_distances(hand_xyz):
    """每手 23 維距離（已做尺度正規化）。"""
    scale = _hand_scale_ref(hand_xyz)
    feats = []
    for i, j in EDGES:
        d = _safe_dist(hand_xyz[i], hand_xyz[j]) / scale
        feats.append(d)
    return np.array(feats, dtype=np.float32)  # (23,)

def compute_hand_edge_directions(hand_xyz):
    """（可選）每手 23×3 = 69 維方向向量（已做尺度正規化）。"""
    scale = _hand_scale_ref(hand_xyz)
    vecs = []
    for i, j in EDGES:
        if hand_xyz[i].sum() == 0.0 or hand_xyz[j].sum() == 0.0:
            vecs.extend([0.0, 0.0, 0.0])
        else:
            v = (hand_xyz[j] - hand_xyz[i]) / scale
            vecs.extend(v.tolist())
    return np.array(vecs, dtype=np.float32)  # (69,)

def extract_features_from_mediapipe(results, feat_dim, scaler=None):
    """
    依模型輸入維度產生每幀特徵：
    - 126: 直接左右手座標 (lh+rh) 21*3*2
    - 46 : 左右手骨架邊距離 23*2
    - 138: 左右手骨架邊方向向量 23*3*2
    """
    lh_xyz = _hand_xyz_from_results(results.left_hand_landmarks)
    rh_xyz = _hand_xyz_from_results(results.right_hand_landmarks)

    if feat_dim == 126:
        feat = np.concatenate([lh_xyz.flatten(), rh_xyz.flatten()])
    elif feat_dim == 46:
        feat = np.concatenate([
            compute_hand_edge_distances(lh_xyz),
            compute_hand_edge_distances(rh_xyz)
        ])
    elif feat_dim == 138:
        feat = np.concatenate([
            compute_hand_edge_directions(lh_xyz),
            compute_hand_edge_directions(rh_xyz)
        ])
    else:
        raise ValueError(f"不支援的輸入維度：{feat_dim}")

    if scaler is not None:
        try:
            feat = scaler.transform(feat.reshape(1, -1))[0]
        except Exception:
            pass
    return feat.astype(np.float32)

# 主要的手語辨識函數
def start():
    print("🚀 啟動手語辨識系統...")
    
    from tensorflow.keras.models import load_model
    
    # 初始化變數
    new_model = None
    actions = None
    sequence_length = 60
    
    print("📥 嘗試加載模型...")
    
        # ---------- 載入模型（支援多個候選檔名） ----------
    model_candidates = [
        "./Model/yu2_2da_atten_0907.keras",   # ← 你的新版模型（46維）可放第一順位
        "./Model/yu1_2da_0907.keras",
        "./Model/j1_0907_noise.keras",
        "./Model/j1_0819_noise.keras",
    ]
    new_model = None
    for mpath in model_candidates:
        try:
            new_model = load_model(
                mpath, custom_objects={'SelfAttention': SelfAttention, 'CustomLSTM': CustomLSTM},
                compile=False
            )
            print(f"✅ 成功加載模型：{mpath}")
            break
        except Exception as e:
            print(f"❌ 加載失敗：{mpath} ；{e}")
    
    # 檢查模型是否成功加載
    if new_model is None:
        print("❌ 無法加載任何模型，程式終止")
        return
    
    print("✅ 模型加載成功！")
    new_model.summary()
    
    # ---------- 解析模型輸入/輸出 ----------
    try:
        in_shape = new_model.input.shape  # TensorShape([None, T, F])
        sequence_length = int(in_shape[1]) if in_shape[1] is not None else 60
        feat_dim = int(in_shape[2]) if in_shape[2] is not None else 126
    except Exception as e:
        print(f"⚠️ 無法解析模型輸入形狀：{e}，預設 T=60, F=126")
        sequence_length, feat_dim = 60, 126

    try:
        num_classes = int(new_model.output.shape[-1])
    except Exception:
        num_classes = 12

    print(f"🔍 模型期望序列長度: {sequence_length}")
    print(f"🔍 模型每幀特徵維度: {feat_dim}")
    print(f"🔍 模型輸出類別數: {num_classes}")

    # ---------- 類別名稱（依輸出數量） ----------
    if num_classes == 10:
        actions = np.array(['complete', 'this', 'id_card', 'paper', 'sign',
                            'cover_name', 'various', 'use', 'life', 'want'])
        print("🔍 使用10個動作")  # ← 修正你原本寫成「5個」的筆誤
    elif num_classes == 12:
        actions = np.array(['complete', 'apply_for', 'invest', 'cover_name', 'me',
                            'passbook', 'use', 'various', 'want', 'what', 'id_card', 'paper'])
    elif num_classes == 14:
        actions = np.array(['complete', 'apply_for', 'invest', 'cover_name', 'me',
                            'passbook', 'use', 'various', 'want', 'what', 'id_card', 'paper',
                            'action_13', 'action_14'])
    else:
        print(f"⚠️ 未知類別數 {num_classes}，使用預設12個動作")
        actions = np.array(['complete', 'apply_for', 'invest', 'cover_name', 'me',
                            'passbook', 'use', 'various', 'want', 'what', 'id_card', 'paper'])
    print(f"📋 動作標籤: {{label: idx for idx, label in enumerate(actions)}}")
    print(f"📊 總共 {actions.shape[0]} 個動作")

    # ---------- 可選：載入 scaler（若 46 維距離有做標準化） ----------
    scaler = None
    if feat_dim == 46:
        try:
            import joblib
            scaler_path = "./Model/edges_scaler.joblib"
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"✅ 已載入距離特徵 scaler：{scaler_path}")
        except Exception as e:
            print(f"ℹ️ 未載入 scaler（將以未標準化特徵推論）：{e}")
    
    # 建立標籤映射
    label_map = {label: num for num, label in enumerate(actions)}
    print(f"📋 動作標籤: {label_map}")
    print(f"📊 總共 {actions.shape[0]} 個動作")

    import cv2
    import mediapipe as mp
    from collections import Counter

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # 檢測影像中的關鍵點
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        return results

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

    def prob_viz(res, actions, input_frame, color=(245, 117, 16)):
        """在左上方畫出機率條。"""
        output = input_frame.copy()
        h = 18  # 每個 bar 高度
        pad = 6
        for i, p in enumerate(res):
            y1 = 60 + i * (h + pad)
            y2 = y1 + h
            x2 = int(max(1, min(output.shape[1]-1, p * (output.shape[1] // 3))))
            cv2.rectangle(output, (0, y1), (x2, y2), color, -1)
            cv2.putText(output, f"{actions[i]}: {p:.2f}", (5, y2-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return output

    

    # ---------- 推論參數 ----------
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7
    alarm_set = False
    trans_result = ""
    last_updated_time = time.time()
    colors = [(245, 117, 16)] * len(actions)

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

            # ---- 抽每幀特徵（自動相容 126/46/138）----
            try:
                feats = extract_features_from_mediapipe(results, feat_dim, scaler)
            except Exception as e:
                print(f"⚠️ 特徵擷取錯誤: {e}")
                feats = None

            if feats is not None and np.count_nonzero(feats) > 10:
                sequence.append(feats.astype(np.float32))
                sequence = sequence[-sequence_length:]

            # ---- 達到序列長度就做一次預測 ----
            if len(sequence) == sequence_length:
                try:
                    inp = np.expand_dims(sequence, axis=0)  # (1, T, F)
                    res = new_model.predict(inp, verbose=0)[0]  # (C,)
                    if res[np.argmax(res)] > threshold:
                        predictions.append(np.argmax(res))

                    most_common = Counter(predictions[-10:]).most_common(1)
                    if most_common and most_common[0][0] == np.argmax(res):
                        idx = int(np.argmax(res))
                        if idx < len(actions):
                            if len(sentence) == 0 or actions[idx] != sentence[-1]:
                                sentence.append(actions[idx])
                                sequence = []
                                last_updated_time = time.time()
                                alarm_set = True

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    frame = prob_viz(res, actions, frame, colors[0])

                except Exception as e:
                    print(f"❌ 模型預測時發生錯誤: {e}")

            # ---- 3 秒出一次結果到後端 & 清空 ----
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
                        response = requests.post(url, data=data, timeout=2.0)
                        print(f'---response status---: {response.status_code}')

                    alarm_set = False
                    sequence = []
                    sentence = []
                except Exception as e:
                    print(f'❌ 處理結果時發生錯誤: {e}')
                    alarm_set = False
                    sequence = []
                    sentence = []

            # ---- 下方字幕區 ----
            img = np.zeros((40, 640, 3), dtype='uint8')
            try:
                fontpath = 'NotoSerifCJKtc-Regular.otf'
                font = ImageFont.truetype(fontpath, 20)
                imgPil = Image.fromarray(img)
                draw = ImageDraw.Draw(imgPil)
                draw.text((5, 5), trans_result, fill=(255, 255, 255), font=font)
                img = np.array(imgPil)
            except Exception as e:
                # 若字型載入失敗，退回 cv2 字型
                cv2.putText(img, trans_result, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 1, cv2.LINE_AA)

            if frame.shape[1] != img.shape[1]:
                img = cv2.resize(img, (frame.shape[1], img.shape[0]))
            if frame.dtype != img.dtype:
                img = img.astype(frame.dtype)

            outputframe = cv2.vconcat([frame, img])
            ret, buffer = cv2.imencode('.jpg', outputframe)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # 串流輸出（供 Flask Response 使用）
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # 可選：鍵盤中斷（若有視窗）
            if cv2.waitKey(10) & 0xFF == ord('x'):
                break

        cap.release()
        cv2.destroyAllWindows()