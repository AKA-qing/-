import csv
import os
from flask import Response
from flask import Flask, render_template, request, redirect, url_for, session, flash,  jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from tool import get_user_from_csv, save_user_to_csv
from datetime import datetime
import json  # ✅ 加上这句即可使用 json 模块
import torch
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v4.0/chat'
SPARKAI_APP_ID = 'efe2fd9d'
SPARKAI_API_SECRET = 'ODI5NmI2YTcyNTQ2MzZhYzBmZmE3Nzk0'
SPARKAI_API_KEY = 'a484b5d4d03d92a003affaf4ac2a520b'
SPARKAI_DOMAIN = '4.0Ultra'
spark = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=False,
)

app = Flask(__name__)
app.secret_key = '123'  # 用于加密 session 数据
USER_CSV_FILE = 'users.csv'

# 初始化用户文件
if not os.path.exists(USER_CSV_FILE):
    with open(USER_CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['email', 'password'])  # 写入标题行

# 定义 before_request 钩子
@app.before_request
def require_login():
    allowed_routes = ['login', 'register']  # 不需要登录验证的路由
    if request.endpoint not in allowed_routes and 'user' not in session:
        flash('请先登录以访问此页面。', 'info')
        return redirect(url_for('login'))

@app.route('/')
def home():
    if 'user' in session:
        # 检查是否已经显示过欢迎消息
        if 'welcome_shown' not in session:
            flash(f'欢迎回来, {session["user"]}!', 'success')
            session['welcome_shown'] = True  # 设置标志位，表示已经显示过欢迎消息
    return render_template('index.html')

@app.route('/psychologyknowledge')
def know():
    return render_template('psychologyknowledge.html')

DIARY_FILE = "diary.json"

# 读取全部日记
def load_diary_entries():
    try:
        with open(DIARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# 保存单条日记
def save_entry(entry):
    data = load_diary_entries()
    data.insert(0, entry)  # 最新日记在最前
    with open(DIARY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 覆盖保存所有日记（如用于删除后）
def save_all_entries(entries):
    with open(DIARY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

from TTS.api import TTS
from TTS.utils.radam import RAdam
from collections import defaultdict
from torch.serialization import add_safe_globals
add_safe_globals([RAdam, defaultdict, dict])

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)

def synthesize_speech(text, output_path, emotion="neutral"):
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        emotion=emotion
    )

@app.route("/api/diary_ai", methods=["POST"])
def diary_ai():
    from ollama import chat
    from datetime import datetime

    data = request.get_json()
    text = data.get("text") or f"{data.get('mood', '')} {data.get('content', '')}"

    # 构建温柔 prompt
    prompt = f"你是一位温柔、耐心的心理陪伴者。请安慰并理解下面的情绪内容，用温柔的话语回应：\n\n{text}"

    # LLaMA 回应（本地）
    result = chat(model="llama3.1:latest", messages=[{"role": "user", "content": prompt}])
    ai_text = result["message"]["content"]

    # 合成语音
    output_filename = f"ai_diary_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    output_path = os.path.join("static", "audio", output_filename)
    synthesize_speech(ai_text, output_path)

    return jsonify({
        "text": ai_text,
        "audio_url": url_for('static', filename=f"audio/{output_filename}")
    })
@app.route("/save_diary", methods=["POST"])
def save_diary():
    mood = request.form.get("mood")
    if mood == "custom":
        mood = request.form.get("custom_mood", "未命名心情")
    content = request.form["content"]
    date = request.form.get("date") or datetime.now().strftime("%Y-%m-%d %H:%M")  # 支持自定义日期
    save_entry({"date": date, "mood": mood, "content": content})
    return redirect(url_for("diary"))
@app.route("/delete_diary/<int:entry_id>", methods=["POST"])
def delete_diary(entry_id):
    entries = load_diary_entries()
    if 0 <= entry_id < len(entries):
        entries.pop(entry_id)
        save_all_entries(entries)
    return redirect(url_for("diary"))
@app.route("/diary")
def diary():
    entries = load_diary_entries()
    return render_template("diary.html", entries=entries)

import random
def random_color():
    colors = ["#FFD54F", "#FFAB91", "#A5D6A7", "#81D4FA", "#CE93D8", "#FFF176", "#90CAF9"]
    return random.choice(colors)
@app.route("/api/diary_events")
def diary_events():
    entries = load_diary_entries()
    events = [
        {
            "title": entry["mood"],
            "start": entry["date"].split(" ")[0],
            "id": i,
            "backgroundColor": random_color(),  # 💛 添加随机背景色
            "borderColor": "#f0f0f0",
            "textColor": "#333"
        }
        for i, entry in enumerate(entries)
    ]
    return jsonify(events)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if get_user_from_csv(email):
            flash('邮箱已被注册，请登录。', 'warning')
            return redirect(url_for('login'))
        elif password != confirm_password:
            flash('两次输入的密码不一致，请重试。', 'danger')
        else:
            save_user_to_csv(email, password)
            flash('注册成功，请登录。', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = get_user_from_csv(email)

        if user and check_password_hash(user['password'], password):
            session['user'] = email
            flash('登录成功！', 'success')
            return redirect(url_for('home'))
        else:
            flash('邮箱或密码错误，请重试。', 'danger')

    return render_template('login.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/logout')
def logout():
    session.pop('user', None)  # 清除用户登录状态
    session.pop('welcome_shown', None)  # 清除欢迎消息标志位
    flash('您已成功退出登录。', 'info')
    return redirect(url_for('home'))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        # 在这里处理留言信息，例如保存到数据库或发送到邮箱
        flash('您的留言已成功发送！我们会尽快与您联系。', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if request.method == 'POST':
        return redirect(url_for('quiz_result'))
    return render_template('quiz.html')

# 测评结果页面
@app.route('/quiz_result', methods=['POST'])
def quiz_result():
    try:
        # 获取问题答案并计算总分
        answers = [int(request.form[f'q{i}']) for i in range(1, 11)]
        total_score = sum(answers)

        if total_score >= 25:
            evaluation = "您的心理状态非常健康，请继续保持！"
        elif 15 <= total_score < 25:
            evaluation = "您的心理状态较为良好，但可以适当关注自己的情绪波动。"
        elif 5 <= total_score < 15:
            evaluation = "您的心理状态存在一定压力，请注意调整心态并寻求帮助。"
        else:
            evaluation = "您的心理状态较为紧张或存在问题，建议及时咨询专业人士。"

        return render_template('quiz_result.html', score=total_score, evaluation=evaluation)
    except KeyError:
        flash("请回答所有问题后再提交！", "warning")
        return redirect(url_for('quiz'))

@app.route("/api/ai_interpretation", methods=["POST"])
def ai_interpretation():
    from ollama import chat
    data = request.get_json()
    score = data.get("score")
    summary = data.get("summary")

    prompt = f"请根据以下心理测评分数 {score} 以及总结 \"{summary}\" ，用温柔、专业的语言做出心理学解释与建议，不使用诊断性词语。"

    result = chat(model="deepseek-r1:7b", messages=[{"role": "user", "content": prompt}])
    return jsonify({"interpretation": result["message"]["content"]})

@app.route('/consult_platform')
def consult_platform():
    return render_template('consult_platform.html')

def stream_response(messages):
    for chunk in spark.generate([messages], stream=True):
        yield chunk.text + " "

@app.route('/ai_consult_stream', methods=['POST'])
def ai_consult_stream():
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({"response": "请提供您的问题。"})

    messages = [ChatMessage(role="user", content=user_input)]
    return Response(stream_response(messages), content_type='text/event-stream')
@app.route('/ai_consult', methods=['GET', 'POST'])
def ai_consult():
    if request.method == 'POST':
        user_input = request.json.get('message', '').strip()
        if not user_input:
            return jsonify({"response": "请提供您的问题。"})
        messages = [ChatMessage(role="user", content=user_input)]
        handler = ChunkPrintHandler()
        result = spark.generate([messages], callbacks=[handler])
        if result and hasattr(result, 'generations') and result.generations:
            ai_response = result.generations[0][0].text  # 从ChatGeneration对象中提取text
        else:
            ai_response = "抱歉，我暂时无法回答这个问题。"
        return jsonify({"response": ai_response})

    return render_template('ai_consult.html')

from flask_socketio import SocketIO, emit, join_room, leave_room
socketio = SocketIO(app)

# 存储每个用户会话的状态
user_sessions = {}

# 路由: 人工咨询页面
@app.route('/human_consult')
def human_consult():
    return render_template('human_consult.html')

# WebSocket: 用户连接
@socketio.on('connect_user')
def handle_connect_user():
    join_room('user')
    print("用户已连接")

# WebSocket: 咨询师连接
@socketio.on('connect_agent')
def handle_connect_agent():
    join_room('agent')
    print("人工咨询师已连接")

# WebSocket: 用户消息
@socketio.on('user_message')
def handle_user_message(data):
    user_message = data.get('message')
    print(f"用户消息: {user_message}")
    # 广播用户消息给人工咨询师
    emit('new_user_message', {'message': user_message}, room='agent')

# WebSocket: 咨询师回复
@socketio.on('agent_reply')
def handle_agent_reply(data):
    agent_reply = data.get('reply')
    print(f"咨询师回复: {agent_reply}")
    # 广播回复给用户
    emit('new_agent_reply', {'reply': agent_reply}, room='user')

# WebSocket: 当用户断开连接
@socketio.on('disconnect_user')
def handle_disconnect_user():
    leave_room('user')
    print("用户已断开连接")

# WebSocket: 当人工咨询师断开连接
@socketio.on('disconnect_agent')
def handle_disconnect_agent():
    leave_room('agent')
    print("人工咨询师已断开连接")



# 路由: 人工咨询页面 (咨询师端)
@app.route('/human_agent')
def human_agent():
    return render_template('human_agent.html')

# WebSocket: 咨询师连接
@socketio.on('connect_agent')
def handle_connect_agent():
    join_room('agent')
    print("人工咨询师已连接")

# WebSocket: 用户连接
@socketio.on('connect_user')
def handle_connect_user():
    join_room('user')
    print("用户已连接")

# WebSocket: 用户发送消息
@socketio.on('user_message')
def handle_user_message(data):
    user_message = data.get('message')
    print(f"用户消息: {user_message}")
    # 广播用户消息给人工咨询师
    emit('new_user_message', {'message': user_message}, room='agent')

# WebSocket: 咨询师发送消息
@socketio.on('agent_reply')
def handle_agent_reply(data):
    agent_reply = data.get('reply')
    print(f"咨询师回复: {agent_reply}")
    # 广播回复给用户
    emit('new_agent_reply', {'reply': agent_reply}, room='user')

# WebSocket: 用户断开连接
@socketio.on('disconnect_user')
def handle_disconnect_user():
    leave_room('user')
    print("用户已断开连接")

# WebSocket: 咨询师断开连接
@socketio.on('disconnect_agent')
def handle_disconnect_agent():
    leave_room('agent')
    print("人工咨询师已断开连接")

# 模拟咨询师账号信息
CONSULTANT_CREDENTIALS = {
    "consultant1": "123",
    "consultant2": "456"
}

@app.route('/agent_platform', methods=['GET', 'POST'])
def agent_platform():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # 检查用户名和密码是否正确
        if username in CONSULTANT_CREDENTIALS and CONSULTANT_CREDENTIALS[username] == password:
            # 登录成功，跳转到咨询师平台
            return redirect(url_for('human_agent'))
        else:
            # 登录失败，显示错误消息
            error = "用户名或密码错误！"
            return render_template('agent_platform.html', error=error)

    return render_template('agent_platform.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
