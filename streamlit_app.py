import streamlit as st
import os
import base64

# --- Minimal CSS for chat, buttons, and input bar ---
st.markdown("""
    <style>
    .input-bar-container {display: flex; align-items: center; justify-content: center; margin-bottom: 2.5rem;}
    .input-bar {background: #fff; border-radius: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); padding: 0.5rem 1.2rem; display: flex; align-items: center; width: 100%; max-width: 500px;}
    .input-bar input {border: none; outline: none; font-size: 1.1rem; background: transparent; flex: 1; padding: 0.5rem;}
    .input-bar .img-icon {margin-right: 0.7rem; font-size: 1.3rem; color: #888;}
    .mini-btn {background: #222; color: #fff; border: none; border-radius: 6px; padding: 0.3rem 0.7rem; font-size: 0.92rem; cursor: pointer; transition: background 0.2s; margin-right: 0.5rem;}
    .mini-btn:last-child {margin-right: 0;}
    .mini-btn:hover {background: #444;}
    .button-row {display: flex; gap: 0.5rem; justify-content: center; margin-top: 0.7rem; margin-bottom: 0.7rem;}
    .chatbox {background: #fff; border-radius: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); padding: 1.5rem; max-width: 500px; margin: 0 auto;}
    .chat-history {max-height: 300px; overflow-y: auto; margin-bottom: 1rem;}
    .chat-row {display: flex; margin-bottom: 0.7rem;}
    .chat-bubble {padding: 0.7rem 1.2rem; border-radius: 16px; font-size: 1rem; max-width: 80%;}
    .chat-user {justify-content: flex-end;}
    .chat-user .chat-bubble {background: #222; color: #fff; border-bottom-right-radius: 4px;}
    .chat-bot {justify-content: flex-start;}
    .chat-bot .chat-bubble {background: #f5f5f7; color: #222; border-bottom-left-radius: 4px;}
    .selected-img {border: 3px solid #007aff !important;}
    </style>
""", unsafe_allow_html=True)

# --- Catalog Images (load from your local dataset) ---
CATALOG_PATH = r"C:\Users\User\Work\College\AIMS\Summer Project 2\clothes_tryon_dataset\train\cloth"
if not os.path.exists(CATALOG_PATH):
    st.error(f"Catalog path not found: {CATALOG_PATH}")
    st.stop()

all_catalog_files = [f for f in os.listdir(CATALOG_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
show_files = all_catalog_files[:20]  # Show first 20

# --- State for selected image and chat ---
if 'selected_catalog_image' not in st.session_state:
    st.session_state['selected_catalog_image'] = None
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

# --- Helper to display images as base64 ---
def img_to_base64(img_path):
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Use os.path.join for all image paths ---
CATALOG_PATH = r"C:\Users\User\Work\College\AIMS\Summer Project 2\clothes_tryon_dataset\train\cloth"
query_img_filename = "00000_00.jpg"
top5_filenames = ["00000_00.jpg", "03085_00.jpg", "10820_00.jpg", "00860_00.jpg", "00005_00.jpg"]

query_img_path = os.path.join(CATALOG_PATH, query_img_filename)
top5_img_paths = [os.path.join(CATALOG_PATH, fname) for fname in top5_filenames]
top5_scores = [1.000, 0.958, 0.953, 0.946, 0.941]

inference_text = """
<b>SEARCH STRATEGY:</b> Focus on the t-shirt style (crew neck, short sleeve, graphic tee), brand (Levi's), color scheme (white and red), and material (cotton). Include items that would broaden the search beyond the specific Levi's logo for variations.<br><br>
<b>User Intent:</b> Find clothes that are similar in style and design to the provided image, potentially looking for alternatives to the specific Levi's tee.<br>
<b>Style Analysis:</b> The image depicts a casual, crew-neck, short-sleeve t-shirt. It's a classic design with a prominent Levi's logo in red and white. The style is simple and versatile, suitable for everyday wear.<br>
<b>Top CLIP Search Terms:</b> ['female crew neck t-shirt with graphic logo', 'red and white graphic tee for women', 'casual cotton t-shirt with logo', "Levi's style t-shirt"]<br>
<b>Detected Gender:</b> female<br>
<b>Skin Tone:</b> unknown<br>
<b>Age Group:</b> adult<br>
<b>Season:</b> all-season<br>
<b>Occasion:</b> casual<br>
<b>Color Palette:</b> ['white', 'red']<br>
<b>Fashion Era:</b> modern<br>
<b>Patterns:</b> []<br>
<b>Materials:</b> ['cotton']<br>
<b>Brands:</b> ['Levi's']<br><br>
<b>CATALOG RESULTS:</b> 5 items<br>
#1 (Score: 1.000)<br>
#2 (Score: 0.958)<br>
#3 (Score: 0.953)<br>
#4 (Score: 0.946)<br>
#5 (Score: 0.941)<br><br>
<b>LLM ANALYSIS:</b><br>
<b>Style:</b> The provided t-shirt is a versatile basic. It pairs well with a wide variety of bottoms like jeans, skirts, or shorts. It can also be layered under jackets or over dresses. The recommendations aim to provide complementary pieces for creating casual and stylish outfits.<br>
<b>Catalog Feedback:</b> The catalog results are highly similar.<br>
<b>Sample Suggestions:</b><br>
- Jeans: "light wash, denim jacket"<br>
- Skirts: "high waisted light wash jeans"<br>
- Shoes: "white sneakers"<br>
- Search: "white leather mini-skirt", "black denim jacket", "black canvas crossbody bag"<br>
"""

# --- Chatbot UI at the top ---
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# --- Input bar with icon-only file upload, prompt, and send ---
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input("Type your message...", key="chat_input", label_visibility="collapsed")
with col2:
    send_clicked = st.button("Send", key="send_btn", use_container_width=True)

if send_clicked:
    if user_input.strip():
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        # Compose the bot reply as a single HTML block
        bot_html = '''
        <div style="background:#fff;border-radius:16px;box-shadow:0 2px 8px rgba(0,0,0,0.04);padding:1.5rem 2rem;max-width:1100px;margin:0 auto 2rem auto;">
            <div style="font-weight:600;margin-bottom:0.7rem;">Your Photo</div>
            <img src="data:image/jpeg;base64,''' + img_to_base64(query_img_path) + '''" style="width:140px; margin-bottom:1.2rem; border-radius:8px; border:1px solid #eee;"/>
            <div style="display:flex;flex-direction:row;gap:18px;margin-bottom:1.5rem;justify-content:center;">
        '''
        for i, (img, score) in enumerate(zip(top5_img_paths, top5_scores)):
            img_b64 = img_to_base64(img)
            if img_b64:
                bot_html += f'<div style="text-align:center;"><img src="data:image/jpeg;base64,{img_b64}" style="width:110px; border-radius:8px; border:1px solid #eee;"/><br><span style="font-size:0.95rem;">#{i+1} (Score: {score:.3f})</span></div>'
        bot_html += '</div>'
        bot_html += f'<div style="margin-top:1.2rem;font-size:1.07rem;">{inference_text}</div></div>'
        st.session_state['chat_history'].append({'role': 'bot', 'content': bot_html})
        st.rerun()

# --- Display chat history ---
st.markdown('<div class="chatbox">', unsafe_allow_html=True)
for msg in st.session_state['chat_history']:
    if msg['role'] == 'user':
        st.markdown(f'<div class="chat-row chat-user"><div class="chat-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-row chat-bot"><div class="chat-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Catalog Grid BELOW the chat area ---
st.markdown("### Catalog")
cols_per_row = 4
rows = [show_files[i:i+cols_per_row] for i in range(0, len(show_files), cols_per_row)]
for row in rows:
    cols = st.columns(cols_per_row)
    for idx, filename in enumerate(row):
        img_path = os.path.join(CATALOG_PATH, filename)
        with open(img_path, "rb") as img_file:
            img_bytes = img_file.read()
            img_b64 = base64.b64encode(img_bytes).decode()
        selected = (st.session_state['selected_catalog_image'] == filename)
        with cols[idx]:
            st.image(f"data:image/jpeg;base64,{img_b64}", use_container_width=True, caption=filename)
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)  # Add spacing
            if st.button("Add to Recommender", key=f"add_{filename}"):
                st.session_state['selected_catalog_image'] = filename
                st.session_state['uploaded_image'] = None  # Clear uploaded image if selecting from catalog
            if selected:
                st.markdown('<div style="color:#007aff;font-size:0.9rem;">Selected</div>', unsafe_allow_html=True) 