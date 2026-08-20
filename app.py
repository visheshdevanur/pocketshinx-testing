import streamlit as st
import json
import random
import plotly.express as px
import plotly.graph_objects as go
from stt_engine import STTEngine
from evaluator import Evaluator
from history_manager import HistoryManager

# Set page layout
st.set_page_config(page_title="PocketSphinx STT Evaluator", layout="wide")

# Initialize managers in session state so they persist across reruns
@st.cache_resource
def get_managers():
    return STTEngine(), HistoryManager()

stt_engine, history_manager = get_managers()

# Load sentences
@st.cache_data
def load_sentences():
    with open('sentences.json', 'r') as f:
        return json.load(f)

sentences = load_sentences()

st.title("🎤 PocketSphinx STT Evaluation Harness")

# Sidebar for controls
with st.sidebar:
    st.header("Test Configuration")
    
    # Sentence Selection
    st.subheader("Reference Sentence")
    ref_mode = st.radio("Selection Mode", ["From Bank", "Custom"])
    
    if ref_mode == "From Bank":
        sentence_options = [f"{i+1}. {s}" for i, s in enumerate(sentences)]
        selected_option = st.selectbox("Select a sentence", sentence_options, index=0)
        
        if st.button("Shuffle Random Sentence"):
            st.session_state['random_sentence'] = random.choice(sentence_options)
        
        if 'random_sentence' in st.session_state:
            # Override if random was clicked
            selected_option = st.session_state['random_sentence']
            st.info(f"Randomly selected: {selected_option}")
            
        selected_sentence = selected_option.split(". ", 1)[1]
    else:
        selected_sentence = st.text_area("Enter custom reference sentence")

    st.subheader("Audio Input")
    audio_source = st.radio("Source", ["Browser Microphone", "Upload File"])
    
    audio_bytes = None
    if audio_source == "Browser Microphone":
        audio_val = st.audio_input("Record Audio (Browser)")
        if audio_val:
            audio_bytes = audio_val.getvalue()
    else:
        audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
        if audio_file:
            audio_bytes = audio_file.getvalue()

# Main Area - Transcription and Evaluation
if audio_bytes and selected_sentence:
    
    # Generate a unique key for the current combination of audio and sentence
    current_test_key = str(hash(audio_bytes)) + selected_sentence
    
    colA, colB = st.columns([1, 4])
    with colA:
        if st.button("▶️ Run Evaluation", type="primary"):
            st.session_state['active_test_key'] = current_test_key
    with colB:
        if st.session_state.get('active_test_key') == current_test_key:
            if st.button("❌ Close Results"):
                st.session_state['active_test_key'] = None
                st.rerun()
                
    # Only show results if the user explicitly ran evaluation for THIS specific sentence and audio
    if st.session_state.get('active_test_key') == current_test_key:
        st.header("Evaluation Results")
        
        st.subheader("Playback Recording")
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("Transcribing..."):
            # Run STT
            transcribed_text, latency = stt_engine.transcribe(audio_bytes)
            
            # Run Evaluator
            metrics = Evaluator.evaluate(selected_sentence, transcribed_text)
            
            # Log to history
            history_manager.add_run(
                reference=selected_sentence,
                transcribed=transcribed_text,
                wer=metrics['wer'],
                cer=metrics['cer'],
                accuracy=metrics['accuracy'],
                latency=latency
            )
            
        # Display Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Word Accuracy", f"{metrics['accuracy']:.2f}%")
        col2.metric("WER", f"{metrics['wer']:.2f}%")
        col3.metric("CER", f"{metrics['cer']:.2f}%")
        col4.metric("Latency", f"{latency:.2f}s")
        
        # Display Alignment
        with st.expander("Word Alignment Analysis", expanded=True):
            alignment_html = ""
            for status, word in metrics['alignment']:
                if status == 'correct':
                    alignment_html += f"<span style='color: green; margin-right: 5px; padding: 2px; border-radius: 3px;'>{word}</span>"
                elif status == 'substituted':
                    alignment_html += f"<span style='color: orange; font-weight: bold; margin-right: 5px; padding: 2px; border-radius: 3px; background-color: #fff3e0;' title='Substituted'>{word}</span>"
                elif status == 'extra':
                    alignment_html += f"<span style='color: purple; font-style: italic; margin-right: 5px; padding: 2px; border-radius: 3px; background-color: #f3e5f5;' title='Extra word'>{word}</span>"
                elif status == 'missing':
                    alignment_html += f"<span style='color: red; text-decoration: line-through; margin-right: 5px; padding: 2px; border-radius: 3px; background-color: #ffebee;' title='Missing word'>{word}</span>"
                    
            st.markdown(f"<div style='font-size: 18px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>{alignment_html}</div>", unsafe_allow_html=True)
            
            st.markdown("""
            **Legend:** 
            <span style='color: green;'>Correct</span> | 
            <span style='color: orange;'>Substituted</span> | 
            <span style='color: purple;'>Extra</span> | 
            <span style='color: red; text-decoration: line-through;'>Missing</span>
            """, unsafe_allow_html=True)
            
        st.divider()

# History and Analytics
st.header("Session History & Analytics")
history_df = history_manager.get_history()

if not history_df.empty:
    avg_metrics = history_manager.get_averages()
    
    # Running averages
    st.subheader("Running Averages")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Word Accuracy", f"{avg_metrics['accuracy']:.2f}%")
    c2.metric("Avg WER", f"{avg_metrics['wer']:.2f}%")
    c3.metric("Avg CER", f"{avg_metrics['cer']:.2f}%")
    
    # Charts
    st.subheader("Performance Graphs")
    tab1, tab2, tab3 = st.tabs(["WER Trend", "CER Trend", "Accuracy vs Latency"])
    
    with tab1:
        fig_wer = px.bar(history_df, x=history_df.index, y="WER (%)", 
                         title="WER (%) Per Test Run", labels={'index': 'Test Run ID'})
        st.plotly_chart(fig_wer, use_container_width=True)
        
    with tab2:
        fig_cer = px.bar(history_df, x=history_df.index, y="CER (%)", 
                         title="CER (%) Per Test Run", labels={'index': 'Test Run ID'})
        st.plotly_chart(fig_cer, use_container_width=True)
        
    with tab3:
        fig_acc = px.scatter(history_df, x="Latency (s)", y="Accuracy (%)", 
                             title="Accuracy vs Latency", hover_data=['Reference'])
        st.plotly_chart(fig_acc, use_container_width=True)
        
    # Table
    st.subheader("History Table")
    st.dataframe(history_df, use_container_width=True)
    
    # Download button
    csv_data = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download History as CSV",
        data=csv_data,
        file_name='stt_test_history.csv',
        mime='text/csv',
    )
else:
    st.info("No test runs logged yet. Complete a test to see history and analytics.")
