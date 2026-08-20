import pandas as pd
import os
import datetime

class HistoryManager:
    def __init__(self, history_file='history.csv'):
        self.history_file = history_file
        self.columns = ['Timestamp', 'Reference', 'Transcribed', 'WER (%)', 'CER (%)', 'Accuracy (%)', 'Latency (s)']
        if not os.path.exists(self.history_file):
            pd.DataFrame(columns=self.columns).to_csv(self.history_file, index=False)
            
    def add_run(self, reference, transcribed, wer, cer, accuracy, latency):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame([[timestamp, reference, transcribed, wer, cer, accuracy, latency]], columns=self.columns)
        new_row.to_csv(self.history_file, mode='a', header=False, index=False)
        
    def get_history(self):
        return pd.read_csv(self.history_file)
        
    def get_averages(self):
        df = self.get_history()
        if df.empty:
            return {"wer": 0.0, "cer": 0.0, "accuracy": 0.0}
        return {
            "wer": df['WER (%)'].mean(),
            "cer": df['CER (%)'].mean(),
            "accuracy": df['Accuracy (%)'].mean()
        }
