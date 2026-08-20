import jiwer
import string

class Evaluator:
    @staticmethod
    def _clean_text(text):
        """Remove punctuation and convert to lowercase for fair comparison."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return " ".join(text.split())

    @staticmethod
    def evaluate(reference, hypothesis):
        if not reference.strip():
            return {
                "wer": 0.0, "cer": 0.0, "accuracy": 100.0, "alignment": []
            }
            
        # Clean both strings
        reference_clean = Evaluator._clean_text(reference)
        hypothesis_clean = Evaluator._clean_text(hypothesis)
        
        if not reference_clean:
             return {
                "wer": 0.0, "cer": 0.0, "accuracy": 100.0, "alignment": []
            }
        
        # Calculate metrics
        wer = jiwer.wer(reference_clean, hypothesis_clean)
        cer = jiwer.cer(reference_clean, hypothesis_clean)
        accuracy = max(0.0, 1.0 - wer) * 100.0
        
        # Generate word-by-word alignment
        out = jiwer.process_words(reference_clean, hypothesis_clean)
        
        ref_words = reference_clean.split()
        hyp_words = hypothesis_clean.split()
        
        aligned_words = []
        for elem in out.alignments[0]:
            if elem.type == 'equal':
                for i in range(elem.ref_start_idx, elem.ref_end_idx):
                    if i < len(ref_words):
                        aligned_words.append(("correct", ref_words[i]))
            elif elem.type == 'substitute':
                # Show what was substituted with what
                sub_hyp_words = [hyp_words[i] for i in range(elem.hyp_start_idx, elem.hyp_end_idx) if i < len(hyp_words)]
                aligned_words.append(("substituted", " ".join(sub_hyp_words)))
            elif elem.type == 'insert':
                ins_hyp_words = [hyp_words[i] for i in range(elem.hyp_start_idx, elem.hyp_end_idx) if i < len(hyp_words)]
                aligned_words.append(("extra", " ".join(ins_hyp_words)))
            elif elem.type == 'delete':
                del_ref_words = [ref_words[i] for i in range(elem.ref_start_idx, elem.ref_end_idx) if i < len(ref_words)]
                aligned_words.append(("missing", " ".join(del_ref_words)))
                    
        return {
            "wer": wer * 100,
            "cer": cer * 100,
            "accuracy": accuracy,
            "alignment": aligned_words
        }
