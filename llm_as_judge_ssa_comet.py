import json
import argparse
import torch
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub.hf_api import HfFolder
import time
import glob
import numpy as np
import random

# SSA-COMET model imports
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: COMET library not available. Install with: pip install unbabel-comet")

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def setup_llama_model(model_name, hf_token):
    """Setup Llama model for inference"""
    HfFolder.save_token(hf_token)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        max_memory={0: "20GiB"},
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        dtype=dtype,
        model_kwargs={"temperature": 0.0, "do_sample": False}
    )

    return pipe

def setup_aya_model(model_name, hf_token):
    """Setup AYA model for inference"""
    HfFolder.save_token(hf_token)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        max_memory={0: "20GiB"},
        trust_remote_code=True
    )

    # AYA doesn't use pipeline, we'll use model and tokenizer directly
    return {'model': model, 'tokenizer': tokenizer}

def setup_ssa_comet_model(model_type='qe', gpus=1):
    """
    Setup SSA-COMET model for evaluation.
    
    Args:
        model_type: 'qe' for Quality Estimation (no reference) or 'mte' for MTE (with reference)
        gpus: Number of GPUs to use
    """
    if not COMET_AVAILABLE:
        raise ImportError("COMET library not available. Install with: pip install unbabel-comet")
    
    if model_type == 'qe':
        model_path = download_model("McGill-NLP/ssa-comet-qe")
    elif model_type == 'mte':
        model_path = download_model("McGill-NLP/ssa-comet-mtl")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'qe' or 'mte'")
    
    model = load_from_checkpoint(model_path)
    return model

def setup_xcomet_model(model_name='Unbabel/XCOMET-XL', gpus=1):
    """
    Setup XCOMET model for evaluation.
    
    Args:
        model_name: XCOMET model name (default: 'Unbabel/XCOMET-XL')
        gpus: Number of GPUs to use
    """
    if not COMET_AVAILABLE:
        raise ImportError("COMET library not available. Install with: pip install unbabel-comet")
    
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model

def evaluate_with_ssa_comet(model, english_source, machine_translation, reference_translation=None, batch_size=8, gpus=1):
    """
    Evaluate translation using SSA-COMET model.
    
    Args:
        model: SSA-COMET model instance
        english_source: Source text
        machine_translation: Machine translation to evaluate
        reference_translation: Reference translation (optional, for MTE mode)
        batch_size: Batch size for prediction
        gpus: Number of GPUs to use
    """
    if reference_translation:
        # MTE mode (with reference)
        data = [{
            "src": english_source,
            "mt": machine_translation,
            "ref": reference_translation
        }]
    else:
        # QE mode (without reference)
        data = [{
            "src": english_source,
            "mt": machine_translation
        }]
    
    try:
        model_output = model.predict(data, batch_size=batch_size, gpus=gpus)
        # COMET models typically return a list of scores or a dict with 'scores' key
        # Handle different output formats
        if isinstance(model_output, list):
            if len(model_output) > 0:
                if isinstance(model_output[0], (int, float)):
                    score = float(model_output[0])
                elif isinstance(model_output[0], dict):
                    score = model_output[0].get('score', model_output[0].get('scores', [0.5])[0])
                else:
                    score = float(model_output[0]) if hasattr(model_output[0], '__float__') else 0.5
            else:
                score = 0.5
        elif isinstance(model_output, dict):
            # Sometimes returns dict with 'scores' key
            if 'scores' in model_output:
                scores = model_output['scores']
                score = float(scores[0]) if len(scores) > 0 else 0.5
            elif 'score' in model_output:
                score = float(model_output['score'])
            else:
                score = 0.5
        elif isinstance(model_output, (int, float)):
            score = float(model_output)
        else:
            # Try to convert to float
            try:
                score = float(model_output)
            except (ValueError, TypeError):
                score = 0.5
        
        # SSA-COMET scores are typically in a reasonable range (often 0-1 or similar)
        # Normalize to 0-1 if it seems to be in 0-100 range
        if score > 1.0 and score <= 100.0:
            score = score / 100.0
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"Error in SSA-COMET evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 0.5  # Default score on error

def evaluate_with_xcomet(model, english_source, machine_translation, reference_translation=None, batch_size=8, gpus=1):
    """
    Evaluate translation using XCOMET model.
    Returns segment-level scores, system-level score, and error spans.
    
    Args:
        model: XCOMET model instance
        english_source: Source text
        machine_translation: Machine translation to evaluate
        reference_translation: Reference translation (optional, not used for QE mode)
        batch_size: Batch size for prediction
        gpus: Number of GPUs to use
    
    Returns:
        dict with keys: 'score' (segment-level score), 'system_score', 'error_spans'
    """
    # XCOMET can work without reference (QE mode)
    data = [{
        "src": english_source,
        "mt": machine_translation
    }]
    
    # Optionally add reference if provided
    if reference_translation:
        data[0]["ref"] = reference_translation
    
    try:
        model_output = model.predict(data, batch_size=batch_size, gpus=gpus)
        
        # Extract segment-level score
        segment_score = None
        if hasattr(model_output, 'scores') and len(model_output.scores) > 0:
            segment_score = float(model_output.scores[0])
        elif isinstance(model_output, list) and len(model_output) > 0:
            segment_score = float(model_output[0])
        elif isinstance(model_output, dict) and 'scores' in model_output:
            segment_score = float(model_output['scores'][0]) if len(model_output['scores']) > 0 else 0.5
        else:
            segment_score = 0.5
        
        # Extract system-level score
        system_score = None
        if hasattr(model_output, 'system_score'):
            system_score = float(model_output.system_score)
        elif isinstance(model_output, dict) and 'system_score' in model_output:
            system_score = float(model_output['system_score'])
        else:
            # If no system score available, use segment score
            system_score = segment_score
        
        # Extract error spans
        error_spans = None
        if hasattr(model_output, 'metadata') and hasattr(model_output.metadata, 'error_spans'):
            error_spans = model_output.metadata.error_spans
        elif isinstance(model_output, dict) and 'metadata' in model_output:
            metadata = model_output['metadata']
            if isinstance(metadata, dict) and 'error_spans' in metadata:
                error_spans = metadata['error_spans']
        elif isinstance(model_output, dict) and 'error_spans' in model_output:
            error_spans = model_output['error_spans']
        
        # Normalize score to 0-1 range if needed
        if segment_score > 1.0 and segment_score <= 100.0:
            segment_score = segment_score / 100.0
        segment_score = max(0.0, min(1.0, segment_score))
        
        if system_score and (system_score > 1.0 and system_score <= 100.0):
            system_score = system_score / 100.0
        if system_score:
            system_score = max(0.0, min(1.0, system_score))
        
        return {
            'score': segment_score,
            'system_score': system_score,
            'error_spans': error_spans
        }
    except Exception as e:
        print(f"Error in XCOMET evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            'score': 0.5,
            'system_score': 0.5,
            'error_spans': None
        }

def create_ssa_comet_prompt_with_error_span(english_source, machine_translation, reference_translation, 
                                            target_language, few_shot_examples=None):
    """
    Create SSA-COMET prompt with error span detection (Figure 7 from paper)
    """
    prompt = """You are asked to compare the meaning of a source segment and its translation. You will be presented with one pair of segments at a time, where a segment may contain one or more sentences. For each pair, you are asked to read the text closely and do the following:
1. Highlight the text spans that convey different meaning in the compared segments. After highlighting a span in the text, you will be asked to select the category that best describes the meaning difference using the following categories:
Source Text:
Omission: The highlighted span in the source text corresponds to information that does not exist in the translated text.
Mistranslation: The highlighted span in the source does not have the exact same meaning as the highlighted span in the translated text.
Translation Text:
Addition: The highlighted span in the translation corresponds to information that does not exist in the source text.
Mistranslation: The highlighted span in the translation does not have the exact same meaning as the highlighted span in the source segment.
Untranslated: The highlighted span in the translation is a copy of the corresponding source segment but should be translated in the target language.
You can highlight as many spans as needed.
2. Assess the translation adequacy on a continuous scale [0 ~ 100] using the quality levels described below:
[0] Nonsense/No meaning preserved: Nearly all information is lost between the translation and source.
[34] Some meaning preserved: The translation preserves some of the meaning of the source but misses significant parts.
[67] Most meaning preserved: The translation retains most of the meaning of the source.
[100] Perfect meaning: The meaning of the translation is completely consistent with the source.
Instruction: Using the provided source and reference sentences, assess the quality of the machine translation from English to """ + target_language + """ on a continuous scale from 0 to 1, where a higher score indicates better translation quality. Please detect the word-level translation errors before giving the score.
Given examples:
"""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            prompt += f"""Example {i}:
Source: {example['source']}
Translation: {example['translation']}
Reference: {example['reference']}
Output:
The following errors are detected:
{example.get('error_spans', 'No errors detected.')}
Based on the {example.get('num_errors', 0)} error detected, the score of translation is: {example['score']:.2f}

"""
    
    prompt += """Based on the examples given, generate the output in exactly the same format, give the error spans and the score, do not give any commentary response.
Source: """ + english_source + """
Translation: """ + machine_translation + """
Reference: """ + reference_translation + """
Output:
"""
    
    return prompt

def create_ssa_comet_prompt_without_error_span(english_source, machine_translation, reference_translation,
                                               target_language, few_shot_examples=None):
    """
    Create SSA-COMET prompt without error span detection (Figure 8 from paper)
    """
    prompt = """Assess the translation adequacy on a continuous scale [0 ~ 100] using the quality levels described below:
[0] Nonsense/No meaning preserved: Nearly all information is lost between the translation and source.
[34] Some meaning preserved: The translation preserves some of the meaning of the source but misses significant parts.
[67] Most meaning preserved: The translation retains most of the meaning of the source.
[100] Perfect meaning: The meaning of the translation is completely consistent with the source.
Instruction: Please assess the given machine translation based on the source sentence. Note that you should only output the final score
Given examples:
"""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            prompt += f"""Example {i}:
Source: {example['source']}
Translation: {example['translation']}
Reference: {example['reference']}
Score: {example['score']:.2f}

"""
    
    prompt += """Based on the examples given, generate the output in exactly the same format, give the score and do not give any commentary response.
Source: """ + english_source + """
Translation: """ + machine_translation + """
Reference: """ + reference_translation + """
Score:
"""
    
    return prompt

def create_ssa_comet_prompt_qe(english_source, machine_translation, target_language, few_shot_examples=None):
    """
    Create SSA-COMET prompt for Quality Estimation (QE) without reference (Figure 9 from paper)
    """
    prompt = """Assess the translation adequacy on a continuous scale [0 ~ 100] using the quality levels described below:
[0] Nonsense/No meaning preserved: Nearly all information is lost between the translation and source.
[34] Some meaning preserved: The translation preserves some of the meaning of the source but misses significant parts.
[67] Most meaning preserved: The translation retains most of the meaning of the source.
[100] Perfect meaning: The meaning of the translation is completely consistent with the source.
Instruction: Please assess the given machine translation based on the source sentence. Note that you should only output the final score
Given examples:
"""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            prompt += f"""Example {i}:
Source: {example['source']}
Translation: {example['translation']}
Score: {example['score']:.2f}

"""
    
    prompt += """Based on the examples given, generate the output in exactly the same format, give the score and do not give any commentary response.
Source: """ + english_source + """
Translation: """ + machine_translation + """
Score:
"""
    
    return prompt

def sample_few_shot_examples(training_data, num_shots=5):
    """
    Sample few-shot examples from training data.
    Following SSA-COMET: divide score range into 5 equal intervals and sample one from each.
    """
    if not training_data or len(training_data) == 0:
        return []
    
    # Extract scores and normalize to 0-1 if needed
    scores = [ex.get('score', 0) for ex in training_data]
    min_score = min(scores)
    max_score = max(scores)
    
    # Normalize scores to 0-1 range
    normalized_scores = [(s - min_score) / (max_score - min_score) if max_score > min_score else 0.5 
                        for s in scores]
    
    # Divide into 5 equal intervals
    interval_size = 1.0 / num_shots
    examples = []
    
    for i in range(num_shots):
        interval_min = i * interval_size
        interval_max = (i + 1) * interval_size if i < num_shots - 1 else 1.0
        
        # Find examples in this interval
        candidates = [j for j, ns in enumerate(normalized_scores) 
                     if interval_min <= ns < interval_max or (i == num_shots - 1 and ns == 1.0)]
        
        if candidates:
            # Sample one randomly from this interval
            idx = random.choice(candidates)
            examples.append(training_data[idx])
        else:
            # If no examples in interval, sample closest one
            closest_idx = min(range(len(normalized_scores)), 
                            key=lambda j: abs(normalized_scores[j] - (interval_min + interval_max) / 2))
            examples.append(training_data[closest_idx])
    
    return examples

def extract_score_from_response(response_text, with_error_span=False):
    """
    Extract adequacy score from LLM response.
    Score should be in range [0, 1] (normalized from 0-100 scale).
    """
    response_text = response_text.strip()
    
    # Try to find score patterns
    import re
    
    # Pattern 1: "score: 0.67" or "score is: 0.67"
    patterns = [
        r'score\s*:?\s*([0-9]+\.?[0-9]*)',
        r'score\s+is\s*:?\s*([0-9]+\.?[0-9]*)',
        r'([0-9]+\.?[0-9]*)\s*$',  # Just a number at the end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # Normalize to 0-1 if it's in 0-100 range
            if score > 1.0:
                score = score / 100.0
            return max(0.0, min(1.0, score))
    
    # If no pattern found, try to extract any number
    numbers = re.findall(r'[0-9]+\.?[0-9]*', response_text)
    if numbers:
        score = float(numbers[-1])  # Take last number
        if score > 1.0:
            score = score / 100.0
        return max(0.0, min(1.0, score))
    
    # Default to 0.5 if parsing fails (as per SSA-COMET paper)
    return 0.5

def query_llama(pipe, prompt, max_retries=3, with_error_span=False):
    """Query Llama model with retry logic"""
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            result = pipe(
                messages,
                max_new_tokens=200 if with_error_span else 50,
                return_full_text=False
            )[0]["generated_text"].strip()
            
            score = extract_score_from_response(result, with_error_span)
            return score

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return 0.5  # Default score on error

    return 0.5

def query_aya(model_dict, prompt, max_retries=3, with_error_span=False):
    """Query AYA model with retry logic"""
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    
    for attempt in range(max_retries):
        try:
            # Format the message with the chat template
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            # Move to same device as model
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                device = list(model.hf_device_map.values())[0]
            elif torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            
            input_ids = input_ids.to(device)
            
            # Generate response
            gen_tokens = model.generate(
                input_ids,
                max_new_tokens=200 if with_error_span else 50,
                do_sample=False,
                temperature=0.0,
            )
            
            # Decode only the new tokens
            gen_text = tokenizer.decode(gen_tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            score = extract_score_from_response(gen_text, with_error_span)
            return score

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return 0.5  # Default score on error

    return 0.5

def get_segment_majority_data(df):
    """Get segment-level data with majority voting"""
    segment_data = []

    for seg_id, group in df.groupby('segmentID'):
        # Get the English source (should be same for all rows in group)
        english_source = group['english_source'].iloc[0]

        # Get target1 and target2 translations
        target1_text = ""
        target2_text = ""

        for _, row in group.iterrows():
            if row['target1ID'] == 'mt-text':
                target1_text = row['target1Text']
            elif row['target1ID'] == 'he-text':
                target1_text = row['target1Text']

            if row['target2ID'] == 'mt-text':
                target2_text = row['target2Text']
            elif row['target2ID'] == 'he-text':
                target2_text = row['target2Text']

        # Determine majority vote
        he_votes = (group['winner_type'] == 'HE').sum()
        mt_votes = (group['winner_type'] == 'MT').sum()

        if he_votes >= 2:
            human_majority = 'HE'
        else:
            human_majority = 'MT'

        segment_data.append({
            'segmentID': seg_id,
            'english_source': english_source,
            'target1_text': target1_text,
            'target2_text': target2_text,
            'target1_id': group['target1ID'].iloc[0],
            'target2_id': group['target2ID'].iloc[0],
            'human_majority': human_majority,
            'he_votes': he_votes,
            'mt_votes': mt_votes
        })

    return segment_data

def load_training_data_for_few_shot(data_dir, language, num_examples=100):
    """
    Load training data for few-shot examples.
    In practice, this would load from a training split.
    For now, we'll sample from the same dataset.
    """
    # This is a placeholder - in real implementation, load from training split
    # For now, return empty list (will use no few-shot if not available)
    return []

def process_language_data(data_dir, language, judge_type, args, pipe=None, training_data=None, comet_model=None):
    """Process data for one language using SSA-COMET prompts or model"""

    # Load non_filtered_clean data
    if language == 'Hausa':
        file_pattern = os.path.join(data_dir, f"WikiPairwise{language}*_non_filtered_clean.csv")
    else:
        file_pattern = os.path.join(data_dir, f"WikiPairwise{language}_non_filtered_clean.csv")
    files = glob.glob(file_pattern)

    if not files:
        print(f"No non_filtered_clean data found for {language}")
        return None

    df = pd.read_csv(files[0])
    print(f"Loaded {len(df)} rows for {language}")
    
    # Apply debug limit if specified
    if args.debug_rows and args.debug_rows > 0:
        first_segments = list(df['segmentID'].drop_duplicates().head(args.debug_rows))
        df = df[df['segmentID'].isin(first_segments)].copy()
        print(f"Limited to first {len(first_segments)} segments ({len(df)} rows) for debugging")

    # Get segment-level data
    segment_data = get_segment_majority_data(df)
    print(f"Processing {len(segment_data)} segments for {language}")

    # Sample few-shot examples if training data provided (only for LLM judges)
    few_shot_examples = None
    if judge_type in ['llama', 'aya'] and training_data and len(training_data) > 0:
        few_shot_examples = sample_few_shot_examples(training_data, num_shots=5)
        print(f"Using {len(few_shot_examples)} few-shot examples")

    # Process each segment
    results = []
    for seg_data in tqdm(segment_data, desc=f"Processing {language}"):
        # For pairwise comparison, we evaluate both translations separately
        xcomet_result1 = None
        xcomet_result2 = None
        
        if judge_type in ['llama', 'aya']:
            # LLM-as-judge: use SSA-COMET prompts
            # Since we don't have reference translations, we use QE (Quality Estimation) prompts
            # Note: SSA-COMET paper (Figure 9) shows QE prompts without error span detection
            
            # Evaluate target1 (without reference)
            prompt1 = create_ssa_comet_prompt_qe(
                seg_data['english_source'],
                seg_data['target1_text'],
                language,
                few_shot_examples
            )
            
            # Evaluate target2 (without reference)
            prompt2 = create_ssa_comet_prompt_qe(
                seg_data['english_source'],
                seg_data['target2_text'],
                language,
                few_shot_examples
            )

            # Query LLM model for both translations
            if judge_type == 'llama':
                score1 = query_llama(pipe, prompt1, with_error_span=args.use_error_span)
                score2 = query_llama(pipe, prompt2, with_error_span=args.use_error_span)
            elif judge_type == 'aya':
                score1 = query_aya(pipe, prompt1, with_error_span=args.use_error_span)
                score2 = query_aya(pipe, prompt2, with_error_span=args.use_error_span)
            else:
                score1 = 0.5
                score2 = 0.5
                
        elif judge_type in ['ssa-comet-qe', 'ssa-comet-mte']:
            # SSA-COMET model: use the learned metric directly
            model_type = 'qe' if judge_type == 'ssa-comet-qe' else 'mte'
            
            # Evaluate target1
            score1 = evaluate_with_ssa_comet(
                comet_model,
                seg_data['english_source'],
                seg_data['target1_text'],
                reference_translation=None if model_type == 'qe' else seg_data['target1_text'],  # For MTE, we'd need actual references
                batch_size=args.comet_batch_size,
                gpus=args.comet_gpus
            )
            
            # Evaluate target2
            score2 = evaluate_with_ssa_comet(
                comet_model,
                seg_data['english_source'],
                seg_data['target2_text'],
                reference_translation=None if model_type == 'qe' else seg_data['target2_text'],
                batch_size=args.comet_batch_size,
                gpus=args.comet_gpus
            )
        elif judge_type == 'xcomet':
            # XCOMET model: use the learned metric directly (QE mode, no reference)
            # Evaluate target1
            xcomet_result1 = evaluate_with_xcomet(
                comet_model,
                seg_data['english_source'],
                seg_data['target1_text'],
                reference_translation=None,  # QE mode, no reference
                batch_size=args.comet_batch_size,
                gpus=args.comet_gpus
            )
            score1 = xcomet_result1['score']
            
            # Evaluate target2
            xcomet_result2 = evaluate_with_xcomet(
                comet_model,
                seg_data['english_source'],
                seg_data['target2_text'],
                reference_translation=None,  # QE mode, no reference
                batch_size=args.comet_batch_size,
                gpus=args.comet_gpus
            )
            score2 = xcomet_result2['score']
        else:
            score1 = 0.5
            score2 = 0.5

        # Determine preference based on scores
        if score1 > score2:
            preference = "MT" if seg_data['target1_id'] == 'mt-text' else "HE"
        elif score2 > score1:
            preference = "MT" if seg_data['target2_id'] == 'mt-text' else "HE"
        else:
            preference = "TIE"

        # Store results with appropriate naming
        result = {
            **seg_data,
            'score_target1': score1,
            'score_target2': score2,
            'preference': preference
        }
        
        # For LLM judges, also store with llm_ prefix for compatibility
        if judge_type in ['llama', 'aya']:
            result['llm_score_target1'] = score1
            result['llm_score_target2'] = score2
            result['llm_preference'] = preference
        
        # For XCOMET, store additional fields (system scores and error spans)
        if judge_type == 'xcomet' and xcomet_result1 is not None and xcomet_result2 is not None:
            result['xcomet_score_target1'] = score1
            result['xcomet_score_target2'] = score2
            result['xcomet_system_score_target1'] = xcomet_result1.get('system_score')
            result['xcomet_system_score_target2'] = xcomet_result2.get('system_score')
            result['xcomet_error_spans_target1'] = xcomet_result1.get('error_spans')
            result['xcomet_error_spans_target2'] = xcomet_result2.get('error_spans')
            result['xcomet_preference'] = preference
        
        results.append(result)

    return results

def main():
    parser = argparse.ArgumentParser(description="SSA-COMET LLM-as-judge evaluation of MT vs HE translations")

    # Data arguments
    parser.add_argument('--data_dir', default='/fs/nexus-scratch/eloirghi/Wiki-LLM-as-judge-DATA-SCRATCH/non_filtered_clean/',
                       help="Directory containing non_filtered_clean data")
    parser.add_argument('--output_dir', default='./llm_judge_results_ssa_comet/',
                       help="Output directory for results")
    parser.add_argument('--languages', nargs='+', 
                       default=['Hausa', 'Igbo', 'Swahili', 'Yoruba', 'Zulu'],
                       help="Languages to process")
    parser.add_argument('--training_data_dir', default=None,
                       help="Directory containing training data for few-shot examples")

    # Model arguments  
    parser.add_argument('--judge_type', choices=['llama', 'aya', 'ssa-comet-qe', 'ssa-comet-mte', 'xcomet'], required=True,
                       help="Type of judge to use: LLM (llama/aya), SSA-COMET model (ssa-comet-qe/ssa-comet-mte), or XCOMET (xcomet)")
    parser.add_argument('--llama_model', default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Llama model name")
    parser.add_argument('--aya_model', default="CohereLabs/aya-expanse-8b",
                       help="AYA model name")
    parser.add_argument('--xcomet_model', default="Unbabel/XCOMET-XL",
                       help="XCOMET model name (default: Unbabel/XCOMET-XL)")
    parser.add_argument('--hf_token', help="Hugging Face token (required for LLM models)")
    parser.add_argument('--debug_rows', type=int, default=None,
                       help="Limit processing to top N segments for debugging")
    parser.add_argument('--use_error_span', action='store_true',
                       help="Use error span detection in prompts (default: False, only for LLM judges)")
    parser.add_argument('--comet_batch_size', type=int, default=8,
                       help="Batch size for SSA-COMET model evaluation")
    parser.add_argument('--comet_gpus', type=int, default=1,
                       help="Number of GPUs for SSA-COMET model evaluation")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup model
    pipe = None
    comet_model = None

    if args.judge_type == 'llama':
        if not args.hf_token:
            raise ValueError("Hugging Face token required for Llama models")
        print("Setting up Llama model...")
        pipe = setup_llama_model(args.llama_model, args.hf_token)

    elif args.judge_type == 'aya':
        if not args.hf_token:
            raise ValueError("Hugging Face token required for AYA models")
        print("Setting up AYA model...")
        pipe = setup_aya_model(args.aya_model, args.hf_token)
    
    elif args.judge_type in ['ssa-comet-qe', 'ssa-comet-mte']:
        print(f"Setting up SSA-COMET model ({args.judge_type})...")
        model_type = 'qe' if args.judge_type == 'ssa-comet-qe' else 'mte'
        comet_model = setup_ssa_comet_model(model_type=model_type, gpus=args.comet_gpus)
    
    elif args.judge_type == 'xcomet':
        print(f"Setting up XCOMET model ({args.xcomet_model})...")
        comet_model = setup_xcomet_model(model_name=args.xcomet_model, gpus=args.comet_gpus)

    # Process each language
    all_results = {}

    for language in args.languages:
        print(f"\n=== Processing {language} ===")

        # Load training data for few-shot if available (only for LLM judges)
        training_data = None
        if args.judge_type in ['llama', 'aya'] and args.training_data_dir:
            training_data = load_training_data_for_few_shot(args.training_data_dir, language)

        results = process_language_data(
            args.data_dir, language, args.judge_type, args,
            pipe=pipe, training_data=training_data, comet_model=comet_model
        )

        if results:
            all_results[language] = results

            # Save language-specific results
            judge_name = args.judge_type.replace('-', '_')
            output_file = os.path.join(args.output_dir, f"{language}_{judge_name}_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                serializable_results = convert_to_json_serializable(results)
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            print(f"Saved results for {language}: {output_file}")

            # Print summary statistics
            total_segs = len(results)
            # Handle both llm_preference and preference fields
            pref_key = 'llm_preference' if 'llm_preference' in results[0] else 'preference'
            mt_pref = sum(1 for r in results if r.get(pref_key) == 'MT')
            he_pref = sum(1 for r in results if r.get(pref_key) == 'HE')
            ties = sum(1 for r in results if r.get(pref_key) == 'TIE')

            judge_name = args.judge_type.replace('-', ' ').title()
            print(f"Summary for {language}:")
            print(f"  Total segments: {total_segs}")
            print(f"  {judge_name} prefers MT: {mt_pref} ({mt_pref/total_segs*100:.1f}%)")
            print(f"  {judge_name} prefers HE: {he_pref} ({he_pref/total_segs*100:.1f}%)")
            print(f"  Ties: {ties}")

    # Save combined results
    if all_results:
        judge_name = args.judge_type.replace('-', '_')
        combined_file = os.path.join(args.output_dir, f"combined_{judge_name}_results.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            serializable_all_results = convert_to_json_serializable(all_results)
            json.dump(serializable_all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved combined results: {combined_file}")

    # Clean up GPU memory
    if pipe:
        torch.cuda.empty_cache()
    if comet_model:
        del comet_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

