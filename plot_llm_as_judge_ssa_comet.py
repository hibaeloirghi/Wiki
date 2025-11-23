import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def load_ssa_results(results_dir, judge_name="ssa_comet_qe", debug_rows=None):
    """
    Load SSA-COMET results. Preference JSON files contain per-language arrays.
    """
    judge_key = judge_name.replace("-", "_")
    combined_path = os.path.join(results_dir, f"combined_{judge_key}_results.json")
    if os.path.exists(combined_path):
        with open(combined_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
        for fp in os.listdir(results_dir):
            if fp.endswith(f"_{judge_key}_results.json"):
                lang = fp.replace(f"_{judge_key}_results.json", "")
                with open(os.path.join(results_dir, fp), "r", encoding="utf-8") as f:
                    data[lang] = json.load(f)

    if debug_rows and debug_rows > 0:
        limited = {}
        for lang, rows in data.items():
            seen = set()
            subset = []
            for row in rows:
                seg_id = row.get("segmentID")
                if not seg_id:
                    continue
                if seg_id not in seen:
                    seen.add(seg_id)
                    subset.append(row)
                    if len(seen) >= debug_rows:
                        break
            limited[lang] = subset
        return limited

    return data


def load_comet_data_robust(file_path, lang):
    """
    Attempt to load COMET data with error recovery.
    Tries multiple strategies to handle corrupted JSON files.
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Loading COMET data for {lang} (file size: {file_size_mb:.2f} MB)...")
    
    lang_entries = {}
    
    # Strategy 1: Try normal JSON parsing
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload:
            items = payload["data"]
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("Unexpected JSON structure")
        
        for item in items:
            item_id = item.get("id")
            if not item_id:
                continue
            try:
                comet_mt = float(item.get("comet_source_mt", 0) or 0)
            except (ValueError, TypeError):
                comet_mt = 0.0
            try:
                comet_ref = float(item.get("comet_source_target", 0) or 0)
            except (ValueError, TypeError):
                comet_ref = 0.0
            lang_entries[item_id] = {
                "id": item_id,
                "file_id": item.get("file_id", item_id.split("/")[0] if "/" in item_id else None),
                "comet_source_mt": comet_mt,
                "comet_source_target": comet_ref,
            }
        print(f"  ✓ Successfully loaded {len(lang_entries)} COMET entries for {lang} (normal parsing)")
        return lang_entries
    except (json.JSONDecodeError, MemoryError) as e:
        print(f"  ⚠ Normal parsing failed ({type(e).__name__}), trying robust parsing...")
    
    # Strategy 2: Try streaming parser (ijson) if available
    try:
        import ijson
        with open(file_path, "rb") as f:
            parser = ijson.items(f, "data.item")
            items_parsed = 0
            items_skipped = 0
            for item in parser:
                try:
                    item_id = item.get("id")
                    if not item_id:
                        items_skipped += 1
                        continue
                    try:
                        comet_mt = float(item.get("comet_source_mt", 0) or 0)
                    except (ValueError, TypeError):
                        comet_mt = 0.0
                    try:
                        comet_ref = float(item.get("comet_source_target", 0) or 0)
                    except (ValueError, TypeError):
                        comet_ref = 0.0
                    lang_entries[item_id] = {
                        "id": item_id,
                        "file_id": item.get("file_id", item_id.split("/")[0] if "/" in item_id else None),
                        "comet_source_mt": comet_mt,
                        "comet_source_target": comet_ref,
                    }
                    items_parsed += 1
                except Exception:
                    items_skipped += 1
                    continue
        if items_parsed > 0:
            print(f"  ✓ Successfully loaded {len(lang_entries)} COMET entries for {lang} (streaming parser, skipped {items_skipped} invalid items)")
            return lang_entries
    except ImportError:
        pass  # ijson not available, try next strategy
    except Exception as e:
        print(f"  ⚠ Streaming parser failed: {type(e).__name__}, trying chunk-based parsing...")
    
    # Strategy 3: Try to read file in chunks and parse what we can
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Find the data array
        data_start = content.find('"data"')
        if data_start == -1:
            raise ValueError("Could not find 'data' key")
        
        array_start = content.find('[', data_start)
        if array_start == -1:
            raise ValueError("Could not find array start")
        
        # Try to extract and parse JSON objects one by one
        # Use regex to find potential JSON objects (simplified approach)
        import re
        # Pattern to match JSON objects (simplified - looks for { ... } pairs)
        # This is a heuristic approach
        items_parsed = 0
        items_skipped = 0
        
        # Try to split by common patterns and parse individual items
        # Look for item boundaries (}, followed by whitespace and { or ])
        array_content = content[array_start + 1:]
        
        # Try to find complete JSON objects by matching braces
        brace_level = 0
        in_string = False
        escape_next = False
        item_start = None
        pos = 0
        
        while pos < len(array_content):
            char = array_content[pos]
            
            if escape_next:
                escape_next = False
                pos += 1
                continue
            
            if char == '\\':
                escape_next = True
                pos += 1
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                pos += 1
                continue
            
            if in_string:
                pos += 1
                continue
            
            if char == '{':
                if brace_level == 0:
                    item_start = pos
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0 and item_start is not None:
                    # Found a complete object
                    try:
                        item_str = array_content[item_start:pos + 1]
                        item = json.loads(item_str)
                        item_id = item.get("id")
                        if item_id:
                            try:
                                comet_mt = float(item.get("comet_source_mt", 0) or 0)
                            except (ValueError, TypeError):
                                comet_mt = 0.0
                            try:
                                comet_ref = float(item.get("comet_source_target", 0) or 0)
                            except (ValueError, TypeError):
                                comet_ref = 0.0
                            lang_entries[item_id] = {
                                "id": item_id,
                                "file_id": item.get("file_id", item_id.split("/")[0] if "/" in item_id else None),
                                "comet_source_mt": comet_mt,
                                "comet_source_target": comet_ref,
                            }
                            items_parsed += 1
                        else:
                            items_skipped += 1
                    except (json.JSONDecodeError, ValueError):
                        items_skipped += 1
                    item_start = None
            elif char == ']' and brace_level == 0:
                # End of array
                break
            
            pos += 1
        
        if items_parsed > 0:
            print(f"  ✓ Successfully loaded {len(lang_entries)} COMET entries for {lang} (chunk-based parsing, skipped {items_skipped} invalid items)")
            return lang_entries
        else:
            raise ValueError("No valid items could be parsed")
            
    except Exception as e:
        print(f"  ✗ All parsing strategies failed: {type(e).__name__}: {e}")
        return lang_entries  # Return whatever we managed to parse (might be empty)


def load_comet_data(comet_dir):
    """
    Load COMET reference JSON files for each language.
    Handles corrupted JSON files by attempting to read valid portions.
    """
    lang_map = {
        "Hausa": "comet_en2ha.json",
        "Igbo": "comet_en2ig.json",
        "Swahili": "comet_en2sw.json",
        "Yoruba": "comet_en2yo.json",
        "Zulu": "comet_en2zu.json",
    }

    comet_data = {}

    for lang, filename in lang_map.items():
        file_path = os.path.join(comet_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: COMET file not found for {lang}: {file_path}")
            continue
        
        lang_entries = load_comet_data_robust(file_path, lang)
        if lang_entries:
            comet_data[lang] = lang_entries
        else:
            print(f"  ✗ Could not load any COMET data for {lang}")
    
    return comet_data


def match_segment(segment_id, comet_entries):
    if not segment_id or not comet_entries:
        return None
    if segment_id in comet_entries:
        return comet_entries[segment_id]
    if "/" in segment_id:
        file_id = segment_id.split("/")[0]
        for cid, entry in comet_entries.items():
            if cid.startswith(file_id + "/"):
                return entry
        for entry in comet_entries.values():
            if str(entry.get("file_id")) == file_id:
                return entry
    return None


def extract_scores(row):
    t1 = row.get("target1_id")
    t2 = row.get("target2_id")
    s1 = row.get("score_target1", row.get("llm_score_target1"))
    s2 = row.get("score_target2", row.get("llm_score_target2"))
    mt = he = None
    if t1 == "mt-text":
        mt = s1
    elif t1 == "he-text":
        he = s1
    if t2 == "mt-text":
        mt = s2
    elif t2 == "he-text":
        he = s2
    return mt, he


def compute_corr(diffs, labels):
    """Compute correlations and agreement rate.
    
    Note: Correlating continuous (diff) with binary (label) is a point-biserial
    correlation, which is valid but may be less interpretable than agreement rate.
    """
    if len(diffs) < 2:
        return {"count": len(diffs), "spearman": None, "pearson": None, "agreement_rate": None}
    
    # Correlation (point-biserial: continuous vs binary)
    spearman_val = spearmanr(diffs, labels).correlation
    pearson_val = pearsonr(diffs, labels)[0]
    
    # Agreement rate: what % of the time does metric preference match human preference?
    # Metric prefers MT when diff > 0, prefers HE when diff < 0
    # Human prefers MT when label = 1, prefers HE when label = 0
    agreements = sum(1 for d, l in zip(diffs, labels) if (d > 0 and l == 1) or (d < 0 and l == 0))
    agreement_rate = agreements / len(diffs) if len(diffs) > 0 else None
    
    return {
        "count": len(diffs),
        "spearman": spearman_val,
        "pearson": pearson_val,
        "agreement_rate": agreement_rate
    }


def analyze_ssa(ssa_results):
    per_lang = {}
    agg_diffs = []
    agg_labels = []

    for lang, rows in ssa_results.items():
        diffs = []
        labels = []
        for row in rows:
            human_majority = row.get("human_majority")
            if human_majority not in ("MT", "HE"):
                continue
            score_mt, score_he = extract_scores(row)
            if score_mt is None or score_he is None:
                continue
            diffs.append(score_mt - score_he)
            labels.append(1 if human_majority == "MT" else 0)
        per_lang[lang] = compute_corr(diffs, labels)
        agg_diffs.extend(diffs)
        agg_labels.extend(labels)

    overall = compute_corr(agg_diffs, agg_labels)
    return per_lang, overall, agg_diffs, agg_labels


def analyze_comet(ssa_results, comet_data):
    per_lang = {}
    agg_diffs = []
    agg_labels = []

    for lang, rows in ssa_results.items():
        lang_comet = comet_data.get(lang)
        diffs = []
        labels = []
        if not lang_comet:
            per_lang[lang] = {"count": 0, "spearman": None, "pearson": None, "agreement_rate": None}
            continue
        for row in rows:
            human_majority = row.get("human_majority")
            if human_majority not in ("MT", "HE"):
                continue
            comet_entry = match_segment(row.get("segmentID"), lang_comet)
            if not comet_entry:
                continue
            comet_mt = comet_entry.get("comet_source_mt")
            comet_ref = comet_entry.get("comet_source_target")
            if comet_mt is None or comet_ref is None:
                continue
            diffs.append(comet_mt - comet_ref)
            labels.append(1 if human_majority == "MT" else 0)
        per_lang[lang] = compute_corr(diffs, labels)
        agg_diffs.extend(diffs)
        agg_labels.extend(labels)

    overall = compute_corr(agg_diffs, agg_labels)
    return per_lang, overall


def build_summary(ssa_per_lang, comet_per_lang):
    languages = sorted(set(ssa_per_lang.keys()) | set(comet_per_lang.keys()))
    summary = {}
    for lang in languages:
        summary[lang] = {
            "ssa": ssa_per_lang.get(lang, {"count": 0, "spearman": None, "pearson": None, "agreement_rate": None}),
            "comet": comet_per_lang.get(lang, {"count": 0, "spearman": None, "pearson": None, "agreement_rate": None}),
        }
    return summary


def save_summary(summary, overall_ssa, overall_comet, output_dir):
    payload = {
        "per_language": summary,
        "overall": {
            "ssa": overall_ssa,
            "comet": overall_comet,
        },
    }
    out_path = os.path.join(output_dir, "ssa_comet_qe_correlation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved correlation summary JSON to {out_path}")


def print_table(summary, overall_ssa, overall_comet):
    languages = sorted(summary.keys())
    print("\n" + "=" * 120)
    print("SSA-COMET-QE vs Human vs COMET: Correlations and Agreement Rates")
    print("=" * 120)
    print("\nNote: Correlations use point-biserial correlation (continuous diff vs binary preference).")
    print("      Agreement rate shows % of time metric preference matches human preference.")
    print()
    header = (
        f"{'Language':<12}"
        f"{'SSA Spearman':>15}{'SSA Pearson':>15}{'SSA Agree%':>12}"
        f"{'COMET Spearman':>18}{'COMET Pearson':>17}{'COMET Agree%':>15}"
    )
    print(header)
    print("-" * 120)
    for lang in languages:
        ssa = summary[lang]["ssa"]
        comet = summary[lang]["comet"]
        ssa_agree = f"{ssa['agreement_rate']*100:.1f}%" if ssa.get('agreement_rate') is not None else "NA"
        comet_agree = f"{comet['agreement_rate']*100:.1f}%" if comet.get('agreement_rate') is not None else "NA"
        print(
            f"{lang:<12}"
            f"{(ssa['spearman'] if ssa['spearman'] is not None else 'NA'):>15}"
            f"{(ssa['pearson'] if ssa['pearson'] is not None else 'NA'):>15}"
            f"{ssa_agree:>12}"
            f"{(comet['spearman'] if comet['spearman'] is not None else 'NA'):>18}"
            f"{(comet['pearson'] if comet['pearson'] is not None else 'NA'):>17}"
            f"{comet_agree:>15}"
        )
    print("-" * 120)
    overall_ssa_agree = f"{overall_ssa['agreement_rate']*100:.1f}%" if overall_ssa.get('agreement_rate') is not None else "NA"
    overall_comet_agree = f"{overall_comet['agreement_rate']*100:.1f}%" if overall_comet.get('agreement_rate') is not None else "NA"
    print(
        f"{'Overall':<12}"
        f"{(overall_ssa['spearman'] if overall_ssa['spearman'] is not None else 'NA'):>15}"
        f"{(overall_ssa['pearson'] if overall_ssa['pearson'] is not None else 'NA'):>15}"
        f"{overall_ssa_agree:>12}"
        f"{(overall_comet['spearman'] if overall_comet['spearman'] is not None else 'NA'):>18}"
        f"{(overall_comet['pearson'] if overall_comet['pearson'] is not None else 'NA'):>17}"
        f"{overall_comet_agree:>15}"
    )
    print("=" * 120 + "\n")


def plot_bar(summary, output_dir):
    sns.set(style="whitegrid")
    languages = sorted(summary.keys())
    ssa_spear = [summary[lang]["ssa"]["spearman"] or 0 for lang in languages]
    comet_spear = [summary[lang]["comet"]["spearman"] or 0 for lang in languages]
    ssa_pear = [summary[lang]["ssa"]["pearson"] or 0 for lang in languages]
    comet_pear = [summary[lang]["comet"]["pearson"] or 0 for lang in languages]
    ssa_agree = [summary[lang]["ssa"].get("agreement_rate") or 0 for lang in languages]
    comet_agree = [summary[lang]["comet"].get("agreement_rate") or 0 for lang in languages]

    x = np.arange(len(languages))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

    # Plot 1: Spearman correlation
    axes[0].bar(x - width / 2, ssa_spear, width, label="SSA Spearman", color="#4C72B0")
    axes[0].bar(x + width / 2, comet_spear, width, label="COMET Spearman", color="#C44E52")
    axes[0].set_title("Spearman Correlation by Language")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(languages, rotation=45, ha="right")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].legend()
    axes[0].set_ylabel("Correlation")

    # Plot 2: Pearson correlation
    axes[1].bar(x - width / 2, ssa_pear, width, label="SSA Pearson", color="#55A868")
    axes[1].bar(x + width / 2, comet_pear, width, label="COMET Pearson", color="#8172B2")
    axes[1].set_title("Pearson Correlation by Language")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(languages, rotation=45, ha="right")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend()
    axes[1].set_ylabel("Correlation")

    # Plot 3: Agreement rate
    axes[2].bar(x - width / 2, [a * 100 for a in ssa_agree], width, label="SSA Agreement", color="#CCB974")
    axes[2].bar(x + width / 2, [a * 100 for a in comet_agree], width, label="COMET Agreement", color="#64B5CD")
    axes[2].set_title("Agreement Rate by Language")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(languages, rotation=45, ha="right")
    axes[2].axhline(50, color="black", linewidth=0.5, linestyle="--", label="Random (50%)")
    axes[2].legend()
    axes[2].set_ylabel("Agreement Rate (%)")
    axes[2].set_ylim(0, 100)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "ssa_vs_comet_correlation_bars.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved bar plot to {out_path}")


def plot_heatmap(summary, output_dir):
    sns.set(style="white")
    languages = sorted(summary.keys())
    data = []
    cols = ["SSA Spearman", "SSA Pearson", "COMET Spearman", "COMET Pearson"]
    for lang in languages:
        ssa = summary[lang]["ssa"]
        comet = summary[lang]["comet"]
        data.append([
            ssa["spearman"] if ssa["spearman"] is not None else 0,
            ssa["pearson"] if ssa["pearson"] is not None else 0,
            comet["spearman"] if comet["spearman"] is not None else 0,
            comet["pearson"] if comet["pearson"] is not None else 0,
        ])

    plt.figure(figsize=(8, max(4, len(languages) * 0.5)))
    sns.heatmap(
        np.array(data),
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=cols,
        yticklabels=languages,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("SSA-COMET-QE vs COMET correlation heatmap")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "ssa_vs_comet_correlation_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved heatmap to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SSA-COMET-QE results vs human annotations and COMET metrics."
    )
    parser.add_argument("--ssa_results_dir", required=True, help="Directory with SSA-COMET-QE JSON results")
    parser.add_argument("--comet_dir", required=True, help="Directory containing COMET JSON files")
    parser.add_argument("--output_dir", default="./ssa_comet_qe_analysis", help="Directory to write plots and tables")
    parser.add_argument("--debug_rows", type=int, default=0, help="Optional limit per language")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading SSA results from {args.ssa_results_dir}")
    ssa_results = load_ssa_results(args.ssa_results_dir, judge_name="ssa_comet_qe", debug_rows=args.debug_rows)
    if not ssa_results:
        raise ValueError("No SSA-COMET-QE results found.")

    print(f"Loading COMET reference metrics from {args.comet_dir}")
    comet_data = load_comet_data(args.comet_dir)

    print("Computing correlations for SSA-COMET-QE...")
    ssa_per_lang, ssa_overall, _, _ = analyze_ssa(ssa_results)

    print("Computing correlations for COMET metrics...")
    comet_per_lang, comet_overall = analyze_comet(ssa_results, comet_data)

    summary = build_summary(ssa_per_lang, comet_per_lang)
    save_summary(summary, ssa_overall, comet_overall, args.output_dir)
    print_table(summary, ssa_overall, comet_overall)
    plot_bar(summary, args.output_dir)
    plot_heatmap(summary, args.output_dir)

    print(f"Finished analysis. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
import argparse
import json
import os
import glob
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

def load_results(results_dir, debug_rows=None, model_type=None):
    """Load results from directory."""
    if model_type:
        # Handle both formats: model_type and model_type with underscores
        model_name = model_type.replace('-', '_')
        combined_pattern = os.path.join(results_dir, f"combined_{model_name}_results.json")
        if os.path.exists(combined_pattern):
            with open(combined_pattern, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            results = defaultdict(list)
            # Try both formats
            for pattern in [f"*_{model_name}_results.json", f"*_{model_type}_results.json"]:
                for fp in glob.glob(os.path.join(results_dir, pattern)):
                    with open(fp, "r", encoding="utf-8") as f:
                        arr = json.load(f)
                    lang = os.path.basename(fp).split("_")[0]
                    results[lang].extend(arr)
            data = dict(results)
    else:
        combined_paths = glob.glob(os.path.join(results_dir, "combined_*_results.json"))
        if combined_paths:
            llama_path = os.path.join(results_dir, "combined_llama_results.json")
            aya_path = os.path.join(results_dir, "combined_aya_results.json")
            if os.path.exists(llama_path):
                with open(llama_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif os.path.exists(aya_path):
                with open(aya_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(combined_paths[0], "r", encoding="utf-8") as f:
                    data = json.load(f)
        else:
            results = defaultdict(list)
            for fp in glob.glob(os.path.join(results_dir, "*_*_results.json")):
                with open(fp, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                lang = os.path.basename(fp).split("_")[0]
                results[lang].extend(arr)
            data = dict(results)
    
    if debug_rows and debug_rows > 0:
        debug_data = {}
        for lang, rows in data.items():
            seen_segments = set()
            debug_rows_list = []
            for row in rows:
                seg_id = row.get('segmentID', '')
                if seg_id not in seen_segments:
                    seen_segments.add(seg_id)
                    debug_rows_list.append(row)
                    if len(seen_segments) >= debug_rows:
                        break
            debug_data[lang] = [r for r in rows if r.get('segmentID', '') in seen_segments]
        return debug_data
    
    return data

def compute_correlations(results_by_lang):
    """
    Compute Spearman and Pearson correlations following SSA-COMET methodology.
    For pairwise comparison, we convert human preferences to scores and compare with LLM scores.
    """
    correlations = {}
    
    for lang, rows in results_by_lang.items():
        # Extract human and LLM scores
        human_scores = []
        llm_scores = []
        
        for row in rows:
            # Human preference: convert MT/HE to numeric score
            # For pairwise: if human prefers MT, assign higher score to MT translation
            human_majority = row.get('human_majority', '')
            target1_id = row.get('target1_id', '')
            target2_id = row.get('target2_id', '')
            
            # Get scores for both translations (handle both llm_score and score fields)
            score1 = row.get('llm_score_target1', row.get('score_target1', 0.5))
            score2 = row.get('llm_score_target2', row.get('score_target2', 0.5))
            
            # Determine which translation human prefers
            if human_majority == 'MT':
                if target1_id == 'mt-text':
                    human_score = 1.0  # Prefer target1
                    model_score = score1
                else:
                    human_score = 0.0  # Prefer target2
                    model_score = score2
            elif human_majority == 'HE':
                if target1_id == 'he-text':
                    human_score = 1.0  # Prefer target1
                    model_score = score1
                else:
                    human_score = 0.0  # Prefer target2
                    model_score = score2
            else:
                continue  # Skip invalid entries
            
            human_scores.append(human_score)
            llm_scores.append(model_score)
        
        if len(human_scores) > 1:
            # Compute Spearman and Pearson correlations
            spearman_corr, spearman_p = spearmanr(human_scores, llm_scores)
            pearson_corr, pearson_p = pearsonr(human_scores, llm_scores)
            
            correlations[lang] = {
                'spearman': spearman_corr if not np.isnan(spearman_corr) else 0.0,
                'spearman_p': spearman_p if not np.isnan(spearman_p) else 1.0,
                'pearson': pearson_corr if not np.isnan(pearson_corr) else 0.0,
                'pearson_p': pearson_p if not np.isnan(pearson_p) else 1.0,
                'n': len(human_scores)
            }
        else:
            correlations[lang] = {
                'spearman': 0.0,
                'spearman_p': 1.0,
                'pearson': 0.0,
                'pearson_p': 1.0,
                'n': 0
            }
    
    return correlations

def plot_correlation_comparison(all_correlations, outdir):
    """Plot Spearman and Pearson correlations for all models"""
    sns.set(style="whitegrid")
    
    # Get all languages from all correlation dicts
    all_languages = set()
    for corr_dict in all_correlations.values():
        all_languages.update(corr_dict.keys())
    languages = sorted(all_languages)
    
    # Prepare data for each model
    model_names = list(all_correlations.keys())
    n_models = len(model_names)
    
    if n_models == 0:
        print("No correlation data to plot")
        return
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(languages))
    width = 0.8 / n_models  # Adjust width based on number of models
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
    
    # Plot Spearman correlations
    for i, model_name in enumerate(model_names):
        spearman_vals = [all_correlations[model_name].get(lang, {}).get('spearman', 0.0) for lang in languages]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax1.bar(x + offset, spearman_vals, width, label=model_name.replace('_', ' ').title(), 
                       color=colors[i % len(colors)], alpha=0.7)
        
        # Add value labels
        for j, val in enumerate(spearman_vals):
            if val != 0:
                ax1.text(j + offset, val + 0.01 if val >= 0 else val - 0.03, f'{val:.3f}', 
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)
    
    ax1.set_xlabel('Language')
    ax1.set_ylabel('Spearman Correlation')
    ax1.set_title('Spearman Correlation: Model Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(languages, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot Pearson correlations
    for i, model_name in enumerate(model_names):
        pearson_vals = [all_correlations[model_name].get(lang, {}).get('pearson', 0.0) for lang in languages]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax2.bar(x + offset, pearson_vals, width, label=model_name.replace('_', ' ').title(), 
                      color=colors[i % len(colors)], alpha=0.7)
        
        # Add value labels
        for j, val in enumerate(pearson_vals):
            if val != 0:
                ax2.text(j + offset, val + 0.01 if val >= 0 else val - 0.03, f'{val:.3f}', 
                        ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)
    
    ax2.set_xlabel('Language')
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Pearson Correlation: Model Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(languages, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "correlation_comparison.png"), dpi=200)
    plt.close()
    
    # Plot 1: Spearman correlation comparison
    x = np.arange(len(languages))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    bars1 = ax1.bar(x - width/2, llama_spearman, width, label='Llama', color='#4C72B0', alpha=0.7)
    bars2 = ax1.bar(x + width/2, aya_spearman, width, label='AYA', color='#55A868', alpha=0.7)
    
    # Add value labels on bars
    for i, (ls, as_) in enumerate(zip(llama_spearman, aya_spearman)):
        if ls != 0:
            ax1.text(i - width/2, ls + 0.01 if ls >= 0 else ls - 0.03, f'{ls:.3f}', 
                    ha='center', va='bottom' if ls >= 0 else 'top', fontsize=8)
        if as_ != 0:
            ax1.text(i + width/2, as_ + 0.01 if as_ >= 0 else as_ - 0.03, f'{as_:.3f}', 
                    ha='center', va='bottom' if as_ >= 0 else 'top', fontsize=8)
    
    ax1.set_xlabel('Language')
    ax1.set_ylabel('Spearman Correlation')
    ax1.set_title('Spearman Correlation: Llama vs AYA')
    ax1.set_xticks(x)
    ax1.set_xticklabels(languages, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Pearson correlation comparison
    bars3 = ax2.bar(x - width/2, llama_pearson, width, label='Llama', color='#4C72B0', alpha=0.7)
    bars4 = ax2.bar(x + width/2, aya_pearson, width, label='AYA', color='#55A868', alpha=0.7)
    
    # Add value labels on bars
    for i, (lp, ap) in enumerate(zip(llama_pearson, aya_pearson)):
        if lp != 0:
            ax2.text(i - width/2, lp + 0.01 if lp >= 0 else lp - 0.03, f'{lp:.3f}', 
                    ha='center', va='bottom' if lp >= 0 else 'top', fontsize=8)
        if ap != 0:
            ax2.text(i + width/2, ap + 0.01 if ap >= 0 else ap - 0.03, f'{ap:.3f}', 
                    ha='center', va='bottom' if ap >= 0 else 'top', fontsize=8)
    
    ax2.set_xlabel('Language')
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Pearson Correlation: Llama vs AYA')
    ax2.set_xticks(x)
    ax2.set_xticklabels(languages, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "correlation_comparison.png"), dpi=200)
    plt.close()

def plot_correlation_heatmap(all_correlations, outdir):
    """Plot heatmap of correlations across languages and metrics"""
    sns.set(style="white")
    
    # Get all languages
    all_languages = set()
    for corr_dict in all_correlations.values():
        all_languages.update(corr_dict.keys())
    languages = sorted(all_languages)
    
    # Prepare data matrix
    model_names = sorted(all_correlations.keys())
    columns = []
    data = []
    
    for lang in languages:
        row = []
        for model_name in model_names:
            row.append(all_correlations[model_name].get(lang, {}).get('spearman', 0.0))
            row.append(all_correlations[model_name].get(lang, {}).get('pearson', 0.0))
        data.append(row)
    
    # Create column names
    for model_name in model_names:
        display_name = model_name.replace('_', ' ').title()
        columns.append(f'{display_name} Spearman')
        columns.append(f'{display_name} Pearson')
    
    df = pd.DataFrame(data, index=languages, columns=columns)
    
    # Plot heatmap
    plt.figure(figsize=(10, max(6, len(languages) * 0.5)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                vmin=-1, vmax=1, center=0, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Heatmap: Llama vs AYA (SSA-COMET Metrics)')
    plt.xlabel('Model and Metric')
    plt.ylabel('Language')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "correlation_heatmap.png"), dpi=200)
    plt.close()

def plot_average_correlations(all_correlations, outdir):
    """Plot average correlations across all languages"""
    sns.set(style="whitegrid")
    
    # Get all languages
    all_languages = set()
    for corr_dict in all_correlations.values():
        all_languages.update(corr_dict.keys())
    languages = sorted(all_languages)
    
    # Calculate averages for each model
    model_names = sorted(all_correlations.keys())
    n_models = len(model_names)
    
    if n_models == 0:
        print("No correlation data to plot")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 6))
    
    x = np.arange(2)  # Spearman and Pearson
    width = 0.8 / n_models
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
    
    for i, model_name in enumerate(model_names):
        spearman_avg = np.mean([all_correlations[model_name].get(lang, {}).get('spearman', 0.0) for lang in languages])
        pearson_avg = np.mean([all_correlations[model_name].get(lang, {}).get('pearson', 0.0) for lang in languages])
        
        offset = (i - n_models/2 + 0.5) * width
        display_name = model_name.replace('_', ' ').title()
        
        bars = ax.bar(x + offset, [spearman_avg, pearson_avg], width, 
                     label=display_name, color=colors[i % len(colors)], alpha=0.7)
        
        # Add value labels
        ax.text(0 + offset, spearman_avg + 0.01, f'{spearman_avg:.3f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(1 + offset, pearson_avg + 0.01, f'{pearson_avg:.3f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Average Correlation')
    ax.set_title('Average Correlation Across All Languages (SSA-COMET Metrics)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Spearman', 'Pearson'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "average_correlations.png"), dpi=200)
    plt.close()

def print_correlation_table(all_correlations):
    """Print correlation table in SSA-COMET format"""
    # Get all languages
    all_languages = set()
    for corr_dict in all_correlations.values():
        all_languages.update(corr_dict.keys())
    languages = sorted(all_languages)
    
    model_names = sorted(all_correlations.keys())
    
    print("\n" + "="*100)
    print("SSA-COMET Correlation Results")
    print("="*100)
    
    # Print header
    header = f"{'Language':<15}"
    for model_name in model_names:
        display_name = model_name.replace('_', ' ').title()
        header += f" {display_name} Spearman{'':<10} {display_name} Pearson{'':<10}"
    print(header)
    print("-"*100)
    
    # Print per-language results
    for lang in languages:
        row = f"{lang:<15}"
        for model_name in model_names:
            s = all_correlations[model_name].get(lang, {}).get('spearman', 0.0)
            p = all_correlations[model_name].get(lang, {}).get('pearson', 0.0)
            row += f" {s:>17.3f} {p:>17.3f}"
        print(row)
    
    # Print averages
    print("-"*100)
    avg_row = f"{'Average':<15}"
    for model_name in model_names:
        s_avg = np.mean([all_correlations[model_name].get(lang, {}).get('spearman', 0.0) for lang in languages])
        p_avg = np.mean([all_correlations[model_name].get(lang, {}).get('pearson', 0.0) for lang in languages])
        avg_row += f" {s_avg:>17.3f} {p_avg:>17.3f}"
    print(avg_row)
    print("="*100 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Plot SSA-COMET evaluation results (LLM-as-judge and SSA-COMET model)")
    
    parser.add_argument('--results_dirs', nargs='+', required=True,
                       help="Directories containing results (can specify multiple)")
    parser.add_argument('--model_types', nargs='+', required=True,
                       help="Model types corresponding to each results_dir (e.g., llama aya ssa_comet_qe)")
    parser.add_argument('--output_dir', default='./llm_as_judge_plots_ssa_comet/',
                       help="Output directory for plots")
    parser.add_argument('--debug_rows', type=int, default=0,
                       help="Limit to first N segments per language (0 for all)")
    
    args = parser.parse_args()
    
    if len(args.results_dirs) != len(args.model_types):
        raise ValueError("Number of results_dirs must match number of model_types")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results for all models
    all_results = {}
    for results_dir, model_type in zip(args.results_dirs, args.model_types):
        print(f"Loading {model_type} results from {results_dir}...")
        results = load_results(results_dir, 
                              debug_rows=args.debug_rows if args.debug_rows > 0 else None,
                              model_type=model_type)
        all_results[model_type] = results
    
    # Compute correlations for all models
    print("Computing correlations...")
    all_correlations = {}
    for model_type, results in all_results.items():
        correlations = compute_correlations(results)
        all_correlations[model_type] = correlations
    
    # Print correlation table
    print_correlation_table(all_correlations)
    
    # Generate plots
    print("Generating plots...")
    plot_correlation_comparison(all_correlations, args.output_dir)
    plot_correlation_heatmap(all_correlations, args.output_dir)
    plot_average_correlations(all_correlations, args.output_dir)
    
    print(f"Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

