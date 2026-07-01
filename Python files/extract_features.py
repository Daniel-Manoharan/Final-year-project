"""
=============================================================
  Hybrid Malware Detection — Feature Extraction Script
  Extracts all 113 columns from PE binary files

  Features Extracted:
  - Entropy Analytics        (11 columns)
  - Opcode Density & Flow    ( 9 columns)
  - Top 50 Monograms         (50 columns)
  - Top 30 Bigrams           (30 columns)
  - PE Structural Metadata   ( 9 columns)
  - Packer Indicators        ( 3 columns)
  - Label                    ( 1 column)
  ─────────────────────────────────────
  TOTAL                      113 columns

  Usage:
    python extract_features.py

  Requirements:
    pip install pefile capstone numpy pandas tqdm
=============================================================
"""

import os
import math
import struct
import hashlib
import warnings
import numpy as np
import pandas as pd
import pefile
import capstone
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  CONFIGURATION — Update these paths to match your setup
# ─────────────────────────────────────────────────────────
MALWARE_DIR   = r"dataset\malware"    # Folder with malware .exe files
BENIGN_DIR    = r"dataset\benign"     # Folder with benign  .exe files
OUTPUT_CSV    = r"features\combined_features.csv"
ERROR_LOG     = r"features\errors.log"

os.makedirs("features", exist_ok=True)

# ─────────────────────────────────────────────────────────
#  TOP 50 OPCODES TO TRACK (monograms)
# ─────────────────────────────────────────────────────────
TOP_50_OPCODES = [
    "mov", "push", "pop", "call", "ret", "jmp", "nop",
    "xor", "add", "sub", "and", "or",  "not", "inc",
    "dec", "cmp", "test", "lea", "imul", "idiv",
    "mul", "div", "shl", "shr", "rol", "ror",
    "je",  "jne", "jz",  "jnz", "jl",  "jle",
    "jg",  "jge", "jb",  "jbe", "ja",  "jae",
    "int", "hlt", "leave", "enter", "retn",
    "movzx", "movsx", "cdq", "cwde", "cbw",
    "fld", "fst", "fstp", "fadd"
]

# ─────────────────────────────────────────────────────────
#  TOP 30 BIGRAMS TO TRACK
# ─────────────────────────────────────────────────────────
TOP_30_BIGRAMS = [
    "push_call",   "mov_ret",    "xor_xor",    "jmp_nop",
    "cmp_jne",     "mov_push",   "push_mov",   "call_mov",
    "mov_mov",     "push_push",  "add_mov",    "sub_mov",
    "mov_cmp",     "cmp_je",     "lea_mov",    "mov_lea",
    "push_ret",    "call_ret",   "nop_nop",    "mov_add",
    "mov_sub",     "mov_xor",    "xor_mov",    "push_xor",
    "call_push",   "ret_push",   "inc_cmp",    "dec_jnz",
    "mov_jmp",     "cmp_jz"
]

# ─────────────────────────────────────────────────────────
#  SUSPICIOUS IMPORTS LIST
# ─────────────────────────────────────────────────────────
SUSPICIOUS_IMPORTS = {
    "VirtualAlloc", "VirtualAllocEx", "WriteProcessMemory",
    "CreateRemoteThread", "NtUnmapViewOfSection", "VirtualProtect",
    "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
    "GetTickCount", "QueryPerformanceCounter", "NtQueryInformationProcess",
    "WSAStartup", "connect", "HttpSendRequest", "InternetOpen",
    "CreateFileA", "WriteFile", "DeleteFileA", "CopyFileA",
    "RegSetValueEx", "RegCreateKeyEx", "RegDeleteKeyA",
    "ShellExecuteA", "WinExec", "CreateProcessA", "CreateProcessW",
    "LoadLibraryA", "GetProcAddress", "SetWindowsHookEx",
    "OpenProcess", "TerminateProcess"
}

# ─────────────────────────────────────────────────────────
#  KNOWN PACKER SIGNATURES (section names)
# ─────────────────────────────────────────────────────────
PACKER_SECTION_NAMES = {
    ".upx0", ".upx1", ".upx2",           # UPX
    ".aspack", ".adata",                  # ASPack
    ".themida", ".winlice",               # Themida/WinLicense
    ".packed", ".pack",                   # Generic
    ".nsp0", ".nsp1", ".nsp2",           # NsPack
    ".petite",                            # Petite
    "!epacked", "!gambit",               # Various
    ".wwpack", ".svkp",                  # WWPack/SVKP
    "pec1", "pec2",                      # PECompact
}


# ═════════════════════════════════════════════════════════
#  BLOCK 1: ENTROPY ANALYTICS (11 features)
# ═════════════════════════════════════════════════════════

def calculate_entropy(data):
    """Calculate Shannon entropy of byte data."""
    if not data or len(data) == 0:
        return 0.0
    freq = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 6)


def sliding_window_entropy(data, window_size=256):
    """Calculate entropy over sliding windows."""
    if len(data) < window_size:
        return calculate_entropy(data), calculate_entropy(data)
    entropies = []
    for i in range(0, len(data) - window_size, window_size // 2):
        window = data[i:i + window_size]
        entropies.append(calculate_entropy(window))
    if not entropies:
        return 0.0, 0.0
    return round(float(np.mean(entropies)), 6), round(float(np.max(entropies)), 6)


def extract_entropy_features(pe, file_data):
    """Extract all 11 entropy features."""
    features = {}

    # Per-section entropy
    section_entropies = []
    entropy_text  = None
    entropy_data  = None
    section_names = []

    for section in pe.sections:
        try:
            sec_data = section.get_data()
            ent      = calculate_entropy(sec_data)
            section_entropies.append(ent)
            name = section.Name.decode("utf-8", errors="ignore").rstrip("\x00").lower()
            section_names.append(name)
            if ".text" in name:
                entropy_text = ent
            if ".data" in name:
                entropy_data = ent
        except:
            continue

    if not section_entropies:
        section_entropies = [0.0]

    entropy_mean = round(float(np.mean(section_entropies)), 6)

    # Fallback if .text or .data not found
    if entropy_text is None:
        entropy_text = entropy_mean
    if entropy_data is None:
        entropy_data = entropy_mean

    # Section name entropy
    if section_names:
        name_bytes = " ".join(section_names).encode()
        sec_name_entropy = calculate_entropy(name_bytes)
    else:
        sec_name_entropy = 0.0

    # Entry point entropy
    try:
        ep  = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        ep_data = file_data[ep:ep + 256] if ep < len(file_data) else b""
        entry_point_entropy = calculate_entropy(ep_data)
    except:
        entry_point_entropy = 0.0

    # Sliding window entropy on full file
    sw_mean, sw_max = sliding_window_entropy(file_data)

    features["entropy_mean"]            = entropy_mean
    features["entropy_max"]             = round(float(np.max(section_entropies)), 6)
    features["entropy_min"]             = round(float(np.min(section_entropies)), 6)
    features["entropy_std"]             = round(float(np.std(section_entropies)), 6)
    features["entropy_text_section"]    = round(entropy_text, 6)
    features["entropy_data_section"]    = round(entropy_data, 6)
    features["high_entropy_sections"]   = int(sum(1 for e in section_entropies if e > 7.0))
    features["sliding_window_mean"]     = sw_mean
    features["sliding_window_max"]      = sw_max
    features["entry_point_entropy"]     = round(entry_point_entropy, 6)
    features["section_name_entropy"]    = round(sec_name_entropy, 6)

    return features


# ═════════════════════════════════════════════════════════
#  BLOCK 2 & 3 & 4: OPCODE FEATURES (89 features)
# ═════════════════════════════════════════════════════════

def extract_opcode_features(pe, file_data):
    """Extract opcode density, top 50 monograms, top 30 bigrams."""
    features     = {}
    opcode_list  = []
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    md.detail = False

    # Disassemble all executable sections
    for section in pe.sections:
        try:
            if section.Characteristics & 0x20000000:  # Executable flag
                sec_data = section.get_data()
                sec_va   = section.VirtualAddress
                for insn in md.disasm(sec_data, sec_va):
                    opcode_list.append(insn.mnemonic.lower().strip())
        except:
            continue

    total_count  = len(opcode_list)
    unique_count = len(set(opcode_list))
    safe_total   = total_count if total_count > 0 else 1

    # ── Block 2: Opcode Density & Flow (9 features) ──────
    features["opcode_total_count"]  = total_count
    features["opcode_unique_count"] = unique_count
    features["opcode_diversity"]    = round(unique_count / safe_total, 6)
    features["nop_ratio"]           = round(opcode_list.count("nop")  / safe_total, 6)
    features["call_ratio"]          = round(opcode_list.count("call") / safe_total, 6)
    features["jmp_ratio"]           = round(opcode_list.count("jmp")  / safe_total, 6)
    features["xor_ratio"]           = round(opcode_list.count("xor")  / safe_total, 6)
    features["push_ratio"]          = round(opcode_list.count("push") / safe_total, 6)
    features["mov_ratio"]           = round(opcode_list.count("mov")  / safe_total, 6)

    # ── Block 3: Top 50 Monograms (50 features) ──────────
    opcode_counts = Counter(opcode_list)
    for op in TOP_50_OPCODES:
        features[f"op_{op}"] = round(opcode_counts.get(op, 0) / safe_total, 6)

    # ── Block 4: Top 30 Bigrams (30 features) ────────────
    bigram_counts = Counter()
    for i in range(len(opcode_list) - 1):
        pair = f"{opcode_list[i]}_{opcode_list[i+1]}"
        bigram_counts[pair] += 1

    total_bigrams = sum(bigram_counts.values())
    safe_bigrams  = total_bigrams if total_bigrams > 0 else 1

    for bg in TOP_30_BIGRAMS:
        features[f"bg_{bg}"] = round(bigram_counts.get(bg, 0) / safe_bigrams, 6)

    return features


# ═════════════════════════════════════════════════════════
#  BLOCK 5: PE STRUCTURAL METADATA (9 features)
# ═════════════════════════════════════════════════════════

def extract_pe_features(pe, filepath):
    """Extract PE structural metadata features."""
    features = {}

    # File size
    features["file_size"] = os.path.getsize(filepath)

    # Section count
    features["section_count"] = len(pe.sections)

    # Imports count
    try:
        imports = sum(
            len(entry.imports)
            for entry in pe.DIRECTORY_ENTRY_IMPORT
        ) if hasattr(pe, "DIRECTORY_ENTRY_IMPORT") else 0
        features["imports_count"] = imports
    except:
        features["imports_count"] = 0

    # Exports count
    try:
        exports = len(pe.DIRECTORY_ENTRY_EXPORT.symbols) \
                  if hasattr(pe, "DIRECTORY_ENTRY_EXPORT") else 0
        features["exports_count"] = exports
    except:
        features["exports_count"] = 0

    # Virtual size ratio
    try:
        virtual_size = sum(s.Misc_VirtualSize for s in pe.sections)
        raw_size     = sum(s.SizeOfRawData    for s in pe.sections)
        features["virtual_size_ratio"] = round(
            virtual_size / raw_size if raw_size > 0 else 0.0, 6
        )
    except:
        features["virtual_size_ratio"] = 0.0

    # Average bytes per section
    try:
        features["avg_bytes_per_section"] = round(
            features["file_size"] / features["section_count"]
            if features["section_count"] > 0 else 0.0, 6
        )
    except:
        features["avg_bytes_per_section"] = 0.0

    # Has TLS
    try:
        features["has_tls"] = int(
            hasattr(pe, "DIRECTORY_ENTRY_TLS") and
            pe.DIRECTORY_ENTRY_TLS is not None
        )
    except:
        features["has_tls"] = 0

    # Has debug
    try:
        features["has_debug"] = int(
            hasattr(pe, "DIRECTORY_ENTRY_DEBUG") and
            pe.DIRECTORY_ENTRY_DEBUG is not None
        )
    except:
        features["has_debug"] = 0

    # Import entropy (entropy of imported function names)
    try:
        import_names = []
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        import_names.append(imp.name.decode("utf-8", errors="ignore"))
        name_str = " ".join(import_names).encode()
        features["import_entropy"] = round(calculate_entropy(name_str), 6)
    except:
        features["import_entropy"] = 0.0

    return features


# ═════════════════════════════════════════════════════════
#  BLOCK 6: PACKER INDICATORS (3 features)
# ═════════════════════════════════════════════════════════

def extract_packer_features(pe):
    """Extract packer/obfuscation indicator features."""
    features = {}

    # Packer signature (0 or 1)
    packer_found = 0
    for section in pe.sections:
        try:
            name = section.Name.decode("utf-8", errors="ignore")\
                         .rstrip("\x00").lower()
            if name in PACKER_SECTION_NAMES:
                packer_found = 1
                break
        except:
            continue
    features["packer_signature"] = packer_found

    # Suspicious import count
    suspicious_count = 0
    try:
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        name = imp.name.decode("utf-8", errors="ignore")
                        if name in SUSPICIOUS_IMPORTS:
                            suspicious_count += 1
    except:
        pass
    features["suspicious_import_count"] = suspicious_count

    # Suspicious section count (non-standard + high entropy)
    standard_sections = {".text",".data",".rdata",".bss",".rsrc",
                         ".reloc",".idata",".edata",".pdata",".tls"}
    suspicious_sections = 0
    for section in pe.sections:
        try:
            name = section.Name.decode("utf-8", errors="ignore")\
                         .rstrip("\x00").lower()
            if name not in standard_sections:
                suspicious_sections += 1
        except:
            continue
    features["suspicious_section_count"] = suspicious_sections

    return features


# ═════════════════════════════════════════════════════════
#  MAIN EXTRACTOR — Combines all blocks
# ═════════════════════════════════════════════════════════

def extract_all_features(filepath, label):
    """
    Extract all 113 features from a PE file.
    Returns a dict with all features + label.
    """
    row = {"file_name": os.path.basename(filepath)}

    try:
        # Load raw bytes
        with open(filepath, "rb") as f:
            file_data = f.read()

        # Parse PE
        pe = pefile.PE(data=file_data, fast_load=False)
        pe.parse_data_directories()

        # Extract all blocks
        row.update(extract_entropy_features(pe, file_data))  # 11 features
        row.update(extract_opcode_features(pe, file_data))   # 89 features
        row.update(extract_pe_features(pe, filepath))        #  9 features
        row.update(extract_packer_features(pe))              #  3 features

        pe.close()

    except Exception as e:
        # Return zeroed row on failure
        row["error"] = str(e)
        for col in get_all_column_names():
            if col not in row:
                row[col] = 0

    row["label"] = label
    return row


def get_all_column_names():
    """Return the complete list of 113 feature column names."""
    cols = []
    # Entropy (11)
    cols += ["entropy_mean","entropy_max","entropy_min","entropy_std",
             "entropy_text_section","entropy_data_section",
             "high_entropy_sections","sliding_window_mean",
             "sliding_window_max","entry_point_entropy","section_name_entropy"]
    # Opcode density (9)
    cols += ["opcode_total_count","opcode_unique_count","opcode_diversity",
             "nop_ratio","call_ratio","jmp_ratio","xor_ratio","push_ratio","mov_ratio"]
    # Top 50 monograms
    cols += [f"op_{op}" for op in TOP_50_OPCODES]
    # Top 30 bigrams
    cols += [f"bg_{bg}" for bg in TOP_30_BIGRAMS]
    # PE structure (9)
    cols += ["file_size","section_count","imports_count","exports_count",
             "virtual_size_ratio","avg_bytes_per_section",
             "has_tls","has_debug","import_entropy"]
    # Packer (3)
    cols += ["packer_signature","suspicious_import_count","suspicious_section_count"]
    return cols


# ═════════════════════════════════════════════════════════
#  MAIN — Process all files
# ═════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Hybrid Malware Detection — Feature Extraction")
    print("  113 columns × 8000 files")
    print("=" * 62)

    # Collect all file paths
    malware_files = [
        (os.path.join(MALWARE_DIR, f), 1)
        for f in os.listdir(MALWARE_DIR)
        if f.endswith(".exe")
    ]
    benign_files  = [
        (os.path.join(BENIGN_DIR, f), 0)
        for f in os.listdir(BENIGN_DIR)
        if f.endswith(".exe")
    ]

    all_files = malware_files + benign_files
    print(f"\n  Malware files : {len(malware_files)}")
    print(f"  Benign files  : {len(benign_files)}")
    print(f"  Total files   : {len(all_files)}")
    print(f"\n  Output CSV    : {OUTPUT_CSV}")
    print(f"  Error log     : {ERROR_LOG}")
    print()

    if not all_files:
        print("❌ No .exe files found! Check your dataset paths.")
        print(f"   Malware dir: {MALWARE_DIR}")
        print(f"   Benign dir : {BENIGN_DIR}")
        return

    # Extract features
    rows   = []
    errors = []

    for filepath, label in tqdm(all_files, desc="Extracting features"):
        row = extract_all_features(filepath, label)
        if "error" in row:
            errors.append(f"{filepath}: {row['error']}")
        rows.append(row)

    # Build DataFrame
    print(f"\n[*] Building dataset...")
    df = pd.DataFrame(rows)

    # Reorder columns: file_name first, label last
    feature_cols = get_all_column_names()
    available    = [c for c in feature_cols if c in df.columns]
    df = df[["file_name"] + available + ["label"]]

    # Fill any NaN with 0
    df.fillna(0, inplace=True)

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)

    # Save error log
    if errors:
        with open(ERROR_LOG, "w") as f:
            f.write("\n".join(errors))

    # ── Final Summary ─────────────────────────────────────
    print("\n" + "=" * 62)
    print("  EXTRACTION COMPLETE")
    print("=" * 62)
    print(f"  ✅ Total rows      : {len(df)}")
    print(f"  ✅ Total columns   : {len(df.columns)} (incl. file_name + label)")
    print(f"  ✅ Malware rows    : {len(df[df['label']==1])}")
    print(f"  ✅ Benign rows     : {len(df[df['label']==0])}")
    print(f"  ❌ Errors          : {len(errors)}")
    print(f"  📄 CSV saved to    : {OUTPUT_CSV}")
    print("=" * 62)
    print()
    print("  Sample of your dataset:")
    print(df[["file_name","entropy_mean","entropy_max",
              "opcode_total_count","packer_signature","label"]].head(5).to_string())
    print()
    print("  Next step: Run train_model.py to train XGBoost!")
    print("=" * 62)


if __name__ == "__main__":
    main()
