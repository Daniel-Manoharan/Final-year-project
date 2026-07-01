"""
Static feature extractor for Machine Learning-Based Metamorphic Malware Detection.

Produces the exact 114-feature vector documented in the interim report:
    - 12  Entropy features           (global + section + sliding-window + entry-point)
    - 63  Opcode frequency features  (50 opcode counts + 10 ratios + 3 aggregates)
    - 20  Opcode bigram features
    - 19  PE structural features
    = 114 features + sha256 + label

Static-only: never executes the sample. Designed to be run inside the isolated
analysis VM described in the proposal (Section 8.1).
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64


# ---------------------------------------------------------------------------
# Feature vocabulary (fixed order — do not reorder without retraining)
# ---------------------------------------------------------------------------

OPCODES_TRACKED: List[str] = [
    "mov", "push", "pop", "call", "ret", "jmp", "je", "jne", "jz", "jnz",
    "jl", "jle", "jg", "jge", "ja", "jb", "cmp", "test", "add", "sub",
    "mul", "div", "inc", "dec", "and", "or", "xor", "not", "shl", "shr",
    "lea", "nop", "int", "in", "out", "rep", "movs", "stos", "lods", "cmps",
    "scas", "xchg", "loop", "hlt", "cli", "sti", "pushf", "popf", "cdq", "leave",
]  # 50

BIGRAMS_TRACKED: List[Tuple[str, str]] = [
    ("push", "call"), ("mov", "ret"), ("cmp", "jne"), ("cmp", "je"),
    ("test", "jz"),  ("test", "jnz"), ("push", "push"), ("mov", "mov"),
    ("xor", "xor"),  ("call", "mov"), ("mov", "call"), ("push", "mov"),
    ("mov", "push"), ("lea", "mov"),  ("mov", "lea"),  ("add", "mov"),
    ("sub", "mov"),  ("cmp", "ja"),   ("cmp", "jb"),   ("xor", "mov"),
]  # 20

# Common packer / obfuscator section-name signatures.
PACKER_SECTIONS = {
    b"UPX0", b"UPX1", b"UPX2", b".UPX0", b".UPX1", b".aspack", b".adata",
    b"ASPack", b".packed", b".petite", b".WWPACK", b".yP", b".y0da",
    b"PEBundle", b"MEW", b"FSG!", b".mackt", b".MPRESS1", b".MPRESS2",
    b"nsp0", b"nsp1", b"nsp2", b".vmp0", b".vmp1", b".vmp2", b".enigma1",
    b".enigma2", b".themida", b"Themida",
}

# Imports commonly abused by malware.
SUSPICIOUS_IMPORTS = {
    b"virtualalloc",       b"virtualprotect",   b"writeprocessmemory",
    b"createremotethread", b"loadlibrarya",     b"loadlibraryw",
    b"getprocaddress",     b"winexec",          b"shellexecutea",
    b"createprocessa",     b"createprocessw",   b"setwindowshookexa",
    b"regopenkeyexa",      b"regsetvalueexa",   b"regcreatekeyexa",
    b"internetopena",      b"internetopenw",    b"internetopenurla",
    b"httpsendrequesta",   b"urldownloadtofilea", b"wsastartup",
    b"socket",             b"connect",          b"send", b"recv",
    b"cryptencrypt",       b"cryptdecrypt",     b"cryptacquirecontexta",
    b"ntunmapviewofsection", b"zwunmapviewofsection",
    b"setthreadcontext",   b"resumethread",     b"queueuserapc",
    b"isdebuggerpresent",  b"checkremotedebuggerpresent",
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """Holds the 114-feature vector plus identity + label."""
    sha256: str
    label: Optional[int] = None            # 1 = malware, 0 = benign, None = unknown
    features: Dict[str, float] = field(default_factory=dict)

    def to_row(self, feature_order: List[str]) -> List:
        return [self.sha256, *[self.features.get(k, 0.0) for k in feature_order], self.label]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature_order() -> List[str]:
    """Canonical order for the 114 feature columns. Frozen once trained."""
    entropy_cols = [
        "entropy_mean", "entropy_max", "entropy_min", "entropy_std",
        "entropy_text", "entropy_data", "entropy_rdata", "entropy_rsrc",
        "sliding_window_max", "sliding_window_mean", "sliding_window_std",
        "entry_point_entropy",
    ]  # 12
    opcode_count_cols  = [f"op_{o}" for o in OPCODES_TRACKED]  # 50
    opcode_ratio_cols  = [
        "mov_ratio", "call_ratio", "jmp_ratio", "xor_ratio", "push_ratio",
        "pop_ratio", "ret_ratio", "cmp_ratio", "arithmetic_ratio", "control_flow_ratio",
    ]  # 10
    opcode_agg_cols    = ["opcode_diversity", "total_opcodes", "unique_opcodes"]  # 3
    bigram_cols        = [f"bg_{a}_{b}" for (a, b) in BIGRAMS_TRACKED]  # 20
    pe_cols = [
        "section_count", "imports_count", "exports_count", "suspicious_import_count",
        "has_tls", "has_debug", "has_resources", "has_relocations",
        "packer_signature", "text_size", "data_size", "rdata_size", "rsrc_size",
        "text_vsize_ratio", "data_vsize_ratio", "image_size",
        "entry_point_in_text", "number_of_symbols", "timestamp_valid",
    ]  # 19
    order = (
        entropy_cols + opcode_count_cols + opcode_ratio_cols +
        opcode_agg_cols + bigram_cols + pe_cols
    )
    assert len(order) == 114, f"expected 114 feature columns, got {len(order)}"
    return order


def extract_features(path: str, label: Optional[int] = None) -> Optional[FeatureVector]:
    """Extract the full 114-feature vector for a single PE file.

    Returns None if the file is not a parseable PE. All exceptions are caught
    per-sample so that batch extraction over thousands of files doesn't abort
    on a single malformed binary.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        sha256 = hashlib.sha256(raw).hexdigest()

        pe = pefile.PE(data=raw, fast_load=False)

        fv = FeatureVector(sha256=sha256, label=label)
        _entropy_features(fv, raw, pe)
        _opcode_features(fv, pe)
        _pe_structural_features(fv, pe, raw)
        return fv
    except (pefile.PEFormatError, OSError, ValueError):
        return None
    except Exception:                      # pragma: no cover — last-line defensive
        return None


# ---------------------------------------------------------------------------
# Entropy channel
# ---------------------------------------------------------------------------

def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    return float(-(nz * np.log2(nz)).sum())


def _sliding_window_entropy(data: bytes, window: int = 4096, step: int = 2048) -> np.ndarray:
    if len(data) < window:
        return np.array([_shannon_entropy(data)])
    out = []
    for i in range(0, len(data) - window + 1, step):
        out.append(_shannon_entropy(data[i:i + window]))
    return np.array(out) if out else np.array([0.0])


def _entropy_features(fv: FeatureVector, raw: bytes, pe: pefile.PE) -> None:
    # Global — per-section entropies used for the summary stats.
    section_entropies = [s.get_entropy() for s in pe.sections] or [0.0]
    fv.features["entropy_mean"] = float(np.mean(section_entropies))
    fv.features["entropy_max"]  = float(np.max(section_entropies))
    fv.features["entropy_min"]  = float(np.min(section_entropies))
    fv.features["entropy_std"]  = float(np.std(section_entropies))

    # Named sections.
    for feat_name, section_name in (
        ("entropy_text",  b".text"),
        ("entropy_data",  b".data"),
        ("entropy_rdata", b".rdata"),
        ("entropy_rsrc",  b".rsrc"),
    ):
        fv.features[feat_name] = 0.0
        for s in pe.sections:
            if s.Name.rstrip(b"\x00").lower() == section_name.lower():
                fv.features[feat_name] = float(s.get_entropy())
                break

    # Sliding window over raw bytes — sensitive to localised packing.
    sw = _sliding_window_entropy(raw)
    fv.features["sliding_window_max"]  = float(sw.max())
    fv.features["sliding_window_mean"] = float(sw.mean())
    fv.features["sliding_window_std"]  = float(sw.std())

    # Entry-point neighbourhood — packers concentrate randomness here.
    fv.features["entry_point_entropy"] = _entry_point_entropy(pe)


def _entry_point_entropy(pe: pefile.PE, size: int = 512) -> float:
    try:
        ep_rva = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        for s in pe.sections:
            start = s.VirtualAddress
            end   = start + max(s.Misc_VirtualSize, s.SizeOfRawData)
            if start <= ep_rva < end:
                offset = s.PointerToRawData + (ep_rva - start)
                chunk = pe.__data__[offset:offset + size]
                return _shannon_entropy(chunk)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Opcode channel — static disassembly via Capstone
# ---------------------------------------------------------------------------

def _pick_disassembler(pe: pefile.PE) -> Cs:
    is_64 = bool(pe.OPTIONAL_HEADER.Magic == 0x20B) or (
        hasattr(pe, "FILE_HEADER")
        and pe.FILE_HEADER.Machine in (0x8664, 0xAA64)      # AMD64 / ARM64
    )
    md = Cs(CS_ARCH_X86, CS_MODE_64 if is_64 else CS_MODE_32)
    md.detail = False
    return md


def _iter_opcodes(pe: pefile.PE, max_bytes: int = 2_000_000) -> List[str]:
    """Return the ordered list of opcode mnemonics across all executable sections.

    Capped at `max_bytes` of disassembly to keep extraction bounded on
    pathological binaries (some packers inflate .text to hundreds of MB).
    """
    md = _pick_disassembler(pe)
    out: List[str] = []
    budget = max_bytes
    for s in pe.sections:
        if not (s.Characteristics & 0x20000000):    # IMAGE_SCN_MEM_EXECUTE
            continue
        code = s.get_data()[:budget]
        if not code:
            continue
        base = pe.OPTIONAL_HEADER.ImageBase + s.VirtualAddress
        for insn in md.disasm(code, base):
            out.append(insn.mnemonic.lower())
        budget -= len(code)
        if budget <= 0:
            break
    return out


def _opcode_features(fv: FeatureVector, pe: pefile.PE) -> None:
    mnemonics = _iter_opcodes(pe)
    total = len(mnemonics)
    counts = Counter(mnemonics)

    # Per-opcode counts.
    for op in OPCODES_TRACKED:
        fv.features[f"op_{op}"] = float(counts.get(op, 0))

    # Ratios — guard against div-by-zero.
    denom = float(total) if total else 1.0
    fv.features["mov_ratio"]           = counts.get("mov", 0)  / denom
    fv.features["call_ratio"]          = counts.get("call", 0) / denom
    fv.features["jmp_ratio"]           = counts.get("jmp", 0)  / denom
    fv.features["xor_ratio"]           = counts.get("xor", 0)  / denom
    fv.features["push_ratio"]          = counts.get("push", 0) / denom
    fv.features["pop_ratio"]           = counts.get("pop", 0)  / denom
    fv.features["ret_ratio"]           = counts.get("ret", 0)  / denom
    fv.features["cmp_ratio"]           = counts.get("cmp", 0)  / denom

    arith = sum(counts.get(o, 0) for o in ("add", "sub", "mul", "div", "inc", "dec"))
    fv.features["arithmetic_ratio"] = arith / denom

    ctrl = sum(
        counts.get(o, 0)
        for o in ("jmp", "je", "jne", "jz", "jnz", "jl", "jle", "jg", "jge",
                  "ja", "jb", "call", "ret", "loop")
    )
    fv.features["control_flow_ratio"] = ctrl / denom

    # Aggregate opcode statistics.
    fv.features["total_opcodes"]    = float(total)
    fv.features["unique_opcodes"]   = float(len(counts))
    fv.features["opcode_diversity"] = (len(counts) / denom) if total else 0.0

    # Bigrams — capture short instruction sequences that survive obfuscation.
    if len(mnemonics) >= 2:
        bg_counter: Counter = Counter(zip(mnemonics[:-1], mnemonics[1:]))
    else:
        bg_counter = Counter()
    for (a, b) in BIGRAMS_TRACKED:
        fv.features[f"bg_{a}_{b}"] = float(bg_counter.get((a, b), 0))


# ---------------------------------------------------------------------------
# PE structural channel
# ---------------------------------------------------------------------------

def _pe_structural_features(fv: FeatureVector, pe: pefile.PE, raw: bytes) -> None:
    fv.features["section_count"] = float(len(pe.sections))

    # Imports.
    imports_count = 0
    suspicious = 0
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                imports_count += 1
                if imp.name and imp.name.lower() in SUSPICIOUS_IMPORTS:
                    suspicious += 1
    fv.features["imports_count"]           = float(imports_count)
    fv.features["suspicious_import_count"] = float(suspicious)

    # Exports.
    exports = 0
    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        exports = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
    fv.features["exports_count"] = float(exports)

    # Directory presence flags.
    fv.features["has_tls"]         = float(hasattr(pe, "DIRECTORY_ENTRY_TLS"))
    fv.features["has_debug"]       = float(hasattr(pe, "DIRECTORY_ENTRY_DEBUG"))
    fv.features["has_resources"]   = float(hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"))
    fv.features["has_relocations"] = float(hasattr(pe, "DIRECTORY_ENTRY_BASERELOC"))

    # Packer signature via section-name blacklist.
    packer = 0
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00")
        if name in PACKER_SECTIONS or any(sig in name for sig in PACKER_SECTIONS):
            packer = 1
            break
    fv.features["packer_signature"] = float(packer)

    # Section sizes.
    section_sizes = {b".text": 0, b".data": 0, b".rdata": 0, b".rsrc": 0}
    vsize_by_name: Dict[bytes, Tuple[int, int]] = {}
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").lower()
        if name in section_sizes:
            section_sizes[name] = int(s.SizeOfRawData)
        vsize_by_name[name] = (int(s.Misc_VirtualSize), int(s.SizeOfRawData))
    fv.features["text_size"]  = float(section_sizes[b".text"])
    fv.features["data_size"]  = float(section_sizes[b".data"])
    fv.features["rdata_size"] = float(section_sizes[b".rdata"])
    fv.features["rsrc_size"]  = float(section_sizes[b".rsrc"])

    # Virtual / raw size ratios — very large ratios suggest packed sections.
    fv.features["text_vsize_ratio"] = _safe_ratio(*vsize_by_name.get(b".text", (0, 0)))
    fv.features["data_vsize_ratio"] = _safe_ratio(*vsize_by_name.get(b".data", (0, 0)))

    fv.features["image_size"] = float(pe.OPTIONAL_HEADER.SizeOfImage)

    # Entry point sits inside .text?
    ep_rva = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    ep_in_text = 0
    for s in pe.sections:
        if s.Name.rstrip(b"\x00").lower() == b".text":
            if s.VirtualAddress <= ep_rva < s.VirtualAddress + max(
                s.Misc_VirtualSize, s.SizeOfRawData
            ):
                ep_in_text = 1
            break
    fv.features["entry_point_in_text"] = float(ep_in_text)

    fv.features["number_of_symbols"] = float(pe.FILE_HEADER.NumberOfSymbols)

    # Timestamp sanity — future / zero / pre-1995 is suspicious.
    ts = pe.FILE_HEADER.TimeDateStamp
    fv.features["timestamp_valid"] = float(788918400 <= ts <= 1893456000)  # 1995..2030


def _safe_ratio(vsize: int, rsize: int) -> float:
    if rsize <= 0:
        return 0.0
    return float(vsize) / float(rsize)
