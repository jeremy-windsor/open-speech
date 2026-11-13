"""Convert Parakeet's published fp32 ONNX encoder to fp16 for the GPU, once, into a cache folder.

There is no published fp16 export that onnx-asr loads, and a naive conversion fails: without shape inference the
model doesn't load (type errors), and without fp32 islands the encoded lengths come out wrong or the encoder output
is garbage. This conversion keeps in fp32 every node outside the conformer layers (subsampling, the length maths,
the positional encoding), plus Softmax, Div, Where and LayerNormalization, and casts the encoder output back to
fp32. On a 0.6B TDT encoder it gives the fp32 model's word error rate at half the memory.

The converted files are written to a temporary folder beside their final one and renamed into place last, so a
conversion that fails or is stopped half way never leaves a folder a later load trusts. The folder's name carries
the source revision and CONVERTER_VERSION: a new revision or a changed conversion builds a new folder, and the old
ones are removed after it.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump when the conversion changes, so caches made by an older one are rebuilt.
CONVERTER_VERSION = 1
# Beside the Hugging Face snapshot folders of the source repository.
FP16_DIR_NAME = "open-speech-fp16"
TEMP_PREFIX = ".tmp-"
# A temporary folder older than this is left over from a conversion that died; a running one is never this old.
STALE_TEMP_S = 6 * 3600

ENCODER = "encoder-model.onnx"
ENCODER_FP16 = "encoder-model.fp16.onnx"
DECODER = "decoder_joint-model.onnx"
# onnx-asr loads the files of one quantization by name: the decoder stays fp32 under the fp16 name.
DECODER_FP16 = "decoder_joint-model.fp16.onnx"
COPIED_FILES = ("config.json", "vocab.txt", "nemo128.onnx")
# The source files the conversion reads (encoder-model.onnx.data holds the encoder's weights).
SOURCE_FILES = (ENCODER, ENCODER + ".data", DECODER, *COPIED_FILES)
FP16_FILES = (ENCODER_FP16, ENCODER_FP16 + ".data", DECODER_FP16, *COPIED_FILES)

# Ops kept in fp32 inside the conformer layers, on top of onnxconverter-common's default list.
FP32_OPS = ("Softmax", "Div", "Where", "LayerNormalization")
# Nodes outside the conformer layers stay fp32: their names don't start with this.
FP16_NODE_PREFIX = "/layers."


def fp16_folder_name(source_revision: str) -> str:
    return f"{source_revision}-c{CONVERTER_VERSION}"


def is_complete(folder: Path) -> bool:
    return folder.is_dir() and all((folder / name).is_file() for name in FP16_FILES)


def cached_fp16_dir(repo_dir: Path, source_revision: str) -> Path | None:
    """The finished fp16 folder for this revision and converter, or None when there is none yet."""
    folder = repo_dir / FP16_DIR_NAME / fp16_folder_name(source_revision)
    return folder if is_complete(folder) else None


def convert_encoder(source: Path, target: Path) -> None:
    """Write ``target`` (an .onnx with its weights in ``target.name + '.data'``), the fp16 copy of ``source``.

    Needs onnx and onnxconverter-common (the cuda extra), and RAM for the fp32 weights and their fp16 copy.
    """
    import onnx
    from onnx import TensorProto, helper
    from onnx.external_data_helper import load_external_data_for_model
    from onnxconverter_common import float16

    # The graph without its weights stays under protobuf's 2 GB limit, so shape inference runs in memory; it reads
    # each initializer's type and dims, not its data.
    model = onnx.load(str(source), load_external_data=False)
    model = onnx.shape_inference.infer_shapes(model)
    load_external_data_for_model(model, str(source.parent))

    block_nodes = [node.name for node in model.graph.node if not node.name.startswith(FP16_NODE_PREFIX)]
    with warnings.catch_warnings():
        # One warning per weight below fp16's smallest normal value: thousands of lines saying it is clamped.
        warnings.simplefilter("ignore", UserWarning)
        model = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=True,
            node_block_list=block_nodes,
            op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + list(FP32_OPS),
        )

    # The converter leaves some Cast(to=float) nodes whose output the inferred value_info now types as fp16; make
    # each Cast agree with the type its output is declared with.
    declared = {value.name: value.type.tensor_type.elem_type for value in [*model.graph.value_info, *model.graph.output]}
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        to = next(attribute for attribute in node.attribute if attribute.name == "to")
        want = declared.get(node.output[0])
        if want is not None and want != to.i:
            to.i = want
    # An output the converter left fp16 is cast back, so callers get fp32 as from the source model.
    for output in model.graph.output:
        if output.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        renamed = output.name + "_fp16"
        for node in model.graph.node:
            node.output[:] = [renamed if name == output.name else name for name in node.output]
            node.input[:] = [renamed if name == output.name else name for name in node.input]
        model.graph.node.append(
            helper.make_node("Cast", [renamed], [output.name], to=TensorProto.FLOAT, name=f"final_cast_{output.name}")
        )
        output.type.tensor_type.elem_type = TensorProto.FLOAT

    onnx.save_model(
        model,
        str(target),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=target.name + ".data",
        size_threshold=1024,
    )


def _remove_stale(fp16_root: Path, keep: str) -> None:
    """Remove fp16 folders of other revisions or converter versions, and temporary folders a dead conversion left."""
    now = time.time()
    for entry in fp16_root.iterdir():
        if entry.name == keep or not entry.is_dir():
            continue
        if entry.name.startswith(TEMP_PREFIX):
            try:
                if now - entry.stat().st_mtime < STALE_TEMP_S:
                    continue  # perhaps another process converting now
            except OSError:
                continue
        shutil.rmtree(entry, ignore_errors=True)
        logger.info("Removed old Parakeet fp16 cache %s", entry)


def _stage_encoder(source_dir: Path, staging: Path) -> Path:
    """Put the fp32 encoder and its weights file in ``staging`` as plain files, and return the encoder's path there.

    The Hugging Face cache keeps each snapshot file as a symbolic link into the repository's ``blobs`` folder, and
    onnx refuses external data whose real path leaves the model's own folder, or that has more than one hard link.
    So both files are copied (about 2.4 GB, once; the folder is removed after the conversion).
    """
    staging.mkdir()
    for name in (ENCODER, ENCODER + ".data"):
        shutil.copyfile(os.path.realpath(source_dir / name), staging / name)
    return staging / ENCODER


def ensure_fp16(source_dir: Path, repo_dir: Path, source_revision: str, *, convert=convert_encoder) -> Path:
    """Return the fp16 folder for ``source_revision``, converting ``source_dir``'s fp32 files into it first if needed.

    ``repo_dir`` is the source repository's cache folder; the fp16 folders live in its FP16_DIR_NAME subfolder. The
    conversion writes to a temporary folder there and renames it into place only when every file is written.
    """
    existing = cached_fp16_dir(repo_dir, source_revision)
    if existing is not None:
        return existing
    missing = [name for name in SOURCE_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Parakeet source files missing from {source_dir}: {', '.join(missing)}")

    fp16_root = repo_dir / FP16_DIR_NAME
    fp16_root.mkdir(parents=True, exist_ok=True)
    final = fp16_root / fp16_folder_name(source_revision)
    temp = fp16_root / f"{TEMP_PREFIX}{final.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    temp.mkdir()
    started = time.monotonic()
    logger.info("Converting Parakeet's encoder to fp16 (once; about a minute and 4 GB of RAM)")
    try:
        staging = temp.with_name(temp.name + "-source")
        try:
            convert(_stage_encoder(source_dir, staging), temp / ENCODER_FP16)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        shutil.copyfile(source_dir / DECODER, temp / DECODER_FP16)
        for name in COPIED_FILES:
            shutil.copyfile(source_dir / name, temp / name)
        if not is_complete(temp):
            raise RuntimeError("Parakeet fp16 conversion did not write every file")
        if final.exists():
            # A folder without every file is a leftover no load trusts (is_complete); replace it.
            if is_complete(final):
                shutil.rmtree(temp, ignore_errors=True)
                return final
            shutil.rmtree(final)
        os.replace(temp, final)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    logger.info("Parakeet fp16 encoder written to %s in %.0f s", final, time.monotonic() - started)
    _remove_stale(fp16_root, keep=final.name)
    return final
