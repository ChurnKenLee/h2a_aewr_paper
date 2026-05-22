import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import difflib
    import json

    import marimo as mo
    import polars as pl

    from h2a.paths import CODE
    import split_do_r_scripts as rsplit

    return CODE, difflib, mo, pl, rsplit


@app.cell
def _(CODE):
    project_root = CODE.parent
    split_dir = CODE / "r_split"
    metadata = split_dir / "_metadata"
    patch_path = CODE / "r_split_patches.jsonl"
    return metadata, patch_path, project_root, split_dir


@app.cell
def _(mo):
    regenerate_button = mo.ui.run_button(
        label="Regenerate split modules",
        kind="success",
        tooltip="Parse Do/*.R, apply enabled patches, rewrite code/r_split, and refresh metadata.",
    )
    regenerate_button
    return (regenerate_button,)


@app.cell
def _(patch_path, project_root, regenerate_button, rsplit):
    regenerate_summary = None
    if regenerate_button.value:
        regenerate_summary = rsplit.generate_split(
            project_root=project_root,
            patch_path=patch_path,
        )
    return


@app.cell
def _(metadata, pl):
    modules = pl.read_parquet(metadata / "modules.parquet")
    sections = pl.read_parquet(metadata / "sections.parquet")
    artifacts = pl.read_parquet(metadata / "artifacts.parquet")
    tokens = pl.read_parquet(metadata / "tokens.parquet")
    return (modules,)


@app.cell
def _(mo, modules):
    module_picker = mo.ui.dropdown(
        options=modules["path"].to_list(),
        value=modules["path"].to_list()[0],
        searchable=True,
        label="Module",
    )
    module_picker
    return (module_picker,)


@app.cell
def _(mo, patch_path, rsplit):
    patches = rsplit.read_patches(patch_path)
    mo.md("### Active Patch Table")
    patch_table = mo.ui.data_editor(
        patches,
        editable_columns=[],
    )
    patch_table
    return


@app.cell
def _(mo):
    op_picker = mo.ui.dropdown(
        options=["insert_before", "insert_after", "replace_regex", "replace_lines"],
        value="insert_before",
        label="Patch op",
    )
    selector_input = mo.ui.text_area(
        label="Selector",
        rows=4,
        full_width=True,
        placeholder="Text, regex, or 1-based line range such as 10:14",
    )
    payload_input = mo.ui.text_area(
        label="Payload",
        rows=10,
        full_width=True,
        placeholder="R code to insert or use as replacement",
    )
    reason_input = mo.ui.text(
        label="Reason",
        full_width=True,
        placeholder="Short audit note",
    )
    enabled_input = mo.ui.checkbox(value=True, label="Enabled")

    mo.vstack([op_picker, selector_input, payload_input, reason_input, enabled_input])
    return (
        enabled_input,
        op_picker,
        payload_input,
        reason_input,
        selector_input,
    )


@app.cell
def _(
    enabled_input,
    module_picker,
    op_picker,
    payload_input,
    reason_input,
    selector_input,
):
    proposed_patch = {
        "module": module_picker.value,
        "op": op_picker.value,
        "selector": selector_input.value,
        "payload": payload_input.value,
        "reason": reason_input.value,
        "enabled": enabled_input.value,
    }
    return (proposed_patch,)


@app.cell
def _(mo):
    preview_patch_button = mo.ui.run_button(
        label="Preview patch against current module",
        tooltip="Show a diff only; do not write the patch table or regenerate.",
    )
    apply_patch_button = mo.ui.run_button(
        label="Append patch and regenerate",
        kind="warn",
        tooltip="Append this patch to code/r_split_patches.jsonl and regenerate code/r_split.",
    )
    mo.hstack([preview_patch_button, apply_patch_button])
    return apply_patch_button, preview_patch_button


@app.cell
def _(difflib, preview_patch_button, proposed_patch, rsplit, split_dir):
    patch_preview_diff = ""
    if preview_patch_button.value:
        module_file = split_dir / proposed_patch["module"]
        current = rsplit.strip_generated_header(module_file.read_text(encoding="utf-8"))
        patched = rsplit.apply_patch_text(current, proposed_patch)
        patch_preview_diff = "\n".join(
            difflib.unified_diff(
                current.splitlines(),
                patched.splitlines(),
                fromfile=f"current/{proposed_patch['module']}",
                tofile=f"patched/{proposed_patch['module']}",
                lineterm="",
            )
        )
    return (patch_preview_diff,)


@app.cell
def _(apply_patch_button, patch_path, project_root, proposed_patch, rsplit):
    apply_summary = None
    if apply_patch_button.value:
        rsplit.append_patch(patch_path, proposed_patch)
        apply_summary = rsplit.generate_split(
            project_root=project_root,
            patch_path=patch_path,
        )
    return (apply_summary,)


@app.cell
def _(apply_summary, mo, patch_preview_diff):
    if patch_preview_diff:
        patch_output = mo.ui.code_editor(
            value=patch_preview_diff,
            language="diff",
            disabled=True,
            min_height=350,
            show_copy_button=True,
        )
    elif apply_summary is not None:
        patch_output = mo.md(
            f"Regenerated {apply_summary['generated_modules']} modules with "
            f"{apply_summary['enabled_patches']} enabled patches."
        )
    else:
        patch_output = mo.md("Patch preview and apply output will appear here.")
    patch_output
    return


@app.cell
def _(module_picker, project_root, rsplit):
    source_vs_final_diff = rsplit.module_diff(
        module_picker.value,
        project_root=project_root,
    )
    return (source_vs_final_diff,)


@app.cell
def _(mo, source_vs_final_diff):
    mo.md("### Source Slice vs Final Generated Module")
    mo.ui.code_editor(
        value=source_vs_final_diff,
        language="diff",
        disabled=True,
        min_height=500,
        show_copy_button=True,
    )
    return


@app.cell
def _(mo, module_picker, rsplit, split_dir):
    final_code = rsplit.strip_generated_header(
        (split_dir / module_picker.value).read_text(encoding="utf-8")
    )
    mo.md("### Final Generated Module Body")
    mo.ui.code_editor(
        value=final_code,
        language="r",
        disabled=True,
        min_height=500,
        show_copy_button=True,
    )
    return


if __name__ == "__main__":
    app.run()
