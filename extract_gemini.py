import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import sys

    with open('conversation.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data['chunkedPrompt']['chunks']

    lines = []
    lines.append("# Gemini AI Studio Conversation Transcript\n")

    for i, chunk in enumerate(chunks):
        # Skip thoughts
        if chunk.get('isThought', False):
            continue
        # Skip code execution chunks
        if 'executableCode' in chunk or 'codeExecutionResult' in chunk:
            continue
        # Skip chunks with no text
        text = chunk.get('text', '')
        if not text or not text.strip():
            continue

        role = chunk.get('role', 'unknown').upper()
        if role == 'MODEL':
            label = '## Assistant'
        elif role == 'USER':
            label = '## User'
        else:
            label = f'## {role}'

        lines.append(label)
        lines.append('')
        lines.append(text.strip())
        lines.append('')
        lines.append('---')
        lines.append('')

    output = '\n'.join(lines)
    with open('gemini_transcript.md', 'w',  encoding='utf-8') as f:
        f.write(output)

    print(f"Done. {len([c for c in chunks if not c.get('isThought') and 'executableCode' not in c and 'codeExecutionResult' not in c and c.get('text','').strip()])} chunks written.")
    print(f"Output size: {len(output):,} chars")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
