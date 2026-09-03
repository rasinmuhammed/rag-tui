# Regenerate the README demo with:  vhs assets/demo/doctor.tape
# Run it from the repository root.

Output assets/doctor-demo.gif

Set Shell bash
Set FontSize 15
Set Width 1250
Set Height 780
Set Padding 18
Set TypingSpeed 35ms
Set Theme "Catppuccin Mocha"

# Put the local build on PATH without showing the plumbing.
Hide
Type "export PATH=$PWD/.venv/bin:$PATH; clear"
Enter
Sleep 1s
Show

Sleep 1s
Type "rag-tui doctor --file assets/demo/knowledge-base.md --chunk-size 25 --strategy paragraph --top 3"
Sleep 800ms
Enter
Sleep 6s

Type "clear"
Enter
Sleep 500ms

# Same document, structure-aware chunking. The headings stop becoming chunks.
Type "rag-tui doctor --file assets/demo/knowledge-base.md --strategy markdown"
Sleep 800ms
Enter
Sleep 6s
