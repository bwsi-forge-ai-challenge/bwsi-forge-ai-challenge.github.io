# Course Material Conventions

## Page Structure

Each lesson page follows this fixed order:

1. **Title** (H1): `Month X: Week Y - Topic Title`
2. **Opening paragraph**: One to two sentences connecting previous material to this lesson's purpose.
3. **Numbered sections** (H2): Core content, each separated by `<hr>`.
4. **Concept checks**: Placed immediately after the section they test.
5. **Interactive sandbox** (if applicable): One per page maximum.
6. **Recommended resources**: Three external links with one-sentence descriptions.
7. **Navigation buttons**: Previous/Next at the bottom.

## Writing Rules

- No emoji anywhere in the content.
- No analogies or metaphors. State what things are, not what they are "like."
- No filler phrases ("Let's dive in", "Without further ado", "Think of it as").
- No first-person plural cheerleading ("We're going to love this", "This is exciting").
- Define every technical term on first use in bold.
- Every claim should be verifiable from the math or the code.

## Concept Checks

- One per major section.
- Must require computation or derivation, not recall.
- Answer should reference the specific formula that produces the result.
- Keep question and answer each to two sentences maximum.

## Interactive Sandboxes

- One per page maximum.
- Must demonstrate a single mathematical operation.
- Input on the left, output on the right in a two-column table.
- Include a reset/clear button.
- Include a status line explaining what the output represents.
- Support both mouse and touch events.

## Math Notation

- Use monospace spans for inline expressions.
- Use bold for named equations (y = xW + b).
- Always state matrix/vector shapes in [rows, cols] format.
- Use Unicode subscripts (x₁) over HTML sub tags when possible.

## Navigation

- Previous button: left-aligned, normal weight.
- Next button: right-aligned, bold weight.
- File naming: `NN_topic_name.html` (zero-padded two-digit prefix, underscores).

## Section Numbering

- Sections use sequential integers starting at 1.
- Subsections use H3 with descriptive titles, no numbering.
