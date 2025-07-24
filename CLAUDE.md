# CLAUDE.md - Transformer LLM From Scratch

Educational GPT-style transformer implementation from scratch.

## Commands

```bash
# Setup
pip install -r requirements.txt

# Test
python test_model.py

# Train
python scripts/train.py --model_size tiny --epochs 5 --batch_size 4     # Fast
python scripts/train.py --model_size small --epochs 10 --batch_size 8   # Better

# Generate
python scripts/generate.py --model_path checkpoints/best_model.pt --prompt "AI future"
python scripts/generate.py --model_path checkpoints/best_model.pt --interactive

# Notebooks
jupyter notebook notebooks/01_attention_mechanism.ipynb
```

## Architecture

- **Purpose**: Educational GPT implementation
- **Type**: Decoder-only transformer 
- **Sizes**: Tiny (2L), Small (6L), Medium (12L), Large (24L)
- **Features**: Attention visualization, multiple sampling, educational notebooks

## Jupyter Notebook Rules

**CRITICAL - NEVER VIOLATE:**

### Required Pattern
```
[Markdown Cell] ← Explanation
[Code Cell]     ← Pure code
[Markdown Cell] ← Next explanation  
[Code Cell]     ← Next code
```

### Code Cells Rules
1. **Every code cell MUST have preceding markdown explanation**
2. **Code cells: ONLY executable code - NO EXPLANATIONS**
3. **Every code cell MUST be verified runnable before completion**
4. **Code cells must be concise, minimal comments**
5. **Break large code blocks into smaller focused cells**
6. **Always check for missing explanatory markdown cells before code blocks**

### Markdown Cells Rules
0. **Markdown cells: explain things from first principles**
1. **Markdown cells  must be concise - remove verbose explanations**
2. **Markdown cells: ONLY text/explanations - NO CODE**

### Validation Checklist
- [ ] Every code cell has preceding markdown cell explaining the topic
- [ ] No code in markdown cells
- [ ] No explanations in code cells
- [ ] No duplicate cells
- [ ] No orphaned code cells
- [ ] All code cells execute without errors
- [ ] Markdown cells are concise, not verbose

### Detection Patterns
- **Problem**: Code cell starting with `# [Description]` without preceding markdown
- **Problem**: Markdown cell containing executable code  
- **Problem**: Multiple cells with identical content
- **Problem**: Code cell with descriptive text like "This code implements..."
- **Problem**: Markdown cell with `def function():` or similar executable code

**Fix**: Always add explanatory markdown cell before code cells. Verify cell types are correct.
